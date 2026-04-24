"""
baselines/graphrag_official/retriever.py
------------------------------------------
Microsoft GraphRAG 官方基线：检索器。

功能：
  1. local_search(query)  -> 微观检索（实体级别，适合具体问题）
  2. global_search(query) -> 宏观检索（社区级别，适合全局问题）
  3. 返回答案 + context token 数

前置条件：
  - 已运行 indexer.py 构建索引
  - graphrag_workspace/output/ 中有 parquet 文件

用法：
  from baselines.graphrag_official.retriever import GraphRAGRetriever
  retriever = GraphRAGRetriever()
  result = retriever.local_search("What is the relationship between X and Y?")
  result = retriever.global_search("What are the main themes in the corpus?")
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# GraphRAG 工作目录（相对于本文件）
WORKSPACE_DIR = Path(__file__).parent / "graphrag_workspace"


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class GraphRAGSearchResult:
    """GraphRAG 检索结果。"""
    query: str
    answer: str
    search_type: str          # "local" 或 "global"
    context_text: str = ""    # 检索到的上下文文本
    context_tokens: int = 0   # context token 数
    latency_ms: float = 0.0   # 检索延迟（毫秒）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"查询：{self.query}\n"
            f"类型：{self.search_type}\n"
            f"答案：{self.answer[:200]}{'...' if len(self.answer) > 200 else ''}\n"
            f"Context tokens：{self.context_tokens}\n"
            f"延迟：{self.latency_ms:.1f} ms"
        )


# ---------------------------------------------------------------------------
# Token 计数工具
# ---------------------------------------------------------------------------

def _get_token_counter():
    """获取 tiktoken 计数器。"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count_tokens(text: str) -> int:
            return len(enc.encode(text))
        return count_tokens
    except ImportError:
        logger.warning("tiktoken 未安装，使用简单空格分词计数")
        def count_tokens(text: str) -> int:
            return len(text.split())
        return count_tokens


# ---------------------------------------------------------------------------
# GraphRAG 检索器
# ---------------------------------------------------------------------------

class GraphRAGRetriever:
    """
    Microsoft GraphRAG 官方检索器。

    封装 local search 和 global search 两种检索模式。

    Parameters
    ----------
    workspace_dir : str, optional
        GraphRAG 工作目录（默认为 ./graphrag_workspace）
    api_key : str, optional
        OpenAI API key（若未设置环境变量则需要传入）
    """

    def __init__(
        self,
        workspace_dir: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.workspace_dir = Path(workspace_dir) if workspace_dir else WORKSPACE_DIR
        self._token_counter = _get_token_counter()

        # 设置 API key
        if api_key:
            os.environ["GRAPHRAG_API_KEY"] = api_key

        # 延迟加载 graphrag 组件
        self._local_search_engine = None
        self._global_search_engine = None
        self._context_builder_local = None
        self._context_builder_global = None
        self._llm = None
        self._token_encoder = None

        # 检查工作目录
        self._check_workspace()

    def _check_workspace(self) -> None:
        """检查工作目录和索引文件是否存在。"""
        if not self.workspace_dir.exists():
            raise FileNotFoundError(
                f"工作目录不存在：{self.workspace_dir}\n"
                "请先运行 setup.sh 初始化工作目录"
            )

        output_dir = self.workspace_dir / "output"
        if not output_dir.exists():
            raise FileNotFoundError(
                f"索引输出目录不存在：{output_dir}\n"
                "请先运行 indexer.py 构建索引"
            )

        parquet_files = list(output_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"未找到索引文件（*.parquet）：{output_dir}\n"
                "请先运行 indexer.py 构建索引"
            )

        logger.info(f"  工作目录：{self.workspace_dir}")
        logger.info(f"  找到 {len(parquet_files)} 个 parquet 文件")

    def _load_graphrag_components(self) -> None:
        """
        加载 GraphRAG 的搜索组件。

        GraphRAG 的 API 在不同版本间有较大变化，这里做了兼容处理。
        """
        if self._llm is not None:
            return

        try:
            import graphrag
            graphrag_version = getattr(graphrag, "__version__", "unknown")
            logger.info(f"  GraphRAG 版本：{graphrag_version}")
        except ImportError:
            raise ImportError(
                "graphrag 未安装，请运行：\n"
                "  pip install graphrag"
            )

        # 加载配置
        settings_file = self.workspace_dir / "settings.yaml"
        if not settings_file.exists():
            raise FileNotFoundError(f"配置文件不存在：{settings_file}")

        try:
            self._load_search_engines_v1()
        except Exception as e1:
            logger.warning(f"  v1 API 加载失败：{e1}，尝试 v2 API...")
            try:
                self._load_search_engines_v2()
            except Exception as e2:
                logger.warning(f"  v2 API 加载失败：{e2}，尝试命令行模式...")
                # 回退到命令行模式
                self._use_cli_mode = True
                logger.info("  将使用命令行模式执行搜索")

    def _load_search_engines_v1(self) -> None:
        """
        加载 GraphRAG 搜索引擎（v0.3.x API）。
        """
        import pandas as pd
        from graphrag.config import load_config
        from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
        from graphrag.query.indexer_adapters import (
            read_indexer_covariates,
            read_indexer_entities,
            read_indexer_relationships,
            read_indexer_reports,
            read_indexer_text_units,
        )
        from graphrag.query.llm.oai.chat_openai import ChatOpenAI
        from graphrag.query.llm.oai.embedding import OpenAIEmbedding
        from graphrag.query.llm.oai.typing import OpenaiApiType
        from graphrag.query.structured_search.global_search.community_context import (
            GlobalCommunityContext,
        )
        from graphrag.query.structured_search.global_search.search import GlobalSearch
        from graphrag.query.structured_search.local_search.mixed_context import (
            LocalSearchMixedContext,
        )
        from graphrag.query.structured_search.local_search.search import LocalSearch
        from graphrag.vector_stores.lancedb import LanceDBVectorStore

        # 加载配置
        config = load_config(self.workspace_dir)
        api_key = os.environ.get("GRAPHRAG_API_KEY", config.llm.api_key or "")

        # 初始化 LLM
        self._llm = ChatOpenAI(
            api_key=api_key,
            model=config.llm.model,
            api_type=OpenaiApiType.OpenAI,
            max_retries=config.llm.max_retries,
        )

        # 初始化 Embedding
        text_embedder = OpenAIEmbedding(
            api_key=api_key,
            api_base=None,
            api_type=OpenaiApiType.OpenAI,
            model=config.embeddings.llm.model,
            deployment_name=config.embeddings.llm.deployment_name,
            max_retries=config.embeddings.llm.max_retries,
        )

        # 加载索引数据
        output_dir = self.workspace_dir / "output"
        # 找到最新的输出目录
        run_dirs = sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        if not run_dirs:
            raise FileNotFoundError(f"输出目录为空：{output_dir}")

        artifacts_dir = run_dirs[0] / "artifacts"
        if not artifacts_dir.exists():
            artifacts_dir = run_dirs[0]

        # 加载各类数据
        entity_df = pd.read_parquet(artifacts_dir / "create_final_nodes.parquet")
        entity_embedding_df = pd.read_parquet(artifacts_dir / "create_final_entities.parquet")
        relationship_df = pd.read_parquet(artifacts_dir / "create_final_relationships.parquet")
        report_df = pd.read_parquet(artifacts_dir / "create_final_community_reports.parquet")
        text_unit_df = pd.read_parquet(artifacts_dir / "create_final_text_units.parquet")

        entities = read_indexer_entities(entity_df, entity_embedding_df, community_level=2)
        relationships = read_indexer_relationships(relationship_df)
        reports = read_indexer_reports(report_df, entity_df, community_level=2)
        text_units = read_indexer_text_units(text_unit_df)

        # 向量存储
        description_embedding_store = LanceDBVectorStore(
            collection_name="default-entity-description",
        )
        description_embedding_store.connect(
            db_uri=str(artifacts_dir / "lancedb")
        )

        # Local Search
        local_context_builder = LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=text_embedder,
        )

        self._local_search_engine = LocalSearch(
            llm=self._llm,
            context_builder=local_context_builder,
            token_encoder=None,
            llm_params={"max_tokens": 2000, "temperature": 0},
            context_builder_params={
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 5,
                "conversation_history_user_turns_only": True,
                "top_k_mapped_entities": 10,
                "top_k_relationships": 10,
                "include_entity_rank": True,
                "include_relationship_weight": True,
                "include_community_rank": False,
                "return_candidate_context": False,
                "max_tokens": 12000,
            },
        )

        # Global Search
        global_context_builder = GlobalCommunityContext(
            community_reports=reports,
            entities=entities,
            token_encoder=None,
        )

        self._global_search_engine = GlobalSearch(
            llm=self._llm,
            context_builder=global_context_builder,
            token_encoder=None,
            max_data_tokens=12000,
            map_llm_params={"max_tokens": 1000, "temperature": 0},
            reduce_llm_params={"max_tokens": 2000, "temperature": 0},
            allow_general_knowledge=False,
            json_mode=True,
            context_builder_params={"use_community_summary": False, "shuffle_data": True},
            concurrent_coroutines=32,
            response_type="multiple paragraphs",
        )

        self._use_cli_mode = False
        logger.info("  ✓ GraphRAG 搜索引擎加载完成（v1 API）")

    def _load_search_engines_v2(self) -> None:
        """
        加载 GraphRAG 搜索引擎（v0.4+ API，使用新的模块结构）。
        """
        # v0.4+ 的 API 结构有所变化，尝试新的导入路径
        from graphrag.config.load_config import load_config
        from graphrag.query.factory import get_local_search_engine, get_global_search_engine

        config = load_config(self.workspace_dir)

        output_dir = self.workspace_dir / "output"
        run_dirs = sorted(output_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
        artifacts_dir = run_dirs[0] / "artifacts" if (run_dirs[0] / "artifacts").exists() else run_dirs[0]

        self._local_search_engine = get_local_search_engine(
            config=config,
            reports=None,
            text_units=None,
            entities=None,
            relationships=None,
            covariates=None,
            response_type="multiple paragraphs",
            description_embedding_store=None,
        )

        self._global_search_engine = get_global_search_engine(
            config=config,
            reports=None,
            entities=None,
            response_type="multiple paragraphs",
        )

        self._use_cli_mode = False
        logger.info("  ✓ GraphRAG 搜索引擎加载完成（v2 API）")

    def _cli_search(self, query: str, search_type: str) -> str:
        """
        使用命令行模式执行搜索（graphrag 3.0.x CLI 语法）。

        graphrag 3.0.x 语法：
            graphrag query <QUERY> --root <dir> --method <local|global>
        注意：query 是位置参数，不是 --query 选项。
        """
        import subprocess

        # graphrag 3.0.x: query 是位置参数
        cmd = [
            sys.executable, "-m", "graphrag", "query",
            query,
            "--root", str(self.workspace_dir),
            "--method", search_type,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env={**os.environ},
            )

            if result.returncode != 0:
                error_msg = result.stderr[-2000:] if result.stderr else "未知错误"
                logger.warning(f"  CLI 搜索失败：{error_msg}")
                return f"[搜索失败] {error_msg}"

            # graphrag 3.0.x 输出格式：直接输出答案文本（无特殊前缀）
            # 过滤掉 INFO/WARNING 日志行，保留实际答案
            output = result.stdout
            lines = output.split("\n")
            answer_lines = []
            skip_prefixes = ("INFO", "WARNING", "ERROR", "DEBUG", "SUCCESS:")
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # 跳过日志行
                if any(stripped.startswith(p) for p in skip_prefixes):
                    continue
                answer_lines.append(line)

            answer = "\n".join(answer_lines).strip()
            return answer if answer else output.strip()

        except subprocess.TimeoutExpired:
            return "[搜索超时]"
        except Exception as e:
            return f"[搜索异常] {e}"

    def local_search(
        self,
        query: str,
        conversation_history: Optional[List[dict]] = None,
    ) -> GraphRAGSearchResult:
        """
        执行 Local Search（微观检索，适合具体实体相关问题）。

        Parameters
        ----------
        query : str
            查询文本
        conversation_history : List[dict], optional
            对话历史

        Returns
        -------
        GraphRAGSearchResult
            检索结果，包含答案和 context token 数
        """
        t0 = time.perf_counter()

        try:
            self._load_graphrag_components()
        except Exception as e:
            logger.warning(f"  组件加载失败：{e}，使用 CLI 模式")
            self._use_cli_mode = True

        if getattr(self, "_use_cli_mode", True):
            answer = self._cli_search(query, "local")
            latency_ms = (time.perf_counter() - t0) * 1000
            context_tokens = self._token_counter(answer)
            return GraphRAGSearchResult(
                query=query,
                answer=answer,
                search_type="local",
                context_text=answer,
                context_tokens=context_tokens,
                latency_ms=latency_ms,
            )

        try:
            import asyncio

            async def _run():
                return await self._local_search_engine.asearch(
                    query,
                    conversation_history=conversation_history,
                )

            # 在同步上下文中运行异步搜索
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, _run())
                        search_result = future.result(timeout=120)
                else:
                    search_result = loop.run_until_complete(_run())
            except RuntimeError:
                search_result = asyncio.run(_run())

            latency_ms = (time.perf_counter() - t0) * 1000

            answer = str(search_result.response)
            context_text = ""
            context_tokens = 0

            # 提取 context 信息
            if hasattr(search_result, "context_data"):
                context_data = search_result.context_data
                if isinstance(context_data, dict):
                    context_parts = []
                    for key, val in context_data.items():
                        if hasattr(val, "to_string"):
                            context_parts.append(val.to_string())
                        elif isinstance(val, str):
                            context_parts.append(val)
                    context_text = "\n".join(context_parts)
                elif isinstance(context_data, str):
                    context_text = context_data

            if hasattr(search_result, "prompt_tokens"):
                context_tokens = search_result.prompt_tokens
            else:
                context_tokens = self._token_counter(context_text or answer)

            return GraphRAGSearchResult(
                query=query,
                answer=answer,
                search_type="local",
                context_text=context_text,
                context_tokens=context_tokens,
                latency_ms=latency_ms,
                metadata={
                    "completion_time": getattr(search_result, "completion_time", 0),
                    "llm_calls": getattr(search_result, "llm_calls", 0),
                },
            )

        except Exception as e:
            logger.warning(f"  Local Search Python API 失败：{e}，回退到 CLI 模式")
            answer = self._cli_search(query, "local")
            latency_ms = (time.perf_counter() - t0) * 1000
            context_tokens = self._token_counter(answer)
            return GraphRAGSearchResult(
                query=query,
                answer=answer,
                search_type="local",
                context_text=answer,
                context_tokens=context_tokens,
                latency_ms=latency_ms,
            )

    def global_search(
        self,
        query: str,
    ) -> GraphRAGSearchResult:
        """
        执行 Global Search（宏观检索，适合全局主题相关问题）。

        Parameters
        ----------
        query : str
            查询文本

        Returns
        -------
        GraphRAGSearchResult
            检索结果，包含答案和 context token 数
        """
        t0 = time.perf_counter()

        try:
            self._load_graphrag_components()
        except Exception as e:
            logger.warning(f"  组件加载失败：{e}，使用 CLI 模式")
            self._use_cli_mode = True

        if getattr(self, "_use_cli_mode", True):
            answer = self._cli_search(query, "global")
            latency_ms = (time.perf_counter() - t0) * 1000
            context_tokens = self._token_counter(answer)
            return GraphRAGSearchResult(
                query=query,
                answer=answer,
                search_type="global",
                context_text=answer,
                context_tokens=context_tokens,
                latency_ms=latency_ms,
            )

        try:
            import asyncio

            async def _run():
                return await self._global_search_engine.asearch(query)

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        future = pool.submit(asyncio.run, _run())
                        search_result = future.result(timeout=180)
                else:
                    search_result = loop.run_until_complete(_run())
            except RuntimeError:
                search_result = asyncio.run(_run())

            latency_ms = (time.perf_counter() - t0) * 1000

            answer = str(search_result.response)
            context_tokens = getattr(search_result, "prompt_tokens", 0)
            if context_tokens == 0:
                context_tokens = self._token_counter(answer)

            return GraphRAGSearchResult(
                query=query,
                answer=answer,
                search_type="global",
                context_text=answer,
                context_tokens=context_tokens,
                latency_ms=latency_ms,
                metadata={
                    "completion_time": getattr(search_result, "completion_time", 0),
                    "llm_calls": getattr(search_result, "llm_calls", 0),
                },
            )

        except Exception as e:
            logger.warning(f"  Global Search Python API 失败：{e}，回退到 CLI 模式")
            answer = self._cli_search(query, "global")
            latency_ms = (time.perf_counter() - t0) * 1000
            context_tokens = self._token_counter(answer)
            return GraphRAGSearchResult(
                query=query,
                answer=answer,
                search_type="global",
                context_text=answer,
                context_tokens=context_tokens,
                latency_ms=latency_ms,
            )


# ---------------------------------------------------------------------------
# CLI 入口（快速测试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GraphRAG 官方检索器（快速测试）")
    parser.add_argument("--workspace_dir", type=str, default=None, help="工作目录")
    parser.add_argument("--query", type=str, required=True, help="查询文本")
    parser.add_argument(
        "--search_type",
        type=str,
        choices=["local", "global"],
        default="local",
        help="搜索类型",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        retriever = GraphRAGRetriever(workspace_dir=args.workspace_dir)
        if args.search_type == "local":
            result = retriever.local_search(args.query)
        else:
            result = retriever.global_search(args.query)
        print(result.summary())
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)
