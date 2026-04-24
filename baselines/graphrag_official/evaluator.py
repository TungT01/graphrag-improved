"""
baselines/graphrag_official/evaluator.py
------------------------------------------
Microsoft GraphRAG 官方基线：评估器。

功能：
  在 MultiHop-RAG 数据集上批量评估，分宏观/微观两组输出指标：
  - Local Search（微观）：P@K, Recall@K, MRR, NDCG@K, ROUGE-L, 平均 context tokens
  - Global Search（宏观）：ROUGE-L, 平均 context tokens（无检索指标，因为 global 不返回 doc_id）

注意：
  - GraphRAG 需要 LLM API（OpenAI），每次查询都会消耗 API 调用
  - 建议先用 --num_samples 小批量测试
  - 设置 GRAPHRAG_API_KEY 环境变量

用法：
  python evaluator.py \
    --data_dir /path/to/multihop_rag \
    --search_type local \
    --num_samples 50 \
    --output_dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 评估指标计算（与 naive_rag/evaluator.py 保持一致）
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_rouge_l(prediction: str, gold: str) -> float:
    """计算 ROUGE-L（基于最长公共子序列）。"""
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    m, n = len(pred_tokens), len(gold_tokens)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, [0] * (n + 1)

    lcs_len = prev[n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / k


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / len(relevant_set)


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(top_k, start=1)
        if rid in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 评估结果数据结构
# ---------------------------------------------------------------------------

@dataclass
class GraphRAGEvalResult:
    """GraphRAG 官方基线评估结果。"""
    system_name: str = "graphrag_official"
    search_type: str = "local"   # "local" 或 "global"
    num_queries: int = 0
    num_valid_queries: int = 0

    # 检索指标（仅 local search 有意义）
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)

    # 文本匹配指标
    avg_rouge_l: float = 0.0

    # 效率指标
    avg_context_tokens: float = 0.0
    avg_latency_ms: float = 0.0

    # 按问题类型分组
    by_question_type: Dict[str, dict] = field(default_factory=dict)

    # 错误统计
    num_errors: int = 0

    def to_dict(self) -> dict:
        return {
            "system_name": self.system_name,
            "search_type": self.search_type,
            "num_queries": self.num_queries,
            "num_valid_queries": self.num_valid_queries,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "avg_rouge_l": self.avg_rouge_l,
            "avg_context_tokens": self.avg_context_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "by_question_type": self.by_question_type,
            "num_errors": self.num_errors,
        }

    def summary(self) -> str:
        k_values = sorted(self.precision_at_k.keys())
        lines = [
            "=" * 60,
            f"  GraphRAG 官方基线评估报告（{self.search_type} search）",
            "=" * 60,
            f"  查询总数       : {self.num_queries}",
            f"  有效查询数     : {self.num_valid_queries}",
            f"  错误数         : {self.num_errors}",
        ]

        if self.search_type == "local" and k_values:
            lines.append(f"  MRR            : {self.mrr:.4f}")
            for k in k_values:
                lines.append(
                    f"  P@{k:<2} / R@{k:<2} / F1@{k:<2} : "
                    f"{self.precision_at_k.get(k, 0):.4f} / "
                    f"{self.recall_at_k.get(k, 0):.4f} / "
                    f"{self.f1_at_k.get(k, 0):.4f}"
                )
            for k in k_values:
                lines.append(f"  NDCG@{k:<2}         : {self.ndcg_at_k.get(k, 0):.4f}")

        lines.extend([
            f"  ROUGE-L        : {self.avg_rouge_l:.4f}",
            f"  平均 context tokens : {self.avg_context_tokens:.1f}",
            f"  平均延迟       : {self.avg_latency_ms:.1f} ms",
        ])

        if self.by_question_type:
            lines.append("\n  按问题类型分组：")
            for qtype, metrics in sorted(self.by_question_type.items()):
                lines.append(
                    f"    [{qtype}] n={metrics.get('n', 0)} "
                    f"ROUGE-L={metrics.get('rouge_l', 0):.4f} "
                    f"tokens={metrics.get('avg_tokens', 0):.0f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主评估函数
# ---------------------------------------------------------------------------

def evaluate_graphrag(
    data_dir: str,
    search_type: str = "local",
    workspace_dir: Optional[str] = None,
    k_values: Optional[List[int]] = None,
    num_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    filter_indexed_only: bool = False,
) -> GraphRAGEvalResult:
    """
    在 MultiHop-RAG 数据集上评估 GraphRAG 官方基线。

    Parameters
    ----------
    data_dir : str
        MultiHop-RAG 数据集目录
    search_type : str
        搜索类型："local"（微观）或 "global"（宏观）
    workspace_dir : str, optional
        GraphRAG 工作目录
    k_values : List[int], optional
        评估的 K 值列表（仅 local search 使用）
    num_samples : int, optional
        随机采样的 QA 对数量
    output_dir : str, optional
        结果输出目录
    seed : int
        随机种子
    max_retries : int
        API 调用失败时的最大重试次数
    retry_delay : float
        重试间隔（秒）
    filter_indexed_only : bool
        若为 True，则只评估 supporting_doc_ids 全部在 GraphRAG 索引中的 QA 对。
        适用于 GraphRAG 只索引了部分文档的场景，确保检索指标有意义。

    Returns
    -------
    GraphRAGEvalResult
        评估结果
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    if search_type not in ("local", "global"):
        raise ValueError(f"search_type 必须是 'local' 或 'global'，当前：{search_type}")

    # 1. 加载数据集
    logger.info("步骤 1/3：加载 MultiHop-RAG 数据集")
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from graphrag_improved.experiments.data_loader import load_multihop_dataset
        dataset = load_multihop_dataset(data_dir)
    except ImportError:
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data_loader import load_multihop_dataset
            dataset = load_multihop_dataset(data_dir)
        except ImportError:
            logger.error("无法导入 data_loader")
            raise

    qa_pairs = dataset.qa_pairs
    logger.info(f"  加载了 {len(qa_pairs)} 条 QA 对，{dataset.num_docs} 篇文档")

    # 若启用过滤，只保留 supporting_doc_ids 全部在 GraphRAG 索引中的 QA 对
    if filter_indexed_only:
        _ws_dir_for_filter = workspace_dir or str(Path(__file__).parent / "graphrag_workspace")
        _input_dir = Path(_ws_dir_for_filter) / "input"
        indexed_titles: set = set()
        if _input_dir.exists():
            for _f in _input_dir.glob("*.txt"):
                try:
                    with open(_f, "r", encoding="utf-8") as _fp:
                        _line = _fp.readline().strip()
                    if _line.startswith("Title:"):
                        indexed_titles.add(_line[6:].strip())
                except Exception:
                    pass
        if indexed_titles:
            before = len(qa_pairs)
            qa_pairs = [
                qa for qa in qa_pairs
                if not qa.supporting_doc_ids
                or all(d in indexed_titles for d in qa.supporting_doc_ids)
            ]
            logger.info(
                f"  [filter_indexed_only] 过滤后：{len(qa_pairs)} 条 QA 对"
                f"（原 {before} 条，GraphRAG 索引覆盖 {len(indexed_titles)} 篇文档）"
            )

    # 随机采样
    if num_samples is not None and num_samples < len(qa_pairs):
        import random
        rng = random.Random(seed)
        qa_pairs = rng.sample(qa_pairs, num_samples)
        logger.info(f"  随机采样 {num_samples} 条 QA 对（seed={seed}）")

    # 2. 初始化检索器
    logger.info("步骤 2/3：初始化 GraphRAG 检索器")
    try:
        from baselines.graphrag_official.retriever import GraphRAGRetriever
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from graphrag_official.retriever import GraphRAGRetriever

    retriever = GraphRAGRetriever(workspace_dir=workspace_dir)

    # 预构建 text_unit human_readable_id → MultiHop-RAG doc_id 的映射表
    _ws_dir = workspace_dir or str(Path(__file__).parent / "graphrag_workspace")
    graphrag_doc_mapping = _build_graphrag_doc_mapping(_ws_dir)

    # 3. 批量评估
    logger.info(f"步骤 3/3：评估 {len(qa_pairs)} 条 QA 对（search_type={search_type}）")
    logger.warning(
        "  [注意] GraphRAG 每次查询都会调用 LLM API，请确保 API key 有效且有足够配额"
    )

    # 初始化累加器
    precision_sum = {k: 0.0 for k in k_values}
    recall_sum = {k: 0.0 for k in k_values}
    f1_sum = {k: 0.0 for k in k_values}
    ndcg_sum = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    rouge_sum = 0.0
    token_sum = 0.0
    latency_sum = 0.0
    valid_n = 0
    error_count = 0

    type_accum: Dict[str, dict] = {}

    for i, qa in enumerate(qa_pairs):
        if (i + 1) % 10 == 0:
            logger.info(f"  进度：{i + 1}/{len(qa_pairs)}（错误：{error_count}）")

        # 执行检索（带重试）
        search_result = None
        for attempt in range(max_retries):
            try:
                if search_type == "local":
                    search_result = retriever.local_search(qa.query)
                else:
                    search_result = retriever.global_search(qa.query)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"  查询 {i} 第 {attempt + 1} 次失败：{e}，"
                        f"{retry_delay}s 后重试..."
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(f"  查询 {i} 最终失败：{e}")
                    error_count += 1

        if search_result is None:
            continue

        token_sum += search_result.context_tokens
        latency_sum += search_result.latency_ms

        # ROUGE-L（答案与 gold answer 的匹配）
        rouge_val = compute_rouge_l(search_result.answer, qa.answer)
        rouge_sum += rouge_val

        # 检索指标（仅 local search，且需要有 supporting_doc_ids）
        relevant_doc_ids = qa.supporting_doc_ids
        if search_type == "local" and relevant_doc_ids:
            valid_n += 1

            # 从答案文本中提取 [Data: Sources (x,y,z)] 引用，映射到 MultiHop-RAG doc_id
            retrieved_doc_ids = _extract_doc_ids_from_context(
                search_result.context_text,
                dataset.corpus,
                graphrag_mapping=graphrag_doc_mapping,
            )

            mrr_val = compute_mrr(retrieved_doc_ids, relevant_doc_ids)
            mrr_sum += mrr_val

            for k in k_values:
                p_i = compute_precision_at_k(retrieved_doc_ids, relevant_doc_ids, k)
                r_i = compute_recall_at_k(retrieved_doc_ids, relevant_doc_ids, k)
                f1_i = (2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) > 0 else 0.0
                precision_sum[k] += p_i
                recall_sum[k] += r_i
                f1_sum[k] += f1_i
                ndcg_sum[k] += compute_ndcg_at_k(retrieved_doc_ids, relevant_doc_ids, k)

        # 按问题类型分组
        qtype = qa.question_type
        if qtype not in type_accum:
            type_accum[qtype] = {"n": 0, "rouge_l": 0.0, "tokens": 0.0}
        type_accum[qtype]["n"] += 1
        type_accum[qtype]["rouge_l"] += rouge_val
        type_accum[qtype]["tokens"] += search_result.context_tokens

    # 汇总结果
    total_n = len(qa_pairs) - error_count
    n = valid_n if valid_n > 0 else 1

    result_obj = GraphRAGEvalResult(
        system_name="graphrag_official",
        search_type=search_type,
        num_queries=len(qa_pairs),
        num_valid_queries=valid_n,
        mrr=mrr_sum / n if search_type == "local" else 0.0,
        avg_rouge_l=rouge_sum / total_n if total_n > 0 else 0.0,
        avg_context_tokens=token_sum / total_n if total_n > 0 else 0.0,
        avg_latency_ms=latency_sum / total_n if total_n > 0 else 0.0,
        num_errors=error_count,
    )

    if search_type == "local":
        for k in k_values:
            result_obj.precision_at_k[k] = precision_sum[k] / n
            result_obj.recall_at_k[k] = recall_sum[k] / n
            result_obj.f1_at_k[k] = f1_sum[k] / n
            result_obj.ndcg_at_k[k] = ndcg_sum[k] / n

    for qtype, accum in type_accum.items():
        cnt = accum["n"] if accum["n"] > 0 else 1
        result_obj.by_question_type[qtype] = {
            "n": accum["n"],
            "rouge_l": accum["rouge_l"] / cnt,
            "avg_tokens": accum["tokens"] / cnt,
        }

    # 输出结果
    logger.info("\n" + result_obj.summary())

    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / f"graphrag_{search_type}_eval_results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_obj.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"  ✓ 结果已保存：{result_file}")

    return result_obj


def _build_graphrag_doc_mapping(workspace_dir: str) -> Dict[int, str]:
    """
    构建 GraphRAG text_unit human_readable_id → MultiHop-RAG doc_id 的映射表。

    映射链路：
      Sources(N) → text_units[human_readable_id=N].document_id
                → documents[id=document_id].title  (文件名，如 "010_SBF's trial...txt")
                → 去掉序号前缀和 .txt 后缀 → 文章标题（即 MultiHop-RAG doc_id）

    Parameters
    ----------
    workspace_dir : str
        GraphRAG 工作目录

    Returns
    -------
    Dict[int, str]
        human_readable_id → MultiHop-RAG doc_id 的映射
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas 未安装，无法构建 GraphRAG 文档映射")
        return {}

    output_dir = Path(workspace_dir) / "output"
    tu_path = output_dir / "text_units.parquet"
    docs_path = output_dir / "documents.parquet"

    if not tu_path.exists() or not docs_path.exists():
        logger.warning(f"parquet 文件不存在：{tu_path} 或 {docs_path}")
        return {}

    try:
        tu_df = pd.read_parquet(tu_path, columns=["human_readable_id", "document_id"])
        docs_df = pd.read_parquet(docs_path, columns=["id", "title"])

        # 构建 document_id → 文章标题 的映射
        # GraphRAG 文件名格式："NNN_文章标题截断.txt"
        # 需要从原始文件名中提取完整标题，但文件名被截断了
        # 改用 documents.parquet 的 text 字段第一行（Title: xxx）来提取标题
        # 实际上 documents.parquet 的 title 字段就是文件名，需要另一种方式
        # 最可靠的方式：读取 input/ 目录下的 txt 文件，从 "Title: xxx" 行提取标题
        input_dir = Path(workspace_dir) / "input"
        filename_to_title: Dict[str, str] = {}

        if input_dir.exists():
            for txt_file in sorted(input_dir.glob("*.txt")):
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        first_line = f.readline().strip()
                    if first_line.startswith("Title:"):
                        article_title = first_line[len("Title:"):].strip()
                        filename_to_title[txt_file.name] = article_title
                except Exception:
                    pass

        # 构建 doc_id（GraphRAG hash）→ 文章标题 的映射
        doc_hash_to_title: Dict[str, str] = {}
        for _, row in docs_df.iterrows():
            fname = row["title"]  # 文件名，如 "010_SBF's trial...txt"
            # 先尝试从 input/ 目录的 txt 文件中查找完整标题
            if fname in filename_to_title:
                doc_hash_to_title[row["id"]] = filename_to_title[fname]
            else:
                # 回退：从文件名中提取（去掉序号前缀 NNN_ 和 .txt 后缀）
                import re as _re
                clean = _re.sub(r"^\d+_", "", fname)  # 去掉 "010_"
                clean = _re.sub(r"\.txt$", "", clean)  # 去掉 ".txt"
                doc_hash_to_title[row["id"]] = clean

        # 构建 human_readable_id → 文章标题 的映射
        mapping: Dict[int, str] = {}
        for _, row in tu_df.iterrows():
            hrid = int(row["human_readable_id"])
            doc_hash = row["document_id"]
            if doc_hash in doc_hash_to_title:
                mapping[hrid] = doc_hash_to_title[doc_hash]

        logger.info(f"  ✓ 构建 GraphRAG 文档映射：{len(mapping)} 条 text_unit → doc_id")
        return mapping

    except Exception as e:
        logger.warning(f"构建 GraphRAG 文档映射失败：{e}")
        return {}


def _extract_doc_ids_from_context(
    context_text: str,
    corpus,
    graphrag_mapping: Optional[Dict[int, str]] = None,
) -> List[str]:
    """
    从 GraphRAG local search 的答案文本中提取文档 ID。

    GraphRAG 答案中包含 [Data: Sources (x, y, z)] 格式的引用，
    其中数字是 text_units.parquet 的 human_readable_id。
    通过预构建的映射表将其转换为 MultiHop-RAG 的 doc_id。

    若映射表不可用，则回退到标题匹配的启发式方法。

    Parameters
    ----------
    context_text : str
        GraphRAG 返回的答案/context 文本
    corpus : List[CorpusDoc]
        语料库文档列表
    graphrag_mapping : Dict[int, str], optional
        human_readable_id → MultiHop-RAG doc_id 的映射表

    Returns
    -------
    List[str]
        检索到的文档 ID 列表（按出现顺序）
    """
    if not context_text:
        return []

    found_ids = []
    seen = set()

    # 方法一：解析 [Data: Sources (x, y, z)] 格式（GraphRAG CLI 输出）
    if graphrag_mapping:
        source_pattern = re.compile(r"\[Data:.*?Sources\s*\(([^)]+)\)", re.IGNORECASE)
        for match in source_pattern.finditer(context_text):
            ids_str = match.group(1)
            for part in ids_str.split(","):
                part = part.strip()
                # 可能有 "123+" 格式，取数字部分
                num_match = re.match(r"(\d+)", part)
                if num_match:
                    hrid = int(num_match.group(1))
                    doc_id = graphrag_mapping.get(hrid)
                    if doc_id and doc_id not in seen:
                        found_ids.append(doc_id)
                        seen.add(doc_id)

        if found_ids:
            return found_ids

    # 方法二：回退到标题匹配（原始逻辑）
    if corpus:
        context_lower = context_text.lower()
        for doc in corpus:
            title = doc.title.lower()
            if title and title in context_lower and doc.doc_id not in seen:
                found_ids.append(doc.doc_id)
                seen.add(doc.doc_id)

    return found_ids


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="GraphRAG 官方基线评估器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="MultiHop-RAG 数据集目录",
    )
    parser.add_argument(
        "--search_type",
        type=str,
        choices=["local", "global"],
        default="local",
        help="搜索类型",
    )
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=None,
        help="GraphRAG 工作目录",
    )
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10],
        help="评估的 K 值列表",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="随机采样的 QA 对数量",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="结果输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="API 调用失败时的最大重试次数",
    )
    parser.add_argument(
        "--filter_indexed_only",
        action="store_true",
        default=False,
        help="只评估 supporting_doc_ids 全部在 GraphRAG 索引中的 QA 对（适用于部分索引场景）",
    )

    args = parser.parse_args()

    try:
        evaluate_graphrag(
            data_dir=args.data_dir,
            search_type=args.search_type,
            workspace_dir=args.workspace_dir,
            k_values=args.k_values,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            seed=args.seed,
            max_retries=args.max_retries,
            filter_indexed_only=args.filter_indexed_only,
        )
    except Exception as e:
        logger.error(f"评估失败：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
