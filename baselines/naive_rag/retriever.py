"""
baselines/naive_rag/retriever.py
---------------------------------
Naive RAG 基线：检索器。

功能：
  1. 加载由 indexer.py 构建的索引
  2. 编码 query 为向量
  3. 计算余弦相似度（向量已 L2 归一化，点积即余弦相似度）
  4. 返回 Top-K 句子，同时统计 context token 数（用 tiktoken cl100k_base）

用法：
  from baselines.naive_rag.retriever import NaiveRAGRetriever
  retriever = NaiveRAGRetriever("./naive_rag_index.pkl")
  results = retriever.retrieve("What is GraphRAG?", top_k=5)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RetrievedSentence:
    """单条检索结果。"""
    sentence: str           # 句子文本
    doc_id: str             # 来源文档 ID
    score: float            # 余弦相似度分数
    rank: int               # 排名（从 1 开始）
    token_count: int = 0    # 该句子的 token 数


@dataclass
class RetrievalResult:
    """检索结果集合。"""
    query: str
    hits: List[RetrievedSentence] = field(default_factory=list)
    total_context_tokens: int = 0   # 所有命中句子的 token 总数
    top_k: int = 5

    def get_context_text(self) -> str:
        """将所有命中句子拼接为上下文文本。"""
        return "\n".join(hit.sentence for hit in self.hits)

    def get_doc_ids(self) -> List[str]:
        """返回命中的 doc_id 列表（去重，保持顺序）。"""
        seen = set()
        result = []
        for hit in self.hits:
            if hit.doc_id not in seen:
                seen.add(hit.doc_id)
                result.append(hit.doc_id)
        return result

    def summary(self) -> str:
        lines = [
            f"查询：{self.query}",
            f"Top-{self.top_k} 结果（context tokens: {self.total_context_tokens}）：",
        ]
        for hit in self.hits:
            lines.append(
                f"  [{hit.rank}] score={hit.score:.4f} doc={hit.doc_id} "
                f"tokens={hit.token_count}\n"
                f"      {hit.sentence[:100]}{'...' if len(hit.sentence) > 100 else ''}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token 计数工具
# ---------------------------------------------------------------------------

def _get_token_counter():
    """
    获取 tiktoken 计数器。若未安装则返回简单的空格分词计数器。
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        def count_tokens(text: str) -> int:
            return len(enc.encode(text))
        return count_tokens
    except ImportError:
        logger.warning("tiktoken 未安装，使用简单空格分词计数（结果可能不准确）")
        def count_tokens(text: str) -> int:
            return len(text.split())
        return count_tokens


# ---------------------------------------------------------------------------
# 检索器
# ---------------------------------------------------------------------------

class NaiveRAGRetriever:
    """
    Naive RAG 检索器。

    基于句子级向量检索，使用余弦相似度排序。

    Parameters
    ----------
    index_path : str
        由 indexer.py 构建的索引文件路径
    model_name : str, optional
        SentenceTransformer 模型名称（若与索引不一致会发出警告）
    """

    def __init__(
        self,
        index_path: str,
        model_name: Optional[str] = None,
    ):
        self.index_path = str(index_path)
        self._index: Optional[dict] = None
        self._model = None
        self._token_counter = _get_token_counter()
        self._model_name = model_name

        # 延迟加载
        self._load_index()

    def _load_index(self) -> None:
        """加载索引文件。"""
        from baselines.naive_rag.indexer import load_index
        self._index = load_index(self.index_path)

        # 检查模型名称一致性
        index_model = self._index.get("model_name", "all-MiniLM-L6-v2")
        if self._model_name is None:
            self._model_name = index_model
        elif self._model_name != index_model:
            logger.warning(
                f"指定模型 {self._model_name} 与索引模型 {index_model} 不一致，"
                f"将使用索引模型 {index_model}"
            )
            self._model_name = index_model

    def _load_model(self) -> None:
        """延迟加载 SentenceTransformer 模型。"""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "请先安装 sentence-transformers：\n"
                "  pip install sentence-transformers"
            )
        logger.info(f"加载模型：{self._model_name}")
        self._model = SentenceTransformer(self._model_name)

    def _encode_query(self, query: str) -> np.ndarray:
        """将 query 编码为 L2 归一化向量。"""
        self._load_model()
        embedding = self._model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0]  # shape: (D,)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> RetrievalResult:
        """
        检索与 query 最相关的 Top-K 句子。

        Parameters
        ----------
        query : str
            查询文本
        top_k : int
            返回的最大句子数
        min_score : float
            最低相似度阈值（低于此分数的结果被过滤）

        Returns
        -------
        RetrievalResult
            检索结果，包含命中句子和 context token 数
        """
        if self._index is None:
            raise RuntimeError("索引未加载，请先调用 _load_index()")

        sentences: List[str] = self._index["sentences"]
        sentence_doc_ids: List[str] = self._index["sentence_doc_ids"]
        embeddings: np.ndarray = self._index["embeddings"]  # (N, D)

        if len(sentences) == 0:
            logger.warning("索引为空，返回空结果")
            return RetrievalResult(query=query, top_k=top_k)

        # 编码 query
        query_vec = self._encode_query(query)  # (D,)

        # 计算余弦相似度（向量已 L2 归一化，点积即余弦相似度）
        scores = embeddings @ query_vec  # (N,)

        # 取 Top-K（使用 argpartition 加速，避免全排序）
        actual_k = min(top_k, len(sentences))
        if actual_k < len(sentences):
            top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores)[::-1]

        # 构建结果
        hits = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score < min_score:
                break
            sentence = sentences[idx]
            doc_id = sentence_doc_ids[idx]
            token_count = self._token_counter(sentence)
            hits.append(RetrievedSentence(
                sentence=sentence,
                doc_id=doc_id,
                score=score,
                rank=rank,
                token_count=token_count,
            ))

        # 统计 context token 总数
        total_tokens = sum(hit.token_count for hit in hits)

        return RetrievalResult(
            query=query,
            hits=hits,
            total_context_tokens=total_tokens,
            top_k=top_k,
        )

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[RetrievalResult]:
        """
        批量检索多个 query。

        Parameters
        ----------
        queries : List[str]
            查询文本列表
        top_k : int
            每个查询返回的最大句子数
        min_score : float
            最低相似度阈值

        Returns
        -------
        List[RetrievalResult]
            每个查询对应的检索结果
        """
        if not queries:
            return []

        self._load_model()

        sentences: List[str] = self._index["sentences"]
        sentence_doc_ids: List[str] = self._index["sentence_doc_ids"]
        embeddings: np.ndarray = self._index["embeddings"]  # (N, D)

        # 批量编码所有 query
        logger.info(f"批量编码 {len(queries)} 个查询...")
        query_vecs = self._model.encode(
            queries,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(queries) > 10,
            batch_size=32,
        )  # (Q, D)

        # 批量计算相似度矩阵
        score_matrix = query_vecs @ embeddings.T  # (Q, N)

        results = []
        actual_k = min(top_k, len(sentences))

        for q_idx, query in enumerate(queries):
            scores = score_matrix[q_idx]  # (N,)

            if actual_k < len(sentences):
                top_indices = np.argpartition(scores, -actual_k)[-actual_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            else:
                top_indices = np.argsort(scores)[::-1]

            hits = []
            for rank, idx in enumerate(top_indices, start=1):
                score = float(scores[idx])
                if score < min_score:
                    break
                sentence = sentences[idx]
                doc_id = sentence_doc_ids[idx]
                token_count = self._token_counter(sentence)
                hits.append(RetrievedSentence(
                    sentence=sentence,
                    doc_id=doc_id,
                    score=score,
                    rank=rank,
                    token_count=token_count,
                ))

            total_tokens = sum(hit.token_count for hit in hits)
            results.append(RetrievalResult(
                query=query,
                hits=hits,
                total_context_tokens=total_tokens,
                top_k=top_k,
            ))

        return results

    @property
    def num_sentences(self) -> int:
        """索引中的句子总数。"""
        if self._index is None:
            return 0
        return self._index.get("num_sentences", 0)

    @property
    def model_name(self) -> str:
        """使用的模型名称。"""
        return self._model_name or "unknown"


# ---------------------------------------------------------------------------
# CLI 入口（快速测试）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Naive RAG 检索器（快速测试）")
    parser.add_argument("--index_path", type=str, required=True, help="索引文件路径")
    parser.add_argument("--query", type=str, required=True, help="查询文本")
    parser.add_argument("--top_k", type=int, default=5, help="返回结果数")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        retriever = NaiveRAGRetriever(args.index_path)
        result = retriever.retrieve(args.query, top_k=args.top_k)
        print(result.summary())
    except Exception as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)
