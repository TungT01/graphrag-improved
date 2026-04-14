"""
retrieval/retriever.py
----------------------
U-Retrieval 双轨检索模块（v3：适配物理优先架构）。

v3 变更：
  - 自底向上检索的物理锚点从 chunk_id（段落级）升级为 sent_id（句子级）
  - 社区的 doc_ids 列用于高层检索时的文档级定位
  - 检索结果携带完整物理路径（sent_id → para_id → doc_id），支持精确溯源
  - entity_chunks 索引改为 node_id → sent_id 映射

U-Retrieval 双轨检索路径：
  1. 自顶向下（Top-Down）：从最高层社区出发，逐层向下导航，
     找到与查询最相关的社区，通过社区的 doc_ids 定位原始文档。
  2. 自底向上（Bottom-Up）：通过实体的 sent_id 直接定位原始句子，
     获取精确的局部上下文，补充自顶向下的全局视角。

当前实现：
  - 基于 TF-IDF 的轻量级相似度计算（无需向量数据库）
  - 社区层次导航（Top-Down）
  - 句子级物理锚点精确检索（Bottom-Up）
  - 结果融合与去重（doc_id 级别去重，避免同文档重复）
  - 可选：BM25 后端（需安装 rank_bm25）
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    单次检索的结果容器。

    Attributes
    ----------
    query           : 原始查询字符串
    top_down_hits   : 自顶向下检索命中的社区列表（按相关性排序）
    bottom_up_hits  : 自底向上检索命中的文本块列表（按相关性排序）
    merged_context  : 融合后的上下文文本（供 LLM 使用）
    metadata        : 检索过程的统计信息
    """
    query: str
    top_down_hits: List["CommunityHit"] = field(default_factory=list)
    bottom_up_hits: List["TextUnitHit"] = field(default_factory=list)
    merged_context: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class CommunityHit:
    """自顶向下检索命中的社区。"""
    community_id: int
    level: int
    title: str
    score: float
    entity_ids: List[str] = field(default_factory=list)
    text_unit_ids: List[str] = field(default_factory=list)   # sent_id 列表
    doc_ids: List[str] = field(default_factory=list)          # v3 新增：文档 ID 列表
    structural_entropy: float = 0.0
    summary: str = ""          # 社区摘要（若有 LLM 生成）


@dataclass
class TextUnitHit:
    """自底向上检索命中的文本块（v3：精确到句子或段落级）。"""
    chunk_id: str          # sent_id 或 para_id（取决于检索粒度）
    text: str
    score: float
    doc_title: str = ""
    doc_id: str = ""       # v3 新增：文档 ID（用于评估时的 ground-truth 对齐）
    sent_id: str = ""      # v3 新增：句子 ID（精确物理坐标）
    entity_mentions: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TF-IDF 工具
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """简单分词：小写化 + 按非字母数字分割。"""
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_tfidf_index(documents: List[str]) -> Tuple[List[Counter], Dict[str, float]]:
    """
    构建 TF-IDF 索引。

    Parameters
    ----------
    documents : List[str]
        文档列表

    Returns
    -------
    Tuple[List[Counter], Dict[str, float]]
        - tf_list  : 每个文档的词频 Counter
        - idf_dict : 每个词的 IDF 值
    """
    N = len(documents)
    tf_list: List[Counter] = []
    df: Counter = Counter()

    for doc in documents:
        tokens = _tokenize(doc)
        tf = Counter(tokens)
        tf_list.append(tf)
        for word in set(tokens):
            df[word] += 1

    idf_dict: Dict[str, float] = {}
    for word, count in df.items():
        idf_dict[word] = math.log((N + 1) / (count + 1)) + 1.0

    return tf_list, idf_dict


def _tfidf_score(query_tokens: List[str], doc_tf: Counter, idf_dict: Dict[str, float]) -> float:
    """计算查询与单个文档的 TF-IDF 余弦相似度（简化版）。"""
    score = 0.0
    doc_len = sum(doc_tf.values()) or 1
    for token in query_tokens:
        if token in doc_tf:
            tf = doc_tf[token] / doc_len
            idf = idf_dict.get(token, 1.0)
            score += tf * idf
    return score


# ---------------------------------------------------------------------------
# BM25 后端（可选）
# ---------------------------------------------------------------------------

def _try_bm25_score(
    query_tokens: List[str],
    corpus_tokens: List[List[str]],
    doc_idx: int,
) -> Optional[float]:
    """
    尝试使用 BM25 计算相关性分数。
    需要安装 rank_bm25：pip install rank-bm25
    若未安装，返回 None（调用方降级到 TF-IDF）。
    """
    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus_tokens)
        scores = bm25.get_scores(query_tokens)
        return float(scores[doc_idx])
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 自顶向下检索（Top-Down Community Navigation）
# ---------------------------------------------------------------------------

class TopDownRetriever:
    """
    自顶向下社区层次导航检索器。

    从最高层社区出发，逐层向下找到与查询最相关的社区，
    返回多个层次的社区命中结果，提供从全局到局部的上下文。

    Parameters
    ----------
    communities_df : pd.DataFrame
        社区表，必须包含 level, community_id, title, entity_ids,
        text_unit_ids, structural_entropy 列
    top_k_per_level : int
        每层返回的最大社区数，默认 3
    max_levels : int
        最多向下导航的层数，默认 3（从顶层往下数）
    """

    def __init__(
        self,
        communities_df: pd.DataFrame,
        top_k_per_level: int = 3,
        max_levels: int = 3,
    ):
        self.communities_df = communities_df
        self.top_k_per_level = top_k_per_level
        self.max_levels = max_levels
        self._index: Dict[int, List[Tuple[int, str, Counter]]] = {}
        self._idf: Dict[str, float] = {}
        self._build_index()

    def _build_index(self) -> None:
        """为每个层级的社区构建 TF-IDF 索引。"""
        if self.communities_df.empty:
            return

        # 按层级分组
        level_groups: Dict[int, List] = defaultdict(list)
        for _, row in self.communities_df.iterrows():
            level_groups[int(row["level"])].append(row)

        # 构建全局 IDF（跨所有层级）
        all_docs = []
        for rows in level_groups.values():
            for row in rows:
                text = self._row_to_text(row)
                all_docs.append(text)

        _, self._idf = _build_tfidf_index(all_docs)

        # 为每层构建 TF 索引
        for level, rows in level_groups.items():
            level_index = []
            for row in rows:
                text = self._row_to_text(row)
                tokens = _tokenize(text)
                tf = Counter(tokens)
                level_index.append((int(row["community_id"]), text, tf))
            self._index[level] = level_index

    @staticmethod
    def _row_to_text(row) -> str:
        """将社区行转换为可检索的文本。"""
        parts = [str(row.get("title", ""))]
        entity_ids = row.get("entity_ids", [])
        if isinstance(entity_ids, list):
            parts.extend([str(e) for e in entity_ids[:20]])
        summary = row.get("summary", "")
        if summary:
            parts.append(str(summary))
        return " ".join(parts)

    def retrieve(self, query: str) -> List[CommunityHit]:
        """
        执行自顶向下检索。

        Parameters
        ----------
        query : str
            查询字符串

        Returns
        -------
        List[CommunityHit]
            按相关性排序的社区命中列表（跨多个层级）
        """
        if not self._index:
            return []

        query_tokens = _tokenize(query)
        all_levels = sorted(self._index.keys(), reverse=True)  # 从高层到低层

        # 限制导航层数
        target_levels = all_levels[:self.max_levels]

        hits: List[CommunityHit] = []
        seen_communities: Set[int] = set()

        for level in target_levels:
            level_entries = self._index[level]
            scored = []
            for comm_id, text, tf in level_entries:
                score = _tfidf_score(query_tokens, tf, self._idf)
                scored.append((comm_id, score))

            # 取 top-k
            scored.sort(key=lambda x: x[1], reverse=True)
            for comm_id, score in scored[:self.top_k_per_level]:
                if comm_id in seen_communities:
                    continue
                seen_communities.add(comm_id)

                # 从 DataFrame 获取完整信息
                row_mask = (
                    (self.communities_df["level"] == level) &
                    (self.communities_df["community_id"] == comm_id)
                )
                matching = self.communities_df[row_mask]
                if matching.empty:
                    continue
                row = matching.iloc[0]

                entity_ids = row.get("entity_ids", [])
                if not isinstance(entity_ids, list):
                    entity_ids = []
                text_unit_ids = row.get("text_unit_ids", [])
                if not isinstance(text_unit_ids, list):
                    text_unit_ids = []

                # v3：获取 doc_ids（文档级定位）
                doc_ids = row.get("doc_ids", [])
                if not isinstance(doc_ids, list):
                    doc_ids = []

                hits.append(CommunityHit(
                    community_id=comm_id,
                    level=level,
                    title=str(row.get("title", "")),
                    score=score,
                    entity_ids=entity_ids,
                    text_unit_ids=text_unit_ids,
                    doc_ids=doc_ids,
                    structural_entropy=float(row.get("structural_entropy", 0.0)),
                    summary=str(row.get("summary", "")),
                ))

        # 全局按分数排序
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits


# ---------------------------------------------------------------------------
# 自底向上检索（Bottom-Up Physical Anchor Retrieval）
# ---------------------------------------------------------------------------

class BottomUpRetriever:
    """
    自底向上物理锚点检索器（v3：句子级精确定位）。

    通过实体的 sent_id 直接定位原始句子/段落，
    提供精确的局部上下文，补充自顶向下检索的全局视角。

    v3 变更：
      - text_units 可以是句子级（sent_id）或段落级（para_id）
      - entity_chunks 索引改为 node_id → sent_id 映射
      - 检索结果携带 doc_id，支持评估时的 ground-truth 对齐

    Parameters
    ----------
    text_units : List[dict]
        文本块列表，每个元素包含 chunk_id, text, doc_title, doc_id 等字段
        v3 中 chunk_id 优先为 sent_id（句子级），也兼容 para_id（段落级）
    entities_df : pd.DataFrame
        实体表，包含 id（node_id）, sent_id, text_unit_ids 列
    top_k : int
        返回的最大文本块数，默认 5
    """

    def __init__(
        self,
        text_units: List[dict],
        entities_df: pd.DataFrame,
        top_k: int = 5,
    ):
        self.text_units = text_units
        self.entities_df = entities_df
        self.top_k = top_k

        # 构建 chunk_id → TextUnit 的快速索引
        self._chunk_index: Dict[str, dict] = {
            unit["chunk_id"]: unit for unit in text_units
            if "chunk_id" in unit
        }

        # 构建实体索引：node_id → sent_id，title → sent_id 集合
        # v3：优先使用 sent_id（精确句子级锚点）
        self._entity_chunks: Dict[str, Set[str]] = {}   # title → chunk_id 集合
        self._node_to_sent: Dict[str, str] = {}          # node_id → sent_id
        if not entities_df.empty:
            for _, row in entities_df.iterrows():
                # node_id → sent_id 映射
                node_id = str(row.get("id", ""))
                sent_id = str(row.get("sent_id", "") or row.get("primary_chunk_id", ""))
                if node_id and sent_id:
                    self._node_to_sent[node_id] = sent_id

                # title → chunk_id 集合（兼容旧检索逻辑）
                if "title" in entities_df.columns:
                    title = str(row["title"])
                    # 优先使用 sent_id
                    if sent_id:
                        self._entity_chunks.setdefault(title, set()).add(sent_id)
                    # 兼容旧格式
                    raw = row.get("text_unit_ids", [])
                    if isinstance(raw, list):
                        for x in raw:
                            self._entity_chunks.setdefault(title, set()).add(str(x))
                    elif isinstance(raw, str):
                        for x in raw.split(";"):
                            if x.strip():
                                self._entity_chunks.setdefault(title, set()).add(x.strip())

        # 构建文本块的 TF-IDF 索引
        docs = [unit.get("text", "") for unit in text_units]
        self._tf_list, self._idf = _build_tfidf_index(docs)
        self._chunk_ids = [unit.get("chunk_id", "") for unit in text_units]

    def retrieve(self, query: str, entity_mentions: Optional[List[str]] = None) -> List[TextUnitHit]:
        """
        执行自底向上检索。

        先通过实体提及找到相关 chunk_id（物理锚点），
        再对这些 chunk 做 TF-IDF 精排，返回最相关的文本块。

        Parameters
        ----------
        query : str
            查询字符串
        entity_mentions : List[str], optional
            查询中提及的实体名称列表（用于物理锚点定位）

        Returns
        -------
        List[TextUnitHit]
            按相关性排序的文本块命中列表
        """
        if not self.text_units:
            return []

        query_tokens = _tokenize(query)

        # Step 1：通过实体提及找到候选 chunk_id 集合（物理锚点）
        anchor_chunk_ids: Set[str] = set()
        if entity_mentions:
            for mention in entity_mentions:
                # 精确匹配
                if mention in self._entity_chunks:
                    anchor_chunk_ids.update(self._entity_chunks[mention])
                else:
                    # 模糊匹配（包含关系）
                    mention_lower = mention.lower()
                    for title, chunks in self._entity_chunks.items():
                        if mention_lower in title.lower() or title.lower() in mention_lower:
                            anchor_chunk_ids.update(chunks)

        # Step 2：对所有文本块打分（物理锚点命中的加权）
        scored: List[Tuple[int, float]] = []
        for idx, (chunk_id, tf) in enumerate(zip(self._chunk_ids, self._tf_list)):
            base_score = _tfidf_score(query_tokens, tf, self._idf)
            # 物理锚点加权：命中锚点的 chunk 分数乘以 1.5
            anchor_bonus = 1.5 if chunk_id in anchor_chunk_ids else 1.0
            scored.append((idx, base_score * anchor_bonus))

        # Step 3：排序取 top-k
        scored.sort(key=lambda x: x[1], reverse=True)

        hits: List[TextUnitHit] = []
        for idx, score in scored[:self.top_k]:
            if score <= 0:
                break
            unit = self.text_units[idx]
            chunk_id = unit.get("chunk_id", "")

            # 找出该 chunk 中出现的实体
            mentions_in_chunk: List[str] = []
            for title, chunks in self._entity_chunks.items():
                if chunk_id in chunks:
                    mentions_in_chunk.append(title)

            hits.append(TextUnitHit(
                chunk_id=chunk_id,
                text=unit.get("text", ""),
                score=score,
                doc_title=unit.get("doc_title", ""),
                doc_id=unit.get("doc_id", ""),       # v3 新增
                sent_id=unit.get("sent_id", chunk_id),  # v3 新增
                entity_mentions=mentions_in_chunk[:10],
            ))

        return hits


# ---------------------------------------------------------------------------
# U-Retrieval 融合检索器
# ---------------------------------------------------------------------------

class URetriever:
    """
    U-Retrieval 双轨融合检索器。

    将自顶向下（社区层次导航）和自底向上（物理锚点定位）
    两条检索路径的结果融合，提供兼顾全局语义和局部精确性的上下文。

    Parameters
    ----------
    communities_df : pd.DataFrame
        社区表
    text_units : List[dict]
        文本块列表（每个元素含 chunk_id, text, doc_title）
    entities_df : pd.DataFrame
        实体表
    top_k_communities : int
        自顶向下检索返回的最大社区数，默认 5
    top_k_chunks : int
        自底向上检索返回的最大文本块数，默认 5
    max_context_chars : int
        融合上下文的最大字符数，默认 4000
    """

    def __init__(
        self,
        communities_df: pd.DataFrame,
        text_units: List[dict],
        entities_df: pd.DataFrame,
        top_k_communities: int = 5,
        top_k_chunks: int = 5,
        max_context_chars: int = 4000,
    ):
        self.max_context_chars = max_context_chars

        self._top_down = TopDownRetriever(
            communities_df=communities_df,
            top_k_per_level=max(1, top_k_communities // 2),
            max_levels=3,
        )
        self._bottom_up = BottomUpRetriever(
            text_units=text_units,
            entities_df=entities_df,
            top_k=top_k_chunks,
        )

    def retrieve(
        self,
        query: str,
        entity_mentions: Optional[List[str]] = None,
        alpha: float = 0.5,
    ) -> RetrievalResult:
        """
        执行 U-Retrieval 双轨融合检索。

        Parameters
        ----------
        query : str
            查询字符串
        entity_mentions : List[str], optional
            查询中提及的实体（用于物理锚点加权）
        alpha : float
            自顶向下结果在融合上下文中的权重比例（0~1），默认 0.5

        Returns
        -------
        RetrievalResult
            包含双轨命中结果和融合上下文的检索结果
        """
        # 执行双轨检索
        top_down_hits = self._top_down.retrieve(query)

        # bottom-up 检索独立于 top-down 社区划分结果：
        # entity_mentions 只来自调用方显式传入，不从 top-down 社区继承。
        # 这样两个版本（baseline/ours）的 bottom-up 使用完全相同的输入，
        # 确保对照实验的控制变量原则。
        bottom_up_hits = self._bottom_up.retrieve(query, entity_mentions)

        # 融合上下文
        merged_context = self._merge_context(
            top_down_hits, bottom_up_hits, alpha
        )

        result = RetrievalResult(
            query=query,
            top_down_hits=top_down_hits,
            bottom_up_hits=bottom_up_hits,
            merged_context=merged_context,
            metadata={
                "num_community_hits": len(top_down_hits),
                "num_chunk_hits": len(bottom_up_hits),
                "context_chars": len(merged_context),
                "alpha": alpha,
            },
        )
        return result

    def _merge_context(
        self,
        top_down_hits: List[CommunityHit],
        bottom_up_hits: List[TextUnitHit],
        alpha: float,
    ) -> str:
        """
        将双轨检索结果融合为单一上下文字符串。

        策略：
        - 按 alpha 分配字符预算
        - 自顶向下部分：优先使用社区摘要，无摘要时使用实体列表
        - 自底向上部分：直接使用原始文本块
        - 去重：相同 chunk_id 的内容只出现一次
        """
        budget_top = int(self.max_context_chars * alpha)
        budget_bottom = self.max_context_chars - budget_top

        parts: List[str] = []
        used_chunks: Set[str] = set()

        # 自顶向下部分
        top_chars = 0
        if top_down_hits:
            parts.append("=== 社区上下文（全局视角）===")
            for hit in top_down_hits:
                if top_chars >= budget_top:
                    break
                if hit.summary:
                    snippet = f"[Level {hit.level} | 社区 {hit.community_id}] {hit.summary}"
                else:
                    entities_str = ", ".join(hit.entity_ids[:10])
                    snippet = f"[Level {hit.level} | 社区 {hit.community_id}] 实体：{entities_str}"
                parts.append(snippet)
                top_chars += len(snippet)
                # 记录该社区覆盖的 chunk_id
                used_chunks.update(hit.text_unit_ids)

        # 自底向上部分
        bottom_chars = 0
        if bottom_up_hits:
            parts.append("\n=== 原文片段（局部精确）===")
            for hit in bottom_up_hits:
                if bottom_chars >= budget_bottom:
                    break
                if hit.chunk_id in used_chunks:
                    continue  # 去重
                snippet = f"[{hit.doc_title}] {hit.text}"
                parts.append(snippet)
                bottom_chars += len(snippet)
                used_chunks.add(hit.chunk_id)

        return "\n".join(parts)

    @classmethod
    def from_pipeline_result(
        cls,
        pipeline_result,
        text_units: List[dict],
        **kwargs,
    ) -> "URetriever":
        """
        从 PipelineResult 直接构建 URetriever 的便捷工厂方法。

        Parameters
        ----------
        pipeline_result : PipelineResult
            run_pipeline() 的返回值
        text_units : List[dict]
            原始文本块列表（需要从 ingestion 模块获取）
        """
        return cls(
            communities_df=pipeline_result.communities_df,
            text_units=text_units,
            entities_df=pipeline_result.entities_df,
            **kwargs,
        )
