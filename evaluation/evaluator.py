"""
evaluation/evaluator.py
-----------------------
评估框架：衡量带结构熵约束的 GraphRAG 改进效果。

评估维度：
  1. 检索质量（Retrieval Quality）
     - Precision@K / Recall@K / F1@K
     - MRR（Mean Reciprocal Rank）
     - NDCG（Normalized Discounted Cumulative Gain）

  2. 社区质量（Community Quality）
     - 模块度 Q（Modularity）
     - 平均结构熵（越低越好，表示物理边界保持越好）
     - 物理纯净率（Level 0 中熵 < 阈值的社区比例）
     - 社区大小分布（均匀性）

  3. 文本匹配（Text Matching，用于 QA 评估）
     - Exact Match（EM）
     - Token-level F1
     - ROUGE-L

使用方式：
  from graphrag_improved.evaluation.evaluator import Evaluator, QAPair
  evaluator = Evaluator()
  metrics = evaluator.evaluate_community_quality(communities_df)
  qa_metrics = evaluator.evaluate_retrieval(qa_pairs, retriever)
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    """
    问答对，用于检索和生成质量评估。

    Attributes
    ----------
    question    : 问题文本
    answer      : 标准答案（gold answer）
    context_ids : 相关文本块的 chunk_id 列表（用于检索评估）
    metadata    : 附加信息（如来源、难度等级）
    """
    question: str
    answer: str
    context_ids: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """检索质量指标。"""
    precision_at_k: Dict[int, float] = field(default_factory=dict)   # {k: precision}
    recall_at_k: Dict[int, float] = field(default_factory=dict)      # {k: recall}
    f1_at_k: Dict[int, float] = field(default_factory=dict)          # {k: f1}
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)        # {k: ndcg}
    num_queries: int = 0

    def summary(self) -> str:
        lines = [
            f"检索质量评估（{self.num_queries} 个查询）",
            f"  MRR          : {self.mrr:.4f}",
        ]
        for k in sorted(self.precision_at_k.keys()):
            lines.append(
                f"  P@{k:<2} / R@{k:<2} / F1@{k:<2} : "
                f"{self.precision_at_k[k]:.4f} / "
                f"{self.recall_at_k[k]:.4f} / "
                f"{self.f1_at_k[k]:.4f}"
            )
        for k in sorted(self.ndcg_at_k.keys()):
            lines.append(f"  NDCG@{k:<2}      : {self.ndcg_at_k[k]:.4f}")
        return "\n".join(lines)


@dataclass
class CommunityMetrics:
    """社区质量指标。"""
    modularity: float = 0.0
    avg_structural_entropy: float = 0.0
    level0_purity_rate: float = 0.0       # Level 0 物理纯净社区比例
    avg_community_size: float = 0.0
    size_std: float = 0.0                 # 社区大小标准差（越小越均匀）
    num_communities: int = 0
    num_levels: int = 0
    entropy_by_level: Dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"社区质量评估",
            f"  社区总数       : {self.num_communities}",
            f"  层次数量       : {self.num_levels}",
            f"  模块度 Q       : {self.modularity:.4f}",
            f"  平均结构熵     : {self.avg_structural_entropy:.4f}",
            f"  Level 0 纯净率 : {self.level0_purity_rate:.2%}",
            f"  平均社区大小   : {self.avg_community_size:.1f} ± {self.size_std:.1f}",
        ]
        if self.entropy_by_level:
            lines.append("  各层平均结构熵 :")
            for level in sorted(self.entropy_by_level.keys()):
                lines.append(f"    Level {level:<2} : {self.entropy_by_level[level]:.4f}")
        return "\n".join(lines)


@dataclass
class TextMatchMetrics:
    """文本匹配指标（用于 QA 评估）。"""
    exact_match: float = 0.0
    token_f1: float = 0.0
    rouge_l: float = 0.0
    num_samples: int = 0

    def summary(self) -> str:
        return (
            f"文本匹配评估（{self.num_samples} 个样本）\n"
            f"  Exact Match : {self.exact_match:.4f}\n"
            f"  Token F1    : {self.token_f1:.4f}\n"
            f"  ROUGE-L     : {self.rouge_l:.4f}"
        )


# ---------------------------------------------------------------------------
# 文本匹配工具函数
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """标准化文本：小写、去标点、去多余空格。"""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_exact_match(prediction: str, gold: str) -> float:
    """计算 Exact Match（完全匹配）。"""
    return float(_normalize_text(prediction) == _normalize_text(gold))


def compute_token_f1(prediction: str, gold: str) -> float:
    """
    计算 Token-level F1。
    将预测和标准答案分词后，计算词级别的 Precision、Recall 和 F1。
    """
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    pred_counter = Counter(pred_tokens)
    gold_counter = Counter(gold_tokens)

    # 计算共同 token 数
    common = sum((pred_counter & gold_counter).values())

    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def compute_rouge_l(prediction: str, gold: str) -> float:
    """
    计算 ROUGE-L（基于最长公共子序列）。

    使用动态规划计算 LCS 长度，然后计算 F1。
    """
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()

    if not pred_tokens or not gold_tokens:
        return 0.0

    # 动态规划计算 LCS
    m, n = len(pred_tokens), len(gold_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == gold_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    if lcs_len == 0:
        return 0.0

    precision = lcs_len / m
    recall = lcs_len / n
    rouge_l = 2 * precision * recall / (precision + recall)
    return rouge_l


# ---------------------------------------------------------------------------
# 检索评估工具函数
# ---------------------------------------------------------------------------

def compute_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """计算 Precision@K。"""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / k


def compute_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """计算 Recall@K。"""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]
    hits = sum(1 for rid in top_k if rid in relevant_set)
    return hits / len(relevant_set)


def compute_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """计算 MRR（Mean Reciprocal Rank）中单个查询的 RR。"""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """计算 NDCG@K（二元相关性）。"""
    relevant_set = set(relevant_ids)
    top_k = retrieved_ids[:k]

    # DCG
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, rid in enumerate(top_k, start=1)
        if rid in relevant_set
    )

    # IDCG（理想情况：所有相关文档排在最前面）
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# 主评估器
# ---------------------------------------------------------------------------

class Evaluator:
    """
    GraphRAG Improved 评估器。

    提供三类评估：
    1. 社区质量评估（无需标注数据）
    2. 检索质量评估（需要 QA 对和相关 chunk_id 标注）
    3. 文本匹配评估（需要预测答案和标准答案）
    """

    def evaluate_community_quality(
        self,
        communities_df: pd.DataFrame,
        relationships_df: Optional[pd.DataFrame] = None,
        purity_threshold: float = 0.01,
    ) -> CommunityMetrics:
        """
        评估社区检测质量。

        Parameters
        ----------
        communities_df : pd.DataFrame
            社区表，必须包含 level, structural_entropy, entity_ids 列
        relationships_df : pd.DataFrame, optional
            关系表，用于计算模块度（若提供）
        purity_threshold : float
            物理纯净社区的熵阈值，默认 0.01

        Returns
        -------
        CommunityMetrics
            社区质量指标
        """
        if communities_df.empty:
            return CommunityMetrics()

        metrics = CommunityMetrics()
        metrics.num_communities = len(communities_df)
        metrics.num_levels = communities_df["level"].nunique()

        # 平均结构熵
        metrics.avg_structural_entropy = float(
            communities_df["structural_entropy"].mean()
        )

        # 各层平均结构熵
        for level in sorted(communities_df["level"].unique()):
            level_df = communities_df[communities_df["level"] == level]
            metrics.entropy_by_level[int(level)] = float(
                level_df["structural_entropy"].mean()
            )

        # Level 0 物理纯净率
        level0_df = communities_df[communities_df["level"] == 0]
        if not level0_df.empty:
            pure = (level0_df["structural_entropy"] < purity_threshold).sum()
            metrics.level0_purity_rate = pure / len(level0_df)

        # 社区大小统计（基于 entity_ids 列）
        sizes = []
        for _, row in communities_df.iterrows():
            entity_ids = row.get("entity_ids", [])
            if isinstance(entity_ids, list):
                sizes.append(len(entity_ids))
            elif isinstance(entity_ids, str):
                sizes.append(len([x for x in entity_ids.split(";") if x.strip()]))

        if sizes:
            metrics.avg_community_size = sum(sizes) / len(sizes)
            mean = metrics.avg_community_size
            variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
            metrics.size_std = math.sqrt(variance)

        # 模块度（简化计算，需要关系表）
        if relationships_df is not None and not relationships_df.empty:
            metrics.modularity = self._compute_modularity(
                communities_df, relationships_df
            )

        return metrics

    def _compute_modularity(
        self,
        communities_df: pd.DataFrame,
        relationships_df: pd.DataFrame,
    ) -> float:
        """
        计算 Level 0 社区的模块度 Q。

        Q = (1/2m) * Σ_ij [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)

        使用简化版本：基于社区内部边数和总边数。
        """
        # 只计算 Level 0
        level0 = communities_df[communities_df["level"] == 0]
        if level0.empty:
            return 0.0

        # 构建实体 → 社区 ID 的映射
        entity_to_comm: Dict[str, int] = {}
        for _, row in level0.iterrows():
            comm_id = int(row["community_id"])
            entity_ids = row.get("entity_ids", [])
            if isinstance(entity_ids, list):
                for eid in entity_ids:
                    entity_to_comm[str(eid)] = comm_id

        if not entity_to_comm:
            return 0.0

        total_weight = 0.0
        internal_weight = 0.0
        degree: Dict[str, float] = Counter()

        for _, row in relationships_df.iterrows():
            src = str(row.get("source", ""))
            tgt = str(row.get("target", ""))
            w = float(row.get("weight", 1.0))
            total_weight += w
            degree[src] += w
            degree[tgt] += w
            if (src in entity_to_comm and tgt in entity_to_comm and
                    entity_to_comm[src] == entity_to_comm[tgt]):
                internal_weight += w

        if total_weight == 0:
            return 0.0

        m = total_weight
        # Q = Σ_c [L_c/m - (d_c/(2m))^2]
        comm_internal: Dict[int, float] = Counter()
        comm_degree: Dict[int, float] = Counter()

        for _, row in relationships_df.iterrows():
            src = str(row.get("source", ""))
            tgt = str(row.get("target", ""))
            w = float(row.get("weight", 1.0))
            if src in entity_to_comm and tgt in entity_to_comm:
                cs, ct = entity_to_comm[src], entity_to_comm[tgt]
                if cs == ct:
                    comm_internal[cs] += w

        for entity, comm_id in entity_to_comm.items():
            comm_degree[comm_id] += degree.get(entity, 0.0)

        q = 0.0
        for comm_id in set(entity_to_comm.values()):
            lc = comm_internal.get(comm_id, 0.0)
            dc = comm_degree.get(comm_id, 0.0)
            q += lc / m - (dc / (2 * m)) ** 2

        return q

    def evaluate_retrieval(
        self,
        qa_pairs: List[QAPair],
        retriever,
        k_values: List[int] = None,
    ) -> RetrievalMetrics:
        """
        评估检索质量。

        Parameters
        ----------
        qa_pairs : List[QAPair]
            问答对列表，每个 QAPair 需要包含 context_ids（相关 chunk_id）
        retriever : URetriever
            检索器实例
        k_values : List[int]
            评估的 K 值列表，默认 [1, 3, 5, 10]

        Returns
        -------
        RetrievalMetrics
            检索质量指标
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        metrics = RetrievalMetrics(num_queries=len(qa_pairs))
        if not qa_pairs:
            return metrics

        # 初始化累加器
        precision_sum = {k: 0.0 for k in k_values}
        recall_sum = {k: 0.0 for k in k_values}
        f1_sum = {k: 0.0 for k in k_values}   # 每个查询单独计算 F1 再求均
        ndcg_sum = {k: 0.0 for k in k_values}
        mrr_sum = 0.0
        valid_n = 0  # 实际参与计算的查询数（排除 context_ids 为空的）

        # 两个版本的 top_down_ids 截断上限：确保检索预算对等
        # 每个社区最多贡献 1 个 doc_id，总上限 top_k_communities 个
        TOP_DOWN_PER_COMM = 1   # 每社区贡献的 doc_id 数
        TOP_DOWN_CAP = 5        # top_down_ids 总上限（与 top_k_communities 一致）

        for qa in qa_pairs:
            if not qa.context_ids:
                continue
            valid_n += 1

            # 执行检索
            result = retriever.retrieve(qa.question)

            # 收集检索到的 chunk_id：
            # 融合策略：交替合并自顶向下（社区）和自底向上（TF-IDF）的结果。
            # 两个版本的 top_down_ids 均截断到相同总长度，确保检索预算对等。
            seen_ids: set = set()
            retrieved_ids: List[str] = []

            # 自底向上：直接 TF-IDF 命中（精确局部检索）
            bottom_ids = [hit.chunk_id for hit in result.bottom_up_hits]

            # 自顶向下：每个社区贡献最多 TOP_DOWN_PER_COMM 个 doc_id，总上限 TOP_DOWN_CAP 个
            # 两个版本均使用相同的截断规则，确保对等性
            top_down_ids: List[str] = []
            for comm_hit in result.top_down_hits:
                count = 0
                for tid in comm_hit.text_unit_ids:
                    if count >= TOP_DOWN_PER_COMM or len(top_down_ids) >= TOP_DOWN_CAP:
                        break
                    top_down_ids.append(tid)
                    count += 1
                if len(top_down_ids) >= TOP_DOWN_CAP:
                    break

            # 交替合并：bottom_up 优先（精确），top_down 补充（社区覆盖）
            max_len = max(len(bottom_ids), len(top_down_ids))
            for i in range(max_len):
                if i < len(bottom_ids):
                    cid = bottom_ids[i]
                    if cid not in seen_ids:
                        retrieved_ids.append(cid)
                        seen_ids.add(cid)
                if i < len(top_down_ids):
                    cid = top_down_ids[i]
                    if cid not in seen_ids:
                        retrieved_ids.append(cid)
                        seen_ids.add(cid)

            # 计算各指标（每个查询单独计算，再求均）
            mrr_sum += compute_mrr(retrieved_ids, qa.context_ids)
            for k in k_values:
                p_i = compute_precision_at_k(retrieved_ids, qa.context_ids, k)
                r_i = compute_recall_at_k(retrieved_ids, qa.context_ids, k)
                f1_i = (2 * p_i * r_i / (p_i + r_i)) if (p_i + r_i) > 0 else 0.0
                precision_sum[k] += p_i
                recall_sum[k] += r_i
                f1_sum[k] += f1_i
                ndcg_sum[k] += compute_ndcg_at_k(retrieved_ids, qa.context_ids, k)

        # 以实际参与计算的查询数为分母（修复无效查询干扰 MRR 的问题）
        n = valid_n if valid_n > 0 else 1
        metrics.mrr = mrr_sum / n
        for k in k_values:
            metrics.precision_at_k[k] = precision_sum[k] / n
            metrics.recall_at_k[k] = recall_sum[k] / n
            metrics.f1_at_k[k] = f1_sum[k] / n   # 先对每个查询计算 F1，再求均
            metrics.ndcg_at_k[k] = ndcg_sum[k] / n

        return metrics

    def evaluate_text_match(
        self,
        predictions: List[str],
        gold_answers: List[str],
    ) -> TextMatchMetrics:
        """
        评估文本匹配质量（用于 QA 生成评估）。

        Parameters
        ----------
        predictions : List[str]
            模型预测答案列表
        gold_answers : List[str]
            标准答案列表

        Returns
        -------
        TextMatchMetrics
            文本匹配指标
        """
        if not predictions or len(predictions) != len(gold_answers):
            return TextMatchMetrics()

        n = len(predictions)
        em_sum = f1_sum = rouge_sum = 0.0

        for pred, gold in zip(predictions, gold_answers):
            em_sum += compute_exact_match(pred, gold)
            f1_sum += compute_token_f1(pred, gold)
            rouge_sum += compute_rouge_l(pred, gold)

        return TextMatchMetrics(
            exact_match=em_sum / n,
            token_f1=f1_sum / n,
            rouge_l=rouge_sum / n,
            num_samples=n,
        )

    def full_report(
        self,
        communities_df: pd.DataFrame,
        relationships_df: Optional[pd.DataFrame] = None,
        qa_pairs: Optional[List[QAPair]] = None,
        retriever=None,
        predictions: Optional[List[str]] = None,
        gold_answers: Optional[List[str]] = None,
    ) -> str:
        """
        生成完整的评估报告文本。

        Parameters
        ----------
        communities_df : pd.DataFrame
            社区表
        relationships_df : pd.DataFrame, optional
            关系表（用于模块度计算）
        qa_pairs : List[QAPair], optional
            问答对（用于检索评估）
        retriever : URetriever, optional
            检索器（用于检索评估）
        predictions : List[str], optional
            预测答案（用于文本匹配评估）
        gold_answers : List[str], optional
            标准答案（用于文本匹配评估）

        Returns
        -------
        str
            完整评估报告文本
        """
        sections = ["=" * 60, "  GraphRAG Improved — 评估报告", "=" * 60]

        # 社区质量
        comm_metrics = self.evaluate_community_quality(communities_df, relationships_df)
        sections.append("\n" + comm_metrics.summary())

        # 检索质量
        if qa_pairs and retriever:
            ret_metrics = self.evaluate_retrieval(qa_pairs, retriever)
            sections.append("\n" + ret_metrics.summary())

        # 文本匹配
        if predictions and gold_answers:
            text_metrics = self.evaluate_text_match(predictions, gold_answers)
            sections.append("\n" + text_metrics.summary())

        sections.append("\n" + "=" * 60)
        return "\n".join(sections)


# ---------------------------------------------------------------------------
# 便捷函数：从 CSV 加载 QA 对
# ---------------------------------------------------------------------------

def load_qa_pairs_from_csv(csv_path: str) -> List[QAPair]:
    """
    从 CSV 文件加载 QA 对。

    CSV 格式：question, answer, context_ids（分号分隔的 chunk_id 列表）

    Parameters
    ----------
    csv_path : str
        CSV 文件路径

    Returns
    -------
    List[QAPair]
        QA 对列表
    """
    df = pd.read_csv(csv_path)
    qa_pairs = []
    for _, row in df.iterrows():
        context_ids = []
        raw = row.get("context_ids", "")
        if isinstance(raw, str) and raw.strip():
            context_ids = [x.strip() for x in raw.split(";") if x.strip()]
        qa_pairs.append(QAPair(
            question=str(row.get("question", "")),
            answer=str(row.get("answer", "")),
            context_ids=context_ids,
        ))
    return qa_pairs
