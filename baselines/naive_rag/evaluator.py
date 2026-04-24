"""
baselines/naive_rag/evaluator.py
----------------------------------
Naive RAG 基线：评估器。

功能：
  在 MultiHop-RAG 数据集上批量评估，输出：
  - P@K, Recall@K, MRR, NDCG@K（检索质量）
  - ROUGE-L（文本匹配质量）
  - 平均 context token 数（效率指标）

用法：
  python evaluator.py \
    --index_path ./naive_rag_index.pkl \
    --data_dir /path/to/multihop_rag \
    --top_k 5 \
    --output_dir ./eval_results \
    --num_samples 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 评估指标计算（独立实现，不依赖项目内部模块）
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """标准化文本：小写、去标点、去多余空格。"""
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
    # 使用滚动数组优化内存
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
    """计算单个查询的 Reciprocal Rank。"""
    relevant_set = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_set:
            return 1.0 / rank
    return 0.0


def compute_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """计算 NDCG@K（二元相关性）。"""
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
class NaiveRAGEvalResult:
    """Naive RAG 评估结果。"""
    system_name: str = "naive_rag"
    num_queries: int = 0
    num_valid_queries: int = 0   # 有 supporting_titles 的查询数

    # 检索指标
    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)

    # 文本匹配指标（基于 context 与 answer 的匹配）
    avg_rouge_l: float = 0.0

    # 效率指标
    avg_context_tokens: float = 0.0
    avg_latency_ms: float = 0.0

    # 按问题类型分组的指标
    by_question_type: Dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "system_name": self.system_name,
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
        }

    def summary(self) -> str:
        k_values = sorted(self.precision_at_k.keys())
        lines = [
            "=" * 60,
            f"  Naive RAG 评估报告",
            "=" * 60,
            f"  查询总数       : {self.num_queries}",
            f"  有效查询数     : {self.num_valid_queries}",
            f"  MRR            : {self.mrr:.4f}",
        ]
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
                    f"MRR={metrics.get('mrr', 0):.4f} "
                    f"R@5={metrics.get('recall_at_5', 0):.4f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主评估函数
# ---------------------------------------------------------------------------

def evaluate_naive_rag(
    index_path: str,
    data_dir: str,
    top_k: int = 5,
    k_values: Optional[List[int]] = None,
    num_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
    filter_doc_titles: Optional[set] = None,
) -> NaiveRAGEvalResult:
    """
    在 MultiHop-RAG 数据集上评估 Naive RAG。

    Parameters
    ----------
    index_path : str
        Naive RAG 索引文件路径
    data_dir : str
        MultiHop-RAG 数据集目录（含 corpus.json 和 MultiHopRAG.json）
    top_k : int
        检索 Top-K 句子
    k_values : List[int], optional
        评估的 K 值列表，默认 [1, 3, 5, 10]
    num_samples : int, optional
        随机采样的 QA 对数量（None 表示全量）
    output_dir : str, optional
        结果输出目录
    seed : int
        随机种子
    filter_doc_titles : set, optional
        若提供，则只评估 supporting_doc_ids 全部在该集合中的 QA 对。
        用于与 GraphRAG 在相同 QA 集上做公平对比。

    Returns
    -------
    NaiveRAGEvalResult
        评估结果
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    # 确保 top_k 覆盖所有 k_values
    effective_top_k = max(top_k, max(k_values))

    # 1. 加载数据集
    logger.info("步骤 1/3：加载 MultiHop-RAG 数据集")
    try:
        # 优先使用项目内的 data_loader
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from graphrag_improved.experiments.data_loader import load_multihop_dataset
        dataset = load_multihop_dataset(data_dir)
    except ImportError:
        # 回退到 baselines 内的 data_loader
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from data_loader import load_multihop_dataset
            dataset = load_multihop_dataset(data_dir)
        except ImportError:
            logger.error("无法导入 data_loader，请确保路径正确")
            raise

    qa_pairs = dataset.qa_pairs
    logger.info(f"  加载了 {len(qa_pairs)} 条 QA 对，{dataset.num_docs} 篇文档")

    # 若提供了文档标题过滤集合，只保留 supporting_doc_ids 全部在其中的 QA 对
    if filter_doc_titles:
        before = len(qa_pairs)
        qa_pairs = [
            qa for qa in qa_pairs
            if not qa.supporting_doc_ids
            or all(d in filter_doc_titles for d in qa.supporting_doc_ids)
        ]
        logger.info(
            f"  [filter_doc_titles] 过滤后：{len(qa_pairs)} 条 QA 对（原 {before} 条）"
        )

    # 随机采样
    if num_samples is not None and num_samples < len(qa_pairs):
        import random
        rng = random.Random(seed)
        qa_pairs = rng.sample(qa_pairs, num_samples)
        logger.info(f"  随机采样 {num_samples} 条 QA 对（seed={seed}）")

    # 2. 初始化检索器
    logger.info("步骤 2/3：初始化 Naive RAG 检索器")
    try:
        from baselines.naive_rag.retriever import NaiveRAGRetriever
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from naive_rag.retriever import NaiveRAGRetriever

    retriever = NaiveRAGRetriever(index_path)
    logger.info(f"  索引包含 {retriever.num_sentences} 个句子")

    # 3. 批量评估
    logger.info(f"步骤 3/3：评估 {len(qa_pairs)} 条 QA 对（top_k={effective_top_k}）")

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

    # 按问题类型分组的累加器
    type_accum: Dict[str, dict] = {}

    for i, qa in enumerate(qa_pairs):
        if (i + 1) % 50 == 0:
            logger.info(f"  进度：{i + 1}/{len(qa_pairs)}")

        # 执行检索并计时
        t0 = time.perf_counter()
        try:
            result = retriever.retrieve(qa.query, top_k=effective_top_k)
        except Exception as e:
            logger.warning(f"  查询 {i} 检索失败：{e}")
            continue
        latency_ms = (time.perf_counter() - t0) * 1000

        latency_sum += latency_ms
        token_sum += result.total_context_tokens

        # 获取检索到的 doc_id 列表（按排名顺序）
        retrieved_doc_ids = result.get_doc_ids()

        # 获取相关 doc_id（来自 supporting_titles）
        relevant_doc_ids = qa.supporting_doc_ids

        if not relevant_doc_ids:
            # 没有标注的相关文档，跳过检索指标计算
            # 但仍计算 ROUGE-L（context 与 answer 的匹配）
            context_text = result.get_context_text()
            rouge_sum += compute_rouge_l(context_text, qa.answer)
            continue

        valid_n += 1

        # 计算检索指标
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

        # ROUGE-L（context 与 answer 的匹配）
        context_text = result.get_context_text()
        rouge_sum += compute_rouge_l(context_text, qa.answer)

        # 按问题类型分组
        qtype = qa.question_type
        if qtype not in type_accum:
            type_accum[qtype] = {
                "n": 0, "mrr": 0.0, "recall_at_5": 0.0, "tokens": 0.0
            }
        type_accum[qtype]["n"] += 1
        type_accum[qtype]["mrr"] += mrr_val
        type_accum[qtype]["recall_at_5"] += compute_recall_at_k(
            retrieved_doc_ids, relevant_doc_ids, 5
        )
        type_accum[qtype]["tokens"] += result.total_context_tokens

    # 汇总结果
    total_n = len(qa_pairs)
    n = valid_n if valid_n > 0 else 1

    result_obj = NaiveRAGEvalResult(
        system_name="naive_rag",
        num_queries=total_n,
        num_valid_queries=valid_n,
        mrr=mrr_sum / n,
        avg_rouge_l=rouge_sum / total_n if total_n > 0 else 0.0,
        avg_context_tokens=token_sum / total_n if total_n > 0 else 0.0,
        avg_latency_ms=latency_sum / total_n if total_n > 0 else 0.0,
    )

    for k in k_values:
        result_obj.precision_at_k[k] = precision_sum[k] / n
        result_obj.recall_at_k[k] = recall_sum[k] / n
        result_obj.f1_at_k[k] = f1_sum[k] / n
        result_obj.ndcg_at_k[k] = ndcg_sum[k] / n

    # 按问题类型汇总
    for qtype, accum in type_accum.items():
        cnt = accum["n"] if accum["n"] > 0 else 1
        result_obj.by_question_type[qtype] = {
            "n": accum["n"],
            "mrr": accum["mrr"] / cnt,
            "recall_at_5": accum["recall_at_5"] / cnt,
            "avg_tokens": accum["tokens"] / cnt,
        }

    # 输出结果
    logger.info("\n" + result_obj.summary())

    # 保存结果
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        result_file = output_path / "naive_rag_eval_results.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_obj.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"  ✓ 结果已保存：{result_file}")

    return result_obj


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
        description="Naive RAG 基线评估器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index_path",
        type=str,
        required=True,
        help="Naive RAG 索引文件路径",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="MultiHop-RAG 数据集目录（含 corpus.json 和 MultiHopRAG.json）",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="检索 Top-K 句子",
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
        help="随机采样的 QA 对数量（None 表示全量）",
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
        "--filter_graphrag_workspace",
        type=str,
        default=None,
        help=(
            "若提供 GraphRAG workspace 目录，则只评估 supporting_doc_ids 全部在该 workspace "
            "input/ 目录中的 QA 对，用于与 GraphRAG 在相同 QA 集上做公平对比"
        ),
    )

    args = parser.parse_args()

    # 若指定了 GraphRAG workspace，读取其 input/ 目录下的文章标题
    filter_doc_titles = None
    if args.filter_graphrag_workspace:
        input_dir = Path(args.filter_graphrag_workspace) / "input"
        if input_dir.exists():
            filter_doc_titles = set()
            for f in input_dir.glob("*.txt"):
                try:
                    with open(f, "r", encoding="utf-8") as fp:
                        line = fp.readline().strip()
                    if line.startswith("Title:"):
                        filter_doc_titles.add(line[6:].strip())
                except Exception:
                    pass
            logger.info(f"  从 GraphRAG workspace 读取了 {len(filter_doc_titles)} 篇文章标题")

    try:
        evaluate_naive_rag(
            index_path=args.index_path,
            data_dir=args.data_dir,
            top_k=args.top_k,
            k_values=args.k_values,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            seed=args.seed,
            filter_doc_titles=filter_doc_titles,
        )
    except Exception as e:
        logger.error(f"评估失败：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
