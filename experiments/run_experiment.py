"""
experiments/run_experiment.py
------------------------------
对照实验主脚本：在 MultiHop-RAG 数据集上对比
  - Baseline : 原版 Leiden（λ=0，无结构熵约束）
  - Ours     : 结构熵约束 Leiden（λ=1000，指数退火）

两个版本使用完全相同的：
  - 实体抽取（规则后端）
  - 检索方式（TF-IDF U-Retrieval）
  - 评估指标（Precision@K / Recall@K / MRR / NDCG@K）

唯一变量：聚类时的 λ 值（0 vs 1000）

用法：
  python -m graphrag_improved.experiments.run_experiment \\
      --data-dir ./data/multihop_rag \\
      --n-qa 200 \\
      --output-dir ./experiments/results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# 将项目根目录加入 sys.path（支持直接运行）
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
from graphrag_improved.constrained_leiden.graphrag_workflow import (
    run_constrained_community_detection,
)
from graphrag_improved.data.ingestion import TextUnit, InputConfig
from graphrag_improved.evaluation.evaluator import (
    Evaluator,
    QAPair as EvalQAPair,
    RetrievalMetrics,
    CommunityMetrics,
)
from graphrag_improved.experiments.data_loader import (
    MultiHopDataset,
    corpus_to_text_units,
    load_multihop_dataset,
    try_download_dataset,
)
from graphrag_improved.extraction.extractor import extract
from graphrag_improved.pipeline_config import ExtractionConfig
from graphrag_improved.retrieval.retriever import URetriever


# ---------------------------------------------------------------------------
# 实验配置
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """单次实验的配置。"""
    name: str                          # 实验名称（如 "baseline" / "ours"）
    lambda_init: float                 # λ 初始值（0 = 无约束）
    lambda_min: float = 0.0
    annealing_schedule: str = "exponential"
    decay_rate: float = 0.5
    max_cluster_size: int = 10
    max_iterations: int = 10
    seed: int = 42
    top_k_communities: int = 5
    top_k_chunks: int = 5
    alpha: float = 0.5                 # U-Retrieval 融合比例
    min_entity_freq: int = 5           # 实体最低出现频次（过滤噪声）


@dataclass
class ExperimentResult:
    """单次实验的完整结果。"""
    config: ExperimentConfig
    community_metrics: CommunityMetrics
    retrieval_metrics: RetrievalMetrics
    elapsed_seconds: float
    num_communities: int
    num_levels: int
    num_entities: int
    num_relationships: int


# ---------------------------------------------------------------------------
# 核心：将 corpus 转换为 TextUnit（保留 doc_id 作为 chunk_id）
# ---------------------------------------------------------------------------

def corpus_to_pipeline_text_units(dataset: MultiHopDataset) -> List[TextUnit]:
    """
    将 MultiHop-RAG corpus 转换为 Pipeline 所需的 TextUnit 列表。

    v3 变更：
        每篇文章先切分为段落，再用 spaCy 切分为句子，
        生成三级 ID 结构：
          doc_id  = doc.doc_id（文章标题，与 MultiHop-RAG 的 evidence 对齐）
          para_id = {doc_id}-p{para_idx:03d}
          sent_id = {para_id}-s{sent_idx:03d}
    """
    from ..data.ingestion import Document, document_to_text_units, SentenceUnit
    from ..data.ingestion import _make_para_id, _make_sent_id
    units = []
    for doc in dataset.corpus:
        document = Document(
            doc_id=doc.doc_id,
            title=doc.title,
            raw_text=doc.body,
            source_path=f"multihop_rag/{doc.doc_id}",
        )
        try:
            doc_units = document_to_text_units(
                document,
                spacy_model="en_core_web_sm",
                use_spacy_sentences=True,
            )
        except Exception:
            # spaCy 不可用时降级为单一 TextUnit
            para_id = _make_para_id(doc.doc_id, 0)
            sent_id = _make_sent_id(para_id, 0)
            doc_units = [TextUnit(
                chunk_id=para_id,
                text=doc.body,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                chunk_index=0,
                sentences=[SentenceUnit(
                    sent_id=sent_id,
                    text=doc.body[:500],
                    doc_id=doc.doc_id,
                    para_id=para_id,
                    sent_index=0,
                    para_index=0,
                )],
                metadata={"source": doc.source},
            )]
        units.extend(doc_units)
    return units


def dataset_qa_to_eval_qa(dataset: MultiHopDataset) -> List[EvalQAPair]:
    """
    将 MultiHopDataset 的 QA 对转换为评估框架所需的 EvalQAPair 格式。

    v3 说明：
        context_ids 使用 doc_id（文章标题）作为 ground-truth。
        MultiHop-RAG 的 supporting_evidence 是文章级的，评估时
        将检索结果中的 doc_id 与此对齐。
    """
    return [
        EvalQAPair(
            question=qa.query,
            answer=qa.answer,
            context_ids=qa.supporting_doc_ids,  # ground-truth doc_id 列表（文章标题）
            metadata={"question_type": qa.question_type},
        )
        for qa in dataset.qa_pairs
        if qa.supporting_doc_ids
    ]


# ---------------------------------------------------------------------------
# 单次实验运行
# ---------------------------------------------------------------------------

def run_single_experiment(
    exp_config: ExperimentConfig,
    text_units: List[TextUnit],
    retrieval_text_units: List[dict],
    eval_qa_pairs: List[EvalQAPair],
    verbose: bool = True,
) -> ExperimentResult:
    """
    运行单次实验（一个 λ 配置）。

    Parameters
    ----------
    exp_config       : 实验配置
    text_units       : Pipeline 用的 TextUnit 列表（含 doc_id）
    retrieval_text_units : URetriever 用的 dict 列表
    eval_qa_pairs    : 评估用的 QA 对
    """
    t_start = time.time()
    if verbose:
        print(f"\n{'─'*50}")
        print(f"  实验：{exp_config.name}  (λ={exp_config.lambda_init})")
        print(f"{'─'*50}")

    # ------------------------------------------------------------------
    # Step 1: 实体与关系抽取（两个版本共享同一份抽取结果）
    # ------------------------------------------------------------------
    if verbose:
        print(f"  [1/3] 实体抽取（{len(text_units)} 篇文章）...")
        t1 = time.time()
    extraction_config = ExtractionConfig(
        backend="rule",
        min_entity_freq=exp_config.min_entity_freq,
        cooccurrence_window=3,
    )
    entities_df, relationships_df = extract(text_units, extraction_config)
    if verbose:
        print(f"        实体 {len(entities_df)} 个，关系 {len(relationships_df)} 条  "
              f"({time.time()-t1:.1f}s)")

    # ------------------------------------------------------------------
    # Step 2: 社区检测（唯一变量：λ）
    # ------------------------------------------------------------------
    if verbose:
        print(f"  [2/3] 社区检测 (λ={exp_config.lambda_init})...")

    annealing_config = AnnealingConfig(
        lambda_init=exp_config.lambda_init,
        lambda_min=exp_config.lambda_min,
        schedule=AnnealingSchedule(exp_config.annealing_schedule),
        decay_rate=exp_config.decay_rate,
        max_level=10,
    )
    communities_df = run_constrained_community_detection(
        entities=entities_df,
        relationships=relationships_df,
        annealing_config=annealing_config,
        max_cluster_size=exp_config.max_cluster_size,
        max_iterations=exp_config.max_iterations,
        seed=exp_config.seed,
    )
    num_levels = communities_df["level"].nunique() if not communities_df.empty else 0
    if verbose:
        print(f"        社区 {len(communities_df)} 个，层次 {num_levels} 层")

    # ------------------------------------------------------------------
    # Step 3: U-Retrieval + 评估
    # ------------------------------------------------------------------
    if verbose:
        print("  [3/3] 检索评估...")

    retriever = URetriever(
        communities_df=communities_df,
        text_units=retrieval_text_units,
        entities_df=entities_df,
        top_k_communities=exp_config.top_k_communities,
        top_k_chunks=exp_config.top_k_chunks,
    )

    evaluator = Evaluator()
    community_metrics = evaluator.evaluate_community_quality(
        communities_df, relationships_df
    )
    retrieval_metrics = evaluator.evaluate_retrieval(
        eval_qa_pairs, retriever, k_values=[1, 3, 5, 10]
    )

    elapsed = time.time() - t_start
    if verbose:
        print(f"        MRR={retrieval_metrics.mrr:.4f}  "
              f"P@5={retrieval_metrics.precision_at_k.get(5, 0):.4f}  "
              f"NDCG@10={retrieval_metrics.ndcg_at_k.get(10, 0):.4f}")
        print(f"        耗时 {elapsed:.1f}s")

    return ExperimentResult(
        config=exp_config,
        community_metrics=community_metrics,
        retrieval_metrics=retrieval_metrics,
        elapsed_seconds=elapsed,
        num_communities=len(communities_df),
        num_levels=num_levels,
        num_entities=len(entities_df),
        num_relationships=len(relationships_df),
    )


# ---------------------------------------------------------------------------
# 对照实验：baseline vs ours
# ---------------------------------------------------------------------------

def run_comparison(
    dataset: MultiHopDataset,
    n_qa: Optional[int] = None,
    output_dir: str = "./experiments/results",
    verbose: bool = True,
) -> Tuple[ExperimentResult, ExperimentResult]:
    """
    在同一份数据上运行 baseline 和 ours 两个版本，返回对比结果。

    Parameters
    ----------
    dataset   : MultiHopDataset
    n_qa      : 使用的 QA 对数量（None = 全部）
    output_dir: 结果保存目录
    """
    # 准备数据
    if n_qa is not None:
        dataset = dataset.subset(n_qa, seed=42)
        if verbose:
            print(f"  使用 {dataset.num_qa} 条 QA 对（随机采样，seed=42）")

    text_units = corpus_to_pipeline_text_units(dataset)
    # v3：检索用的 text_units 使用文章级粒度（每篇文章一个单元）
    # 评估时的 ground-truth 是 doc_id（文章标题），与此对齐
    retrieval_text_units = corpus_to_text_units(dataset.corpus)
    eval_qa_pairs = dataset_qa_to_eval_qa(dataset)

    if verbose:
        print(f"\n  数据规模：{dataset.num_docs} 篇文章，{len(eval_qa_pairs)} 条有效 QA 对")
        # 统计问题类型分布
        type_counts: Dict[str, int] = {}
        for qa in dataset.qa_pairs:
            type_counts[qa.question_type] = type_counts.get(qa.question_type, 0) + 1
        print(f"  问题类型分布：{type_counts}")

    # 实验配置
    baseline_config = ExperimentConfig(
        name="Baseline (λ=0, 原版Leiden)",
        lambda_init=0.0,
        lambda_min=0.0,
    )
    ours_config = ExperimentConfig(
        name="Ours (λ=1000, 结构熵约束)",
        lambda_init=1000.0,
        lambda_min=0.0,
        annealing_schedule="exponential",
        decay_rate=0.5,
    )

    # 运行实验
    if verbose:
        print("\n" + "="*50)
        print("  开始对照实验")
        print("="*50)

    result_baseline = run_single_experiment(
        baseline_config, text_units, retrieval_text_units, eval_qa_pairs, verbose
    )
    result_ours = run_single_experiment(
        ours_config, text_units, retrieval_text_units, eval_qa_pairs, verbose
    )

    # 保存结果
    _save_results(result_baseline, result_ours, output_dir, verbose)

    return result_baseline, result_ours


# ---------------------------------------------------------------------------
# 结果保存与打印
# ---------------------------------------------------------------------------

def _save_results(
    baseline: ExperimentResult,
    ours: ExperimentResult,
    output_dir: str,
    verbose: bool = True,
) -> None:
    """保存实验结果为 JSON 和打印对比表格。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _result_to_dict(r: ExperimentResult) -> dict:
        rm = r.retrieval_metrics
        cm = r.community_metrics
        return {
            "name": r.config.name,
            "lambda_init": r.config.lambda_init,
            "num_entities": r.num_entities,
            "num_relationships": r.num_relationships,
            "num_communities": r.num_communities,
            "num_levels": r.num_levels,
            "elapsed_seconds": round(r.elapsed_seconds, 2),
            # 社区质量
            "modularity": round(cm.modularity, 4),
            "avg_structural_entropy": round(cm.avg_structural_entropy, 4),
            "level0_purity_rate": round(cm.level0_purity_rate, 4),
            "avg_community_size": round(cm.avg_community_size, 2),
            # 检索质量
            "mrr": round(rm.mrr, 4),
            "precision_at_1": round(rm.precision_at_k.get(1, 0), 4),
            "precision_at_3": round(rm.precision_at_k.get(3, 0), 4),
            "precision_at_5": round(rm.precision_at_k.get(5, 0), 4),
            "precision_at_10": round(rm.precision_at_k.get(10, 0), 4),
            "recall_at_5": round(rm.recall_at_k.get(5, 0), 4),
            "recall_at_10": round(rm.recall_at_k.get(10, 0), 4),
            "f1_at_5": round(rm.f1_at_k.get(5, 0), 4),
            "ndcg_at_5": round(rm.ndcg_at_k.get(5, 0), 4),
            "ndcg_at_10": round(rm.ndcg_at_k.get(10, 0), 4),
            "entropy_by_level": {
                str(k): round(v, 4)
                for k, v in cm.entropy_by_level.items()
            },
        }

    results = {
        "baseline": _result_to_dict(baseline),
        "ours": _result_to_dict(ours),
    }

    # 计算提升幅度
    def _delta(key: str) -> str:
        b = results["baseline"].get(key, 0) or 0
        o = results["ours"].get(key, 0) or 0
        if b == 0:
            return "N/A"
        delta = (o - b) / b * 100
        sign = "+" if delta >= 0 else ""
        return f"{sign}{delta:.1f}%"

    results["improvement"] = {
        "mrr": _delta("mrr"),
        "precision_at_5": _delta("precision_at_5"),
        "recall_at_10": _delta("recall_at_10"),
        "ndcg_at_10": _delta("ndcg_at_10"),
        "level0_purity_rate": _delta("level0_purity_rate"),
        "avg_structural_entropy": _delta("avg_structural_entropy"),
    }

    # 保存 JSON
    result_path = out / "comparison_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if verbose:
        _print_comparison_table(results)
        print(f"\n  结果已保存：{result_path}")


def _print_comparison_table(results: dict) -> None:
    """打印对比表格。"""
    b = results["baseline"]
    o = results["ours"]
    imp = results["improvement"]

    print("\n" + "="*70)
    print("  实验结果对比：MultiHop-RAG 检索评估")
    print("="*70)

    rows = [
        ("指标", "Baseline (λ=0)", "Ours (λ=1000)", "提升"),
        ("─"*20, "─"*16, "─"*14, "─"*10),
        # 检索质量
        ("MRR",
         f"{b['mrr']:.4f}", f"{o['mrr']:.4f}", imp["mrr"]),
        ("Precision@1",
         f"{b['precision_at_1']:.4f}", f"{o['precision_at_1']:.4f}", ""),
        ("Precision@3",
         f"{b['precision_at_3']:.4f}", f"{o['precision_at_3']:.4f}", ""),
        ("Precision@5",
         f"{b['precision_at_5']:.4f}", f"{o['precision_at_5']:.4f}", imp["precision_at_5"]),
        ("Recall@5",
         f"{b['recall_at_5']:.4f}", f"{o['recall_at_5']:.4f}", ""),
        ("Recall@10",
         f"{b['recall_at_10']:.4f}", f"{o['recall_at_10']:.4f}", imp["recall_at_10"]),
        ("NDCG@5",
         f"{b['ndcg_at_5']:.4f}", f"{o['ndcg_at_5']:.4f}", ""),
        ("NDCG@10",
         f"{b['ndcg_at_10']:.4f}", f"{o['ndcg_at_10']:.4f}", imp["ndcg_at_10"]),
        ("─"*20, "─"*16, "─"*14, "─"*10),
        # 社区质量
        ("模块度 Q",
         f"{b['modularity']:.4f}", f"{o['modularity']:.4f}", ""),
        ("平均结构熵",
         f"{b['avg_structural_entropy']:.4f}", f"{o['avg_structural_entropy']:.4f}",
         imp["avg_structural_entropy"]),
        ("Level0 纯净率",
         f"{b['level0_purity_rate']:.2%}", f"{o['level0_purity_rate']:.2%}",
         imp["level0_purity_rate"]),
        ("社区数量",
         str(b['num_communities']), str(o['num_communities']), ""),
        ("层次数量",
         str(b['num_levels']), str(o['num_levels']), ""),
    ]

    for row in rows:
        print(f"  {row[0]:<22} {row[1]:<18} {row[2]:<16} {row[3]}")

    print("="*70)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MultiHop-RAG 对照实验：原版 Leiden vs 结构熵约束版本"
    )
    parser.add_argument(
        "--data-dir",
        default="./data/multihop_rag",
        help="MultiHop-RAG 数据集目录（含 corpus.json 和 MultiHopRAG.json）",
    )
    parser.add_argument(
        "--n-qa",
        type=int,
        default=None,
        help="使用的 QA 对数量（默认全部 2556 条，建议先用 200 快速验证）",
    )
    parser.add_argument(
        "--output-dir",
        default="./experiments/results",
        help="结果保存目录",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="尝试从 HuggingFace 自动下载数据集",
    )
    parser.add_argument(
        "--question-type",
        default=None,
        choices=["inference_query", "comparison_query", "temporal_query", "null_query"],
        help="只评估特定类型的问题",
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  GraphRAG Improved — MultiHop-RAG 对照实验")
    print("="*60)

    # 尝试下载数据集
    data_dir = Path(args.data_dir)
    if args.download or not (data_dir / "corpus.json").exists():
        print("\n[数据集] 尝试下载...")
        success = try_download_dataset(str(data_dir))
        if not success:
            print("\n[错误] 无法自动下载数据集，请手动操作：")
            print("  git clone https://github.com/yixuantt/MultiHop-RAG")
            print(f"  cp -r MultiHop-RAG/dataset/* {data_dir}/")
            sys.exit(1)

    # 加载数据集
    print("\n[数据集] 加载中...")
    dataset = load_multihop_dataset(str(data_dir))

    # 按问题类型过滤
    if args.question_type:
        dataset = dataset.filter_by_type(args.question_type)
        print(f"  过滤后：{dataset.num_qa} 条 {args.question_type} 类型问题")

    # 运行对照实验
    baseline_result, ours_result = run_comparison(
        dataset=dataset,
        n_qa=args.n_qa,
        output_dir=args.output_dir,
        verbose=True,
    )

    print("\n✅ 实验完成！")


if __name__ == "__main__":
    main()
