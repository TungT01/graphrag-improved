"""
baselines/run_evaluation.py
-----------------------------
统一评估入口：在 MultiHop-RAG 数据集上对比三个系统的性能。

支持的系统：
  - naive_rag        : 逐句向量化检索（sentence-transformers all-MiniLM-L6-v2）
  - graphrag_local   : Microsoft GraphRAG local search（微观检索）
  - graphrag_global  : Microsoft GraphRAG global search（宏观检索）

用法示例：

  # 评估 Naive RAG（需要先构建索引）
  python run_evaluation.py \
    --system naive_rag \
    --data_path /path/to/multihop_rag \
    --index_path ./naive_rag_index.pkl \
    --top_k 5 \
    --num_samples 200 \
    --output_dir ./eval_results

  # 评估 GraphRAG Local Search
  python run_evaluation.py \
    --system graphrag_local \
    --data_path /path/to/multihop_rag \
    --num_samples 50 \
    --output_dir ./eval_results

  # 评估 GraphRAG Global Search
  python run_evaluation.py \
    --system graphrag_global \
    --data_path /path/to/multihop_rag \
    --num_samples 50 \
    --output_dir ./eval_results

  # 先构建 Naive RAG 索引
  python run_evaluation.py \
    --system naive_rag \
    --build_index \
    --corpus_json /path/to/corpus.json \
    --index_path ./naive_rag_index.pkl

  # 对比所有系统（需要先分别构建索引）
  python run_evaluation.py \
    --system all \
    --data_path /path/to/multihop_rag \
    --index_path ./naive_rag_index.pkl \
    --num_samples 100 \
    --output_dir ./eval_results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# 确保 baselines 目录在 Python 路径中
_BASELINES_DIR = Path(__file__).parent
_PROJECT_ROOT = _BASELINES_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_BASELINES_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 各系统评估函数
# ---------------------------------------------------------------------------

def run_naive_rag_eval(
    data_path: str,
    index_path: str,
    top_k: int = 5,
    k_values: Optional[List[int]] = None,
    num_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """运行 Naive RAG 评估。"""
    logger.info("\n" + "=" * 60)
    logger.info("  运行 Naive RAG 评估")
    logger.info("=" * 60)

    from naive_rag.evaluator import evaluate_naive_rag

    result = evaluate_naive_rag(
        index_path=index_path,
        data_dir=data_path,
        top_k=top_k,
        k_values=k_values or [1, 3, 5, 10],
        num_samples=num_samples,
        output_dir=output_dir,
        seed=seed,
    )
    return result.to_dict()


def run_graphrag_local_eval(
    data_path: str,
    workspace_dir: Optional[str] = None,
    k_values: Optional[List[int]] = None,
    num_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """运行 GraphRAG Local Search 评估。"""
    logger.info("\n" + "=" * 60)
    logger.info("  运行 GraphRAG Local Search 评估")
    logger.info("=" * 60)

    from graphrag_official.evaluator import evaluate_graphrag

    result = evaluate_graphrag(
        data_dir=data_path,
        search_type="local",
        workspace_dir=workspace_dir,
        k_values=k_values or [1, 3, 5, 10],
        num_samples=num_samples,
        output_dir=output_dir,
        seed=seed,
    )
    return result.to_dict()


def run_graphrag_global_eval(
    data_path: str,
    workspace_dir: Optional[str] = None,
    num_samples: Optional[int] = None,
    output_dir: Optional[str] = None,
    seed: int = 42,
) -> dict:
    """运行 GraphRAG Global Search 评估。"""
    logger.info("\n" + "=" * 60)
    logger.info("  运行 GraphRAG Global Search 评估")
    logger.info("=" * 60)

    from graphrag_official.evaluator import evaluate_graphrag

    result = evaluate_graphrag(
        data_dir=data_path,
        search_type="global",
        workspace_dir=workspace_dir,
        num_samples=num_samples,
        output_dir=output_dir,
        seed=seed,
    )
    return result.to_dict()


# ---------------------------------------------------------------------------
# 索引构建
# ---------------------------------------------------------------------------

def build_naive_rag_index(
    corpus_json: Optional[str] = None,
    input_dir: Optional[str] = None,
    index_path: str = "./naive_rag_index.pkl",
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """构建 Naive RAG 索引。"""
    logger.info("\n" + "=" * 60)
    logger.info("  构建 Naive RAG 索引")
    logger.info("=" * 60)

    from naive_rag.indexer import build_index

    build_index(
        input_dir=input_dir,
        corpus_json=corpus_json,
        index_path=index_path,
        model_name=model_name,
    )


def build_graphrag_index(
    corpus_json: Optional[str] = None,
    input_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    max_docs: Optional[int] = None,
) -> None:
    """构建 GraphRAG 官方索引。"""
    logger.info("\n" + "=" * 60)
    logger.info("  构建 GraphRAG 官方索引")
    logger.info("=" * 60)

    from graphrag_official.indexer import build_index

    success = build_index(
        input_dir=input_dir,
        corpus_json=corpus_json,
        workspace_dir=workspace_dir,
        max_docs=max_docs,
    )
    if not success:
        raise RuntimeError("GraphRAG 索引构建失败")


# ---------------------------------------------------------------------------
# 对比报告生成
# ---------------------------------------------------------------------------

def generate_comparison_report(
    results: Dict[str, dict],
    output_dir: str,
) -> str:
    """
    生成多系统对比报告。

    Parameters
    ----------
    results : Dict[str, dict]
        各系统的评估结果字典
    output_dir : str
        输出目录

    Returns
    -------
    str
        报告文本
    """
    lines = [
        "=" * 70,
        "  多系统对比评估报告",
        "=" * 70,
        "",
    ]

    # 检索指标对比表
    k_values = [1, 3, 5, 10]
    systems = list(results.keys())

    if systems:
        lines.append("检索指标对比（P@K / R@K / MRR / NDCG@K）：")
        lines.append("-" * 70)

        # 表头
        header = f"{'指标':<20}" + "".join(f"{s:<20}" for s in systems)
        lines.append(header)
        lines.append("-" * 70)

        # MRR
        row = f"{'MRR':<20}"
        for s in systems:
            mrr = results[s].get("mrr", 0)
            row += f"{mrr:<20.4f}"
        lines.append(row)

        # P@K, R@K, NDCG@K
        for k in k_values:
            for metric_name, metric_key in [("P", "precision_at_k"), ("R", "recall_at_k"), ("NDCG", "ndcg_at_k")]:
                row = f"{metric_name}@{k:<18}"
                for s in systems:
                    val = results[s].get(metric_key, {}).get(k, 0)
                    row += f"{val:<20.4f}"
                lines.append(row)

        lines.append("-" * 70)

        # ROUGE-L
        row = f"{'ROUGE-L':<20}"
        for s in systems:
            val = results[s].get("avg_rouge_l", 0)
            row += f"{val:<20.4f}"
        lines.append(row)

        # Context Tokens
        row = f"{'Avg Context Tokens':<20}"
        for s in systems:
            val = results[s].get("avg_context_tokens", 0)
            row += f"{val:<20.1f}"
        lines.append(row)

        # Latency
        row = f"{'Avg Latency (ms)':<20}"
        for s in systems:
            val = results[s].get("avg_latency_ms", 0)
            row += f"{val:<20.1f}"
        lines.append(row)

        lines.append("=" * 70)

    report_text = "\n".join(lines)
    print(report_text)

    # 保存报告
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_file = output_path / "comparison_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    logger.info(f"  ✓ 对比报告已保存：{report_file}")

    # 保存 JSON 格式
    json_file = output_path / "comparison_results.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✓ JSON 结果已保存：{json_file}")

    return report_text


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG 基线系统统一评估入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 系统选择
    parser.add_argument(
        "--system",
        type=str,
        choices=["naive_rag", "graphrag_local", "graphrag_global", "all"],
        required=True,
        help="要评估的系统",
    )

    # 数据集
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="MultiHop-RAG 数据集目录（含 corpus.json 和 MultiHopRAG.json）",
    )

    # Naive RAG 专用参数
    parser.add_argument(
        "--index_path",
        type=str,
        default="./naive_rag_index.pkl",
        help="Naive RAG 索引文件路径",
    )
    parser.add_argument(
        "--build_index",
        action="store_true",
        help="构建索引（而非评估）",
    )
    parser.add_argument(
        "--corpus_json",
        type=str,
        default=None,
        help="corpus.json 文件路径（用于构建索引）",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="包含 .txt 文件的目录（用于构建索引）",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer 模型名称",
    )

    # GraphRAG 专用参数
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default=None,
        help="GraphRAG 工作目录",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="最大文档数量（用于 GraphRAG 索引构建）",
    )

    # 通用评估参数
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="检索 Top-K（Naive RAG）",
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

    args = parser.parse_args()

    # 自动发现数据集路径
    if args.data_path is None and not args.build_index:
        from data_loader import find_default_data_dir
        args.data_path = find_default_data_dir()
        if args.data_path:
            logger.info(f"自动发现数据集：{args.data_path}")
        else:
            logger.error(
                "未找到数据集，请指定 --data_path\n"
                "或将数据集放在：/Users/ttung/Desktop/个人学习/data/multihop_rag/"
            )
            sys.exit(1)

    # 构建索引模式
    if args.build_index:
        if args.system in ("naive_rag", "all"):
            if args.corpus_json is None and args.input_dir is None:
                # 尝试使用默认 corpus.json
                default_corpus = Path("/Users/ttung/Desktop/个人学习/data/multihop_rag/corpus.json")
                if default_corpus.exists():
                    args.corpus_json = str(default_corpus)
                    logger.info(f"使用默认 corpus.json：{args.corpus_json}")
                else:
                    parser.error("构建 Naive RAG 索引需要 --corpus_json 或 --input_dir")

            build_naive_rag_index(
                corpus_json=args.corpus_json,
                input_dir=args.input_dir,
                index_path=args.index_path,
                model_name=args.model_name,
            )

        if args.system in ("graphrag_local", "graphrag_global", "all"):
            if args.corpus_json is None and args.input_dir is None:
                default_corpus = Path("/Users/ttung/Desktop/个人学习/data/multihop_rag/corpus.json")
                if default_corpus.exists():
                    args.corpus_json = str(default_corpus)
                else:
                    parser.error("构建 GraphRAG 索引需要 --corpus_json 或 --input_dir")

            build_graphrag_index(
                corpus_json=args.corpus_json,
                input_dir=args.input_dir,
                workspace_dir=args.workspace_dir,
                max_docs=args.max_docs,
            )
        return

    # 评估模式
    all_results = {}

    try:
        if args.system in ("naive_rag", "all"):
            if not Path(args.index_path).exists():
                logger.warning(
                    f"Naive RAG 索引不存在：{args.index_path}\n"
                    "请先运行：python run_evaluation.py --system naive_rag --build_index"
                )
                if args.system == "naive_rag":
                    sys.exit(1)
            else:
                result = run_naive_rag_eval(
                    data_path=args.data_path,
                    index_path=args.index_path,
                    top_k=args.top_k,
                    k_values=args.k_values,
                    num_samples=args.num_samples,
                    output_dir=args.output_dir,
                    seed=args.seed,
                )
                all_results["naive_rag"] = result

        if args.system in ("graphrag_local", "all"):
            result = run_graphrag_local_eval(
                data_path=args.data_path,
                workspace_dir=args.workspace_dir,
                k_values=args.k_values,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                seed=args.seed,
            )
            all_results["graphrag_local"] = result

        if args.system in ("graphrag_global", "all"):
            result = run_graphrag_global_eval(
                data_path=args.data_path,
                workspace_dir=args.workspace_dir,
                num_samples=args.num_samples,
                output_dir=args.output_dir,
                seed=args.seed,
            )
            all_results["graphrag_global"] = result

        # 生成对比报告（当有多个系统时）
        if len(all_results) > 1:
            generate_comparison_report(all_results, args.output_dir)

    except KeyboardInterrupt:
        logger.info("\n评估被用户中断")
        if all_results:
            logger.info("保存已完成的结果...")
            generate_comparison_report(all_results, args.output_dir)
        sys.exit(0)
    except Exception as e:
        logger.error(f"评估失败：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
