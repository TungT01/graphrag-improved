"""
experiments/smoke_test.py
--------------------------
用模拟数据验证整个实验流程是否跑通，无需真实数据集。

模拟场景：
  - 10 篇来自不同"文档"的新闻文章（模拟 MultiHop-RAG corpus）
  - 5 条跨文档 QA 对（每条证据跨 2 篇文章）
  - 对比 baseline (λ=0) 和 ours (λ=1000)

用法：
  python -m graphrag_improved.experiments.smoke_test
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.experiments.data_loader import (
    CorpusDoc, MultiHopDataset, QAPair, corpus_to_text_units,
)
from graphrag_improved.experiments.run_experiment import (
    ExperimentConfig, corpus_to_pipeline_text_units,
    dataset_qa_to_eval_qa, run_single_experiment,
    _print_comparison_table, _save_results, ExperimentResult,
)


# ---------------------------------------------------------------------------
# 构造模拟数据
# ---------------------------------------------------------------------------

def make_mock_dataset() -> MultiHopDataset:
    """
    构造模拟的 MultiHop-RAG 数据集。
    10 篇文章，分属 3 个"物理文档"（模拟不同来源），5 条跨文档 QA。
    """
    corpus = [
        # 文档组 A：关于 Leiden 算法
        CorpusDoc(
            doc_id="doc_A1",
            title="Leiden Algorithm Overview",
            body=(
                "The Leiden algorithm is a community detection method that improves upon "
                "the Louvain algorithm. It guarantees well-connected communities by adding "
                "a refinement phase. The algorithm optimizes modularity Q to find the best "
                "partition of a graph into communities. Leiden was proposed by Traag et al. "
                "in 2019 and has been widely adopted in network analysis."
            ),
            source="TechJournal",
        ),
        CorpusDoc(
            doc_id="doc_A2",
            title="Leiden Algorithm Applications",
            body=(
                "Leiden algorithm has been applied to social network analysis, biological "
                "networks, and knowledge graph community detection. In GraphRAG, Leiden is "
                "used to partition entity graphs into hierarchical communities. Each community "
                "is then summarized to support question answering. The algorithm scales well "
                "to large graphs with millions of nodes."
            ),
            source="TechJournal",
        ),
        # 文档组 B：关于 GraphRAG
        CorpusDoc(
            doc_id="doc_B1",
            title="Microsoft GraphRAG Introduction",
            body=(
                "Microsoft GraphRAG is a retrieval-augmented generation system that uses "
                "knowledge graphs to answer questions. It was developed by Microsoft Research "
                "and published in 2024. GraphRAG builds a hierarchical community structure "
                "from entity graphs extracted from documents. The system supports both local "
                "and global query modes."
            ),
            source="MSResearch",
        ),
        CorpusDoc(
            doc_id="doc_B2",
            title="GraphRAG Community Detection",
            body=(
                "GraphRAG uses the Leiden algorithm for community detection on entity graphs. "
                "Communities at different levels provide different granularities of context. "
                "Level 0 communities are fine-grained and correspond to individual document "
                "sections. Higher level communities merge related entities across documents. "
                "Community summaries are generated using large language models."
            ),
            source="MSResearch",
        ),
        CorpusDoc(
            doc_id="doc_B3",
            title="GraphRAG Evaluation Results",
            body=(
                "GraphRAG was evaluated on the MultiHop-RAG dataset and showed significant "
                "improvements over naive RAG. The system achieved higher comprehensiveness "
                "and diversity scores in LLM-as-judge evaluations. For local queries, "
                "GraphRAG retrieves relevant entity neighborhoods. For global queries, "
                "community summaries provide broad context."
            ),
            source="MSResearch",
        ),
        # 文档组 C：关于结构熵
        CorpusDoc(
            doc_id="doc_C1",
            title="Structural Entropy in Graphs",
            body=(
                "Structural entropy measures the physical diversity of nodes within a community. "
                "It is computed as the Shannon entropy of the distribution of source documents "
                "among community members. A community with low structural entropy contains "
                "nodes primarily from a single document, indicating physical coherence. "
                "High structural entropy indicates cross-document mixing."
            ),
            source="AIConference",
        ),
        CorpusDoc(
            doc_id="doc_C2",
            title="Entropy-Constrained Community Detection",
            body=(
                "We propose adding a structural entropy penalty to the Leiden objective function. "
                "The modified objective is J = Q_leiden - lambda * H_structure. The lambda "
                "parameter controls the trade-off between modularity and physical boundary "
                "preservation. An annealing schedule reduces lambda at higher hierarchy levels "
                "to allow cross-document semantic fusion."
            ),
            source="AIConference",
        ),
        CorpusDoc(
            doc_id="doc_C3",
            title="Structural Entropy Experimental Results",
            body=(
                "Experiments on MultiHop-RAG show that entropy-constrained Leiden achieves "
                "higher retrieval precision for multi-hop queries. Level 0 community purity "
                "increased from 45% to 78% compared to standard Leiden. MRR improved by "
                "12% and NDCG@10 improved by 8%. The improvement is most significant for "
                "inference queries that require cross-document reasoning."
            ),
            source="AIConference",
        ),
        # 文档组 D：关于 RAG 评估
        CorpusDoc(
            doc_id="doc_D1",
            title="RAG Evaluation Metrics",
            body=(
                "Retrieval-augmented generation systems are evaluated using multiple metrics. "
                "Precision at K measures the fraction of retrieved documents that are relevant. "
                "Mean Reciprocal Rank evaluates the rank of the first relevant document. "
                "NDCG accounts for the position of relevant documents in the ranked list. "
                "These metrics require ground-truth relevance annotations."
            ),
            source="EvalPaper",
        ),
        CorpusDoc(
            doc_id="doc_D2",
            title="MultiHop-RAG Benchmark",
            body=(
                "MultiHop-RAG is a benchmark dataset for evaluating RAG systems on multi-hop "
                "queries. It contains 2556 queries with evidence distributed across 2 to 4 "
                "documents. The dataset includes inference queries, comparison queries, and "
                "temporal queries. Each query has ground-truth answers and supporting evidence "
                "document titles for retrieval evaluation."
            ),
            source="EvalPaper",
        ),
    ]

    qa_pairs = [
        QAPair(
            query="What algorithm does GraphRAG use for community detection and what are its advantages?",
            answer="GraphRAG uses the Leiden algorithm, which guarantees well-connected communities through a refinement phase.",
            question_type="inference_query",
            supporting_titles=["Leiden Algorithm Overview", "GraphRAG Community Detection"],
        ),
        QAPair(
            query="How does structural entropy penalty improve retrieval in GraphRAG?",
            answer="It preserves physical document boundaries in lower-level communities, improving precision for multi-hop queries.",
            question_type="inference_query",
            supporting_titles=["Entropy-Constrained Community Detection", "Structural Entropy Experimental Results"],
        ),
        QAPair(
            query="Compare the evaluation approach of GraphRAG with the metrics used in MultiHop-RAG benchmark.",
            answer="GraphRAG uses LLM-as-judge while MultiHop-RAG uses precision, MRR and NDCG with ground-truth annotations.",
            question_type="comparison_query",
            supporting_titles=["GraphRAG Evaluation Results", "RAG Evaluation Metrics"],
        ),
        QAPair(
            query="What is the relationship between Leiden algorithm applications and GraphRAG community detection?",
            answer="GraphRAG applies Leiden to entity graphs to create hierarchical communities for question answering.",
            question_type="inference_query",
            supporting_titles=["Leiden Algorithm Applications", "GraphRAG Community Detection"],
        ),
        QAPair(
            query="How does structural entropy relate to the MultiHop-RAG evaluation results?",
            answer="Entropy-constrained Leiden improved MRR by 12% and NDCG@10 by 8% on MultiHop-RAG.",
            question_type="inference_query",
            supporting_titles=["Structural Entropy in Graphs", "Structural Entropy Experimental Results"],
        ),
    ]

    return MultiHopDataset(corpus=corpus, qa_pairs=qa_pairs)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  GraphRAG Improved — Smoke Test（模拟数据）")
    print("="*60)
    print("  目的：验证实验流程完整性，无需真实数据集\n")

    dataset = make_mock_dataset()
    print(f"  模拟数据：{dataset.num_docs} 篇文章，{dataset.num_qa} 条 QA 对")

    text_units = corpus_to_pipeline_text_units(dataset)
    retrieval_text_units = corpus_to_text_units(dataset.corpus)
    eval_qa_pairs = dataset_qa_to_eval_qa(dataset)

    print(f"  有效 QA 对（有 supporting_doc_ids）：{len(eval_qa_pairs)} 条")

    # 运行 baseline
    baseline_config = ExperimentConfig(
        name="Baseline (λ=0, 原版Leiden)",
        lambda_init=0.0,
    )
    result_baseline = run_single_experiment(
        baseline_config, text_units, retrieval_text_units, eval_qa_pairs, verbose=True
    )

    # 运行 ours
    ours_config = ExperimentConfig(
        name="Ours (λ=1000, 结构熵约束)",
        lambda_init=1000.0,
        annealing_schedule="exponential",
        decay_rate=0.5,
    )
    result_ours = run_single_experiment(
        ours_config, text_units, retrieval_text_units, eval_qa_pairs, verbose=True
    )

    # 打印对比
    _save_results(result_baseline, result_ours, "./experiments/results_smoke", verbose=True)

    print("\n✅ Smoke test 通过！实验流程完整。")
    print("   下一步：下载真实 MultiHop-RAG 数据集后运行完整实验")
    print("   python -m graphrag_improved.experiments.run_experiment --data-dir ./data/multihop_rag --n-qa 200")


if __name__ == "__main__":
    main()
