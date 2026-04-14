"""
测试真实数据集规模下纯 Python Leiden 的耗时。
"""
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.experiments.data_loader import load_multihop_dataset, corpus_to_text_units
from graphrag_improved.experiments.run_experiment import corpus_to_pipeline_text_units
from graphrag_improved.extraction.extractor import extract
from graphrag_improved.pipeline_config import ExtractionConfig
from graphrag_improved.constrained_leiden.graphrag_workflow import build_graph_from_graphrag, build_physical_nodes_from_graphrag
from graphrag_improved.constrained_leiden.leiden_constrained import hierarchical_leiden_constrained
from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
import networkx as nx

DATA_DIR = "/Users/ttung/Desktop/个人学习/data/multihop_rag"

print("加载数据集...")
dataset = load_multihop_dataset(DATA_DIR)
print(f"文章数: {dataset.num_docs}, QA数: {dataset.num_qa}")

print("抽取实体...")
text_units = corpus_to_pipeline_text_units(dataset)
extraction_config = ExtractionConfig(backend="rule", min_entity_freq=5, cooccurrence_window=3)
t0 = time.time()
entities_df, relationships_df = extract(text_units, extraction_config)
print(f"实体: {len(entities_df)}, 关系: {len(relationships_df)}, 耗时: {time.time()-t0:.1f}s")

print("构建图...")
graph = build_graph_from_graphrag(entities_df, relationships_df)
physical_nodes = build_physical_nodes_from_graphrag(entities_df)
print(f"图: {graph.number_of_nodes()} 节点, {graph.number_of_edges()} 边")

# 过滤低权重边（与实验一致）
import pandas as pd
rels_filtered = relationships_df[relationships_df["weight"] >= 2.0]
graph_filtered = build_graph_from_graphrag(entities_df, rels_filtered)
if not nx.is_connected(graph_filtered):
    lcc = max(nx.connected_components(graph_filtered), key=len)
    graph_filtered = graph_filtered.subgraph(lcc).copy()
    physical_nodes_filtered = {k: v for k, v in physical_nodes.items() if k in lcc}
else:
    physical_nodes_filtered = physical_nodes
print(f"过滤后图: {graph_filtered.number_of_nodes()} 节点, {graph_filtered.number_of_edges()} 边")

print()
print("测试纯 Python Leiden 耗时（lambda=1000）...")
annealing_config = AnnealingConfig(
    lambda_init=1000.0, lambda_min=0.0, max_level=10, decay_rate=0.5,
    schedule=AnnealingSchedule.EXPONENTIAL,
)
t0 = time.time()
result = hierarchical_leiden_constrained(
    graph_filtered, physical_nodes_filtered, annealing_config,
    max_cluster_size=10, max_iterations=10, seed=42
)
elapsed = time.time() - t0
print(f"纯 Python Leiden 耗时: {elapsed:.2f}s, 层数: {len(result.levels)}")
if result.levels:
    n_comms = len(set(result.levels[0].values()))
    print(f"Level 0 社区数: {n_comms}")
    # 计算平均结构熵
    entropies = list(result.level_entropy[0].values())
    avg_h = sum(entropies) / len(entropies) if entropies else 0
    print(f"Level 0 平均结构熵: {avg_h:.4f}")
