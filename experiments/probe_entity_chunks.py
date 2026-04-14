"""
分析 MultiHop-RAG 数据集中实体的 chunk_ids 分布，
理解为什么结构熵在真实数据上难以降低。
"""
import sys, time
from pathlib import Path
from collections import Counter
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.experiments.data_loader import load_multihop_dataset
from graphrag_improved.experiments.run_experiment import corpus_to_pipeline_text_units
from graphrag_improved.extraction.extractor import extract
from graphrag_improved.pipeline_config import ExtractionConfig
from graphrag_improved.constrained_leiden.graphrag_workflow import build_graph_from_graphrag, build_physical_nodes_from_graphrag
import networkx as nx

DATA_DIR = "/Users/ttung/Desktop/个人学习/data/multihop_rag"

print("加载数据集（仅用前 100 篇文章快速分析）...")
dataset = load_multihop_dataset(DATA_DIR)
# 只用前 100 篇文章
class SubDataset:
    def __init__(self, corpus, qa_pairs):
        self.corpus = corpus
        self.qa_pairs = qa_pairs
        self.num_docs = len(corpus)
        self.num_qa = len(qa_pairs)

sub = SubDataset(dataset.corpus[:100], dataset.qa_pairs[:200])
print(f"使用 {sub.num_docs} 篇文章")

text_units = corpus_to_pipeline_text_units(sub)
print("抽取实体（min_freq=2）...")
t0 = time.time()
extraction_config = ExtractionConfig(backend="rule", min_entity_freq=2, cooccurrence_window=3)
entities_df, relationships_df = extract(text_units, extraction_config)
print(f"实体: {len(entities_df)}, 关系: {len(relationships_df)}, 耗时: {time.time()-t0:.1f}s")

# 分析 chunk_ids 分布
chunk_counts = []
for _, row in entities_df.iterrows():
    raw = row.get("text_unit_ids", [])
    if isinstance(raw, list):
        chunk_counts.append(len(raw))
    else:
        chunk_counts.append(0)

print()
print("=== 实体 chunk_ids 数量分布 ===")
dist = Counter(chunk_counts)
for k in sorted(dist.keys()):
    pct = dist[k] / len(chunk_counts) * 100
    print(f"  {k} 个 chunk: {dist[k]:4d} 个实体 ({pct:.1f}%)")

print(f"  平均 chunk 数: {sum(chunk_counts)/len(chunk_counts):.2f}")
print(f"  最大 chunk 数: {max(chunk_counts)}")
print(f"  跨多个 chunk 的实体比例: {sum(1 for c in chunk_counts if c > 1)/len(chunk_counts)*100:.1f}%")

# 构建图并分析
rels_filtered = relationships_df[relationships_df["weight"] >= 2.0]
graph = build_graph_from_graphrag(entities_df, rels_filtered)
physical_nodes = build_physical_nodes_from_graphrag(entities_df)

if not nx.is_connected(graph):
    lcc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lcc).copy()
    physical_nodes = {k: v for k, v in physical_nodes.items() if k in lcc}

print()
print(f"=== 过滤后图规模 ===")
print(f"  节点: {graph.number_of_nodes()}, 边: {graph.number_of_edges()}")

# 分析图中节点的 chunk_ids 分布
pn_chunk_counts = [len(pn.chunk_ids) for pn in physical_nodes.values()]
dist2 = Counter(pn_chunk_counts)
print()
print("=== 图中节点 chunk_ids 数量分布 ===")
for k in sorted(dist2.keys()):
    pct = dist2[k] / len(pn_chunk_counts) * 100
    print(f"  {k} 个 chunk: {dist2[k]:4d} 个节点 ({pct:.1f}%)")

# 关键问题：如果实体本身就跨多个 chunk，
# 那么即使把同 chunk 的实体聚在一起，社区熵也不会是 0
# 因为每个节点的 chunk_ids 集合本身就包含多个 chunk
print()
print("=== 关键分析 ===")
single_chunk = sum(1 for c in pn_chunk_counts if c == 1)
multi_chunk = sum(1 for c in pn_chunk_counts if c > 1)
print(f"  单 chunk 节点: {single_chunk} ({single_chunk/len(pn_chunk_counts)*100:.1f}%)")
print(f"  多 chunk 节点: {multi_chunk} ({multi_chunk/len(pn_chunk_counts)*100:.1f}%)")
print()
print("  结论：多 chunk 节点的存在使得社区熵无法降为 0，")
print("  即使所有节点都来自同一物理文档，熵也可能 > 0。")
print()
print("  这解释了为什么 λ=1000 在真实数据上的熵降低效果有限：")
print("  实体本身就跨多个文档，结构熵的下界不是 0。")
