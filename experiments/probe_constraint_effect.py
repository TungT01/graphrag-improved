"""
专项测试：物理约束效果验证。
构造一个"语义相似但物理分散"的图，验证 λ=1000 能阻止跨文档合并。
"""
import sys, random, math
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.constrained_leiden.physical_anchor import PhysicalNode, compute_structural_entropy
from graphrag_improved.constrained_leiden.leiden_constrained import hierarchical_leiden_constrained
from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
import networkx as nx

print("=" * 60)
print("测试：语义相似但物理分散的图")
print("设计：2 个 chunk，每个 chunk 5 个节点")
print("      chunk 内部边权重 = 跨 chunk 边权重（语义无差异）")
print("      λ=1000 应能阻止跨 chunk 合并，λ=0 会随机合并")
print("=" * 60)

# 构造图：2 个 chunk，各 5 个节点，所有边权重相同
G = nx.Graph()
physical = {}
n_per_chunk = 5
n_chunks = 2

for c in range(n_chunks):
    for i in range(n_per_chunk):
        nid = f"c{c}_n{i}"
        G.add_node(nid)
        physical[nid] = PhysicalNode(nid, frozenset([f"chunk_{c}"]), 0)

# 所有节点对之间都有相同权重的边（完全图，语义完全相同）
all_nodes = list(G.nodes())
for i in range(len(all_nodes)):
    for j in range(i+1, len(all_nodes)):
        G.add_edge(all_nodes[i], all_nodes[j], weight=1.0)

print(f"图: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边（完全图，所有边权重=1）")
print()

for lam in [0.0, 100.0, 1000.0]:
    cfg = AnnealingConfig(lambda_init=lam, lambda_min=0.0, max_level=10,
                          decay_rate=0.5, schedule=AnnealingSchedule.EXPONENTIAL)
    result = hierarchical_leiden_constrained(G, physical, cfg,
                                             max_cluster_size=20, max_iterations=10, seed=42)
    if result.levels:
        comms = result.levels[0]
        n_comms = len(set(comms.values()))
        entropies = list(result.level_entropy[0].values())
        avg_h = sum(entropies) / len(entropies) if entropies else 0

        comm_to_chunks = {}
        for nid, cid in comms.items():
            chunk = list(physical[nid].chunk_ids)[0]
            comm_to_chunks.setdefault(cid, set()).add(chunk)
        cross_chunk_comms = sum(1 for chunks in comm_to_chunks.values() if len(chunks) > 1)
        pure_comms = sum(1 for chunks in comm_to_chunks.values() if len(chunks) == 1)

        print(f"λ={lam:6.0f}: 社区数={n_comms}, 平均熵={avg_h:.4f}, "
              f"纯净社区={pure_comms}, 跨chunk社区={cross_chunk_comms}")

print()
print("=" * 60)
print("测试2：更大规模，混合边权重")
print("设计：4 个 chunk，各 8 个节点")
print("      chunk 内部边权重 = 5，跨 chunk 边权重 = 3（差距小）")
print("      λ=0 可能合并跨 chunk 节点，λ=1000 应保持物理边界")
print("=" * 60)

G2 = nx.Graph()
physical2 = {}
n_per_chunk2 = 8
n_chunks2 = 4
rng = random.Random(42)

for c in range(n_chunks2):
    for i in range(n_per_chunk2):
        nid = f"c{c}_n{i}"
        G2.add_node(nid)
        physical2[nid] = PhysicalNode(nid, frozenset([f"chunk_{c}"]), 0)

# chunk 内部边（权重 5）
for c in range(n_chunks2):
    for i in range(n_per_chunk2):
        for j in range(i+1, n_per_chunk2):
            G2.add_edge(f"c{c}_n{i}", f"c{c}_n{j}", weight=5.0)

# 跨 chunk 边（权重 3，密集连接）
for c1 in range(n_chunks2):
    for c2 in range(c1+1, n_chunks2):
        for i in range(n_per_chunk2):
            for j in range(n_per_chunk2):
                G2.add_edge(f"c{c1}_n{i}", f"c{c2}_n{j}", weight=3.0)

print(f"图: {G2.number_of_nodes()} 节点, {G2.number_of_edges()} 边")
print()

for lam in [0.0, 100.0, 1000.0]:
    cfg2 = AnnealingConfig(lambda_init=lam, lambda_min=0.0, max_level=10,
                           decay_rate=0.5, schedule=AnnealingSchedule.EXPONENTIAL)
    result2 = hierarchical_leiden_constrained(G2, physical2, cfg2,
                                              max_cluster_size=20, max_iterations=10, seed=42)
    if result2.levels:
        comms2 = result2.levels[0]
        n_comms2 = len(set(comms2.values()))
        entropies2 = list(result2.level_entropy[0].values())
        avg_h2 = sum(entropies2) / len(entropies2) if entropies2 else 0

        comm_to_chunks2 = {}
        for nid, cid in comms2.items():
            chunk = list(physical2[nid].chunk_ids)[0]
            comm_to_chunks2.setdefault(cid, set()).add(chunk)
        cross2 = sum(1 for chunks in comm_to_chunks2.values() if len(chunks) > 1)
        pure2 = sum(1 for chunks in comm_to_chunks2.values() if len(chunks) == 1)

        print(f"λ={lam:6.0f}: 社区数={n_comms2:3d}, 平均熵={avg_h2:.4f}, "
              f"纯净社区={pure2:3d}, 跨chunk社区={cross2:3d}")

print()
print("=" * 60)
print("测试3：MultiHop-RAG 真实场景模拟")
print("设计：609 个 chunk，每个 chunk 1-3 个实体，实体间有共现关系")
print("      部分实体跨多个 chunk（真实情况）")
print("=" * 60)

N_CHUNKS_SIM = 50   # 模拟 50 篇文章
N_ENTITIES_SIM = 200
G3 = nx.Graph()
physical3 = {}
rng3 = random.Random(42)

nodes3 = [f"e{i}" for i in range(N_ENTITIES_SIM)]
for n in nodes3:
    G3.add_node(n)
    # 每个实体来自 1-2 个 chunk（模拟跨文档实体）
    n_c = rng3.choices([1, 2], weights=[0.7, 0.3])[0]
    cids = frozenset(f"chunk_{rng3.randint(0, N_CHUNKS_SIM-1)}" for _ in range(n_c))
    physical3[n] = PhysicalNode(n, cids, 0)

# 同 chunk 实体之间有强连接
chunk_to_entities = {}
for nid, pn in physical3.items():
    for cid in pn.chunk_ids:
        chunk_to_entities.setdefault(cid, []).append(nid)

for cid, ents in chunk_to_entities.items():
    for i in range(len(ents)):
        for j in range(i+1, len(ents)):
            u, v = ents[i], ents[j]
            if G3.has_edge(u, v):
                G3[u][v]["weight"] += 3.0
            else:
                G3.add_edge(u, v, weight=3.0)

# 随机跨 chunk 弱连接
for _ in range(300):
    u = rng3.choice(nodes3)
    v = rng3.choice(nodes3)
    if u != v:
        u_chunks = physical3[u].chunk_ids
        v_chunks = physical3[v].chunk_ids
        if not u_chunks.intersection(v_chunks):  # 确实跨 chunk
            if G3.has_edge(u, v):
                G3[u][v]["weight"] += 1.0
            else:
                G3.add_edge(u, v, weight=1.0)

print(f"图: {G3.number_of_nodes()} 节点, {G3.number_of_edges()} 边")
print()

for lam in [0.0, 1000.0]:
    cfg3 = AnnealingConfig(lambda_init=lam, lambda_min=0.0, max_level=10,
                           decay_rate=0.5, schedule=AnnealingSchedule.EXPONENTIAL)
    result3 = hierarchical_leiden_constrained(G3, physical3, cfg3,
                                              max_cluster_size=10, max_iterations=10, seed=42)
    if result3.levels:
        comms3 = result3.levels[0]
        n_comms3 = len(set(comms3.values()))
        entropies3 = list(result3.level_entropy[0].values())
        avg_h3 = sum(entropies3) / len(entropies3) if entropies3 else 0

        comm_to_chunks3 = {}
        for nid, cid in comms3.items():
            for chunk in physical3[nid].chunk_ids:
                comm_to_chunks3.setdefault(cid, set()).add(chunk)
        cross3 = sum(1 for chunks in comm_to_chunks3.values() if len(chunks) > 1)
        pure3 = sum(1 for chunks in comm_to_chunks3.values() if len(chunks) == 1)

        print(f"λ={lam:6.0f}: 社区数={n_comms3:3d}, 平均熵={avg_h3:.4f}, "
              f"纯净社区={pure3:3d}, 跨chunk社区={cross3:3d}")
