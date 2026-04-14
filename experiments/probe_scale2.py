"""
快速测试：只测图规模和纯 Python Leiden 耗时，跳过慢的实体抽取。
"""
import sys, time, random
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.constrained_leiden.physical_anchor import PhysicalNode
from graphrag_improved.constrained_leiden.leiden_constrained import hierarchical_leiden_constrained, _initialize_state, _local_moving_phase
from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
import networkx as nx

# 根据上次实验日志：baseline 200 社区，ours 907 社区
# 推算实体数约 500-2000，关系数约 2000-10000
# 用不同规模测试耗时

configs = [
    (300, 1500, 609),
    (600, 3000, 609),
    (1000, 5000, 609),
    (2000, 10000, 609),
    (3000, 15000, 609),
]

print(f"{'节点':>6} {'边':>6} {'耗时(s)':>10} {'层数':>6} {'社区数':>8}")
print("-" * 45)

for n_nodes, n_edges, n_chunks in configs:
    rng = random.Random(42)
    G = nx.Graph()
    nodes = [f"e{i}" for i in range(n_nodes)]
    for n in nodes:
        G.add_node(n)
    added = 0
    attempts = 0
    while added < n_edges and attempts < n_edges * 3:
        u = rng.choice(nodes)
        v = rng.choice(nodes)
        attempts += 1
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=float(rng.randint(1, 10)))
            added += 1

    physical_nodes = {}
    for node in nodes:
        n_c = rng.randint(1, 3)
        cids = frozenset(f"c{rng.randint(0, n_chunks-1)}" for _ in range(n_c))
        physical_nodes[node] = PhysicalNode(node_id=node, chunk_ids=cids, level=0)

    cfg = AnnealingConfig(lambda_init=1000.0, lambda_min=0.0, max_level=10,
                          decay_rate=0.5, schedule=AnnealingSchedule.EXPONENTIAL)
    t0 = time.time()
    result = hierarchical_leiden_constrained(G, physical_nodes, cfg,
                                             max_cluster_size=10, max_iterations=10, seed=42)
    elapsed = time.time() - t0
    n_comms = len(set(result.levels[0].values())) if result.levels else 0
    print(f"{n_nodes:>6} {G.number_of_edges():>6} {elapsed:>10.2f} {len(result.levels):>6} {n_comms:>8}")
