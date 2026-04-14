"""
验证增量熵状态的正确性，并对比性能提升。
"""
import sys, time, random, math
from pathlib import Path
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.constrained_leiden.physical_anchor import PhysicalNode, compute_structural_entropy, compute_delta_entropy
from graphrag_improved.constrained_leiden.leiden_constrained import CommunityEntropyState, _initialize_state, _local_moving_phase, hierarchical_leiden_constrained
from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
import networkx as nx

print("=" * 60)
print("1. 增量熵状态正确性验证")
print("=" * 60)

rng = random.Random(42)
N_TESTS = 1000
errors = 0

for _ in range(N_TESTS):
    # 随机构造社区
    n_nodes = rng.randint(1, 20)
    n_chunks = rng.randint(2, 10)
    community = []
    for i in range(n_nodes):
        nc = rng.randint(1, 3)
        cids = frozenset(f"c{rng.randint(0, n_chunks-1)}" for _ in range(nc))
        community.append(PhysicalNode(f"n{i}", cids, 0))

    # 构造增量状态
    es = CommunityEntropyState()
    for node in community:
        es.add_node(node)

    # 验证 entropy() 与 compute_structural_entropy 一致
    h_incr = es.entropy()
    h_orig = compute_structural_entropy(community)
    if abs(h_incr - h_orig) > 1e-9:
        errors += 1
        print(f"  entropy 不一致: incr={h_incr:.6f}, orig={h_orig:.6f}")

    # 验证 delta_entropy_if_add 与 compute_delta_entropy 一致
    new_nc = rng.randint(1, 3)
    new_cids = frozenset(f"c{rng.randint(0, n_chunks-1)}" for _ in range(new_nc))
    new_node = PhysicalNode("new", new_cids, 0)

    delta_incr = es.delta_entropy_if_add(new_node)
    delta_orig = compute_delta_entropy(community, new_node)
    if abs(delta_incr - delta_orig) > 1e-9:
        errors += 1
        print(f"  delta_entropy 不一致: incr={delta_incr:.6f}, orig={delta_orig:.6f}")

    # 验证 remove_node 后 entropy 正确
    if community:
        node_to_remove = rng.choice(community)
        es_copy = es.copy()
        es_copy.remove_node(node_to_remove)
        remaining = [n for n in community if n is not node_to_remove]
        h_after_incr = es_copy.entropy()
        h_after_orig = compute_structural_entropy(remaining)
        if abs(h_after_incr - h_after_orig) > 1e-9:
            errors += 1
            print(f"  remove 后 entropy 不一致: incr={h_after_incr:.6f}, orig={h_after_orig:.6f}")

print(f"正确性测试: {N_TESTS} 组，错误数: {errors}")

print()
print("=" * 60)
print("2. 性能对比：增量 vs 原始 compute_delta_entropy")
print("=" * 60)

# 构造 50 节点社区
community_50 = [PhysicalNode(f"n{i}", frozenset([f"c{i%10}"]), 0) for i in range(50)]
new_node = PhysicalNode("new", frozenset(["c3"]), 0)

es_50 = CommunityEntropyState()
for n in community_50:
    es_50.add_node(n)

N_CALLS = 50000

t0 = time.time()
for _ in range(N_CALLS):
    compute_delta_entropy(community_50, new_node)
t1 = time.time()
orig_ms = (t1 - t0) * 1000
print(f"原始方法 {N_CALLS} 次 (50节点社区): {orig_ms:.1f}ms ({orig_ms/N_CALLS*1000:.2f}μs/次)")

t0 = time.time()
for _ in range(N_CALLS):
    es_50.delta_entropy_if_add(new_node)
t1 = time.time()
incr_ms = (t1 - t0) * 1000
print(f"增量方法 {N_CALLS} 次 (50节点社区): {incr_ms:.1f}ms ({incr_ms/N_CALLS*1000:.2f}μs/次)")
print(f"加速比: {orig_ms/incr_ms:.1f}x")

print()
print("=" * 60)
print("3. 完整算法性能测试（模拟真实规模）")
print("=" * 60)

configs = [
    (300, 1500, 609),
    (600, 3000, 609),
    (1000, 5000, 609),
    (2000, 10000, 609),
]

print(f"{'节点':>6} {'边':>6} {'耗时(s)':>10} {'层数':>6} {'社区数':>8} {'平均熵':>10}")
print("-" * 55)

for n_nodes, n_edges, n_chunks in configs:
    rng2 = random.Random(42)
    G = nx.Graph()
    nodes = [f"e{i}" for i in range(n_nodes)]
    for n in nodes:
        G.add_node(n)
    added = 0
    attempts = 0
    while added < n_edges and attempts < n_edges * 3:
        u = rng2.choice(nodes)
        v = rng2.choice(nodes)
        attempts += 1
        if u != v and not G.has_edge(u, v):
            G.add_edge(u, v, weight=float(rng2.randint(1, 10)))
            added += 1

    physical_nodes = {}
    for node in nodes:
        n_c = rng2.randint(1, 3)
        cids = frozenset(f"c{rng2.randint(0, n_chunks-1)}" for _ in range(n_c))
        physical_nodes[node] = PhysicalNode(node_id=node, chunk_ids=cids, level=0)

    cfg = AnnealingConfig(lambda_init=1000.0, lambda_min=0.0, max_level=10,
                          decay_rate=0.5, schedule=AnnealingSchedule.EXPONENTIAL)
    t0 = time.time()
    result = hierarchical_leiden_constrained(G, physical_nodes, cfg,
                                             max_cluster_size=10, max_iterations=10, seed=42)
    elapsed = time.time() - t0
    n_comms = len(set(result.levels[0].values())) if result.levels else 0
    entropies = list(result.level_entropy[0].values()) if result.level_entropy else [0]
    avg_h = sum(entropies) / len(entropies)
    print(f"{n_nodes:>6} {G.number_of_edges():>6} {elapsed:>10.2f} {len(result.levels):>6} {n_comms:>8} {avg_h:>10.4f}")

print()
print("=" * 60)
print("4. 物理约束效果验证：λ=1000 vs λ=0")
print("=" * 60)

# 构造一个有明确物理边界的图：
# 3 个 chunk，每个 chunk 10 个节点，chunk 内部边权重 10，跨 chunk 边权重 1
G_test = nx.Graph()
n_per_chunk = 10
n_chunks_test = 3
all_nodes = []
physical_test = {}

for c in range(n_chunks_test):
    for i in range(n_per_chunk):
        nid = f"c{c}_n{i}"
        G_test.add_node(nid)
        all_nodes.append(nid)
        physical_test[nid] = PhysicalNode(nid, frozenset([f"chunk_{c}"]), 0)

# chunk 内部强连接
for c in range(n_chunks_test):
    for i in range(n_per_chunk):
        for j in range(i+1, n_per_chunk):
            G_test.add_edge(f"c{c}_n{i}", f"c{c}_n{j}", weight=10.0)

# 跨 chunk 弱连接（少量）
rng3 = random.Random(42)
for _ in range(15):
    c1, c2 = rng3.sample(range(n_chunks_test), 2)
    i, j = rng3.randint(0, n_per_chunk-1), rng3.randint(0, n_per_chunk-1)
    u, v = f"c{c1}_n{i}", f"c{c2}_n{j}"
    if not G_test.has_edge(u, v):
        G_test.add_edge(u, v, weight=1.0)

print(f"测试图: {G_test.number_of_nodes()} 节点, {G_test.number_of_edges()} 边")
print(f"  chunk 内部边权重=10, 跨 chunk 边权重=1")

for lam in [0.0, 1000.0]:
    cfg_test = AnnealingConfig(lambda_init=lam, lambda_min=0.0, max_level=10,
                               decay_rate=0.5, schedule=AnnealingSchedule.EXPONENTIAL)
    result_test = hierarchical_leiden_constrained(G_test, physical_test, cfg_test,
                                                  max_cluster_size=20, max_iterations=10, seed=42)
    if result_test.levels:
        comms = result_test.levels[0]
        n_comms_test = len(set(comms.values()))
        entropies_test = list(result_test.level_entropy[0].values())
        avg_h_test = sum(entropies_test) / len(entropies_test)

        # 统计跨 chunk 社区数
        comm_to_chunks = {}
        for nid, cid in comms.items():
            chunk = physical_test[nid].chunk_ids
            comm_to_chunks.setdefault(cid, set()).update(chunk)
        cross_chunk_comms = sum(1 for chunks in comm_to_chunks.values() if len(chunks) > 1)

        print(f"  λ={lam:6.0f}: 社区数={n_comms_test:3d}, 平均熵={avg_h_test:.4f}, 跨chunk社区数={cross_chunk_comms}")
