"""
性能剖析：找出纯 Python Leiden 的真正瓶颈，并测试 609 篇文章规模的实际耗时。
"""
import time
import cProfile
import pstats
import io
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from graphrag_improved.constrained_leiden.physical_anchor import PhysicalNode, compute_structural_entropy, compute_delta_entropy
from graphrag_improved.constrained_leiden.leiden_constrained import hierarchical_leiden_constrained, _initialize_state, _local_moving_phase
from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
import networkx as nx
import random

print("=" * 60)
print("1. 构造模拟 609 篇文章规模的图")
print("=" * 60)

# MultiHop-RAG 实验中的实际规模（从实验日志推算）
# 609 篇文章，min_entity_freq=5，约 500-800 个实体，2000-5000 条关系
N_ENTITIES = 600
N_CHUNKS = 609
N_EDGES = 3000

rng = random.Random(42)
G = nx.Graph()

# 添加节点
nodes = [f"entity_{i}" for i in range(N_ENTITIES)]
for n in nodes:
    G.add_node(n)

# 添加边（模拟共现关系）
edges_added = 0
while edges_added < N_EDGES:
    u = rng.choice(nodes)
    v = rng.choice(nodes)
    if u != v and not G.has_edge(u, v):
        G.add_edge(u, v, weight=float(rng.randint(1, 10)))
        edges_added += 1

print(f"图规模: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边")

# 构造物理节点（每个实体来自 1-3 个 chunk）
physical_nodes = {}
for i, node in enumerate(nodes):
    n_chunks = rng.randint(1, 3)
    chunk_ids = frozenset(f"chunk_{rng.randint(0, N_CHUNKS-1)}" for _ in range(n_chunks))
    physical_nodes[node] = PhysicalNode(node_id=node, chunk_ids=chunk_ids, level=0)

print(f"物理节点: {len(physical_nodes)} 个")

print()
print("=" * 60)
print("2. 性能剖析：纯 Python Leiden（当前实现）")
print("=" * 60)

annealing_config = AnnealingConfig(
    lambda_init=1000.0,
    lambda_min=0.0,
    max_level=10,
    decay_rate=0.5,
    schedule=AnnealingSchedule.EXPONENTIAL,
)

# 先测一下单层的耗时
state = _initialize_state(G, physical_nodes)
rng_obj = random.Random(42)

t0 = time.time()
moved = _local_moving_phase(G, state, 1000.0, rng_obj)
t1 = time.time()
print(f"单层 local_moving_phase: {(t1-t0)*1000:.1f}ms, moved={moved}")

# 完整运行并计时
t0 = time.time()
result = hierarchical_leiden_constrained(
    G, physical_nodes, annealing_config, max_cluster_size=10, max_iterations=10, seed=42
)
t1 = time.time()
print(f"完整 hierarchical_leiden: {t1-t0:.2f}s, 层数={len(result.levels)}")

print()
print("=" * 60)
print("3. 热点分析：compute_delta_entropy 调用次数")
print("=" * 60)

# 统计 local_moving_phase 中 compute_delta_entropy 的调用次数
state2 = _initialize_state(G, physical_nodes)
call_count = [0]
original_delta = compute_delta_entropy

import graphrag_improved.constrained_leiden.physical_anchor as pa_module

original_fn = pa_module.compute_delta_entropy
def counting_delta_entropy(community_nodes, new_node):
    call_count[0] += 1
    return original_fn(community_nodes, new_node)

pa_module.compute_delta_entropy = counting_delta_entropy

# 重新导入使用新函数
import graphrag_improved.constrained_leiden.leiden_constrained as lc_module
lc_module.compute_delta_entropy = counting_delta_entropy

rng_obj2 = random.Random(42)
t0 = time.time()
moved2 = lc_module._local_moving_phase(G, state2, 1000.0, rng_obj2)
t1 = time.time()
print(f"单层 local_moving_phase: {(t1-t0)*1000:.1f}ms")
print(f"compute_delta_entropy 调用次数: {call_count[0]}")
print(f"平均每次调用耗时: {(t1-t0)*1000/max(call_count[0],1):.4f}ms")

# 恢复
pa_module.compute_delta_entropy = original_fn
lc_module.compute_delta_entropy = original_fn

print()
print("=" * 60)
print("4. 优化方向：增量熵计算的复杂度分析")
print("=" * 60)

# 当前 compute_delta_entropy 的复杂度：O(|community|)
# 每次节点移动需要调用 O(|neighbor_communities|) 次
# 总复杂度：O(|nodes| × avg_neighbors × avg_community_size)

avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
print(f"平均度: {avg_degree:.1f}")
print(f"预估 delta_entropy 调用次数/轮: {N_ENTITIES * avg_degree:.0f}")
print(f"当前实现复杂度: O(N × D × C) 其中 N={N_ENTITIES}, D={avg_degree:.0f}, C=avg_community_size")

print()
print("=" * 60)
print("5. 优化方案：维护增量熵状态（O(1) 更新）")
print("=" * 60)

print("""
当前 compute_delta_entropy 每次都重新计算整个社区的熵：
  H_before = compute_structural_entropy(community_nodes)  # O(|community|)
  H_after  = compute_structural_entropy(community_nodes + [new_node])  # O(|community|)

优化：维护每个社区的 chunk_weight_map（{chunk_id: weight}）和 total_weight
  - 加入节点时：O(|new_node.chunk_ids|) 更新，通常 O(1)
  - 计算 ΔH：O(|new_node.chunk_ids|) 而非 O(|community|)
  
这将把 compute_delta_entropy 从 O(|community|) 降到 O(1)
预期加速：10-50x（取决于平均社区大小）
""")

# 测试增量熵计算的正确性
from collections import defaultdict

def compute_entropy_from_weights(chunk_weights, total_weight):
    """从预计算的权重字典计算熵。"""
    import math
    if total_weight == 0:
        return 0.0
    entropy = 0.0
    for w in chunk_weights.values():
        p = w / total_weight
        if p > 0:
            entropy -= p * math.log(p)
    return entropy

def compute_delta_entropy_incremental(chunk_weights, total_weight, new_node):
    """O(|new_node.chunk_ids|) 的增量熵计算。"""
    import math
    h_before = compute_entropy_from_weights(chunk_weights, total_weight)
    
    # 模拟加入 new_node 后的状态
    new_weights = dict(chunk_weights)
    new_total = total_weight
    weight_per_chunk = 1.0 / len(new_node.chunk_ids)
    for cid in new_node.chunk_ids:
        new_weights[cid] = new_weights.get(cid, 0.0) + weight_per_chunk
        new_total += weight_per_chunk
    
    h_after = compute_entropy_from_weights(new_weights, new_total)
    return h_after - h_before

# 验证正确性
test_nodes = [
    PhysicalNode("a", frozenset(["c1"]), 0),
    PhysicalNode("b", frozenset(["c1"]), 0),
    PhysicalNode("c", frozenset(["c2"]), 0),
]
new_node = PhysicalNode("d", frozenset(["c2"]), 0)

# 原始方法
delta_orig = compute_delta_entropy(test_nodes, new_node)

# 增量方法
chunk_weights = {}
total_weight = 0.0
for n in test_nodes:
    wpc = 1.0 / len(n.chunk_ids)
    for cid in n.chunk_ids:
        chunk_weights[cid] = chunk_weights.get(cid, 0.0) + wpc
        total_weight += wpc
delta_incr = compute_delta_entropy_incremental(chunk_weights, total_weight, new_node)

print(f"原始 ΔH: {delta_orig:.6f}")
print(f"增量 ΔH: {delta_incr:.6f}")
print(f"结果一致: {abs(delta_orig - delta_incr) < 1e-10}")

print()
print("=" * 60)
print("6. 增量方法性能测试")
print("=" * 60)

# 构造一个有 50 个节点的社区
community = [PhysicalNode(f"n{i}", frozenset([f"c{i%10}"]), 0) for i in range(50)]
new_n = PhysicalNode("new", frozenset(["c3"]), 0)

# 预计算 chunk_weights
cw = {}
tw = 0.0
for n in community:
    wpc = 1.0 / len(n.chunk_ids)
    for cid in n.chunk_ids:
        cw[cid] = cw.get(cid, 0.0) + wpc
        tw += wpc

N_CALLS = 10000

t0 = time.time()
for _ in range(N_CALLS):
    compute_delta_entropy(community, new_n)
t1 = time.time()
print(f"原始方法 {N_CALLS} 次: {(t1-t0)*1000:.1f}ms ({(t1-t0)/N_CALLS*1e6:.2f}μs/次)")

t0 = time.time()
for _ in range(N_CALLS):
    compute_delta_entropy_incremental(cw, tw, new_n)
t1 = time.time()
print(f"增量方法 {N_CALLS} 次: {(t1-t0)*1000:.1f}ms ({(t1-t0)/N_CALLS*1e6:.2f}μs/次)")
