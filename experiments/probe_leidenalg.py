"""
探针脚本：调研 leidenalg 的扩展能力，确定过程约束的技术方案。
"""
import leidenalg as la
import igraph as ig
import math
from collections import defaultdict

print("=" * 60)
print("5. 测试 move_nodes_constrained 正确用法")
print("=" * 60)

G = ig.Graph.Famous('Zachary')

# 模拟：节点 0-10 来自 chunk_0，节点 11-20 来自 chunk_1，其余来自 chunk_2
chunk_assignment = {}
for v in range(G.vcount()):
    if v <= 10:
        chunk_assignment[v] = 0
    elif v <= 20:
        chunk_assignment[v] = 1
    else:
        chunk_assignment[v] = 2

# 构造约束分区：同一 chunk 的节点在同一社区
constrained_membership = [chunk_assignment[v] for v in range(G.vcount())]
constrained_partition = la.CPMVertexPartition(G, initial_membership=constrained_membership, resolution_parameter=0.1)
print(f"约束分区社区数: {len(set(constrained_membership))}")

# 在约束分区内运行 move_nodes_constrained
# 初始化为单节点社区
singleton_membership = list(range(G.vcount()))
partition4 = la.CPMVertexPartition(G, initial_membership=singleton_membership, resolution_parameter=0.1)
opt4 = la.Optimiser()
opt4.set_rng_seed(42)
improvement = opt4.move_nodes_constrained(partition4, constrained_partition)
print(f"constrained move_nodes improvement: {improvement:.6f}")
print(f"结果社区数: {len(set(partition4.membership))}")
print(f"membership[:5]: {partition4.membership[:5]}")

# 验证：结果中是否有跨 chunk 的社区？
cross_chunk = 0
for comm_id in set(partition4.membership):
    nodes_in_comm = [v for v, m in enumerate(partition4.membership) if m == comm_id]
    chunks_in_comm = set(chunk_assignment[v] for v in nodes_in_comm)
    if len(chunks_in_comm) > 1:
        cross_chunk += 1
print(f"跨 chunk 社区数: {cross_chunk} (应为 0，因为 constrained 限制了跨 chunk 移动)")

print()
print("=" * 60)
print("6. 测试：Leiden 完整流程 + 物理约束")
print("=" * 60)

# 完整的 Leiden 流程：
# Phase 1: move_nodes（全局移动，无约束）
# Phase 2: refine_partition（细化，用 move_nodes_constrained 限制在当前社区内）
# Phase 3: aggregate

# 我们的目标：在 Phase 1 中加入物理约束
# 方案：用 is_membership_fixed 锁定跨 chunk 的"危险"移动

# 先跑标准 Leiden
partition_std = la.find_partition(G, la.CPMVertexPartition, resolution_parameter=0.1, seed=42)
print(f"标准 Leiden 社区数: {len(set(partition_std.membership))}")
cross_std = 0
for comm_id in set(partition_std.membership):
    nodes_in_comm = [v for v, m in enumerate(partition_std.membership) if m == comm_id]
    chunks_in_comm = set(chunk_assignment[v] for v in nodes_in_comm)
    if len(chunks_in_comm) > 1:
        cross_std += 1
print(f"标准 Leiden 跨 chunk 社区数: {cross_std}")

print()
print("=" * 60)
print("7. 核心方案：igraph 加速的 Python Leiden 变体")
print("=" * 60)

# 测试 igraph 的图操作速度
import time

# 构造一个较大的图（模拟 609 篇文章的实体图规模）
n_nodes = 2000
n_edges = 10000
import random
random.seed(42)
edges = [(random.randint(0, n_nodes-1), random.randint(0, n_nodes-1)) for _ in range(n_edges)]
edges = [(u, v) for u, v in edges if u != v]

G_large = ig.Graph(n=n_nodes, edges=edges, directed=False)
G_large.simplify()
print(f"大图: {G_large.vcount()} 节点, {G_large.ecount()} 边")

# 测试 igraph 邻居查询速度
t0 = time.time()
for v in range(n_nodes):
    _ = G_large.neighbors(v)
t1 = time.time()
print(f"igraph 全图邻居查询: {(t1-t0)*1000:.1f}ms")

# 对比 networkx
import networkx as nx
G_nx = nx.Graph()
G_nx.add_edges_from(G_large.get_edgelist())
t0 = time.time()
for v in range(n_nodes):
    _ = list(G_nx.neighbors(v))
t1 = time.time()
print(f"networkx 全图邻居查询: {(t1-t0)*1000:.1f}ms")

print()
print("=" * 60)
print("8. 结论与技术方案选择")
print("=" * 60)
print("""
leidenalg 的 diff_move 是 C++ 实现，无法在 Python 层覆盖。
因此无法直接在 leidenalg 内部注入结构熵惩罚。

可行方案对比：

方案A：igraph 加速的 Python Leiden（推荐）
  - 用 igraph 替换 networkx 做图操作（快 10-50x）
  - 保留 Python 层的 ΔJ = ΔQ - λ·ΔH 决策逻辑
  - 预期速度：比当前纯 Python 快 10-30x，609 篇文章可在 30s 内完成

方案B：leidenalg 初始化 + Python 迭代修正
  - 用 leidenalg 做初始化（快速得到好的起点）
  - 再用 Python 层做多轮"物理约束修正"迭代
  - 每轮：找出跨 chunk 的社区 → 尝试拆分 → 检查 ΔQ 是否可接受
  - 问题：修正轮次多时仍然慢，且不保证收敛到最优

方案C：修改 leidenalg 源码（C++ 层）
  - 最彻底，但需要编译，不适合快速迭代

选择方案A：igraph 加速的 Python Leiden
""")
