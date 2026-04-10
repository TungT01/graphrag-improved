"""
leiden_constrained.py
---------------------
带结构熵惩罚的 Leiden 变体核心算法。

目标函数：
    J = Q_leiden - λ · H_structure

其中：
    Q_leiden   : 标准 Leiden 模块度增益（衡量语义相似性）
    H_structure: 社区内节点物理来源的香农熵（衡量物理分散度）
    λ          : 退火系数，随层级升高从极大值衰减至 0

算法流程（每一层）：
    1. 局部移动阶段（Local Moving）：
       对每个节点，遍历其邻居所在社区，
       计算移动后的 ΔJ = ΔQ - λ·ΔH，
       若 ΔJ > 0 则执行移动。
    2. 细化阶段（Refinement）：
       在每个社区内部，尝试将节点分裂为更小的子社区，
       提高解的质量，避免陷入局部最优。
    3. 聚合阶段（Aggregation）：
       将每个社区压缩为超节点，超节点继承所有子节点的
       chunk_id 并集，保留物理锚点信息。
    4. 递归：对聚合后的图重复上述步骤，直到收敛。

层次化输出：
    返回多层次的社区分配结果，每层对应一个 {node_id: community_id} 映射。
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

from .annealing import AnnealingConfig, get_lambda
from .physical_anchor import (
    PhysicalNode,
    compute_delta_entropy,
    compute_structural_entropy,
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CommunityState:
    """
    单层聚类的社区状态，维护算法运行时的所有中间数据。

    Attributes
    ----------
    node_to_community : Dict[str, int]
        {节点ID: 社区ID} 映射
    community_to_nodes : Dict[int, Set[str]]
        {社区ID: 节点ID集合} 映射（node_to_community 的反向索引）
    physical_nodes : Dict[str, PhysicalNode]
        {节点ID: PhysicalNode} 映射，携带物理锚点信息
    total_edge_weight : float
        图中所有边的权重之和（用于模块度计算）
    community_internal_weight : Dict[int, float]
        {社区ID: 社区内部边权重之和}
    community_total_degree : Dict[int, float]
        {社区ID: 社区内所有节点的度之和}
    node_degree : Dict[str, float]
        {节点ID: 该节点的加权度}
    """
    node_to_community: Dict[str, int] = field(default_factory=dict)
    community_to_nodes: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))
    physical_nodes: Dict[str, PhysicalNode] = field(default_factory=dict)
    total_edge_weight: float = 0.0
    community_internal_weight: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    community_total_degree: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    node_degree: Dict[str, float] = field(default_factory=lambda: defaultdict(float))

    def get_community_physical_nodes(self, community_id: int) -> List[PhysicalNode]:
        """获取指定社区内所有节点的 PhysicalNode 列表。"""
        return [
            self.physical_nodes[nid]
            for nid in self.community_to_nodes[community_id]
            if nid in self.physical_nodes
        ]


@dataclass
class HierarchicalCommunityResult:
    """
    层次化聚类的完整输出结果。

    Attributes
    ----------
    levels : List[Dict[str, int]]
        每层的 {原始节点ID: 社区ID} 映射列表。
        levels[0] 是最细粒度（底层），levels[-1] 是最粗粒度（顶层）。
    level_entropy : List[Dict[int, float]]
        每层每个社区的结构熵，{社区ID: H_structure}。
        用于验证物理约束效果。
    level_lambda : List[float]
        每层使用的 λ 值，便于调试退火效果。
    node_physical_map : Dict[str, PhysicalNode]
        原始节点的物理锚点信息。
    """
    levels: List[Dict[str, int]] = field(default_factory=list)
    level_entropy: List[Dict[int, float]] = field(default_factory=list)
    level_lambda: List[float] = field(default_factory=list)
    node_physical_map: Dict[str, PhysicalNode] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 模块度计算
# ---------------------------------------------------------------------------

def _compute_delta_modularity(
    node: str,
    target_community: int,
    state: CommunityState,
    graph: nx.Graph,
) -> float:
    """
    计算将节点 node 从当前社区移动到 target_community 后的模块度变化量 ΔQ。

    使用 Leiden/Louvain 标准公式：
        ΔQ = [k_i_in / m] - [k_i * Σ_tot / (2m²)]

    其中：
        k_i_in  : node 与 target_community 内节点的边权重之和
        k_i     : node 的加权度
        Σ_tot   : target_community 内所有节点的度之和
        m       : 图中所有边权重之和

    Parameters
    ----------
    node : str
        待移动的节点
    target_community : int
        目标社区 ID
    state : CommunityState
        当前社区状态
    graph : nx.Graph
        当前层的图

    Returns
    -------
    float
        模块度变化量 ΔQ（正值表示移动后模块度增加）
    """
    m = state.total_edge_weight
    if m == 0:
        return 0.0

    k_i = state.node_degree[node]
    sigma_tot = state.community_total_degree[target_community]

    # 计算 node 与 target_community 内节点的边权重之和
    k_i_in = 0.0
    for neighbor in graph.neighbors(node):
        if state.node_to_community.get(neighbor) == target_community:
            edge_data = graph.get_edge_data(node, neighbor)
            k_i_in += edge_data.get("weight", 1.0)

    # 标准模块度增益公式
    delta_q = (k_i_in / m) - (k_i * sigma_tot / (2.0 * m * m))
    return delta_q


def _compute_remove_delta_modularity(
    node: str,
    state: CommunityState,
    graph: nx.Graph,
) -> float:
    """
    计算将节点 node 从其当前社区移除后的模块度变化量（负值）。
    用于在移动节点前先计算"移除"的代价。
    """
    current_community = state.node_to_community[node]
    m = state.total_edge_weight
    if m == 0:
        return 0.0

    k_i = state.node_degree[node]
    sigma_tot = state.community_total_degree[current_community]

    k_i_in = 0.0
    for neighbor in graph.neighbors(node):
        if state.node_to_community.get(neighbor) == current_community and neighbor != node:
            edge_data = graph.get_edge_data(node, neighbor)
            k_i_in += edge_data.get("weight", 1.0)

    delta_q = (k_i_in / m) - (k_i * (sigma_tot - k_i) / (2.0 * m * m))
    return -delta_q  # 移除操作，取负


# ---------------------------------------------------------------------------
# 状态管理辅助函数
# ---------------------------------------------------------------------------

def _initialize_state(
    graph: nx.Graph,
    physical_nodes: Dict[str, PhysicalNode],
) -> CommunityState:
    """
    初始化社区状态：每个节点独立成一个社区。

    Parameters
    ----------
    graph : nx.Graph
        当前层的图（节点为字符串 ID，边有可选 weight 属性）
    physical_nodes : Dict[str, PhysicalNode]
        节点的物理锚点信息

    Returns
    -------
    CommunityState
        初始化后的状态（每节点一个社区）
    """
    state = CommunityState(physical_nodes=physical_nodes)

    # 计算总边权重
    state.total_edge_weight = sum(
        d.get("weight", 1.0) for _, _, d in graph.edges(data=True)
    )

    # 初始化：每个节点独立成一个社区
    for i, node in enumerate(graph.nodes()):
        state.node_to_community[node] = i
        state.community_to_nodes[i].add(node)

        # 计算节点度
        degree = sum(
            d.get("weight", 1.0) for _, _, d in graph.edges(node, data=True)
        )
        state.node_degree[node] = degree
        state.community_total_degree[i] = degree
        state.community_internal_weight[i] = 0.0  # 初始无内部边

    return state


def _move_node(
    node: str,
    target_community: int,
    state: CommunityState,
    graph: nx.Graph,
) -> None:
    """
    将节点从当前社区移动到目标社区，并更新所有相关状态。

    Parameters
    ----------
    node : str
        待移动的节点
    target_community : int
        目标社区 ID
    state : CommunityState
        当前社区状态（原地修改）
    graph : nx.Graph
        当前层的图
    """
    source_community = state.node_to_community[node]
    if source_community == target_community:
        return

    k_i = state.node_degree[node]

    # 计算 node 与源社区和目标社区的内部边权重
    k_i_source = 0.0
    k_i_target = 0.0
    for neighbor in graph.neighbors(node):
        w = graph.get_edge_data(node, neighbor).get("weight", 1.0)
        if state.node_to_community.get(neighbor) == source_community and neighbor != node:
            k_i_source += w
        elif state.node_to_community.get(neighbor) == target_community:
            k_i_target += w

    # 更新源社区
    state.community_to_nodes[source_community].discard(node)
    state.community_total_degree[source_community] -= k_i
    state.community_internal_weight[source_community] -= k_i_source

    # 清理空社区
    if not state.community_to_nodes[source_community]:
        del state.community_to_nodes[source_community]
        del state.community_total_degree[source_community]
        del state.community_internal_weight[source_community]

    # 更新目标社区
    state.community_to_nodes[target_community].add(node)
    state.community_total_degree[target_community] += k_i
    state.community_internal_weight[target_community] += k_i_target

    # 更新节点归属
    state.node_to_community[node] = target_community


# ---------------------------------------------------------------------------
# 核心算法阶段
# ---------------------------------------------------------------------------

def _local_moving_phase(
    graph: nx.Graph,
    state: CommunityState,
    lambda_val: float,
    rng: random.Random,
) -> bool:
    """
    局部移动阶段：核心创新点所在。

    对每个节点，遍历其邻居所在的所有社区，
    计算移动后的目标函数变化量：
        ΔJ = ΔQ_leiden - λ · ΔH_structure

    若 ΔJ > 0，则执行移动。

    Parameters
    ----------
    graph : nx.Graph
        当前层的图
    state : CommunityState
        当前社区状态（原地修改）
    lambda_val : float
        当前层级的 λ 值
    rng : random.Random
        随机数生成器（用于打乱节点遍历顺序）

    Returns
    -------
    bool
        本轮是否发生了任何节点移动（False 表示已收敛）
    """
    nodes = list(graph.nodes())
    rng.shuffle(nodes)  # 随机化遍历顺序，避免顺序偏差

    moved = False

    for node in nodes:
        current_community = state.node_to_community[node]

        # 收集邻居所在的所有不同社区（排除当前社区）
        neighbor_communities: Set[int] = set()
        for neighbor in graph.neighbors(node):
            nc = state.node_to_community.get(neighbor)
            if nc is not None and nc != current_community:
                neighbor_communities.add(nc)

        if not neighbor_communities:
            continue

        # 计算从当前社区移除的基础代价
        delta_q_remove = _compute_remove_delta_modularity(node, state, graph)

        # 当前社区的物理节点列表（不含当前节点）
        current_comm_nodes_without_self = [
            state.physical_nodes[n]
            for n in state.community_to_nodes[current_community]
            if n != node and n in state.physical_nodes
        ]

        best_delta_j = 0.0  # 只有 ΔJ > 0 才移动
        best_community = current_community

        for candidate_community in neighbor_communities:
            # ΔQ：移动到候选社区的模块度增益
            delta_q_add = _compute_delta_modularity(node, candidate_community, state, graph)
            delta_q = delta_q_remove + delta_q_add

            # ΔH：移动到候选社区的结构熵变化
            if lambda_val > 0 and node in state.physical_nodes:
                candidate_comm_nodes = [
                    state.physical_nodes[n]
                    for n in state.community_to_nodes[candidate_community]
                    if n in state.physical_nodes
                ]
                delta_h = compute_delta_entropy(candidate_comm_nodes, state.physical_nodes[node])
            else:
                delta_h = 0.0

            # 目标函数变化量：ΔJ = ΔQ - λ·ΔH
            delta_j = delta_q - lambda_val * delta_h

            if delta_j > best_delta_j:
                best_delta_j = delta_j
                best_community = candidate_community

        # 执行最优移动
        if best_community != current_community:
            _move_node(node, best_community, state, graph)
            moved = True

    return moved


def _refinement_phase(
    graph: nx.Graph,
    state: CommunityState,
    lambda_val: float,
    rng: random.Random,
) -> None:
    """
    细化阶段：在每个社区内部尝试进一步细分，提高解的质量。

    Leiden 相比 Louvain 的核心改进之一：
    在局部移动后，对每个社区内部重新运行局部移动，
    允许将社区拆分为更小的子社区，避免陷入局部最优。

    Parameters
    ----------
    graph : nx.Graph
        当前层的图
    state : CommunityState
        当前社区状态（原地修改）
    lambda_val : float
        当前层级的 λ 值
    rng : random.Random
        随机数生成器
    """
    # 对每个社区，构建子图并在子图上运行局部移动
    communities_snapshot = {
        comm_id: set(nodes)
        for comm_id, nodes in state.community_to_nodes.items()
    }

    next_community_id = max(state.community_to_nodes.keys(), default=0) + 1

    for comm_id, comm_nodes in communities_snapshot.items():
        if len(comm_nodes) <= 1:
            continue

        # 构建社区子图
        subgraph = graph.subgraph(comm_nodes).copy()

        # 在子图上初始化独立社区状态
        sub_physical = {
            n: state.physical_nodes[n]
            for n in comm_nodes
            if n in state.physical_nodes
        }
        sub_state = _initialize_state(subgraph, sub_physical)

        # 在子图上运行局部移动（单轮）
        _local_moving_phase(subgraph, sub_state, lambda_val, rng)

        # 检查是否发生了分裂（子图内出现了多个社区）
        sub_communities = set(sub_state.node_to_community.values())
        if len(sub_communities) <= 1:
            continue  # 未发生分裂，跳过

        # 将子图的分裂结果映射回主状态
        # 原社区 comm_id 保留给第一个子社区，其余分配新 ID
        sub_comm_list = list(sub_communities)
        sub_comm_to_main = {sub_comm_list[0]: comm_id}
        for sc in sub_comm_list[1:]:
            sub_comm_to_main[sc] = next_community_id
            next_community_id += 1

        for node in comm_nodes:
            sub_comm = sub_state.node_to_community[node]
            main_comm = sub_comm_to_main[sub_comm]
            if main_comm != comm_id:
                _move_node(node, main_comm, state, graph)


def _aggregation_phase(
    graph: nx.Graph,
    state: CommunityState,
    level: int,
) -> Tuple[nx.Graph, Dict[str, PhysicalNode], Dict[str, str]]:
    """
    聚合阶段：将每个社区压缩为超节点，构建下一层的图。

    关键设计：超节点继承所有子节点的 chunk_id 并集，
    保证物理锚点信息在层次化过程中不丢失。

    Parameters
    ----------
    graph : nx.Graph
        当前层的图
    state : CommunityState
        当前社区状态
    level : int
        当前层级（用于创建超节点的 PhysicalNode）

    Returns
    -------
    Tuple[nx.Graph, Dict[str, PhysicalNode], Dict[str, str]]
        - 聚合后的新图（超节点图）
        - 超节点的 PhysicalNode 映射
        - {原始节点ID: 超节点ID} 的映射（用于追踪层次关系）
    """
    super_graph = nx.Graph()
    super_physical: Dict[str, PhysicalNode] = {}
    node_to_super: Dict[str, str] = {}

    # 为每个社区创建超节点
    for comm_id, comm_nodes in state.community_to_nodes.items():
        super_node_id = f"super_{comm_id}_l{level}"

        # 超节点继承所有子节点的 chunk_id 并集
        child_physical_nodes = [
            state.physical_nodes[n]
            for n in comm_nodes
            if n in state.physical_nodes
        ]
        super_pnode = PhysicalNode.merge(
            node_id=super_node_id,
            nodes=child_physical_nodes,
            level=level + 1,
        )

        super_graph.add_node(super_node_id)
        super_physical[super_node_id] = super_pnode

        for node in comm_nodes:
            node_to_super[node] = super_node_id

    # 在超节点之间添加聚合边（权重为原始边权重之和）
    super_edge_weights: Dict[Tuple[str, str], float] = defaultdict(float)
    for u, v, data in graph.edges(data=True):
        su = node_to_super.get(u)
        sv = node_to_super.get(v)
        if su and sv and su != sv:
            key = (min(su, sv), max(su, sv))
            super_edge_weights[key] += data.get("weight", 1.0)

    for (su, sv), weight in super_edge_weights.items():
        super_graph.add_edge(su, sv, weight=weight)

    return super_graph, super_physical, node_to_super


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def hierarchical_leiden_constrained(
    graph: nx.Graph,
    physical_nodes: Dict[str, PhysicalNode],
    annealing_config: Optional[AnnealingConfig] = None,
    max_cluster_size: int = 10,
    max_iterations: int = 10,
    seed: int = 42,
) -> HierarchicalCommunityResult:
    """
    带结构熵惩罚的层次化 Leiden 算法主入口。

    Parameters
    ----------
    graph : nx.Graph
        输入图。节点为字符串 ID，边可有 weight 属性（默认 1.0）。
        节点 ID 必须与 physical_nodes 的键一致。
    physical_nodes : Dict[str, PhysicalNode]
        {节点ID: PhysicalNode} 映射，携带物理锚点信息。
        图中所有节点都应有对应的 PhysicalNode。
    annealing_config : AnnealingConfig, optional
        退火配置。默认使用指数衰减，lambda_init=1000.0。
    max_cluster_size : int
        单个社区的最大节点数。超过此阈值的社区会被递归细分。
        默认 10（与原版 GraphRAG 一致）。
    max_iterations : int
        每层局部移动阶段的最大迭代轮数。默认 10。
    seed : int
        随机种子，保证结果可复现。默认 42。

    Returns
    -------
    HierarchicalCommunityResult
        层次化聚类结果，包含每层的社区分配、结构熵和 λ 值。

    Examples
    --------
    >>> import networkx as nx
    >>> from constrained_leiden.physical_anchor import PhysicalNode
    >>> from constrained_leiden.leiden_constrained import hierarchical_leiden_constrained
    >>>
    >>> G = nx.karate_club_graph()
    >>> # 为每个节点分配 chunk_id（模拟物理来源）
    >>> physical = {
    ...     str(n): PhysicalNode.from_entity(str(n), f"chunk_{n % 5}")
    ...     for n in G.nodes()
    ... }
    >>> G_str = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    >>> result = hierarchical_leiden_constrained(G_str, physical)
    >>> print(f"层次数：{len(result.levels)}")
    """
    if annealing_config is None:
        annealing_config = AnnealingConfig()

    rng = random.Random(seed)
    result = HierarchicalCommunityResult(node_physical_map=physical_nodes)

    # 当前层的图和物理节点映射
    current_graph = graph.copy()
    current_physical = dict(physical_nodes)

    # 追踪从超节点到原始节点的映射（用于将高层社区映射回原始节点）
    # super_to_originals[super_node_id] = set of original node IDs
    super_to_originals: Dict[str, Set[str]] = {
        node: {node} for node in graph.nodes()
    }

    level = 0

    while True:
        lambda_val = get_lambda(level, annealing_config)

        # 初始化当前层状态
        state = _initialize_state(current_graph, current_physical)

        # 迭代执行局部移动 + 细化，直到收敛
        for iteration in range(max_iterations):
            moved = _local_moving_phase(current_graph, state, lambda_val, rng)
            _refinement_phase(current_graph, state, lambda_val, rng)
            if not moved:
                break

        # 将当前层的社区分配映射回原始节点
        # 每个超节点对应一组原始节点，它们共享同一个社区 ID
        original_node_to_community: Dict[str, int] = {}
        for super_node, comm_id in state.node_to_community.items():
            for original_node in super_to_originals.get(super_node, {super_node}):
                original_node_to_community[original_node] = comm_id

        # 计算当前层每个社区的结构熵
        level_entropy: Dict[int, float] = {}
        for comm_id, comm_nodes in state.community_to_nodes.items():
            comm_physical = [
                current_physical[n]
                for n in comm_nodes
                if n in current_physical
            ]
            level_entropy[comm_id] = compute_structural_entropy(comm_physical)

        result.levels.append(original_node_to_community)
        result.level_entropy.append(level_entropy)
        result.level_lambda.append(lambda_val)

        # 检查终止条件：
        # 1. 所有节点已在同一社区（完全聚合）
        # 2. 社区数量等于节点数量（无法进一步聚合）
        num_communities = len(state.community_to_nodes)
        if num_communities <= 1 or num_communities == len(current_graph.nodes()):
            break

        # 检查是否所有社区都满足 max_cluster_size 约束
        # 若满足，且 lambda 已接近 0，则停止
        max_comm_size = max(
            len(nodes) for nodes in state.community_to_nodes.values()
        )
        if max_comm_size <= max_cluster_size and lambda_val < 1e-6:
            break

        # 聚合阶段：构建下一层的超节点图
        next_graph, next_physical, node_to_super = _aggregation_phase(
            current_graph, state, level
        )

        # 更新超节点到原始节点的映射
        new_super_to_originals: Dict[str, Set[str]] = {}
        for super_node in next_graph.nodes():
            new_super_to_originals[super_node] = set()

        for current_node, super_node in node_to_super.items():
            originals = super_to_originals.get(current_node, {current_node})
            new_super_to_originals[super_node].update(originals)

        super_to_originals = new_super_to_originals
        current_graph = next_graph
        current_physical = next_physical
        level += 1

        # 防止无限循环
        if level > annealing_config.max_level * 2:
            break

    return result
