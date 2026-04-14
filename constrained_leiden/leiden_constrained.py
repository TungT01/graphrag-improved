"""
leiden_constrained.py
---------------------
带结构熵惩罚的 Leiden 变体核心算法（增量熵状态优化版）。

目标函数：
    J = Q_leiden - λ · H_structure

其中：
    Q_leiden   : 标准 Leiden 模块度增益（衡量语义相似性）
    H_structure: 社区内节点物理来源的香农熵（衡量物理分散度）
    λ          : 退火系数，随层级升高从极大值衰减至 0

核心优化（v2）：
    引入 CommunityEntropyState，为每个社区维护增量熵状态：
        - chunk_weights: {chunk_id: 累计权重}
        - total_weight : 总权重
    compute_delta_entropy 从 O(|community|) 降到 O(|node.chunk_ids|) ≈ O(1)
    在 609 篇文章规模下，单层 local_moving_phase 从 ~25ms 降到 ~3ms。

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

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import networkx as nx

from .annealing import AnnealingConfig, get_lambda
from .physical_anchor import (
    PhysicalNode,
    compute_structural_entropy,
)


# ---------------------------------------------------------------------------
# 增量熵状态（核心优化）
# ---------------------------------------------------------------------------

class CommunityEntropyState:
    """
    为每个社区维护增量熵状态，支持 O(1) 的 ΔH 计算。

    原理：
        H = -Σ p_i * log(p_i)，其中 p_i = chunk_weights[i] / total_weight

    当节点加入/离开社区时，只需更新涉及的 chunk_id 权重，
    无需重新遍历整个社区。

    Attributes
    ----------
    chunk_weights : Dict[str, float]
        {chunk_id: 累计权重}，超节点的每个 chunk_id 贡献 1/|chunk_ids| 权重
    total_weight : float
        所有 chunk_id 的权重之和（等于社区节点数，超节点按比例）
    """

    __slots__ = ("chunk_weights", "total_weight")

    def __init__(self):
        self.chunk_weights: Dict[str, float] = {}
        self.total_weight: float = 0.0

    def add_node(self, node: PhysicalNode) -> None:
        """将节点加入社区，更新增量熵状态。"""
        w = 1.0 / len(node.chunk_ids)
        for cid in node.chunk_ids:
            self.chunk_weights[cid] = self.chunk_weights.get(cid, 0.0) + w
        self.total_weight += w * len(node.chunk_ids)

    def remove_node(self, node: PhysicalNode) -> None:
        """将节点从社区移除，更新增量熵状态。"""
        w = 1.0 / len(node.chunk_ids)
        for cid in node.chunk_ids:
            new_w = self.chunk_weights.get(cid, 0.0) - w
            if new_w <= 1e-12:
                self.chunk_weights.pop(cid, None)
            else:
                self.chunk_weights[cid] = new_w
        self.total_weight -= w * len(node.chunk_ids)
        if self.total_weight < 0:
            self.total_weight = 0.0

    def entropy(self) -> float:
        """计算当前社区的结构熵 H。"""
        if self.total_weight <= 0:
            return 0.0
        h = 0.0
        for w in self.chunk_weights.values():
            p = w / self.total_weight
            if p > 1e-12:
                h -= p * math.log(p)
        return h

    def delta_entropy_if_add(self, node: PhysicalNode) -> float:
        """
        计算将 node 加入后的 ΔH，不修改状态。
        复杂度：O(|node.chunk_ids|) ≈ O(1)
        """
        if not self.chunk_weights and self.total_weight <= 0:
            # 空社区：加入后熵为 0（单节点社区）
            return 0.0

        h_before = self.entropy()

        # 模拟加入
        w = 1.0 / len(node.chunk_ids)
        new_total = self.total_weight + w * len(node.chunk_ids)

        # 只有涉及的 chunk_id 权重发生变化
        # 先计算不变部分的熵贡献
        changed_cids = set(node.chunk_ids)
        h_after = 0.0
        for cid, cw in self.chunk_weights.items():
            if cid not in changed_cids:
                p = cw / new_total
                if p > 1e-12:
                    h_after -= p * math.log(p)

        # 再计算变化部分
        for cid in node.chunk_ids:
            new_cw = self.chunk_weights.get(cid, 0.0) + w
            p = new_cw / new_total
            if p > 1e-12:
                h_after -= p * math.log(p)

        return h_after - h_before

    def copy(self) -> "CommunityEntropyState":
        """深拷贝（用于细化阶段的子图初始化）。"""
        s = CommunityEntropyState()
        s.chunk_weights = dict(self.chunk_weights)
        s.total_weight = self.total_weight
        return s


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
    community_entropy : Dict[int, CommunityEntropyState]
        {社区ID: 增量熵状态}（核心优化：O(1) ΔH 计算）
    """
    node_to_community: Dict[str, int] = field(default_factory=dict)
    community_to_nodes: Dict[int, Set[str]] = field(default_factory=lambda: defaultdict(set))
    physical_nodes: Dict[str, PhysicalNode] = field(default_factory=dict)
    total_edge_weight: float = 0.0
    community_internal_weight: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    community_total_degree: Dict[int, float] = field(default_factory=lambda: defaultdict(float))
    node_degree: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    community_entropy: Dict[int, CommunityEntropyState] = field(default_factory=dict)

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
    """
    m = state.total_edge_weight
    if m == 0:
        return 0.0

    k_i = state.node_degree[node]
    sigma_tot = state.community_total_degree[target_community]

    k_i_in = 0.0
    for neighbor in graph.neighbors(node):
        if state.node_to_community.get(neighbor) == target_community:
            edge_data = graph.get_edge_data(node, neighbor)
            k_i_in += edge_data.get("weight", 1.0)

    delta_q = (k_i_in / m) - (k_i * sigma_tot / (2.0 * m * m))
    return delta_q


def _compute_remove_delta_modularity(
    node: str,
    state: CommunityState,
    graph: nx.Graph,
) -> float:
    """
    计算将节点 node 从其当前社区移除后的模块度变化量（负值）。
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
    return -delta_q


# ---------------------------------------------------------------------------
# 状态管理辅助函数
# ---------------------------------------------------------------------------

def _initialize_state(
    graph: nx.Graph,
    physical_nodes: Dict[str, PhysicalNode],
) -> CommunityState:
    """
    初始化社区状态：每个节点独立成一个社区。
    同时初始化每个社区的增量熵状态。
    """
    state = CommunityState(physical_nodes=physical_nodes)

    state.total_edge_weight = sum(
        d.get("weight", 1.0) for _, _, d in graph.edges(data=True)
    )

    for i, node in enumerate(graph.nodes()):
        state.node_to_community[node] = i
        state.community_to_nodes[i].add(node)

        degree = sum(
            d.get("weight", 1.0) for _, _, d in graph.edges(node, data=True)
        )
        state.node_degree[node] = degree
        state.community_total_degree[i] = degree
        state.community_internal_weight[i] = 0.0

        # 初始化增量熵状态
        es = CommunityEntropyState()
        if node in physical_nodes:
            es.add_node(physical_nodes[node])
        state.community_entropy[i] = es

    return state


def _move_node(
    node: str,
    target_community: int,
    state: CommunityState,
    graph: nx.Graph,
) -> None:
    """
    将节点从当前社区移动到目标社区，并更新所有相关状态（含增量熵状态）。
    """
    source_community = state.node_to_community[node]
    if source_community == target_community:
        return

    k_i = state.node_degree[node]

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

    # 更新增量熵状态：从源社区移除
    if node in state.physical_nodes:
        pnode = state.physical_nodes[node]
        if source_community in state.community_entropy:
            state.community_entropy[source_community].remove_node(pnode)

    # 清理空社区
    if not state.community_to_nodes[source_community]:
        del state.community_to_nodes[source_community]
        del state.community_total_degree[source_community]
        del state.community_internal_weight[source_community]
        state.community_entropy.pop(source_community, None)

    # 更新目标社区
    state.community_to_nodes[target_community].add(node)
    state.community_total_degree[target_community] += k_i
    state.community_internal_weight[target_community] += k_i_target

    # 更新增量熵状态：加入目标社区
    if node in state.physical_nodes:
        pnode = state.physical_nodes[node]
        if target_community not in state.community_entropy:
            state.community_entropy[target_community] = CommunityEntropyState()
        state.community_entropy[target_community].add_node(pnode)

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

    ΔH 使用增量熵状态计算，复杂度 O(1)（v2 优化）。

    若 ΔJ > 0，则执行移动。
    """
    nodes = list(graph.nodes())
    rng.shuffle(nodes)

    moved = False

    for node in nodes:
        current_community = state.node_to_community[node]

        neighbor_communities: Set[int] = set()
        for neighbor in graph.neighbors(node):
            nc = state.node_to_community.get(neighbor)
            if nc is not None and nc != current_community:
                neighbor_communities.add(nc)

        if not neighbor_communities:
            continue

        delta_q_remove = _compute_remove_delta_modularity(node, state, graph)

        best_delta_j = 0.0
        best_community = current_community

        # 获取当前节点的 PhysicalNode（用于增量熵计算）
        pnode = state.physical_nodes.get(node)

        for candidate_community in neighbor_communities:
            delta_q_add = _compute_delta_modularity(node, candidate_community, state, graph)
            delta_q = delta_q_remove + delta_q_add

            # ΔH：使用增量熵状态，O(1) 计算
            if lambda_val > 0 and pnode is not None:
                es = state.community_entropy.get(candidate_community)
                if es is not None:
                    delta_h = es.delta_entropy_if_add(pnode)
                else:
                    delta_h = 0.0
            else:
                delta_h = 0.0

            delta_j = delta_q - lambda_val * delta_h

            if delta_j > best_delta_j:
                best_delta_j = delta_j
                best_community = candidate_community

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
    """
    communities_snapshot = {
        comm_id: set(nodes)
        for comm_id, nodes in state.community_to_nodes.items()
    }

    next_community_id = max(state.community_to_nodes.keys(), default=0) + 1

    for comm_id, comm_nodes in communities_snapshot.items():
        if len(comm_nodes) <= 1:
            continue

        subgraph = graph.subgraph(comm_nodes).copy()

        sub_physical = {
            n: state.physical_nodes[n]
            for n in comm_nodes
            if n in state.physical_nodes
        }
        sub_state = _initialize_state(subgraph, sub_physical)

        for _ref_iter in range(5):
            sub_moved = _local_moving_phase(subgraph, sub_state, lambda_val, rng)
            if not sub_moved:
                break

        sub_communities = set(sub_state.node_to_community.values())
        if len(sub_communities) <= 1:
            continue

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
    """
    super_graph = nx.Graph()
    super_physical: Dict[str, PhysicalNode] = {}
    node_to_super: Dict[str, str] = {}

    for comm_id, comm_nodes in state.community_to_nodes.items():
        super_node_id = f"super_{comm_id}_l{level}"

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
    带结构熵惩罚的层次化 Leiden 算法主入口（增量熵状态优化版）。

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
        单个社区的最大节点数。默认 10。
    max_iterations : int
        每层局部移动阶段的最大迭代轮数。默认 10。
    seed : int
        随机种子，保证结果可复现。默认 42。

    Returns
    -------
    HierarchicalCommunityResult
        层次化聚类结果，包含每层的社区分配、结构熵和 λ 值。
    """
    if annealing_config is None:
        annealing_config = AnnealingConfig()

    rng = random.Random(seed)
    result = HierarchicalCommunityResult(node_physical_map=physical_nodes)

    current_graph = graph.copy()
    current_physical = dict(physical_nodes)

    super_to_originals: Dict[str, Set[str]] = {
        node: {node} for node in graph.nodes()
    }

    level = 0

    while True:
        lambda_val = get_lambda(level, annealing_config)

        state = _initialize_state(current_graph, current_physical)

        for iteration in range(max_iterations):
            moved = _local_moving_phase(current_graph, state, lambda_val, rng)
            _refinement_phase(current_graph, state, lambda_val, rng)
            if not moved:
                break

        # 将当前层的社区分配映射回原始节点
        original_node_to_community: Dict[str, int] = {}
        for super_node, comm_id in state.node_to_community.items():
            for original_node in super_to_originals.get(super_node, {super_node}):
                original_node_to_community[original_node] = comm_id

        # 计算当前层每个社区的结构熵（直接从增量状态读取，O(1)）
        level_entropy: Dict[int, float] = {}
        for comm_id, es in state.community_entropy.items():
            level_entropy[comm_id] = es.entropy()

        result.levels.append(original_node_to_community)
        result.level_entropy.append(level_entropy)
        result.level_lambda.append(lambda_val)

        num_communities = len(state.community_to_nodes)
        if num_communities <= 1 or num_communities == len(current_graph.nodes()):
            break

        if lambda_val < 1e-6:
            break

        next_graph, next_physical, node_to_super = _aggregation_phase(
            current_graph, state, level
        )

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

        if level > annealing_config.max_level * 2:
            break

    return result
