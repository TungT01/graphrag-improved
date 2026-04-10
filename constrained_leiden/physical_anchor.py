"""
physical_anchor.py
------------------
物理锚点数据结构与结构熵计算模块。

核心概念：
- 每个图节点携带一个 chunk_id 集合，表示其物理来源
- 结构熵 H_structure 衡量一个社区内节点来源的物理分散程度
- H=0 表示社区内所有节点来自同一物理来源（最纯净）
- H=log(k) 表示节点均匀分散于 k 个不同来源（最混乱）
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set


@dataclass
class PhysicalNode:
    """
    图中的一个节点，携带物理锚点信息。

    Attributes
    ----------
    node_id : str
        节点唯一标识符（对应 GraphRAG 中的实体 title）
    chunk_ids : FrozenSet[str]
        该节点来源的物理 chunk ID 集合。
        - 原始叶节点：通常只有 1 个 chunk_id
        - 超节点（聚合后）：继承所有子节点的 chunk_id 并集
    level : int
        节点所在的层次级别（0 = 原始叶节点）
    """
    node_id: str
    chunk_ids: FrozenSet[str]
    level: int = 0

    @classmethod
    def from_entity(cls, node_id: str, chunk_id: str) -> "PhysicalNode":
        """从单个实体创建叶节点（最常见的初始化方式）。"""
        return cls(node_id=node_id, chunk_ids=frozenset([chunk_id]), level=0)

    @classmethod
    def merge(cls, node_id: str, nodes: list["PhysicalNode"], level: int) -> "PhysicalNode":
        """
        将多个节点合并为一个超节点。
        chunk_ids 取所有子节点的并集，保证物理来源信息不丢失。
        """
        merged_chunks: Set[str] = set()
        for n in nodes:
            merged_chunks.update(n.chunk_ids)
        return cls(node_id=node_id, chunk_ids=frozenset(merged_chunks), level=level)


def compute_structural_entropy(community_nodes: list[PhysicalNode]) -> float:
    """
    计算一个社区的结构熵 H_structure。

    定义：以社区内所有节点的 chunk_id 来源分布为概率分布，
    计算香农熵。

    H = -∑ p_i * log(p_i)

    其中 p_i = (来自第 i 个 chunk_id 的节点数) / (社区总节点数)

    Parameters
    ----------
    community_nodes : list[PhysicalNode]
        社区内的所有节点

    Returns
    -------
    float
        结构熵值，范围 [0, log(k)]，k 为不同 chunk_id 数量
        返回 0.0 表示社区物理纯净（所有节点来自同一来源）

    Notes
    -----
    当节点携带多个 chunk_id 时（超节点情况），每个 chunk_id
    贡献 1/|chunk_ids| 的权重，保证总权重归一化。
    """
    if not community_nodes:
        return 0.0

    # 统计每个 chunk_id 的加权出现次数
    # 超节点的每个 chunk_id 贡献 1/|chunk_ids| 的权重
    chunk_weights: Dict[str, float] = {}
    total_weight = 0.0

    for node in community_nodes:
        weight_per_chunk = 1.0 / len(node.chunk_ids)
        for chunk_id in node.chunk_ids:
            chunk_weights[chunk_id] = chunk_weights.get(chunk_id, 0.0) + weight_per_chunk
            total_weight += weight_per_chunk

    if total_weight == 0.0:
        return 0.0

    # 计算香农熵
    entropy = 0.0
    for weight in chunk_weights.values():
        p = weight / total_weight
        if p > 0:
            entropy -= p * math.log(p)  # 使用自然对数（单位：nat）

    return entropy


def compute_delta_entropy(
    community_nodes: list[PhysicalNode],
    new_node: PhysicalNode,
) -> float:
    """
    计算将 new_node 加入社区后，结构熵的变化量 ΔH。

    ΔH = H(community ∪ {new_node}) - H(community)

    正值表示加入后熵增加（物理来源更分散）。
    负值表示加入后熵减少（物理来源更集中，不常见）。

    Parameters
    ----------
    community_nodes : list[PhysicalNode]
        当前社区内的节点列表（不含 new_node）
    new_node : PhysicalNode
        待加入的节点

    Returns
    -------
    float
        结构熵变化量 ΔH
    """
    h_before = compute_structural_entropy(community_nodes)
    h_after = compute_structural_entropy(community_nodes + [new_node])
    return h_after - h_before


def compute_community_entropy_map(
    communities: Dict[int, list[PhysicalNode]]
) -> Dict[int, float]:
    """
    批量计算所有社区的结构熵，返回 {community_id: entropy} 映射。

    Parameters
    ----------
    communities : Dict[int, list[PhysicalNode]]
        {社区ID: 社区内节点列表} 的映射

    Returns
    -------
    Dict[int, float]
        {社区ID: 结构熵} 的映射
    """
    return {
        comm_id: compute_structural_entropy(nodes)
        for comm_id, nodes in communities.items()
    }
