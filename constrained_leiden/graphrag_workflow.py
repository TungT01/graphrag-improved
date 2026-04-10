"""
graphrag_workflow.py
--------------------
将带结构熵惩罚的 Leiden 算法封装为 GraphRAG 可替换的工作流接口。

使用方式：
    在 GraphRAG 的 Pipeline 中，用本模块的 run_workflow 替换
    原有的 create_communities 工作流：

    from constrained_leiden.graphrag_workflow import run_workflow
    WorkflowFactory.register("create_communities", run_workflow)

输入：
    - entities.parquet  : 实体表，必须包含 title, text_unit_ids 列
    - relationships.parquet : 关系表，必须包含 source, target, weight 列

输出：
    - communities.parquet : 社区表，格式与原版 GraphRAG 兼容
      额外新增列：
        - structural_entropy (float) : 社区结构熵，用于验证物理约束效果
        - lambda_used (float)        : 该层使用的 λ 值
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd

from .annealing import AnnealingConfig, AnnealingSchedule
from .leiden_constrained import HierarchicalCommunityResult, hierarchical_leiden_constrained
from .physical_anchor import PhysicalNode


# ---------------------------------------------------------------------------
# 数据转换：GraphRAG Parquet → 算法输入
# ---------------------------------------------------------------------------

def build_graph_from_graphrag(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> nx.Graph:
    """
    从 GraphRAG 的 entities 和 relationships DataFrame 构建 networkx 图。

    Parameters
    ----------
    entities : pd.DataFrame
        必须包含列：title
    relationships : pd.DataFrame
        必须包含列：source, target
        可选列：weight（默认 1.0）

    Returns
    -------
    nx.Graph
        无向图，节点为实体 title，边有 weight 属性
    """
    G = nx.Graph()

    # 添加节点
    for _, row in entities.iterrows():
        G.add_node(str(row["title"]))

    # 添加边
    for _, row in relationships.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        weight = float(row.get("weight", 1.0)) if "weight" in row else 1.0

        if G.has_node(source) and G.has_node(target) and source != target:
            if G.has_edge(source, target):
                # 累加重复边的权重
                G[source][target]["weight"] += weight
            else:
                G.add_edge(source, target, weight=weight)

    return G


def build_physical_nodes_from_graphrag(
    entities: pd.DataFrame,
) -> Dict[str, PhysicalNode]:
    """
    从 GraphRAG 的 entities DataFrame 构建物理节点映射。

    每个实体的 text_unit_ids 列表作为其物理来源（chunk_ids）。
    若 text_unit_ids 为空，则使用实体 title 的 hash 作为占位 chunk_id。

    Parameters
    ----------
    entities : pd.DataFrame
        必须包含列：title
        可选列：text_unit_ids（list of str）

    Returns
    -------
    Dict[str, PhysicalNode]
        {实体title: PhysicalNode} 映射
    """
    physical_nodes: Dict[str, PhysicalNode] = {}

    for _, row in entities.iterrows():
        title = str(row["title"])

        # 提取 chunk_ids
        chunk_ids = set()
        if "text_unit_ids" in row and row["text_unit_ids"] is not None:
            raw = row["text_unit_ids"]
            if isinstance(raw, list):
                chunk_ids = {str(cid) for cid in raw if cid}
            elif isinstance(raw, str) and raw.strip():
                # 可能是逗号分隔的字符串
                chunk_ids = {cid.strip() for cid in raw.split(",") if cid.strip()}

        # 若无 chunk_id，使用 title hash 作为占位符
        if not chunk_ids:
            chunk_ids = {f"placeholder_{hash(title) % 100000}"}

        physical_nodes[title] = PhysicalNode(
            node_id=title,
            chunk_ids=frozenset(chunk_ids),
            level=0,
        )

    return physical_nodes


# ---------------------------------------------------------------------------
# 数据转换：算法输出 → GraphRAG communities.parquet
# ---------------------------------------------------------------------------

def convert_result_to_communities_df(
    result: HierarchicalCommunityResult,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    """
    将层次化聚类结果转换为 GraphRAG 兼容的 communities DataFrame。

    输出格式与原版 GraphRAG 的 communities.parquet 完全兼容，
    额外新增 structural_entropy 和 lambda_used 列。

    Parameters
    ----------
    result : HierarchicalCommunityResult
        层次化聚类结果
    entities : pd.DataFrame
        原始实体表（用于获取 text_unit_ids）
    relationships : pd.DataFrame
        原始关系表（用于获取社区内关系）

    Returns
    -------
    pd.DataFrame
        communities DataFrame，列包括：
        id, community_id, level, title, entity_ids,
        relationship_ids, text_unit_ids, parent_id, children,
        structural_entropy, lambda_used
    """
    # 构建实体 title → id 的映射
    entity_title_to_id: Dict[str, str] = {}
    entity_title_to_text_units: Dict[str, List[str]] = {}
    if "title" in entities.columns:
        for _, row in entities.iterrows():
            title = str(row["title"])
            entity_id = str(row.get("id", title))
            entity_title_to_id[title] = entity_id
            if "text_unit_ids" in row and row["text_unit_ids"] is not None:
                raw = row["text_unit_ids"]
                if isinstance(raw, list):
                    entity_title_to_text_units[title] = [str(x) for x in raw]
                else:
                    entity_title_to_text_units[title] = []

    # 构建关系索引：{(source, target): relationship_id}
    rel_index: Dict[tuple, str] = {}
    if "source" in relationships.columns and "target" in relationships.columns:
        for _, row in relationships.iterrows():
            key = (str(row["source"]), str(row["target"]))
            rel_index[key] = str(row.get("id", f"{row['source']}_{row['target']}"))

    records = []

    for level_idx, node_to_comm in enumerate(result.levels):
        lambda_val = result.level_lambda[level_idx]
        level_entropy = result.level_entropy[level_idx]

        # 反转映射：{community_id: [node_titles]}
        comm_to_nodes: Dict[int, List[str]] = {}
        for node, comm_id in node_to_comm.items():
            comm_to_nodes.setdefault(comm_id, []).append(node)

        # 构建父子关系（当前层社区 → 上一层社区）
        parent_map: Dict[int, Optional[int]] = {}
        if level_idx > 0:
            prev_node_to_comm = result.levels[level_idx - 1]
            for comm_id, nodes in comm_to_nodes.items():
                # 找到当前社区节点在上一层所属的社区
                parent_comms = set()
                for node in nodes:
                    if node in prev_node_to_comm:
                        parent_comms.add(prev_node_to_comm[node])
                # 取最多节点所在的上一层社区作为父社区
                if parent_comms:
                    parent_map[comm_id] = max(
                        parent_comms,
                        key=lambda pc: sum(
                            1 for n in nodes
                            if prev_node_to_comm.get(n) == pc
                        )
                    )
                else:
                    parent_map[comm_id] = None
        else:
            parent_map = {comm_id: None for comm_id in comm_to_nodes}

        # 构建子社区映射（上一层社区 → 当前层子社区列表）
        children_map: Dict[int, List[int]] = {comm_id: [] for comm_id in comm_to_nodes}
        if level_idx > 0:
            for child_comm, parent_comm in parent_map.items():
                if parent_comm is not None and parent_comm in children_map:
                    children_map[parent_comm].append(child_comm)

        for comm_id, nodes in comm_to_nodes.items():
            # 收集实体 ID
            entity_ids = [
                entity_title_to_id.get(n, n) for n in nodes
            ]

            # 收集关系 ID（社区内部的关系）
            node_set = set(nodes)
            relationship_ids = []
            for (src, tgt), rel_id in rel_index.items():
                if src in node_set and tgt in node_set:
                    relationship_ids.append(rel_id)

            # 收集 text_unit_ids（所有节点的 chunk_id 并集）
            text_unit_ids: List[str] = []
            seen_tus: set = set()
            for node in nodes:
                for tu in entity_title_to_text_units.get(node, []):
                    if tu not in seen_tus:
                        text_unit_ids.append(tu)
                        seen_tus.add(tu)

            # 结构熵
            entropy = level_entropy.get(comm_id, 0.0)

            # 父社区 ID（转为字符串格式）
            parent_comm = parent_map.get(comm_id)
            parent_id = f"community_{level_idx - 1}_{parent_comm}" if parent_comm is not None else None

            records.append({
                "id": str(uuid.uuid4()),
                "community_id": comm_id,
                "level": level_idx,
                "title": f"Community {comm_id} (Level {level_idx})",
                "entity_ids": entity_ids,
                "relationship_ids": relationship_ids,
                "text_unit_ids": text_unit_ids,
                "parent_id": parent_id,
                "children": [
                    f"community_{level_idx + 1}_{c}"
                    for c in children_map.get(comm_id, [])
                ],
                # 新增列：物理约束效果验证
                "structural_entropy": entropy,
                "lambda_used": lambda_val,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# GraphRAG 工作流主入口
# ---------------------------------------------------------------------------

def run_constrained_community_detection(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
    annealing_config: Optional[AnnealingConfig] = None,
    max_cluster_size: int = 10,
    max_iterations: int = 10,
    seed: int = 42,
    use_lcc: bool = True,
) -> pd.DataFrame:
    """
    带结构熵惩罚的社区检测主函数，替换 GraphRAG 的 create_communities 工作流。

    Parameters
    ----------
    entities : pd.DataFrame
        GraphRAG 的 entities.parquet，必须包含 title 列
    relationships : pd.DataFrame
        GraphRAG 的 relationships.parquet，必须包含 source, target 列
    annealing_config : AnnealingConfig, optional
        退火配置，默认使用指数衰减 lambda_init=1000.0
    max_cluster_size : int
        社区最大节点数，默认 10
    max_iterations : int
        每层最大迭代轮数，默认 10
    seed : int
        随机种子，默认 42
    use_lcc : bool
        是否仅对最大连通分量运行聚类（过滤孤立节点），默认 True

    Returns
    -------
    pd.DataFrame
        communities DataFrame，与原版 GraphRAG 格式兼容
    """
    if annealing_config is None:
        annealing_config = AnnealingConfig(
            lambda_init=1000.0,
            lambda_min=0.0,
            max_level=10,
            decay_rate=0.5,
            schedule=AnnealingSchedule.EXPONENTIAL,
        )

    # 构建图和物理节点
    graph = build_graph_from_graphrag(entities, relationships)
    physical_nodes = build_physical_nodes_from_graphrag(entities)

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=[
            "id", "community_id", "level", "title",
            "entity_ids", "relationship_ids", "text_unit_ids",
            "parent_id", "children", "structural_entropy", "lambda_used",
        ])

    # 可选：仅对最大连通分量运行聚类
    if use_lcc and not nx.is_connected(graph):
        lcc_nodes = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(lcc_nodes).copy()
        physical_nodes = {k: v for k, v in physical_nodes.items() if k in lcc_nodes}

    # 运行带结构熵惩罚的层次化 Leiden
    result = hierarchical_leiden_constrained(
        graph=graph,
        physical_nodes=physical_nodes,
        annealing_config=annealing_config,
        max_cluster_size=max_cluster_size,
        max_iterations=max_iterations,
        seed=seed,
    )

    # 转换为 GraphRAG 兼容格式
    communities_df = convert_result_to_communities_df(result, entities, relationships)

    return communities_df
