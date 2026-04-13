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
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import pandas as pd

from .annealing import AnnealingConfig, AnnealingSchedule
from .leiden_constrained import HierarchicalCommunityResult, hierarchical_leiden_constrained
from .physical_anchor import PhysicalNode, compute_structural_entropy


# ---------------------------------------------------------------------------
# leidenalg 快速路径（C 实现，比纯 Python 快 100x+）
# ---------------------------------------------------------------------------

def _try_import_leidenalg():
    """尝试导入 leidenalg，返回 (leidenalg, igraph) 或 (None, None)。"""
    try:
        import leidenalg
        import igraph as ig
        return leidenalg, ig
    except ImportError:
        return None, None


def _nx_to_igraph(graph: nx.Graph):
    """将 networkx 图转换为 igraph 图（保留边权重）。"""
    import igraph as ig
    nodes = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    weights = [graph[u][v].get("weight", 1.0) for u, v in graph.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.vs["name"] = nodes
    g.es["weight"] = weights
    return g, nodes


def _split_high_entropy_community(
    comm_nodes: List[str],
    physical_nodes: Dict[str, PhysicalNode],
    graph: nx.Graph,
    entropy_threshold: float = 1.5,
) -> List[List[str]]:
    """
    对高结构熵社区进行后处理拆分。

    策略：按 chunk_id 分组，将来自不同物理来源的节点拆分为子社区。
    这是结构熵约束的简化实现：确保每个社区内节点的物理来源尽量集中。
    """
    if len(comm_nodes) <= 1:
        return [comm_nodes]

    # 计算当前社区的结构熵
    phys_list = [physical_nodes[n] for n in comm_nodes if n in physical_nodes]
    if not phys_list:
        return [comm_nodes]

    entropy = compute_structural_entropy(phys_list)
    if entropy <= entropy_threshold:
        return [comm_nodes]

    # 按主要 chunk_id 分组（取每个节点 chunk_ids 中出现最多的那个）
    chunk_groups: Dict[str, List[str]] = defaultdict(list)
    for node in comm_nodes:
        if node in physical_nodes:
            pn = physical_nodes[node]
            if pn.chunk_ids:
                # 选择该节点最常见的 chunk_id
                primary_chunk = sorted(pn.chunk_ids)[0]
                chunk_groups[primary_chunk].append(node)
            else:
                chunk_groups["__unknown__"].append(node)
        else:
            chunk_groups["__unknown__"].append(node)

    if len(chunk_groups) <= 1:
        return [comm_nodes]

    return list(chunk_groups.values())


def _fast_leiden_with_entropy_constraint(
    graph: nx.Graph,
    physical_nodes: Dict[str, PhysicalNode],
    lambda_init: float,
    max_cluster_size: int,
    seed: int,
) -> HierarchicalCommunityResult:
    """
    基于 leidenalg (C 实现) 的快速社区检测，叠加结构熵后处理。

    两阶段策略：
    1. 用 leidenalg 快速得到初始社区划分（baseline: lambda=0 直接用原版）
    2. 若 lambda > 0，对高结构熵社区做后处理拆分（ours 版本的差异化）

    Parameters
    ----------
    lambda_init : float
        λ 值。0 = 纯 Leiden（baseline），>0 = 叠加结构熵约束（ours）
    """
    leidenalg, ig = _try_import_leidenalg()
    if leidenalg is None:
        raise ImportError("leidenalg not available")

    result = HierarchicalCommunityResult(node_physical_map=physical_nodes)

    # 转换为 igraph
    ig_graph, nodes = _nx_to_igraph(graph)
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # 运行 leidenalg（使用 CPMVertexPartition，支持分辨率参数，社区粒度更可控）
    # resolution_parameter 越大，社区越小越多（类似 GraphRAG 的 max_cluster_size 控制）
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.CPMVertexPartition,
        weights="weight",
        resolution_parameter=0.5,
        seed=seed,
        n_iterations=10,
    )

    # 构建 level 0 的社区分配
    node_to_community: Dict[str, int] = {}
    community_to_nodes: Dict[int, List[str]] = defaultdict(list)
    for comm_id, members in enumerate(partition):
        for idx in members:
            node_name = nodes[idx]
            node_to_community[node_name] = comm_id
            community_to_nodes[comm_id].append(node_name)

    # 若 lambda > 0，对高结构熵社区做后处理拆分（这是 ours 版本的核心差异）
    if lambda_init > 0:
        # 熵阈值：lambda 越大，约束越严，阈值越低
        # 只拆分熵值极高的社区（跨 5+ 篇文档），保持社区数量与 baseline 接近
        entropy_threshold = max(1.5, 3.0 - lambda_init / 1000.0)
        new_comm_id = max(community_to_nodes.keys()) + 1
        new_node_to_community: Dict[str, int] = dict(node_to_community)
        new_community_to_nodes: Dict[int, List[str]] = {}

        for comm_id, comm_nodes in community_to_nodes.items():
            sub_groups = _split_high_entropy_community(
                comm_nodes, physical_nodes, graph, entropy_threshold
            )
            if len(sub_groups) == 1:
                new_community_to_nodes[comm_id] = comm_nodes
            else:
                # 拆分为多个子社区
                for i, sub_group in enumerate(sub_groups):
                    if i == 0:
                        new_community_to_nodes[comm_id] = sub_group
                        for n in sub_group:
                            new_node_to_community[n] = comm_id
                    else:
                        new_community_to_nodes[new_comm_id] = sub_group
                        for n in sub_group:
                            new_node_to_community[n] = new_comm_id
                        new_comm_id += 1

        node_to_community = new_node_to_community
        community_to_nodes = new_community_to_nodes

    # 计算每个社区的结构熵
    level_entropy: Dict[int, float] = {}
    for comm_id, comm_nodes in community_to_nodes.items():
        phys_list = [physical_nodes[n] for n in comm_nodes if n in physical_nodes]
        level_entropy[comm_id] = compute_structural_entropy(phys_list)

    result.levels.append(dict(node_to_community))
    result.level_entropy.append(level_entropy)
    result.level_lambda.append(lambda_init)

    return result


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
    min_edge_weight: float = 2.0,
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
    min_edge_weight : float
        最小边权重阈值，过滤低频共现关系，默认 2.0

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

    # 过滤低权重关系（减少图规模，加速 Leiden）
    if min_edge_weight > 1.0 and "weight" in relationships.columns:
        relationships = relationships[relationships["weight"] >= min_edge_weight]

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

    # 优先使用 leidenalg (C 实现) 快速路径；若不可用则回退到纯 Python 实现
    leidenalg_mod, _ = _try_import_leidenalg()
    if leidenalg_mod is not None:
        result = _fast_leiden_with_entropy_constraint(
            graph=graph,
            physical_nodes=physical_nodes,
            lambda_init=annealing_config.lambda_init,
            max_cluster_size=max_cluster_size,
            seed=seed,
        )
    else:
        # 回退：纯 Python 实现（较慢，适合小图）
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
