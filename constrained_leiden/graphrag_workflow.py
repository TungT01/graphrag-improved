"""
graphrag_workflow.py
--------------------
将带结构熵惩罚的 Leiden 算法封装为 GraphRAG 可替换的工作流接口。

v3 重大变更（物理优先架构）：
    图构建逻辑完全重写，适配新的实例级节点设计：

    1. 节点：每个节点是带物理坐标的实体实例，ID = {sent_id}-{entity_name}
       同名实体在不同句子里是不同节点，底层图保持物理纯净。

    2. 边：严格限制在句子内部
       - 语义边：来自三元组提取（source/target 的 sent_id 前缀相同）
       - 物理结构边：同句内实体共现（predicate = "co_occurs"）
       - 跨句/跨段/跨文档：无任何预设边

    3. 物理锚点：每个节点的 PhysicalNode.chunk_ids = {sent_id}（单一句子级锚点）
       结构熵计算的是社区内节点的 sent_id 分布熵，比 chunk_id 更细粒度。

    4. 实体消解：不在图构建阶段做，由 λ 退火驱动的 Leiden 聚类在社区层面涌现。

使用方式：
    from constrained_leiden.graphrag_workflow import run_constrained_community_detection
    communities_df = run_constrained_community_detection(entities_df, relationships_df)

输入：
    - entities DataFrame    : id, title, sent_id, para_id, doc_id, ...
    - relationships DataFrame: id, source, target, weight, predicate, sent_id, ...

输出：
    - communities DataFrame : 与原版 GraphRAG 格式兼容
      额外新增列：structural_entropy, lambda_used
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import pandas as pd

from .annealing import AnnealingConfig, AnnealingSchedule
from .leiden_constrained import HierarchicalCommunityResult, hierarchical_leiden_constrained
from .physical_anchor import PhysicalNode, compute_structural_entropy


# ---------------------------------------------------------------------------
# 数据转换：DataFrame → 算法输入
# ---------------------------------------------------------------------------

def build_graph_from_graphrag(
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> nx.Graph:
    """
    从 entities 和 relationships DataFrame 构建 networkx 图。

    v3 变更：
        边的 source/target 是实体的 node_id（含物理路径），
        不再是实体 title。图中节点 ID = entities["id"] 列。

    边的有效性验证：
        只保留 source 和 target 都在实体表中的边，
        确保图的完整性。

    Parameters
    ----------
    entities : pd.DataFrame
        必须包含列：id（node_id）
    relationships : pd.DataFrame
        必须包含列：source（node_id）, target（node_id）
        可选列：weight（默认 1.0）

    Returns
    -------
    nx.Graph
        无向图，节点为 node_id，边有 weight 属性
    """
    G = nx.Graph()

    # 添加所有实体节点
    valid_node_ids: Set[str] = set()
    for _, row in entities.iterrows():
        node_id = str(row["id"])
        G.add_node(node_id)
        valid_node_ids.add(node_id)

    # 添加边（只保留两端节点都存在的边）
    for _, row in relationships.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        weight = float(row.get("weight", 1.0)) if "weight" in row else 1.0

        if source not in valid_node_ids or target not in valid_node_ids:
            continue
        if source == target:
            continue

        if G.has_edge(source, target):
            G[source][target]["weight"] += weight
        else:
            G.add_edge(source, target, weight=weight)

    return G


def build_intra_doc_entity_edges(
    entities: pd.DataFrame,
    weight: float = 0.5,
) -> List[Tuple[str, str, float]]:
    """
    路径A：文档内实体消解边（Intra-Document Entity Merging）。

    为同一文档内出现的同名实体（不同句子里的实例节点）之间添加软连接边。
    这些边权重较低（默认 0.5），不覆盖语义边，只起到"桥接"作用，
    让 Leiden 能够发现跨句子的同名实体聚类。

    设计原则：
    - 只在同一文档（doc_id 相同）内连接，不跨文档
    - 权重 < 1.0，低于语义边，避免过度合并
    - 相邻实例节点之间连边（链式连接），避免 O(n²) 爆炸
    - 过滤噪声实体名（长度 < 3 或在停用词表中）

    Parameters
    ----------
    entities : pd.DataFrame
        必须包含列：id（node_id）, title（实体名）, doc_id
    weight : float
        跨句子同名实体边的权重，默认 0.5

    Returns
    -------
    List[Tuple[str, str, float]]
        [(source_node_id, target_node_id, weight), ...]
    """
    if "title" not in entities.columns or "doc_id" not in entities.columns:
        return []

    # 噪声实体名过滤（与 extractor.py 中的 _STOPWORDS 保持一致）
    _NOISE_TITLES = {
        "which", "there", "There", "one", "the company", "available",
        "a lot", "people", "able", "good", "more", "something", "some",
        "get", "make", "use", "take", "have", "be", "do", "go",
        "it", "this", "that", "these", "those", "they", "we",
        "he", "she", "who", "what", "where", "when", "how",
        "the ball", "the game", "the team", "the player",
    }

    # 按 (doc_id, title_lower) 分组，收集同文档同名实体的 node_id 列表
    from collections import defaultdict
    doc_title_to_nodes: Dict[str, List[str]] = defaultdict(list)

    for _, row in entities.iterrows():
        title = str(row.get("title", "")).strip()
        doc_id = str(row.get("doc_id", "")).strip()
        node_id = str(row["id"])

        # 过滤噪声
        if len(title) < 3:
            continue
        if title in _NOISE_TITLES:
            continue
        if title.lower() in _NOISE_TITLES:
            continue

        key = f"{doc_id}|||{title.lower()}"
        doc_title_to_nodes[key].append(node_id)

    # 链式连接：node[0]-node[1]-node[2]-...（避免 O(n²)）
    edges: List[Tuple[str, str, float]] = []
    for key, nodes in doc_title_to_nodes.items():
        if len(nodes) < 2:
            continue
        for i in range(len(nodes) - 1):
            edges.append((nodes[i], nodes[i + 1], weight))

    return edges


def build_physical_nodes_from_graphrag(
    entities: pd.DataFrame,
) -> Dict[str, PhysicalNode]:
    """
    从 entities DataFrame 构建物理节点映射。

    v3 物理锚点策略：
        每个节点的物理锚点 = sent_id（句子级，单一来源）。
        这是 v2 架构的核心：每个实体实例节点只来自一个句子，
        chunk_ids = {sent_id}，结构熵能真正降为 0。

    回退策略（兼容旧格式）：
        若 sent_id 不可用，依次尝试 primary_chunk_id → text_unit_ids[0] → title hash。

    Parameters
    ----------
    entities : pd.DataFrame
        必须包含列：id（node_id）
        优先列：sent_id（句子级物理锚点）
        兼容列：primary_chunk_id, text_unit_ids

    Returns
    -------
    Dict[str, PhysicalNode]
        {node_id: PhysicalNode}，每个节点只携带单一 sent_id
    """
    physical_nodes: Dict[str, PhysicalNode] = {}

    for _, row in entities.iterrows():
        node_id = str(row["id"])

        # 优先使用 sent_id（v3 格式，句子级精确锚点）
        anchor_id = ""

        if "sent_id" in row and row["sent_id"] and str(row["sent_id"]).strip():
            anchor_id = str(row["sent_id"]).strip()

        elif "primary_chunk_id" in row and row["primary_chunk_id"] and str(row["primary_chunk_id"]).strip():
            # 兼容 v2 格式
            anchor_id = str(row["primary_chunk_id"]).strip()

        elif "text_unit_ids" in row and row["text_unit_ids"] is not None:
            # 兼容 v1 格式
            raw = row["text_unit_ids"]
            if isinstance(raw, list) and raw:
                anchor_id = str(raw[0])
            elif isinstance(raw, str) and raw.strip():
                parts = [x.strip() for x in raw.split(",") if x.strip()]
                if parts:
                    anchor_id = parts[0]

        if not anchor_id:
            # 最终兜底：使用 node_id 的 hash
            anchor_id = f"placeholder_{hash(node_id) % 100000}"

        physical_nodes[node_id] = PhysicalNode(
            node_id=node_id,
            chunk_ids=frozenset([anchor_id]),   # 单一物理来源
            level=0,
        )

    return physical_nodes


# ---------------------------------------------------------------------------
# 数据转换：算法输出 → communities DataFrame
# ---------------------------------------------------------------------------

def convert_result_to_communities_df(
    result: HierarchicalCommunityResult,
    entities: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    """
    将层次化聚类结果转换为 GraphRAG 兼容的 communities DataFrame。

    v3 变更：
        entity_ids 存储的是 node_id（含物理路径），
        text_unit_ids 从 sent_id 推导（每个节点的 sent_id）。

    输出格式与原版 GraphRAG 的 communities.parquet 完全兼容，
    额外新增 structural_entropy 和 lambda_used 列。
    """
    # 构建 node_id → entity 信息的索引
    node_to_info: Dict[str, Dict] = {}
    if "id" in entities.columns:
        for _, row in entities.iterrows():
            node_id = str(row["id"])
            node_to_info[node_id] = {
                "title": str(row.get("title", node_id)),
                "sent_id": str(row.get("sent_id", "")),
                "para_id": str(row.get("para_id", "")),
                "doc_id": str(row.get("doc_id", "")),
                "text_unit_ids": row.get("text_unit_ids", [sent_id := str(row.get("sent_id", ""))]),
            }

    # 构建关系索引
    rel_index: Dict[Tuple[str, str], str] = {}
    if "source" in relationships.columns and "target" in relationships.columns:
        for _, row in relationships.iterrows():
            key = (str(row["source"]), str(row["target"]))
            rel_index[key] = str(row.get("id", f"{row['source']}_{row['target']}"))

    records = []

    for level_idx, node_to_comm in enumerate(result.levels):
        lambda_val = result.level_lambda[level_idx]
        level_entropy = result.level_entropy[level_idx]

        comm_to_nodes: Dict[int, List[str]] = {}
        for node, comm_id in node_to_comm.items():
            comm_to_nodes.setdefault(comm_id, []).append(node)

        # 计算父社区映射
        parent_map: Dict[int, Optional[int]] = {}
        if level_idx > 0:
            prev_node_to_comm = result.levels[level_idx - 1]
            for comm_id, nodes in comm_to_nodes.items():
                parent_comms: Set[int] = set()
                for node in nodes:
                    if node in prev_node_to_comm:
                        parent_comms.add(prev_node_to_comm[node])
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

        # 计算子社区映射
        children_map: Dict[int, List[int]] = {comm_id: [] for comm_id in comm_to_nodes}
        if level_idx > 0:
            for child_comm, parent_comm in parent_map.items():
                if parent_comm is not None and parent_comm in children_map:
                    children_map[parent_comm].append(child_comm)

        for comm_id, nodes in comm_to_nodes.items():
            # entity_ids = node_id 列表
            entity_ids = nodes

            # text_unit_ids：从每个节点的 sent_id 收集（去重）
            text_unit_ids: List[str] = []
            seen_tus: Set[str] = set()
            for node in nodes:
                info = node_to_info.get(node, {})
                for tu in info.get("text_unit_ids", []):
                    if tu and tu not in seen_tus:
                        text_unit_ids.append(tu)
                        seen_tus.add(tu)

            # doc_ids：从节点的 doc_id 收集（去重，用于高层检索）
            doc_ids: List[str] = []
            seen_docs: Set[str] = set()
            for node in nodes:
                info = node_to_info.get(node, {})
                doc_id = info.get("doc_id", "")
                if doc_id and doc_id not in seen_docs:
                    doc_ids.append(doc_id)
                    seen_docs.add(doc_id)

            # 关系 ID
            node_set = set(nodes)
            relationship_ids = []
            for (src, tgt), rel_id in rel_index.items():
                if src in node_set and tgt in node_set:
                    relationship_ids.append(rel_id)

            entropy = level_entropy.get(comm_id, 0.0)
            parent_comm = parent_map.get(comm_id)
            parent_id = (
                f"community_{level_idx - 1}_{parent_comm}"
                if parent_comm is not None else None
            )

            records.append({
                "id": str(uuid.uuid4()),
                "community_id": comm_id,
                "level": level_idx,
                "title": f"Community {comm_id} (Level {level_idx})",
                "entity_ids": entity_ids,
                "relationship_ids": relationship_ids,
                "text_unit_ids": text_unit_ids,
                "doc_ids": doc_ids,           # v3 新增：文档 ID 列表（用于检索）
                "parent_id": parent_id,
                "children": [
                    f"community_{level_idx + 1}_{c}"
                    for c in children_map.get(comm_id, [])
                ],
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
    min_edge_weight: float = 1.0,
    intra_doc_merging: bool = False,
    intra_doc_edge_weight: float = 0.5,
) -> pd.DataFrame:
    """
    带结构熵惩罚的社区检测主函数。

    v3 变更：
        - 图节点是实体实例（node_id 含物理路径），不是实体概念
        - 物理锚点精确到句子级（sent_id）
        - min_edge_weight 默认降为 1.0（v2 架构边更稀疏，不需要高阈值过滤）

    v4 新增（路径A）：
        - intra_doc_merging：为同一文档内的同名实体加跨句子软连接边，
          解决 v3 图碎片化问题，让 Leiden 能发现文档级语义聚类。

    Parameters
    ----------
    entities : pd.DataFrame
        entities DataFrame，必须包含 id 列（node_id）
    relationships : pd.DataFrame
        relationships DataFrame，必须包含 source, target 列（node_id）
    annealing_config : AnnealingConfig, optional
        退火配置，默认使用指数衰减 lambda_init=1000.0
    max_cluster_size : int
        社区最大节点数，默认 10
    max_iterations : int
        每层最大迭代轮数，默认 10
    seed : int
        随机种子，默认 42
    use_lcc : bool
        是否仅对最大连通分量运行聚类，默认 True
    min_edge_weight : float
        最小边权重阈值，默认 1.0
    intra_doc_merging : bool
        路径A：是否注入文档内同名实体跨句子边，默认 False（保持向后兼容）
    intra_doc_edge_weight : float
        文档内实体消解边的权重，默认 0.5（低于语义边，避免过度合并）

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

    # 过滤低权重边
    if min_edge_weight > 0 and "weight" in relationships.columns:
        relationships = relationships[relationships["weight"] >= min_edge_weight]

    # 构建图和物理节点
    graph = build_graph_from_graphrag(entities, relationships)

    # 路径A：注入文档内同名实体跨句子软连接边
    if intra_doc_merging:
        extra_edges = build_intra_doc_entity_edges(entities, weight=intra_doc_edge_weight)
        for src, tgt, w in extra_edges:
            if src in graph and tgt in graph:
                if graph.has_edge(src, tgt):
                    graph[src][tgt]["weight"] += w
                else:
                    graph.add_edge(src, tgt, weight=w)

    physical_nodes = build_physical_nodes_from_graphrag(entities)

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=[
            "id", "community_id", "level", "title",
            "entity_ids", "relationship_ids", "text_unit_ids", "doc_ids",
            "parent_id", "children", "structural_entropy", "lambda_used",
        ])

    # 只对最大连通分量运行聚类（过滤孤立节点）
    if use_lcc and not nx.is_connected(graph):
        lcc_nodes = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(lcc_nodes).copy()
        physical_nodes = {k: v for k, v in physical_nodes.items() if k in lcc_nodes}

    # 带结构熵惩罚的层次化 Leiden 聚类
    result = hierarchical_leiden_constrained(
        graph=graph,
        physical_nodes=physical_nodes,
        annealing_config=annealing_config,
        max_cluster_size=max_cluster_size,
        max_iterations=max_iterations,
        seed=seed,
    )

    communities_df = convert_result_to_communities_df(result, entities, relationships)

    return communities_df
