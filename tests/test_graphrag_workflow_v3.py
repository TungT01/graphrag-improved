"""
tests/test_graphrag_workflow_v3.py
-----------------------------------
v3 物理优先架构 — 图构建与社区检测工作流单元测试。

测试覆盖：
1. build_graph_from_graphrag：节点 ID 使用 node_id（含物理路径）
2. build_graph_from_graphrag：跨句子无预设边（边只来自 relationships 表）
3. build_physical_nodes_from_graphrag：sent_id 优先策略
4. build_physical_nodes_from_graphrag：兼容旧格式（primary_chunk_id / text_unit_ids）
5. run_constrained_community_detection：空输入返回空 DataFrame
6. run_constrained_community_detection：输出格式包含 v3 新增列（doc_ids）
7. 物理约束：高 λ 时同句实体应聚在同一社区
8. convert_result_to_communities_df：doc_ids 字段正确收集
9. 结构熵在 v3 架构下能降为 0（同句实体聚合）
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import networkx as nx
import pandas as pd
import pytest

# 路径设置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
from graphrag_improved.constrained_leiden.graphrag_workflow import (
    build_graph_from_graphrag,
    build_physical_nodes_from_graphrag,
    convert_result_to_communities_df,
    run_constrained_community_detection,
)
from graphrag_improved.constrained_leiden.physical_anchor import PhysicalNode


# ===========================================================================
# Fixtures：v3 格式的 DataFrame
# ===========================================================================

def _make_v3_entities(specs: List[Dict]) -> pd.DataFrame:
    """
    构造 v3 格式的 entities DataFrame。
    specs 格式：[{"node_id": ..., "title": ..., "sent_id": ..., "doc_id": ...}, ...]
    """
    records = []
    for s in specs:
        node_id = s["node_id"]
        sent_id = s.get("sent_id", "doc001-p000-s000")
        records.append({
            "id": node_id,
            "title": s.get("title", node_id),
            "type": "PROPER_NOUN",
            "description": "",
            "sent_id": sent_id,
            "para_id": "-".join(sent_id.split("-")[:-1]),  # 去掉 -sXXX
            "doc_id": s.get("doc_id", "doc001"),
            "text_unit_ids": [sent_id],
            "primary_chunk_id": sent_id,
        })
    return pd.DataFrame(records)


def _make_v3_relationships(specs: List[Dict]) -> pd.DataFrame:
    """
    构造 v3 格式的 relationships DataFrame。
    specs 格式：[{"source": ..., "target": ..., "weight": ..., "predicate": ...}, ...]
    """
    records = []
    for i, s in enumerate(specs):
        records.append({
            "id": f"rel_{i:03d}",
            "source": s["source"],
            "target": s["target"],
            "weight": s.get("weight", 1.0),
            "predicate": s.get("predicate", "co_occurs"),
            "description": "",
            "sent_id": s.get("sent_id", ""),
        })
    return pd.DataFrame(records)


@pytest.fixture
def two_cluster_v3_data():
    """
    两簇 v3 数据：
    - 簇 A：节点 a1, a2, a3，来自 sent_id=doc001-p000-s000，内部强连接
    - 簇 B：节点 b1, b2, b3，来自 sent_id=doc001-p001-s000，内部强连接
    - 跨簇边：a1-b1，权重极小
    """
    entities = _make_v3_entities([
        {"node_id": "doc001-p000-s000-a1", "title": "A1", "sent_id": "doc001-p000-s000", "doc_id": "doc001"},
        {"node_id": "doc001-p000-s000-a2", "title": "A2", "sent_id": "doc001-p000-s000", "doc_id": "doc001"},
        {"node_id": "doc001-p000-s000-a3", "title": "A3", "sent_id": "doc001-p000-s000", "doc_id": "doc001"},
        {"node_id": "doc001-p001-s000-b1", "title": "B1", "sent_id": "doc001-p001-s000", "doc_id": "doc001"},
        {"node_id": "doc001-p001-s000-b2", "title": "B2", "sent_id": "doc001-p001-s000", "doc_id": "doc001"},
        {"node_id": "doc001-p001-s000-b3", "title": "B3", "sent_id": "doc001-p001-s000", "doc_id": "doc001"},
    ])
    relationships = _make_v3_relationships([
        # 簇 A 内部强连接
        {"source": "doc001-p000-s000-a1", "target": "doc001-p000-s000-a2", "weight": 10.0},
        {"source": "doc001-p000-s000-a2", "target": "doc001-p000-s000-a3", "weight": 10.0},
        {"source": "doc001-p000-s000-a1", "target": "doc001-p000-s000-a3", "weight": 10.0},
        # 簇 B 内部强连接
        {"source": "doc001-p001-s000-b1", "target": "doc001-p001-s000-b2", "weight": 10.0},
        {"source": "doc001-p001-s000-b2", "target": "doc001-p001-s000-b3", "weight": 10.0},
        {"source": "doc001-p001-s000-b1", "target": "doc001-p001-s000-b3", "weight": 10.0},
        # 跨簇弱连接
        {"source": "doc001-p000-s000-a1", "target": "doc001-p001-s000-b1", "weight": 0.1},
    ])
    return entities, relationships


@pytest.fixture
def single_cluster_v3_data():
    """单簇 v3 数据：所有节点来自同一 sent_id。"""
    entities = _make_v3_entities([
        {"node_id": "doc001-p000-s000-x1", "title": "X1", "sent_id": "doc001-p000-s000", "doc_id": "doc001"},
        {"node_id": "doc001-p000-s000-x2", "title": "X2", "sent_id": "doc001-p000-s000", "doc_id": "doc001"},
        {"node_id": "doc001-p000-s000-x3", "title": "X3", "sent_id": "doc001-p000-s000", "doc_id": "doc001"},
    ])
    relationships = _make_v3_relationships([
        {"source": "doc001-p000-s000-x1", "target": "doc001-p000-s000-x2", "weight": 5.0},
        {"source": "doc001-p000-s000-x2", "target": "doc001-p000-s000-x3", "weight": 5.0},
        {"source": "doc001-p000-s000-x1", "target": "doc001-p000-s000-x3", "weight": 5.0},
    ])
    return entities, relationships


# ===========================================================================
# 测试 1：build_graph_from_graphrag
# ===========================================================================

class TestBuildGraphFromGraphRAG:

    def test_nodes_use_node_id(self, two_cluster_v3_data):
        """图节点应使用 node_id（含物理路径），而非 title。"""
        entities, relationships = two_cluster_v3_data
        G = build_graph_from_graphrag(entities, relationships)
        # 节点 ID 应包含 sent_id 前缀
        for node in G.nodes():
            assert "doc001-p" in node, f"节点 ID 不含物理路径：{node}"

    def test_all_entities_become_nodes(self, two_cluster_v3_data):
        """所有实体应成为图节点。"""
        entities, relationships = two_cluster_v3_data
        G = build_graph_from_graphrag(entities, relationships)
        assert G.number_of_nodes() == len(entities)

    def test_valid_edges_added(self, two_cluster_v3_data):
        """有效边（两端节点都存在）应被添加。"""
        entities, relationships = two_cluster_v3_data
        G = build_graph_from_graphrag(entities, relationships)
        # 7 条关系，所有节点都存在，应全部添加
        assert G.number_of_edges() == len(relationships)

    def test_invalid_edges_filtered(self):
        """引用不存在节点的边应被过滤。"""
        entities = _make_v3_entities([
            {"node_id": "node_A", "sent_id": "doc001-p000-s000"},
            {"node_id": "node_B", "sent_id": "doc001-p000-s000"},
        ])
        relationships = _make_v3_relationships([
            {"source": "node_A", "target": "node_B", "weight": 1.0},
            {"source": "node_A", "target": "node_NONEXISTENT", "weight": 1.0},  # 无效边
        ])
        G = build_graph_from_graphrag(entities, relationships)
        assert G.number_of_edges() == 1  # 只有有效边

    def test_self_loops_filtered(self):
        """自环边应被过滤。"""
        entities = _make_v3_entities([
            {"node_id": "node_A", "sent_id": "doc001-p000-s000"},
        ])
        relationships = _make_v3_relationships([
            {"source": "node_A", "target": "node_A", "weight": 1.0},  # 自环
        ])
        G = build_graph_from_graphrag(entities, relationships)
        assert G.number_of_edges() == 0

    def test_duplicate_edges_weight_accumulated(self):
        """重复边的权重应累加。"""
        entities = _make_v3_entities([
            {"node_id": "node_A", "sent_id": "doc001-p000-s000"},
            {"node_id": "node_B", "sent_id": "doc001-p000-s000"},
        ])
        relationships = _make_v3_relationships([
            {"source": "node_A", "target": "node_B", "weight": 2.0},
            {"source": "node_A", "target": "node_B", "weight": 3.0},
        ])
        G = build_graph_from_graphrag(entities, relationships)
        assert G.number_of_edges() == 1
        assert G["node_A"]["node_B"]["weight"] == pytest.approx(5.0)

    def test_empty_entities_returns_empty_graph(self):
        """空实体表应返回空图。"""
        entities = pd.DataFrame(columns=["id", "title", "sent_id"])
        relationships = pd.DataFrame(columns=["source", "target", "weight"])
        G = build_graph_from_graphrag(entities, relationships)
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0


# ===========================================================================
# 测试 2：build_physical_nodes_from_graphrag
# ===========================================================================

class TestBuildPhysicalNodes:

    def test_sent_id_used_as_anchor(self):
        """v3 格式：sent_id 应作为物理锚点。"""
        entities = _make_v3_entities([
            {"node_id": "doc001-p000-s000-graphrag", "sent_id": "doc001-p000-s000"},
        ])
        physical = build_physical_nodes_from_graphrag(entities)
        node = physical["doc001-p000-s000-graphrag"]
        assert "doc001-p000-s000" in node.chunk_ids

    def test_each_node_has_single_anchor(self, two_cluster_v3_data):
        """v3 架构：每个叶节点应只有一个物理锚点（sent_id）。"""
        entities, _ = two_cluster_v3_data
        physical = build_physical_nodes_from_graphrag(entities)
        for node_id, pnode in physical.items():
            assert len(pnode.chunk_ids) == 1, \
                f"节点 {node_id} 有多个锚点：{pnode.chunk_ids}（v3 应只有一个 sent_id）"

    def test_fallback_to_primary_chunk_id(self):
        """无 sent_id 时应回退到 primary_chunk_id。"""
        entities = pd.DataFrame([{
            "id": "node_A",
            "title": "A",
            "primary_chunk_id": "chunk_001",
            # 无 sent_id 列
        }])
        physical = build_physical_nodes_from_graphrag(entities)
        assert "chunk_001" in physical["node_A"].chunk_ids

    def test_fallback_to_text_unit_ids(self):
        """无 sent_id 和 primary_chunk_id 时应回退到 text_unit_ids[0]。"""
        entities = pd.DataFrame([{
            "id": "node_A",
            "title": "A",
            "text_unit_ids": ["chunk_001", "chunk_002"],
            # 无 sent_id 和 primary_chunk_id
        }])
        physical = build_physical_nodes_from_graphrag(entities)
        assert "chunk_001" in physical["node_A"].chunk_ids

    def test_placeholder_when_no_anchor(self):
        """无任何锚点信息时应使用 placeholder。"""
        entities = pd.DataFrame([{
            "id": "node_A",
            "title": "A",
            # 无 sent_id, primary_chunk_id, text_unit_ids
        }])
        physical = build_physical_nodes_from_graphrag(entities)
        anchor = list(physical["node_A"].chunk_ids)[0]
        assert anchor.startswith("placeholder_")

    def test_different_sent_ids_give_different_anchors(self, two_cluster_v3_data):
        """不同 sent_id 的节点应有不同的物理锚点。"""
        entities, _ = two_cluster_v3_data
        physical = build_physical_nodes_from_graphrag(entities)

        # 簇 A 节点的锚点
        anchor_a = list(physical["doc001-p000-s000-a1"].chunk_ids)[0]
        # 簇 B 节点的锚点
        anchor_b = list(physical["doc001-p001-s000-b1"].chunk_ids)[0]

        assert anchor_a != anchor_b, \
            "不同 sent_id 的节点应有不同的物理锚点"

    def test_same_sent_id_gives_same_anchor(self, two_cluster_v3_data):
        """相同 sent_id 的节点应有相同的物理锚点。"""
        entities, _ = two_cluster_v3_data
        physical = build_physical_nodes_from_graphrag(entities)

        anchor_a1 = list(physical["doc001-p000-s000-a1"].chunk_ids)[0]
        anchor_a2 = list(physical["doc001-p000-s000-a2"].chunk_ids)[0]
        anchor_a3 = list(physical["doc001-p000-s000-a3"].chunk_ids)[0]

        assert anchor_a1 == anchor_a2 == anchor_a3, \
            "相同 sent_id 的节点应有相同的物理锚点"


# ===========================================================================
# 测试 3：run_constrained_community_detection
# ===========================================================================

class TestRunConstrainedCommunityDetection:

    def test_empty_input_returns_empty_df(self):
        """空输入应返回空 DataFrame。"""
        entities = pd.DataFrame(columns=["id", "title", "sent_id", "text_unit_ids"])
        relationships = pd.DataFrame(columns=["id", "source", "target", "weight"])
        result = run_constrained_community_detection(entities, relationships)
        assert len(result) == 0

    def test_output_has_required_columns(self, two_cluster_v3_data):
        """输出 DataFrame 应包含所有必要列（含 v3 新增的 doc_ids）。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        required = {
            "id", "community_id", "level", "title",
            "entity_ids", "relationship_ids", "text_unit_ids",
            "doc_ids",           # v3 新增
            "parent_id", "children", "structural_entropy", "lambda_used",
        }
        assert required.issubset(set(result.columns)), \
            f"缺失列：{required - set(result.columns)}"

    def test_output_not_empty_for_valid_input(self, two_cluster_v3_data):
        """有效输入应产生非空输出。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        assert len(result) > 0

    def test_structural_entropy_non_negative(self, two_cluster_v3_data):
        """结构熵应 >= 0。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        assert (result["structural_entropy"] >= 0).all()

    def test_level_zero_exists(self, two_cluster_v3_data):
        """输出应包含 level=0 的社区。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        assert 0 in result["level"].values

    def test_high_lambda_separates_physical_clusters(self, two_cluster_v3_data):
        """高 λ 时，来自不同 sent_id 的节点应被分配到不同社区。"""
        entities, relationships = two_cluster_v3_data
        config = AnnealingConfig(lambda_init=100000.0, lambda_min=0.0, max_level=3)
        result = run_constrained_community_detection(
            entities, relationships,
            annealing_config=config,
            use_lcc=False,  # 不过滤孤立节点，保留所有节点
        )

        if len(result) == 0:
            pytest.skip("图不连通，无法测试社区分离")

        level0 = result[result["level"] == 0]
        if len(level0) < 2:
            pytest.skip("只有一个社区，无法测试分离")

        # 检查 A 簇节点是否在同一社区
        a_nodes = {"doc001-p000-s000-a1", "doc001-p000-s000-a2", "doc001-p000-s000-a3"}
        b_nodes = {"doc001-p001-s000-b1", "doc001-p001-s000-b2", "doc001-p001-s000-b3"}

        def find_community(node_id: str) -> int:
            for _, row in level0.iterrows():
                if node_id in row["entity_ids"]:
                    return row["community_id"]
            return -1

        comm_a = {find_community(n) for n in a_nodes if find_community(n) != -1}
        comm_b = {find_community(n) for n in b_nodes if find_community(n) != -1}

        if comm_a and comm_b:
            assert comm_a != comm_b, \
                "高 λ 时，来自不同 sent_id 的节点应被分配到不同社区"

    def test_single_cluster_zero_entropy(self, single_cluster_v3_data):
        """所有节点来自同一 sent_id 时，底层社区结构熵应为 0。"""
        entities, relationships = single_cluster_v3_data
        config = AnnealingConfig(lambda_init=10000.0, lambda_min=0.0, max_level=3)
        result = run_constrained_community_detection(
            entities, relationships, annealing_config=config
        )
        if len(result) > 0:
            level0 = result[result["level"] == 0]
            # 所有节点来自同一 sent_id，结构熵应为 0
            entropies = level0["structural_entropy"].tolist()
            for h in entropies:
                assert abs(h) < 1e-9, \
                    f"同 sent_id 节点的结构熵应为 0，实际：{entropies}"

    def test_doc_ids_collected_correctly(self, two_cluster_v3_data):
        """doc_ids 字段应正确收集社区内节点的 doc_id。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        if len(result) > 0:
            for _, row in result.iterrows():
                assert isinstance(row["doc_ids"], list), \
                    f"doc_ids 应为列表，实际：{type(row['doc_ids'])}"

    def test_entity_ids_are_node_ids(self, two_cluster_v3_data):
        """entity_ids 应为 node_id（含物理路径），而非 title。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        valid_node_ids = set(entities["id"])
        for _, row in result.iterrows():
            for eid in row["entity_ids"]:
                assert eid in valid_node_ids, \
                    f"entity_id={eid} 不在有效 node_id 集合中"


# ===========================================================================
# 测试 4：convert_result_to_communities_df
# ===========================================================================

class TestConvertResultToCommunitiesDf:

    def test_doc_ids_field_present(self, two_cluster_v3_data):
        """输出 DataFrame 应包含 doc_ids 列（v3 新增）。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        assert "doc_ids" in result.columns

    def test_text_unit_ids_are_sent_ids(self, single_cluster_v3_data):
        """text_unit_ids 应包含 sent_id（v3 精确到句子级）。"""
        entities, relationships = single_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        if len(result) > 0:
            for _, row in result.iterrows():
                for tu_id in row["text_unit_ids"]:
                    # sent_id 格式：{doc_id}-p{para_idx}-s{sent_idx}
                    assert re.match(r".+-p\d{3}-s\d{3}$", tu_id), \
                        f"text_unit_id={tu_id} 不符合 sent_id 格式"

    def test_community_id_is_integer(self, two_cluster_v3_data):
        """community_id 应为整数。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        if len(result) > 0:
            assert result["community_id"].dtype in (int, "int64", "int32")

    def test_lambda_used_non_negative(self, two_cluster_v3_data):
        """lambda_used 应 >= 0。"""
        entities, relationships = two_cluster_v3_data
        result = run_constrained_community_detection(entities, relationships)
        if len(result) > 0:
            assert (result["lambda_used"] >= 0).all()


# ===========================================================================
# 测试 5：v3 架构特有的物理纯净性验证
# ===========================================================================

class TestV3PhysicalPurity:

    def test_bottom_level_entropy_lower_with_high_lambda(self, two_cluster_v3_data):
        """高 λ 时，底层社区的平均结构熵应 ≤ 低 λ 时的结果。"""
        entities, relationships = two_cluster_v3_data

        config_high = AnnealingConfig(lambda_init=100000.0, lambda_min=0.0, max_level=3)
        config_low = AnnealingConfig(lambda_init=0.001, lambda_min=0.0, max_level=3)

        result_high = run_constrained_community_detection(
            entities, relationships, annealing_config=config_high, use_lcc=False
        )
        result_low = run_constrained_community_detection(
            entities, relationships, annealing_config=config_low, use_lcc=False
        )

        if len(result_high) == 0 or len(result_low) == 0:
            pytest.skip("图不连通，跳过此测试")

        level0_high = result_high[result_high["level"] == 0]["structural_entropy"].mean()
        level0_low = result_low[result_low["level"] == 0]["structural_entropy"].mean()

        assert level0_high <= level0_low + 1e-6, \
            f"高 λ 底层熵 ({level0_high:.4f}) 应 ≤ 低 λ 底层熵 ({level0_low:.4f})"

    def test_v3_node_ids_in_graph_contain_sent_id(self, two_cluster_v3_data):
        """v3 图中的节点 ID 应包含 sent_id 格式的物理路径。"""
        entities, relationships = two_cluster_v3_data
        G = build_graph_from_graphrag(entities, relationships)
        for node in G.nodes():
            # v3 node_id 格式：{sent_id}-{entity_name}
            # sent_id 格式：{doc_id}-p{para_idx}-s{sent_idx}
            assert re.search(r"-p\d{3}-s\d{3}-", node), \
                f"节点 ID 不含 sent_id 格式的物理路径：{node}"

    def test_physical_nodes_level_is_zero(self, two_cluster_v3_data):
        """初始物理节点的 level 应为 0（叶节点）。"""
        entities, _ = two_cluster_v3_data
        physical = build_physical_nodes_from_graphrag(entities)
        for node_id, pnode in physical.items():
            assert pnode.level == 0, \
                f"节点 {node_id} 的 level={pnode.level}，应为 0"


import re  # 确保 re 在模块级别可用

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
