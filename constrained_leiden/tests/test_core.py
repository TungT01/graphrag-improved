"""
test_core.py
------------
核心逻辑单元测试。

测试覆盖：
1. 结构熵计算的正确性（边界情况 + 一般情况）
2. λ 退火机制的单调性和边界值
3. 物理约束的有效性（底层不应出现跨 chunk 合并）
4. 层次化输出的完整性
5. GraphRAG 接口的格式兼容性
"""

from __future__ import annotations

import math
import sys
import os

# 将父目录加入路径，使测试可以直接运行
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import networkx as nx
import pandas as pd
import pytest

from graphrag_improved.constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule, get_lambda
from graphrag_improved.constrained_leiden.graphrag_workflow import (
    build_graph_from_graphrag,
    build_physical_nodes_from_graphrag,
    run_constrained_community_detection,
)
from graphrag_improved.constrained_leiden.leiden_constrained import hierarchical_leiden_constrained
from graphrag_improved.constrained_leiden.physical_anchor import (
    PhysicalNode,
    compute_delta_entropy,
    compute_structural_entropy,
)


# ===========================================================================
# 测试 1：结构熵计算
# ===========================================================================

class TestStructuralEntropy:

    def test_single_source_entropy_is_zero(self):
        """所有节点来自同一 chunk，熵应为 0。"""
        nodes = [
            PhysicalNode("A", frozenset(["chunk_1"])),
            PhysicalNode("B", frozenset(["chunk_1"])),
            PhysicalNode("C", frozenset(["chunk_1"])),
        ]
        entropy = compute_structural_entropy(nodes)
        assert entropy == pytest.approx(0.0, abs=1e-9)

    def test_uniform_distribution_max_entropy(self):
        """节点均匀分布于 k 个 chunk，熵应为 log(k)。"""
        k = 4
        nodes = [
            PhysicalNode(f"node_{i}", frozenset([f"chunk_{i}"]))
            for i in range(k)
        ]
        entropy = compute_structural_entropy(nodes)
        expected = math.log(k)
        assert entropy == pytest.approx(expected, rel=1e-6)

    def test_empty_community_entropy_is_zero(self):
        """空社区的熵应为 0。"""
        assert compute_structural_entropy([]) == 0.0

    def test_single_node_entropy_is_zero(self):
        """单节点社区的熵应为 0。"""
        nodes = [PhysicalNode("A", frozenset(["chunk_1"]))]
        assert compute_structural_entropy(nodes) == pytest.approx(0.0, abs=1e-9)

    def test_two_sources_equal_split(self):
        """两个 chunk 各贡献一半节点，熵应为 log(2)。"""
        nodes = [
            PhysicalNode("A", frozenset(["chunk_1"])),
            PhysicalNode("B", frozenset(["chunk_2"])),
        ]
        entropy = compute_structural_entropy(nodes)
        assert entropy == pytest.approx(math.log(2), rel=1e-6)

    def test_super_node_with_multiple_chunks(self):
        """超节点携带多个 chunk_id，权重应均匀分配。"""
        nodes = [
            PhysicalNode("A", frozenset(["chunk_1", "chunk_2"])),
            PhysicalNode("B", frozenset(["chunk_1"])),
        ]
        entropy = compute_structural_entropy(nodes)
        # p(chunk_1) = 1.5/2.0 = 0.75, p(chunk_2) = 0.5/2.0 = 0.25
        expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        assert entropy == pytest.approx(expected, rel=1e-6)

    def test_delta_entropy_same_source_is_zero(self):
        """向同源社区加入同源节点，ΔH 应为 0。"""
        community = [
            PhysicalNode("A", frozenset(["chunk_1"])),
            PhysicalNode("B", frozenset(["chunk_1"])),
        ]
        new_node = PhysicalNode("C", frozenset(["chunk_1"]))
        delta = compute_delta_entropy(community, new_node)
        assert delta == pytest.approx(0.0, abs=1e-9)

    def test_delta_entropy_cross_source_is_positive(self):
        """向同源社区加入异源节点，ΔH 应为正值。"""
        community = [
            PhysicalNode("A", frozenset(["chunk_1"])),
            PhysicalNode("B", frozenset(["chunk_1"])),
        ]
        new_node = PhysicalNode("C", frozenset(["chunk_2"]))
        delta = compute_delta_entropy(community, new_node)
        assert delta > 0


# ===========================================================================
# 测试 2：λ 退火机制
# ===========================================================================

class TestAnnealing:

    def test_level_zero_returns_lambda_init(self):
        """level=0 时应返回 lambda_init。"""
        config = AnnealingConfig(lambda_init=1000.0, lambda_min=0.0)
        assert get_lambda(0, config) == pytest.approx(1000.0)

    def test_exponential_monotone_decreasing(self):
        """指数退火应单调递减。"""
        config = AnnealingConfig(
            lambda_init=1000.0, lambda_min=0.0,
            max_level=10, decay_rate=0.5,
            schedule=AnnealingSchedule.EXPONENTIAL,
        )
        lambdas = [get_lambda(l, config) for l in range(11)]
        for i in range(len(lambdas) - 1):
            assert lambdas[i] >= lambdas[i + 1], \
                f"退火不单调：level {i} λ={lambdas[i]}, level {i+1} λ={lambdas[i+1]}"

    def test_linear_monotone_decreasing(self):
        """线性退火应单调递减。"""
        config = AnnealingConfig(
            lambda_init=100.0, lambda_min=0.0,
            max_level=5, schedule=AnnealingSchedule.LINEAR,
        )
        lambdas = [get_lambda(l, config) for l in range(6)]
        for i in range(len(lambdas) - 1):
            assert lambdas[i] >= lambdas[i + 1]

    def test_cosine_monotone_decreasing(self):
        """余弦退火应单调递减。"""
        config = AnnealingConfig(
            lambda_init=100.0, lambda_min=0.0,
            max_level=10, schedule=AnnealingSchedule.COSINE,
        )
        lambdas = [get_lambda(l, config) for l in range(11)]
        for i in range(len(lambdas) - 1):
            assert lambdas[i] >= lambdas[i + 1]

    def test_lambda_never_below_min(self):
        """λ 值不应低于 lambda_min。"""
        config = AnnealingConfig(lambda_init=100.0, lambda_min=5.0, max_level=10)
        for level in range(20):
            assert get_lambda(level, config) >= 5.0

    def test_invalid_config_raises(self):
        """无效配置应抛出异常。"""
        with pytest.raises(ValueError):
            AnnealingConfig(lambda_init=-1.0)
        with pytest.raises(ValueError):
            AnnealingConfig(lambda_init=10.0, lambda_min=20.0)


# ===========================================================================
# 测试 3：物理约束有效性
# ===========================================================================

class TestPhysicalConstraint:

    def _build_two_cluster_graph(self):
        """
        构建一个两簇图：
        - 簇 A：节点 a1, a2, a3，来自 chunk_A，内部强连接
        - 簇 B：节点 b1, b2, b3，来自 chunk_B，内部强连接
        - 跨簇边：a1-b1，权重极小
        """
        G = nx.Graph()
        for u, v in [("a1", "a2"), ("a2", "a3"), ("a1", "a3")]:
            G.add_edge(u, v, weight=10.0)
        for u, v in [("b1", "b2"), ("b2", "b3"), ("b1", "b3")]:
            G.add_edge(u, v, weight=10.0)
        G.add_edge("a1", "b1", weight=0.1)

        physical = {
            "a1": PhysicalNode("a1", frozenset(["chunk_A"])),
            "a2": PhysicalNode("a2", frozenset(["chunk_A"])),
            "a3": PhysicalNode("a3", frozenset(["chunk_A"])),
            "b1": PhysicalNode("b1", frozenset(["chunk_B"])),
            "b2": PhysicalNode("b2", frozenset(["chunk_B"])),
            "b3": PhysicalNode("b3", frozenset(["chunk_B"])),
        }
        return G, physical

    def test_high_lambda_separates_physical_clusters(self):
        """高 λ 时，来自不同 chunk 的节点应被分配到不同社区。"""
        G, physical = self._build_two_cluster_graph()
        config = AnnealingConfig(lambda_init=10000.0, lambda_min=0.0, max_level=5)
        result = hierarchical_leiden_constrained(G, physical, config)

        level0 = result.levels[0]
        comm_a = {level0[n] for n in ["a1", "a2", "a3"]}
        comm_b = {level0[n] for n in ["b1", "b2", "b3"]}

        assert len(comm_a) == 1, f"A 簇节点被分到了不同社区：{comm_a}"
        assert len(comm_b) == 1, f"B 簇节点被分到了不同社区：{comm_b}"
        assert comm_a != comm_b, "A 簇和 B 簇被错误地合并到同一社区"

    def test_zero_lambda_does_not_crash(self):
        """λ=0 时退化为标准 Leiden，不应崩溃。"""
        G, physical = self._build_two_cluster_graph()
        config = AnnealingConfig(lambda_init=0.0, lambda_min=0.0, max_level=5)
        result = hierarchical_leiden_constrained(G, physical, config)
        assert len(result.levels) >= 1

    def test_bottom_level_entropy_lower_with_high_lambda(self):
        """高 λ 时，底层社区的平均结构熵应 ≤ λ=0 时的结果。"""
        G, physical = self._build_two_cluster_graph()
        G.add_edge("a2", "b2", weight=0.1)
        G.add_edge("a3", "b3", weight=0.1)

        config_high = AnnealingConfig(lambda_init=10000.0, lambda_min=0.0, max_level=5)
        result_high = hierarchical_leiden_constrained(G, physical, config_high)

        config_zero = AnnealingConfig(lambda_init=0.0, lambda_min=0.0, max_level=5)
        result_zero = hierarchical_leiden_constrained(G, physical, config_zero)

        def avg_entropy(result):
            entropies = list(result.level_entropy[0].values())
            return sum(entropies) / len(entropies) if entropies else 0.0

        assert avg_entropy(result_high) <= avg_entropy(result_zero) + 1e-6


# ===========================================================================
# 测试 4：层次化输出完整性
# ===========================================================================

class TestHierarchicalOutput:

    def test_all_nodes_covered_in_each_level(self):
        """每层的社区分配应覆盖所有原始节点。"""
        G = nx.karate_club_graph()
        G_str = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
        physical = {
            str(n): PhysicalNode(str(n), frozenset([f"chunk_{n % 5}"]))
            for n in G.nodes()
        }
        result = hierarchical_leiden_constrained(G_str, physical)

        all_nodes = set(G_str.nodes())
        for level_idx, level_map in enumerate(result.levels):
            covered = set(level_map.keys())
            assert covered == all_nodes, \
                f"Level {level_idx} 未覆盖所有节点，缺失：{all_nodes - covered}"

    def test_level_count_matches_lambda_count(self):
        """层次数、λ 列表、熵列表长度应一致。"""
        G = nx.path_graph(10)
        G_str = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
        physical = {
            str(n): PhysicalNode(str(n), frozenset([f"chunk_{n % 3}"]))
            for n in G.nodes()
        }
        result = hierarchical_leiden_constrained(G_str, physical)
        assert len(result.levels) == len(result.level_lambda) == len(result.level_entropy)

    def test_result_has_at_least_one_level(self):
        """结果至少应有一层。"""
        G = nx.complete_graph(5)
        G_str = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
        physical = {
            str(n): PhysicalNode(str(n), frozenset([f"chunk_{n}"]))
            for n in G.nodes()
        }
        result = hierarchical_leiden_constrained(G_str, physical)
        assert len(result.levels) >= 1


# ===========================================================================
# 测试 5：GraphRAG 接口兼容性
# ===========================================================================

class TestGraphRAGInterface:

    def _make_sample_dataframes(self):
        """构造最小化的 GraphRAG 格式 DataFrame。"""
        entities = pd.DataFrame([
            {"id": "e1", "title": "Drug X", "text_unit_ids": ["chunk_1", "chunk_2"]},
            {"id": "e2", "title": "Receptor Y", "text_unit_ids": ["chunk_1"]},
            {"id": "e3", "title": "Compound Z", "text_unit_ids": ["chunk_3"]},
            {"id": "e4", "title": "Pathway W", "text_unit_ids": ["chunk_3"]},
        ])
        relationships = pd.DataFrame([
            {"id": "r1", "source": "Drug X", "target": "Receptor Y", "weight": 5.0},
            {"id": "r2", "source": "Compound Z", "target": "Pathway W", "weight": 5.0},
            {"id": "r3", "source": "Drug X", "target": "Compound Z", "weight": 0.5},
        ])
        return entities, relationships

    def test_output_has_required_columns(self):
        """输出 DataFrame 应包含所有必要列。"""
        entities, relationships = self._make_sample_dataframes()
        result = run_constrained_community_detection(entities, relationships)
        required_columns = {
            "id", "community_id", "level", "title",
            "entity_ids", "relationship_ids", "text_unit_ids",
            "parent_id", "children", "structural_entropy", "lambda_used",
        }
        assert required_columns.issubset(set(result.columns)), \
            f"缺失列：{required_columns - set(result.columns)}"

    def test_output_not_empty(self):
        """有效输入应产生非空输出。"""
        entities, relationships = self._make_sample_dataframes()
        result = run_constrained_community_detection(entities, relationships)
        assert len(result) > 0

    def test_empty_input_returns_empty_df(self):
        """空输入应返回空 DataFrame 而非报错。"""
        entities = pd.DataFrame(columns=["id", "title", "text_unit_ids"])
        relationships = pd.DataFrame(columns=["id", "source", "target", "weight"])
        result = run_constrained_community_detection(entities, relationships)
        assert len(result) == 0

    def test_structural_entropy_in_valid_range(self):
        """结构熵值应在合法范围内（>= 0）。"""
        entities, relationships = self._make_sample_dataframes()
        result = run_constrained_community_detection(entities, relationships)
        assert (result["structural_entropy"] >= 0).all()

    def test_physical_constraint_reduces_entropy(self):
        """高 λ 配置下，底层社区的平均结构熵应 ≤ 低 λ 配置。"""
        entities, relationships = self._make_sample_dataframes()

        config_high = AnnealingConfig(lambda_init=10000.0, lambda_min=0.0)
        config_low = AnnealingConfig(lambda_init=0.001, lambda_min=0.0)

        result_high = run_constrained_community_detection(
            entities, relationships, annealing_config=config_high
        )
        result_low = run_constrained_community_detection(
            entities, relationships, annealing_config=config_low
        )

        level0_high = result_high[result_high["level"] == 0]["structural_entropy"].mean()
        level0_low = result_low[result_low["level"] == 0]["structural_entropy"].mean()

        assert level0_high <= level0_low + 1e-6, \
            f"高 λ 底层熵 ({level0_high:.4f}) 应 ≤ 低 λ 底层熵 ({level0_low:.4f})"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
