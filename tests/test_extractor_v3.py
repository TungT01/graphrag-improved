"""
tests/test_extractor_v3.py
--------------------------
v3 物理优先架构 — 实体关系抽取模块单元测试。

测试覆盖：
1. Entity.node_id 包含完整物理路径（sent_id 前缀）
2. 同名实体在不同句子里是不同节点（无实体消解）
3. 代词跳过（方案一）
4. 物理结构边严格限制在句子内部
5. entities_to_dataframe / relations_to_dataframe 输出格式
6. extract() 主入口：空输入、无 SentenceUnit 时的错误处理
7. 关系合并（重复边权重累加）
8. spaCy 三元组提取（需要 spaCy 安装）
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pytest

# 路径设置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graphrag_improved.data.ingestion import (
    Document,
    SentenceUnit,
    TextUnit,
    _make_doc_id,
    _make_para_id,
    _make_sent_id,
    document_to_text_units,
)
from graphrag_improved.extraction.extractor import (
    Entity,
    Relation,
    _extract_physical_structure_edges,
    _is_valid_entity_span,
    _merge_relations,
    entities_to_dataframe,
    extract,
    relations_to_dataframe,
)
from graphrag_improved.pipeline_config import ExtractionConfig


# ===========================================================================
# Fixtures
# ===========================================================================

def _make_sentence_unit(
    text: str,
    doc_id: str = "doc001",
    para_idx: int = 0,
    sent_idx: int = 0,
) -> SentenceUnit:
    """构造测试用 SentenceUnit。"""
    para_id = _make_para_id(doc_id, para_idx)
    sent_id = _make_sent_id(para_id, sent_idx)
    return SentenceUnit(
        sent_id=sent_id,
        text=text,
        doc_id=doc_id,
        para_id=para_id,
        sent_index=sent_idx,
        para_index=para_idx,
    )


def _make_text_unit_with_sentences(
    sentences: List[str],
    doc_id: str = "doc001",
    para_idx: int = 0,
) -> TextUnit:
    """构造包含指定句子的 TextUnit。"""
    para_id = _make_para_id(doc_id, para_idx)
    sent_units = [
        _make_sentence_unit(text, doc_id, para_idx, i)
        for i, text in enumerate(sentences)
    ]
    return TextUnit(
        chunk_id=para_id,
        text=" ".join(sentences),
        doc_id=doc_id,
        doc_title="Test Doc",
        chunk_index=para_idx,
        sentences=sent_units,
    )


@pytest.fixture
def single_sentence_unit():
    """单句 SentenceUnit。"""
    return _make_sentence_unit(
        "GraphRAG uses the Leiden algorithm for community detection.",
        doc_id="doc001",
        para_idx=0,
        sent_idx=0,
    )


@pytest.fixture
def two_sentence_units_same_para():
    """同一段落内的两个 SentenceUnit。"""
    return [
        _make_sentence_unit("GraphRAG uses Leiden.", doc_id="doc001", para_idx=0, sent_idx=0),
        _make_sentence_unit("Leiden improves Louvain.", doc_id="doc001", para_idx=0, sent_idx=1),
    ]


@pytest.fixture
def two_sentence_units_diff_para():
    """不同段落的两个 SentenceUnit。"""
    return [
        _make_sentence_unit("GraphRAG uses Leiden.", doc_id="doc001", para_idx=0, sent_idx=0),
        _make_sentence_unit("Leiden improves Louvain.", doc_id="doc001", para_idx=1, sent_idx=0),
    ]


# ===========================================================================
# 测试 1：Entity.node_id 格式
# ===========================================================================

class TestEntityNodeID:

    def test_node_id_contains_sent_id(self, single_sentence_unit):
        """node_id 应以 sent_id 为前缀。"""
        ent = Entity(
            title="GraphRAG",
            entity_type="PROPER_NOUN",
            sent_id=single_sentence_unit.sent_id,
            para_id=single_sentence_unit.para_id,
            doc_id=single_sentence_unit.doc_id,
        )
        assert ent.node_id.startswith(single_sentence_unit.sent_id), \
            f"node_id={ent.node_id} 应以 sent_id={single_sentence_unit.sent_id} 开头"

    def test_node_id_format(self, single_sentence_unit):
        """node_id 格式应为 {sent_id}-{title_normalized}。"""
        ent = Entity(
            title="GraphRAG",
            entity_type="PROPER_NOUN",
            sent_id=single_sentence_unit.sent_id,
            para_id=single_sentence_unit.para_id,
            doc_id=single_sentence_unit.doc_id,
        )
        # title 归一化：小写 + 非字母数字替换为下划线
        expected_suffix = "graphrag"
        assert ent.node_id.endswith(expected_suffix), \
            f"node_id={ent.node_id} 应以 '{expected_suffix}' 结尾"

    def test_node_id_normalizes_special_chars(self, single_sentence_unit):
        """title 中的特殊字符应被归一化为下划线。"""
        ent = Entity(
            title="Graph-RAG System",
            entity_type="PROPER_NOUN",
            sent_id=single_sentence_unit.sent_id,
            para_id=single_sentence_unit.para_id,
            doc_id=single_sentence_unit.doc_id,
        )
        # node_id 格式：{sent_id}-{title_normalized}
        # "Graph-RAG System" → "graph_rag_system"
        # 提取 title 归一化部分（去掉 sent_id 前缀和分隔符 '-'）
        title_part = ent.node_id[len(single_sentence_unit.sent_id) + 1:]  # +1 跳过分隔符 '-'
        assert "-" not in title_part, f"title 归一化部分不应含 '-'：{title_part}"
        assert " " not in ent.node_id

    def test_same_title_different_sent_id_gives_different_node_id(self):
        """同名实体在不同句子里应有不同的 node_id（无实体消解）。"""
        sent_id_1 = "doc001-p000-s000"
        sent_id_2 = "doc001-p000-s001"

        ent1 = Entity(
            title="Leiden",
            entity_type="PROPER_NOUN",
            sent_id=sent_id_1,
            para_id="doc001-p000",
            doc_id="doc001",
        )
        ent2 = Entity(
            title="Leiden",
            entity_type="PROPER_NOUN",
            sent_id=sent_id_2,
            para_id="doc001-p000",
            doc_id="doc001",
        )
        assert ent1.node_id != ent2.node_id, \
            "同名实体在不同句子里应有不同 node_id（v3 无实体消解）"

    def test_entity_id_alias(self, single_sentence_unit):
        """entity_id 应是 node_id 的别名。"""
        ent = Entity(
            title="Leiden",
            entity_type="PROPER_NOUN",
            sent_id=single_sentence_unit.sent_id,
            para_id=single_sentence_unit.para_id,
            doc_id=single_sentence_unit.doc_id,
        )
        assert ent.entity_id == ent.node_id


# ===========================================================================
# 测试 2：代词过滤（方案一）
# ===========================================================================

class TestPronounFiltering:

    def test_pronoun_tag_filtered(self):
        """代词词性标签应被过滤。"""
        assert not _is_valid_entity_span("it", "PRP")
        assert not _is_valid_entity_span("they", "PRP")
        assert not _is_valid_entity_span("he", "PRP")
        assert not _is_valid_entity_span("she", "PRP")
        assert not _is_valid_entity_span("its", "PRP$")
        assert not _is_valid_entity_span("their", "PRP$")

    def test_wh_pronoun_filtered(self):
        """疑问代词也应被过滤。"""
        assert not _is_valid_entity_span("who", "WP")
        assert not _is_valid_entity_span("whose", "WP$")

    def test_stopwords_filtered(self):
        """停用词应被过滤。"""
        assert not _is_valid_entity_span("the", "DT")
        assert not _is_valid_entity_span("a", "DT")
        assert not _is_valid_entity_span("in", "IN")

    def test_short_text_filtered(self):
        """过短文本（< 2 字符）应被过滤。"""
        assert not _is_valid_entity_span("a", "NN")
        assert not _is_valid_entity_span("", "NN")

    def test_pure_digits_filtered(self):
        """纯数字应被过滤。"""
        assert not _is_valid_entity_span("123", "CD")
        assert not _is_valid_entity_span("42", "CD")

    def test_valid_entity_passes(self):
        """有效实体名称应通过过滤。"""
        assert _is_valid_entity_span("GraphRAG", "NNP")
        assert _is_valid_entity_span("Leiden algorithm", "NNP")
        assert _is_valid_entity_span("community detection", "NN")


# ===========================================================================
# 测试 3：物理结构边严格限制在句子内部
# ===========================================================================

class TestPhysicalStructureEdges:

    def test_same_sent_entities_get_edges(self):
        """同一句子内的实体应生成物理结构边。"""
        sent_id = "doc001-p000-s000"
        entities = [
            Entity("GraphRAG", "PROPER_NOUN", sent_id, "doc001-p000", "doc001"),
            Entity("Leiden", "PROPER_NOUN", sent_id, "doc001-p000", "doc001"),
            Entity("community", "NOUN", sent_id, "doc001-p000", "doc001"),
        ]
        edges = _extract_physical_structure_edges(entities)
        # 3 个节点应生成 C(3,2) = 3 条边
        assert len(edges) == 3

    def test_cross_sent_entities_no_edges(self):
        """不同句子的实体不应生成物理结构边。"""
        entities = [
            Entity("GraphRAG", "PROPER_NOUN", "doc001-p000-s000", "doc001-p000", "doc001"),
            Entity("Leiden", "PROPER_NOUN", "doc001-p000-s001", "doc001-p000", "doc001"),
        ]
        edges = _extract_physical_structure_edges(entities)
        assert len(edges) == 0, \
            "不同句子的实体不应生成物理结构边（v3 底层图严格限制在句子内部）"

    def test_single_entity_no_edges(self):
        """单个实体不应生成任何边。"""
        entities = [
            Entity("GraphRAG", "PROPER_NOUN", "doc001-p000-s000", "doc001-p000", "doc001"),
        ]
        edges = _extract_physical_structure_edges(entities)
        assert len(edges) == 0

    def test_empty_entities_no_edges(self):
        """空实体列表不应生成任何边。"""
        edges = _extract_physical_structure_edges([])
        assert len(edges) == 0

    def test_physical_edge_predicate_is_co_occurs(self):
        """物理结构边的谓词应为 'co_occurs'。"""
        sent_id = "doc001-p000-s000"
        entities = [
            Entity("GraphRAG", "PROPER_NOUN", sent_id, "doc001-p000", "doc001"),
            Entity("Leiden", "PROPER_NOUN", sent_id, "doc001-p000", "doc001"),
        ]
        edges = _extract_physical_structure_edges(entities)
        assert len(edges) == 1
        assert edges[0].predicate == "co_occurs"

    def test_physical_edge_weight_is_one(self):
        """物理结构边的权重应为 1.0。"""
        sent_id = "doc001-p000-s000"
        entities = [
            Entity("GraphRAG", "PROPER_NOUN", sent_id, "doc001-p000", "doc001"),
            Entity("Leiden", "PROPER_NOUN", sent_id, "doc001-p000", "doc001"),
        ]
        edges = _extract_physical_structure_edges(entities)
        assert edges[0].weight == 1.0

    def test_mixed_sent_entities_only_same_sent_edges(self):
        """混合句子的实体列表中，只有同句实体应生成边。"""
        entities = [
            Entity("A", "NOUN", "doc001-p000-s000", "doc001-p000", "doc001"),
            Entity("B", "NOUN", "doc001-p000-s000", "doc001-p000", "doc001"),
            Entity("C", "NOUN", "doc001-p000-s001", "doc001-p000", "doc001"),  # 不同句子
        ]
        edges = _extract_physical_structure_edges(entities)
        # 只有 A-B 应生成边，A-C 和 B-C 不应生成
        assert len(edges) == 1
        node_ids = {edges[0].source_node_id, edges[0].target_node_id}
        assert edges[0].source_node_id != edges[0].target_node_id


# ===========================================================================
# 测试 4：关系合并
# ===========================================================================

class TestRelationMerging:

    def test_duplicate_relations_merged(self):
        """相同 (source, predicate, target) 的关系应合并，权重累加。"""
        relations = [
            Relation("node_A", "node_B", "use", weight=1.0, sent_id="s1"),
            Relation("node_A", "node_B", "use", weight=1.0, sent_id="s2"),
            Relation("node_A", "node_B", "use", weight=1.0, sent_id="s3"),
        ]
        merged = _merge_relations(relations)
        assert len(merged) == 1
        assert merged[0].weight == pytest.approx(3.0)

    def test_different_predicates_not_merged(self):
        """不同谓词的关系不应合并。"""
        relations = [
            Relation("node_A", "node_B", "use", weight=1.0),
            Relation("node_A", "node_B", "improve", weight=1.0),
        ]
        merged = _merge_relations(relations)
        assert len(merged) == 2

    def test_different_targets_not_merged(self):
        """不同目标节点的关系不应合并。"""
        relations = [
            Relation("node_A", "node_B", "use", weight=1.0),
            Relation("node_A", "node_C", "use", weight=1.0),
        ]
        merged = _merge_relations(relations)
        assert len(merged) == 2

    def test_empty_relations_returns_empty(self):
        """空关系列表应返回空列表。"""
        assert _merge_relations([]) == []


# ===========================================================================
# 测试 5：DataFrame 输出格式
# ===========================================================================

class TestDataFrameOutput:

    def _make_entity(self, title: str, sent_id: str = "doc001-p000-s000") -> Entity:
        return Entity(
            title=title,
            entity_type="PROPER_NOUN",
            sent_id=sent_id,
            para_id="doc001-p000",
            doc_id="doc001",
        )

    def test_entities_df_required_columns(self):
        """entities DataFrame 应包含所有必要列。"""
        entities = {
            "e1": self._make_entity("GraphRAG"),
            "e2": self._make_entity("Leiden"),
        }
        # 使用 node_id 作为 key
        entities_by_node_id = {e.node_id: e for e in entities.values()}
        df = entities_to_dataframe(entities_by_node_id)
        required = {"id", "title", "type", "description", "sent_id", "para_id", "doc_id",
                    "text_unit_ids", "primary_chunk_id"}
        assert required.issubset(set(df.columns)), \
            f"缺失列：{required - set(df.columns)}"

    def test_entities_df_id_is_node_id(self):
        """entities DataFrame 的 id 列应为 node_id（含物理路径）。"""
        ent = self._make_entity("GraphRAG")
        df = entities_to_dataframe({ent.node_id: ent})
        assert df.iloc[0]["id"] == ent.node_id

    def test_entities_df_text_unit_ids_is_list(self):
        """text_unit_ids 列应为列表类型。"""
        ent = self._make_entity("GraphRAG")
        df = entities_to_dataframe({ent.node_id: ent})
        assert isinstance(df.iloc[0]["text_unit_ids"], list)

    def test_entities_df_text_unit_ids_contains_sent_id(self):
        """text_unit_ids 应包含 sent_id（v3 精确到句子级）。"""
        ent = self._make_entity("GraphRAG", sent_id="doc001-p000-s000")
        df = entities_to_dataframe({ent.node_id: ent})
        assert "doc001-p000-s000" in df.iloc[0]["text_unit_ids"]

    def test_entities_df_primary_chunk_id_is_sent_id(self):
        """primary_chunk_id 应等于 sent_id。"""
        ent = self._make_entity("GraphRAG", sent_id="doc001-p000-s000")
        df = entities_to_dataframe({ent.node_id: ent})
        assert df.iloc[0]["primary_chunk_id"] == "doc001-p000-s000"

    def test_empty_entities_returns_empty_df(self):
        """空实体字典应返回空 DataFrame（含正确列名）。"""
        df = entities_to_dataframe({})
        assert len(df) == 0
        assert "id" in df.columns

    def test_relations_df_required_columns(self):
        """relations DataFrame 应包含所有必要列。"""
        relations = [
            Relation("node_A", "node_B", "use", weight=1.0, sent_id="s1"),
        ]
        df = relations_to_dataframe(relations)
        required = {"id", "source", "target", "weight", "predicate", "description", "sent_id"}
        assert required.issubset(set(df.columns)), \
            f"缺失列：{required - set(df.columns)}"

    def test_empty_relations_returns_empty_df(self):
        """空关系列表应返回空 DataFrame（含正确列名）。"""
        df = relations_to_dataframe([])
        assert len(df) == 0
        assert "id" in df.columns


# ===========================================================================
# 测试 6：extract() 主入口
# ===========================================================================

class TestExtractMainEntry:

    def test_empty_input_returns_empty_dfs(self):
        """空输入应返回两个空 DataFrame。"""
        config = ExtractionConfig()
        entities_df, rels_df = extract([], config)
        assert len(entities_df) == 0
        assert len(rels_df) == 0
        # 检查列名
        assert "id" in entities_df.columns
        assert "id" in rels_df.columns

    def test_no_sentences_raises_value_error(self):
        """TextUnit 不含 SentenceUnit 时应抛出 ValueError。"""
        config = ExtractionConfig()
        # 构造没有 sentences 的 TextUnit
        unit = TextUnit(
            chunk_id="doc001-p000",
            text="Some text.",
            doc_id="doc001",
            doc_title="Test",
            chunk_index=0,
            sentences=[],  # 空 sentences
        )
        with pytest.raises(ValueError, match="SentenceUnit"):
            extract([unit], config)

    def test_spacy_extract_produces_dataframes(self):
        """spaCy 提取应返回两个 DataFrame。"""
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        config = ExtractionConfig(backend="spacy", spacy_model="en_core_web_sm")
        unit = _make_text_unit_with_sentences(
            ["GraphRAG uses the Leiden algorithm.", "Leiden improves Louvain."],
            doc_id="doc001",
            para_idx=0,
        )
        entities_df, rels_df = extract([unit], config)
        assert isinstance(entities_df, pd.DataFrame)
        assert isinstance(rels_df, pd.DataFrame)

    def test_spacy_entities_have_sent_id(self):
        """spaCy 提取的实体应包含 sent_id 列。"""
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        config = ExtractionConfig(backend="spacy", spacy_model="en_core_web_sm")
        unit = _make_text_unit_with_sentences(
            ["GraphRAG uses the Leiden algorithm for community detection."],
            doc_id="doc001",
            para_idx=0,
        )
        entities_df, _ = extract([unit], config)
        if len(entities_df) > 0:
            assert "sent_id" in entities_df.columns
            # sent_id 应非空
            assert entities_df["sent_id"].notna().all()

    def test_spacy_no_cross_sentence_edges(self):
        """spaCy 提取的关系不应包含跨句子的物理结构边。"""
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        config = ExtractionConfig(backend="spacy", spacy_model="en_core_web_sm")
        # 两个句子在不同 SentenceUnit 中
        unit = _make_text_unit_with_sentences(
            [
                "GraphRAG uses Leiden algorithm.",
                "Shannon entropy measures diversity.",
            ],
            doc_id="doc001",
            para_idx=0,
        )
        entities_df, rels_df = extract([unit], config)

        if len(rels_df) > 0 and len(entities_df) > 0:
            # 构建 node_id → sent_id 映射
            node_to_sent = dict(zip(entities_df["id"], entities_df["sent_id"]))

            # 检查每条关系：source 和 target 的 sent_id 应相同（同句内）
            # 或者 source/target 不在 node_to_sent 中（可能是孤立节点）
            for _, row in rels_df.iterrows():
                src_sent = node_to_sent.get(row["source"], None)
                tgt_sent = node_to_sent.get(row["target"], None)
                if src_sent and tgt_sent:
                    assert src_sent == tgt_sent, \
                        f"跨句子关系：{row['source']} (sent={src_sent}) → {row['target']} (sent={tgt_sent})"

    def test_spacy_node_id_contains_sent_id_prefix(self):
        """spaCy 提取的实体 node_id 应以 sent_id 为前缀。"""
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        config = ExtractionConfig(backend="spacy", spacy_model="en_core_web_sm")
        unit = _make_text_unit_with_sentences(
            ["GraphRAG uses the Leiden algorithm for community detection."],
            doc_id="doc001",
            para_idx=0,
        )
        entities_df, _ = extract([unit], config)

        if len(entities_df) > 0:
            for _, row in entities_df.iterrows():
                node_id = row["id"]
                sent_id = row["sent_id"]
                assert node_id.startswith(sent_id), \
                    f"node_id={node_id} 应以 sent_id={sent_id} 开头"


# ===========================================================================
# 测试 7：Relation.relation_id
# ===========================================================================

class TestRelationID:

    def test_relation_id_is_deterministic(self):
        """相同 (source, predicate, target) 应生成相同 relation_id。"""
        rel1 = Relation("node_A", "node_B", "use")
        rel2 = Relation("node_A", "node_B", "use")
        assert rel1.relation_id == rel2.relation_id

    def test_different_relations_have_different_ids(self):
        """不同关系应有不同 relation_id。"""
        rel1 = Relation("node_A", "node_B", "use")
        rel2 = Relation("node_A", "node_C", "use")
        rel3 = Relation("node_A", "node_B", "improve")
        assert rel1.relation_id != rel2.relation_id
        assert rel1.relation_id != rel3.relation_id

    def test_relation_id_is_12_hex_chars(self):
        """relation_id 应为 12 位十六进制字符串。"""
        rel = Relation("node_A", "node_B", "use")
        assert len(rel.relation_id) == 12
        assert re.fullmatch(r"[0-9a-f]{12}", rel.relation_id)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
