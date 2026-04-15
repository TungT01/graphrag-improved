"""
tests/test_ingestion_v3.py
--------------------------
v3 物理优先架构 — 数据摄入模块单元测试。

测试覆盖：
1. 三级 ID 格式正确性（doc_id / para_id / sent_id）
2. SentenceUnit 字段完整性
3. TextUnit 与 SentenceUnit 的层次关系
4. 段落切分逻辑（_split_paragraphs）
5. 句子切分降级方案（正则 fallback）
6. ID 全局唯一性
7. 工具函数：get_all_sentences / build_sent_index / build_para_index
8. 边界情况：空文本、单句段落、超长文本
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

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
    _split_paragraphs,
    _split_sentences_fallback,
    build_para_index,
    build_sent_index,
    document_to_text_units,
    get_all_sentences,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def simple_doc():
    """包含两个段落、每段两句的简单文档。"""
    return Document(
        doc_id="testdoc001",
        title="Test Document",
        raw_text=(
            "The Leiden algorithm improves upon Louvain. "
            "It guarantees well-connected communities.\n\n"
            "GraphRAG uses Leiden for community detection. "
            "Communities are summarized by LLMs."
        ),
        source_path="/fake/path/test.txt",
    )


@pytest.fixture
def multi_para_doc():
    """包含三个段落的文档，用于测试段落索引。"""
    return Document(
        doc_id="multipara001",
        title="Multi Paragraph Doc",
        raw_text=(
            "First paragraph has enough content to pass the minimum character threshold.\n\n"
            "Second paragraph also has enough content to pass the minimum character threshold.\n\n"
            "Third paragraph with sufficient content to pass the minimum character threshold."
        ),
        source_path="/fake/path/multi.txt",
    )


# ===========================================================================
# 测试 1：ID 格式正确性
# ===========================================================================

class TestIDFormat:

    def test_doc_id_is_12_hex_chars(self):
        """doc_id 应为 12 位十六进制字符串。"""
        doc_id = _make_doc_id("/some/path/file.txt")
        assert len(doc_id) == 12
        assert re.fullmatch(r"[0-9a-f]{12}", doc_id), f"doc_id 格式错误：{doc_id}"

    def test_doc_id_is_deterministic(self):
        """相同路径应生成相同 doc_id。"""
        path = "/some/path/file.txt"
        assert _make_doc_id(path) == _make_doc_id(path)

    def test_doc_id_differs_for_different_paths(self):
        """不同路径应生成不同 doc_id。"""
        assert _make_doc_id("/path/a.txt") != _make_doc_id("/path/b.txt")

    def test_para_id_format(self):
        """para_id 格式应为 {doc_id}-p{para_index:03d}。"""
        para_id = _make_para_id("abc123def456", 2)
        assert para_id == "abc123def456-p002"

    def test_para_id_zero_padded(self):
        """para_index 应零填充到 3 位。"""
        assert _make_para_id("doc001", 0) == "doc001-p000"
        assert _make_para_id("doc001", 9) == "doc001-p009"
        assert _make_para_id("doc001", 99) == "doc001-p099"
        assert _make_para_id("doc001", 100) == "doc001-p100"

    def test_sent_id_format(self):
        """sent_id 格式应为 {para_id}-s{sent_index:03d}。"""
        sent_id = _make_sent_id("doc001-p002", 5)
        assert sent_id == "doc001-p002-s005"

    def test_sent_id_contains_full_physical_path(self):
        """sent_id 应包含完整的物理路径（doc_id + para_idx + sent_idx）。"""
        doc_id = "abc123def456"
        para_id = _make_para_id(doc_id, 3)
        sent_id = _make_sent_id(para_id, 1)
        # sent_id 应包含 doc_id 前缀
        assert sent_id.startswith(doc_id)
        # 格式：{doc_id}-p003-s001
        assert sent_id == f"{doc_id}-p003-s001"


# ===========================================================================
# 测试 2：段落切分
# ===========================================================================

class TestParagraphSplitting:

    def test_split_by_blank_line(self):
        """应按空行切分段落。"""
        text = (
            "First paragraph with enough content here.\n\n"
            "Second paragraph with enough content here.\n\n"
            "Third paragraph with enough content here."
        )
        paras = _split_paragraphs(text, min_chars=20)
        assert len(paras) == 3

    def test_short_paragraphs_merged(self):
        """过短段落（< min_chars）应与下一段合并。"""
        text = "Short.\n\nThis is a longer paragraph with enough content."
        paras = _split_paragraphs(text, min_chars=20)
        # "Short." 只有 6 字符，应与下一段合并
        assert len(paras) == 1
        assert "Short." in paras[0]
        assert "longer paragraph" in paras[0]

    def test_empty_text_returns_single_item(self):
        """空文本应返回包含原文本的单元素列表。"""
        paras = _split_paragraphs("   ", min_chars=20)
        assert len(paras) >= 1

    def test_single_paragraph_no_split(self):
        """无空行的文本应作为单个段落返回。"""
        text = "This is a single paragraph without any blank lines in it."
        paras = _split_paragraphs(text, min_chars=20)
        assert len(paras) == 1
        assert paras[0] == text

    def test_multiple_blank_lines_treated_as_one(self):
        """多个连续空行应等同于单个空行。"""
        text = "Para one with enough content.\n\n\n\nPara two with enough content."
        paras = _split_paragraphs(text, min_chars=20)
        assert len(paras) == 2


# ===========================================================================
# 测试 3：句子切分降级方案
# ===========================================================================

class TestSentenceSplittingFallback:

    def test_basic_sentence_split(self):
        """正则降级方案应能切分基本句子。"""
        text = "First sentence. Second sentence. Third sentence."
        sents = _split_sentences_fallback(text)
        assert len(sents) >= 2

    def test_empty_text_returns_single_item(self):
        """空文本应返回包含原文本的单元素列表。"""
        sents = _split_sentences_fallback("  ")
        assert len(sents) >= 1

    def test_single_sentence_no_split(self):
        """单句文本不应被切分。"""
        text = "This is a single sentence without any terminal punctuation"
        sents = _split_sentences_fallback(text)
        assert len(sents) == 1

    def test_exclamation_and_question_marks(self):
        """感叹号和问号也应作为句子边界。"""
        text = "Is this working? Yes it is! Great."
        sents = _split_sentences_fallback(text)
        assert len(sents) >= 2


# ===========================================================================
# 测试 4：document_to_text_units 核心逻辑
# ===========================================================================

class TestDocumentToTextUnits:

    def test_returns_text_units(self, simple_doc):
        """应返回非空的 TextUnit 列表。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        assert len(units) >= 1
        assert all(isinstance(u, TextUnit) for u in units)

    def test_text_unit_chunk_id_is_para_id(self, simple_doc):
        """TextUnit.chunk_id 应等于 para_id 格式。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for i, unit in enumerate(units):
            expected_para_id = _make_para_id(simple_doc.doc_id, i)
            assert unit.chunk_id == expected_para_id, \
                f"TextUnit[{i}].chunk_id={unit.chunk_id}, 期望={expected_para_id}"

    def test_text_unit_doc_id_matches(self, simple_doc):
        """TextUnit.doc_id 应与文档 doc_id 一致。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            assert unit.doc_id == simple_doc.doc_id

    def test_text_unit_contains_sentences(self, simple_doc):
        """每个 TextUnit 应包含至少一个 SentenceUnit。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            assert len(unit.sentences) >= 1, \
                f"TextUnit '{unit.chunk_id}' 没有 SentenceUnit"

    def test_sentence_unit_fields_complete(self, simple_doc):
        """SentenceUnit 的所有字段应正确填充。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            for sent in unit.sentences:
                assert isinstance(sent, SentenceUnit)
                assert sent.sent_id, "sent_id 不应为空"
                assert sent.text.strip(), "text 不应为空"
                assert sent.doc_id == simple_doc.doc_id
                assert sent.para_id == unit.chunk_id
                assert sent.sent_index >= 0
                assert sent.para_index >= 0

    def test_sent_id_format_in_units(self, simple_doc):
        """SentenceUnit.sent_id 应符合 {para_id}-s{sent_idx:03d} 格式。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            for sent in unit.sentences:
                expected_prefix = unit.chunk_id
                assert sent.sent_id.startswith(expected_prefix), \
                    f"sent_id={sent.sent_id} 应以 para_id={expected_prefix} 开头"
                # 格式：{para_id}-s{sent_idx:03d}
                assert re.match(r".+-s\d{3}$", sent.sent_id), \
                    f"sent_id 格式错误：{sent.sent_id}"

    def test_sent_index_sequential(self, simple_doc):
        """同一段落内的 sent_index 应从 0 开始连续递增。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            indices = [s.sent_index for s in unit.sentences]
            assert indices == list(range(len(indices))), \
                f"sent_index 不连续：{indices}"

    def test_para_index_sequential(self, multi_para_doc):
        """TextUnit 的 chunk_index 应从 0 开始连续递增。"""
        units = document_to_text_units(multi_para_doc, use_spacy_sentences=False)
        indices = [u.chunk_index for u in units]
        assert indices == list(range(len(indices))), \
            f"chunk_index 不连续：{indices}"

    def test_two_paragraphs_produce_two_units(self, simple_doc):
        """两段落文档应产生两个 TextUnit。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        assert len(units) == 2

    def test_physical_path_property(self, simple_doc):
        """SentenceUnit.physical_path 应返回可读的物理路径描述。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            for sent in unit.sentences:
                path = sent.physical_path
                assert "doc=" in path
                assert "para=" in path
                assert "sent=" in path


# ===========================================================================
# 测试 5：ID 全局唯一性
# ===========================================================================

class TestIDUniqueness:

    def test_para_ids_unique_within_document(self, multi_para_doc):
        """同一文档内所有 para_id 应全局唯一。"""
        units = document_to_text_units(multi_para_doc, use_spacy_sentences=False)
        para_ids = [u.chunk_id for u in units]
        assert len(para_ids) == len(set(para_ids)), \
            f"存在重复的 para_id：{para_ids}"

    def test_sent_ids_unique_within_document(self, simple_doc):
        """同一文档内所有 sent_id 应全局唯一。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        sent_ids = [s.sent_id for u in units for s in u.sentences]
        assert len(sent_ids) == len(set(sent_ids)), \
            f"存在重复的 sent_id：{sent_ids}"

    def test_sent_ids_unique_across_documents(self):
        """不同文档的 sent_id 应全局唯一（因为 doc_id 不同）。"""
        doc1 = Document(
            doc_id=_make_doc_id("/path/doc1.txt"),
            title="Doc 1",
            raw_text="First document sentence one. First document sentence two.",
            source_path="/path/doc1.txt",
        )
        doc2 = Document(
            doc_id=_make_doc_id("/path/doc2.txt"),
            title="Doc 2",
            raw_text="Second document sentence one. Second document sentence two.",
            source_path="/path/doc2.txt",
        )
        units1 = document_to_text_units(doc1, use_spacy_sentences=False)
        units2 = document_to_text_units(doc2, use_spacy_sentences=False)

        ids1 = {s.sent_id for u in units1 for s in u.sentences}
        ids2 = {s.sent_id for u in units2 for s in u.sentences}

        overlap = ids1 & ids2
        assert not overlap, f"不同文档存在相同 sent_id：{overlap}"


# ===========================================================================
# 测试 6：工具函数
# ===========================================================================

class TestUtilityFunctions:

    def test_get_all_sentences_flattens(self, simple_doc):
        """get_all_sentences 应将所有 SentenceUnit 展平为一维列表。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        all_sents = get_all_sentences(units)
        total = sum(len(u.sentences) for u in units)
        assert len(all_sents) == total
        assert all(isinstance(s, SentenceUnit) for s in all_sents)

    def test_get_all_sentences_empty_input(self):
        """空输入应返回空列表。"""
        assert get_all_sentences([]) == []

    def test_build_sent_index_keys_are_sent_ids(self, simple_doc):
        """build_sent_index 的键应为 sent_id。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        index = build_sent_index(units)
        all_sents = get_all_sentences(units)
        expected_keys = {s.sent_id for s in all_sents}
        assert set(index.keys()) == expected_keys

    def test_build_sent_index_values_are_sentence_units(self, simple_doc):
        """build_sent_index 的值应为 SentenceUnit 对象。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        index = build_sent_index(units)
        for sent_id, sent in index.items():
            assert isinstance(sent, SentenceUnit)
            assert sent.sent_id == sent_id

    def test_build_para_index_keys_are_chunk_ids(self, multi_para_doc):
        """build_para_index 的键应为 chunk_id（para_id）。"""
        units = document_to_text_units(multi_para_doc, use_spacy_sentences=False)
        index = build_para_index(units)
        expected_keys = {u.chunk_id for u in units}
        assert set(index.keys()) == expected_keys

    def test_build_para_index_values_are_text_units(self, multi_para_doc):
        """build_para_index 的值应为 TextUnit 对象。"""
        units = document_to_text_units(multi_para_doc, use_spacy_sentences=False)
        index = build_para_index(units)
        for chunk_id, unit in index.items():
            assert isinstance(unit, TextUnit)
            assert unit.chunk_id == chunk_id

    def test_build_sent_index_lookup(self, simple_doc):
        """通过 sent_id 应能正确查找 SentenceUnit。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        index = build_sent_index(units)
        # 取第一个 sent_id 进行查找
        first_sent = units[0].sentences[0]
        found = index[first_sent.sent_id]
        assert found.text == first_sent.text


# ===========================================================================
# 测试 7：TextUnit 属性
# ===========================================================================

class TestTextUnitProperties:

    def test_word_count_property(self, simple_doc):
        """TextUnit.word_count 应返回正整数。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            assert unit.word_count > 0

    def test_sentence_count_property(self, simple_doc):
        """TextUnit.sentence_count 应等于 sentences 列表长度。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            assert unit.sentence_count == len(unit.sentences)

    def test_metadata_contains_source_path(self, simple_doc):
        """TextUnit.metadata 应包含 source_path。"""
        units = document_to_text_units(simple_doc, use_spacy_sentences=False)
        for unit in units:
            assert "source_path" in unit.metadata
            assert unit.metadata["source_path"] == simple_doc.source_path


# ===========================================================================
# 测试 8：spaCy 集成（可选，需要 spaCy 安装）
# ===========================================================================

class TestSpaCyIntegration:

    def test_spacy_sentence_split_produces_sentences(self, simple_doc):
        """spaCy 切分应产生有效的 SentenceUnit 列表。"""
        try:
            units = document_to_text_units(simple_doc, use_spacy_sentences=True)
            assert len(units) >= 1
            for unit in units:
                assert len(unit.sentences) >= 1
        except (ImportError, OSError):
            pytest.skip("spaCy 或模型未安装，跳过此测试")

    def test_spacy_sent_ids_unique(self, simple_doc):
        """spaCy 切分后的 sent_id 应全局唯一。"""
        try:
            units = document_to_text_units(simple_doc, use_spacy_sentences=True)
            sent_ids = [s.sent_id for u in units for s in u.sentences]
            assert len(sent_ids) == len(set(sent_ids))
        except (ImportError, OSError):
            pytest.skip("spaCy 或模型未安装，跳过此测试")

    def test_spacy_sent_id_format(self, simple_doc):
        """spaCy 切分后的 sent_id 应符合格式规范。"""
        try:
            units = document_to_text_units(simple_doc, use_spacy_sentences=True)
            for unit in units:
                for sent in unit.sentences:
                    assert sent.sent_id.startswith(unit.chunk_id), \
                        f"sent_id={sent.sent_id} 应以 para_id={unit.chunk_id} 开头"
        except (ImportError, OSError):
            pytest.skip("spaCy 或模型未安装，跳过此测试")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
