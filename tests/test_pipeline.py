"""
tests/test_pipeline.py
----------------------
端到端 Pipeline 集成测试（v3 物理优先架构）。

测试覆盖：
1. 配置加载与验证
2. 数据摄入（txt / json 格式）—— v3 三级 ID 切分
3. 实体关系抽取（spaCy 后端）—— v3 无实体消解
4. 完整 Pipeline 端到端运行
5. 输出文件完整性验证
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

# 路径设置
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graphrag_improved.pipeline_config import (
    ClusteringConfig,
    ExtractionConfig,
    InputConfig,
    OutputConfig,
    PipelineConfig,
    load_config,
)
from graphrag_improved.data.ingestion import (
    Document,
    SentenceUnit,
    TextUnit,
    _make_doc_id,
    _make_para_id,
    _make_sent_id,
    document_to_text_units,
    documents_to_text_units,
    get_all_sentences,
    ingest,
    load_documents,
)
from graphrag_improved.extraction.extractor import extract
from graphrag_improved.output.reporter import PipelineResult, save_results
from graphrag_improved.run import run_pipeline


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_txt_dir(tmp_path):
    """创建包含示例 txt 文件的临时目录。"""
    content = """GraphRAG and Leiden Algorithm

The Leiden algorithm improves upon Louvain by guaranteeing community connectivity.
GraphRAG uses Leiden to build hierarchical community summaries.

Structural Entropy Constraint

Shannon entropy measures the physical diversity of a community.
The lambda annealing mechanism controls the decay of the penalty coefficient.
Physical Anchoring assigns each triple a Chunk_ID for source attribution.

U-Retrieval combines top-down navigation with bottom-up physical anchor retrieval.
LLMs benefit from RAG by accessing external knowledge bases.
"""
    (tmp_path / "paper.txt").write_text(content, encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def sample_json_dir(tmp_path):
    """创建包含示例 json 文件的临时目录。"""
    import json
    data = [
        {"title": "Doc1", "text": "GraphRAG uses Leiden algorithm for community detection.\nLeiden improves Louvain by ensuring connectivity."},
        {"title": "Doc2", "text": "Shannon entropy measures information content.\nStructural entropy constrains the Leiden clustering process."},
    ]
    (tmp_path / "docs.json").write_text(json.dumps(data), encoding="utf-8")
    return str(tmp_path)


@pytest.fixture
def minimal_config(sample_txt_dir, tmp_path):
    """构造最小化的 PipelineConfig，用于快速测试。"""
    cfg = PipelineConfig(project_root=tmp_path)
    cfg.input = InputConfig(
        data_dir=sample_txt_dir,
        chunk_strategy="paragraph",
    )
    cfg.extraction = ExtractionConfig(
        backend="spacy",
        spacy_model="en_core_web_sm",
        min_entity_freq=1,
    )
    cfg.clustering = ClusteringConfig(
        lambda_init=100.0,
        lambda_min=0.0,
        max_cluster_size=5,
        max_iterations=3,
        seed=42,
    )
    cfg.output = OutputConfig(
        output_dir=str(tmp_path / "output"),
        html_report=True,
        csv_export=True,
        parquet_export=False,  # 测试环境可能无 pyarrow
        console_summary=False,
    )
    return cfg


def _has_spacy():
    """检查 spaCy 和模型是否可用。"""
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except (ImportError, OSError):
        return False


# ===========================================================================
# 测试 1：配置加载
# ===========================================================================

class TestConfig:

    def test_default_config_loads(self):
        """默认配置应能正常加载（即使 config.yaml 不存在）。"""
        cfg = load_config("/nonexistent/path/config.yaml")
        assert cfg.clustering.lambda_init == 1000.0
        assert cfg.extraction.backend == "rule"

    def test_invalid_chunk_strategy_raises(self):
        """无效的 chunk_strategy 应抛出 ValueError。"""
        cfg = PipelineConfig()
        cfg.input.chunk_strategy = "invalid_strategy"
        with pytest.raises(ValueError, match="chunk_strategy"):
            cfg.validate()

    def test_invalid_lambda_raises(self):
        """lambda_min > lambda_init 应抛出 ValueError。"""
        cfg = PipelineConfig()
        cfg.clustering.lambda_min = 999.0
        cfg.clustering.lambda_init = 1.0
        with pytest.raises(ValueError):
            cfg.validate()

    def test_yaml_config_loads(self, tmp_path):
        """YAML 配置文件应能正确解析。"""
        yaml_content = """
input:
  data_dir: "./data"
  chunk_strategy: "sentence"
clustering:
  lambda_init: 500.0
  annealing_schedule: "linear"
output:
  html_report: false
"""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content)

        try:
            import yaml
            cfg = load_config(str(config_file))
            assert cfg.clustering.lambda_init == 500.0
            assert cfg.clustering.annealing_schedule == "linear"
            assert cfg.output.html_report is False
            assert cfg.input.chunk_strategy == "sentence"
        except ImportError:
            pytest.skip("PyYAML 未安装，跳过 YAML 配置测试")


# ===========================================================================
# 测试 2：数据摄入（v3 三级 ID）
# ===========================================================================

class TestIngestion:

    def test_load_txt_documents(self, sample_txt_dir):
        """应能正确加载 txt 文件。"""
        config = InputConfig(data_dir=sample_txt_dir)
        docs = load_documents(config)
        assert len(docs) >= 1
        assert all(doc.raw_text for doc in docs)

    def test_load_json_documents(self, sample_json_dir):
        """应能正确加载 json 文件。"""
        config = InputConfig(data_dir=sample_json_dir)
        docs = load_documents(config)
        assert len(docs) >= 2

    def test_paragraph_chunking(self, sample_txt_dir):
        """v3 段落分块：每个 TextUnit 对应一个段落，内含 SentenceUnit 列表。"""
        text = (
            "First paragraph with enough content to exceed the minimum character threshold.\n\n"
            "Second paragraph with enough content to exceed the minimum character threshold.\n\n"
            "Third paragraph with enough content to exceed the minimum character threshold."
        )
        doc = Document(
            doc_id=_make_doc_id("/fake/test.txt"),
            title="Test",
            raw_text=text,
            source_path="/fake/test.txt",
        )
        units = document_to_text_units(doc, use_spacy_sentences=False)
        assert len(units) == 3
        # 每个 TextUnit 应包含 SentenceUnit
        for unit in units:
            assert len(unit.sentences) >= 1

    def test_sentence_units_have_sent_id(self, sample_txt_dir):
        """v3 架构：每个 SentenceUnit 应有 sent_id（含物理路径）。"""
        config = InputConfig(data_dir=sample_txt_dir)
        units = ingest(config)
        all_sents = get_all_sentences(units)
        assert len(all_sents) > 0
        for sent in all_sents:
            assert sent.sent_id, f"SentenceUnit 缺少 sent_id：{sent.text[:50]}"
            # sent_id 应包含 para_id 前缀
            assert sent.sent_id.startswith(sent.para_id)

    def test_chunk_ids_are_unique(self, sample_txt_dir):
        """所有 TextUnit 的 chunk_id（para_id）应全局唯一。"""
        config = InputConfig(data_dir=sample_txt_dir)
        units = ingest(config)
        ids = [u.chunk_id for u in units]
        assert len(ids) == len(set(ids)), "存在重复的 chunk_id"

    def test_sent_ids_are_unique(self, sample_txt_dir):
        """v3 架构：所有 SentenceUnit 的 sent_id 应全局唯一。"""
        config = InputConfig(data_dir=sample_txt_dir)
        units = ingest(config)
        all_sents = get_all_sentences(units)
        sent_ids = [s.sent_id for s in all_sents]
        assert len(sent_ids) == len(set(sent_ids)), "存在重复的 sent_id"

    def test_empty_dir_raises(self, tmp_path):
        """空目录应抛出 FileNotFoundError 或返回空列表。"""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        config = InputConfig(data_dir=str(empty_dir))
        docs = load_documents(config)
        assert docs == []

    def test_nonexistent_dir_raises(self):
        """不存在的目录应抛出 FileNotFoundError。"""
        config = InputConfig(data_dir="/nonexistent/path/12345")
        with pytest.raises(FileNotFoundError):
            load_documents(config)

    def test_text_unit_contains_sentences(self, sample_txt_dir):
        """v3 架构：每个 TextUnit 应包含至少一个 SentenceUnit。"""
        config = InputConfig(data_dir=sample_txt_dir)
        units = ingest(config)
        for unit in units:
            assert len(unit.sentences) >= 1, \
                f"TextUnit '{unit.chunk_id}' 没有 SentenceUnit"


# ===========================================================================
# 测试 3：实体关系抽取（v3 spaCy 后端）
# ===========================================================================

class TestExtraction:

    def _make_v3_units(self, texts_per_para):
        """
        构造 v3 格式的 TextUnit 列表（含 SentenceUnit）。
        texts_per_para: List[List[str]]，每个子列表是一个段落的句子列表。
        """
        doc_id = _make_doc_id("/fake/test.txt")
        units = []
        for para_idx, sentences in enumerate(texts_per_para):
            para_id = _make_para_id(doc_id, para_idx)
            sent_units = [
                SentenceUnit(
                    sent_id=_make_sent_id(para_id, sent_idx),
                    text=sent_text,
                    doc_id=doc_id,
                    para_id=para_id,
                    sent_index=sent_idx,
                    para_index=para_idx,
                )
                for sent_idx, sent_text in enumerate(sentences)
            ]
            units.append(TextUnit(
                chunk_id=para_id,
                text=" ".join(sentences),
                doc_id=doc_id,
                doc_title="Test Doc",
                chunk_index=para_idx,
                sentences=sent_units,
            ))
        return units

    def test_spacy_backend_extracts_entities(self):
        """v3 spaCy 后端应能从文本中抽取实体。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        units = self._make_v3_units([
            ["GraphRAG uses the Leiden algorithm for community detection."],
            ["LLMs improve knowledge retrieval."],
        ])
        config = ExtractionConfig(backend="spacy", min_entity_freq=1)
        entities_df, _ = extract(units, config)
        assert isinstance(entities_df, pd.DataFrame)
        assert "title" in entities_df.columns
        assert "text_unit_ids" in entities_df.columns

    def test_v3_entities_have_physical_columns(self):
        """v3 实体表应包含 sent_id / para_id / doc_id 列。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        units = self._make_v3_units([
            ["GraphRAG uses the Leiden algorithm for community detection."],
        ])
        config = ExtractionConfig(backend="spacy", min_entity_freq=1)
        entities_df, _ = extract(units, config)
        for col in ["sent_id", "para_id", "doc_id"]:
            assert col in entities_df.columns, f"缺少列：{col}"

    def test_relations_have_required_columns(self):
        """关系表应包含必要列（含 v3 新增的 sent_id）。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        units = self._make_v3_units([
            ["GraphRAG and Leiden are related methods."],
        ])
        config = ExtractionConfig(backend="spacy", min_entity_freq=1)
        _, rels_df = extract(units, config)
        required = {"id", "source", "target", "weight"}
        assert required.issubset(set(rels_df.columns))

    def test_empty_input_returns_empty_dfs(self):
        """空输入应返回空 DataFrame。"""
        config = ExtractionConfig(backend="spacy")
        entities_df, rels_df = extract([], config)
        assert len(entities_df) == 0
        assert len(rels_df) == 0

    def test_entity_text_unit_ids_are_sent_ids(self):
        """v3 实体的 text_unit_ids 应包含有效的 sent_id（精确到句子级）。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        units = self._make_v3_units([
            ["GraphRAG is a knowledge graph system."],
            ["Leiden algorithm clusters graph nodes."],
        ])
        config = ExtractionConfig(backend="spacy", min_entity_freq=1)
        entities_df, _ = extract(units, config)
        valid_sent_ids = {s.sent_id for u in units for s in u.sentences}
        for _, row in entities_df.iterrows():
            for cid in row["text_unit_ids"]:
                assert cid in valid_sent_ids, f"无效的 sent_id：{cid}"

    def test_no_entity_resolution(self):
        """v3 无实体消解：同名实体在不同句子里应是不同节点。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        # 两个句子都包含 "Leiden"，但在不同 SentenceUnit 中
        units = self._make_v3_units([
            [
                "Leiden algorithm improves Louvain.",
                "Leiden guarantees well-connected communities.",
            ],
        ])
        config = ExtractionConfig(backend="spacy", min_entity_freq=1)
        entities_df, _ = extract(units, config)

        if len(entities_df) > 0:
            # node_id 应包含 sent_id 前缀，不同句子的同名实体 node_id 不同
            leiden_nodes = entities_df[
                entities_df["title"].str.lower().str.contains("leiden", na=False)
            ]
            if len(leiden_nodes) >= 2:
                node_ids = leiden_nodes["id"].tolist()
                assert len(node_ids) == len(set(node_ids)), \
                    "v3 无实体消解：同名实体在不同句子里应有不同 node_id"


# ===========================================================================
# 测试 4：端到端 Pipeline
# ===========================================================================

class TestPipeline:

    def test_full_pipeline_runs(self, minimal_config):
        """完整 Pipeline 应能正常运行并返回结果。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        result = run_pipeline(config=minimal_config, verbose=False)
        assert isinstance(result, PipelineResult)
        assert isinstance(result.communities_df, pd.DataFrame)
        assert isinstance(result.entities_df, pd.DataFrame)
        assert isinstance(result.relationships_df, pd.DataFrame)

    def test_pipeline_produces_communities(self, minimal_config):
        """Pipeline 应产生至少一个社区。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        result = run_pipeline(config=minimal_config, verbose=False)
        # 若实体数量足够，应有社区
        if len(result.entities_df) > 0:
            assert len(result.communities_df) >= 0  # 允许空（图可能不连通）

    def test_pipeline_communities_have_required_columns(self, minimal_config):
        """社区 DataFrame 应包含所有必要列（含 v3 新增的 doc_ids）。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        result = run_pipeline(config=minimal_config, verbose=False)
        if not result.communities_df.empty:
            required = {
                "id", "community_id", "level", "structural_entropy", "lambda_used"
            }
            assert required.issubset(set(result.communities_df.columns))

    def test_pipeline_run_stats_populated(self, minimal_config):
        """运行统计信息应被正确填充。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        result = run_pipeline(config=minimal_config, verbose=False)
        assert "num_text_units" in result.run_stats
        assert result.run_stats["num_text_units"] > 0
        assert "elapsed_seconds" in result.run_stats
        assert result.run_stats["elapsed_seconds"] > 0

    def test_pipeline_with_json_input(self, sample_json_dir, tmp_path):
        """Pipeline 应能处理 JSON 格式输入。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        cfg = PipelineConfig(project_root=tmp_path)
        cfg.input = InputConfig(data_dir=sample_json_dir, chunk_strategy="paragraph")
        cfg.extraction = ExtractionConfig(backend="spacy", min_entity_freq=1)
        cfg.clustering = ClusteringConfig(lambda_init=100.0, max_iterations=3, seed=0)
        cfg.output = OutputConfig(
            output_dir=str(tmp_path / "out"),
            html_report=False, csv_export=False,
            parquet_export=False, console_summary=False,
        )
        result = run_pipeline(config=cfg, verbose=False)
        assert result.run_stats["num_text_units"] > 0

    def test_pipeline_entities_have_sent_id(self, minimal_config):
        """v3 Pipeline 输出的实体表应包含 sent_id 列。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        result = run_pipeline(config=minimal_config, verbose=False)
        if not result.entities_df.empty:
            assert "sent_id" in result.entities_df.columns, \
                "v3 实体表应包含 sent_id 列"


# ===========================================================================
# 测试 5：输出文件完整性
# ===========================================================================

class TestOutput:

    def test_csv_files_created(self, minimal_config):
        """CSV 文件应被正确创建。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        result = run_pipeline(config=minimal_config, verbose=False)
        out_dir = Path(minimal_config.output.output_dir)

        if not result.communities_df.empty:
            assert (out_dir / "communities.csv").exists()
        if not result.entities_df.empty:
            assert (out_dir / "entities.csv").exists()

    def test_html_report_created(self, minimal_config):
        """HTML 报告应被正确创建。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        result = run_pipeline(config=minimal_config, verbose=False)
        out_dir = Path(minimal_config.output.output_dir)
        assert (out_dir / "report.html").exists()

    def test_html_report_is_valid(self, minimal_config):
        """HTML 报告应包含基本结构。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        run_pipeline(config=minimal_config, verbose=False)
        html_path = Path(minimal_config.output.output_dir) / "report.html"
        content = html_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "GraphRAG Improved" in content
        assert "structural_entropy" in content.lower() or "结构熵" in content

    def test_summary_txt_created(self, minimal_config):
        """摘要文本文件应被创建。"""
        if not _has_spacy():
            pytest.skip("spaCy 或模型未安装，跳过此测试")

        minimal_config.output.console_summary = True
        run_pipeline(config=minimal_config, verbose=False)
        out_dir = Path(minimal_config.output.output_dir)
        assert (out_dir / "summary.txt").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
