"""
data/ingestion.py
-----------------
数据输入模块：从文件系统读取文档，执行三级物理切分，
输出标准化的 Document / Paragraph / SentenceUnit 层次结构。

三级物理结构：
  Document  — 原始文档，ID 基于文件路径 hash
  Paragraph — 按空行分割的段落，ID 格式：{doc_id}-p{para_idx}
  Sentence  — spaCy doc.sents 切分的句子，ID 格式：{para_id}-s{sent_idx}

核心设计原则：
  每个句子都有唯一的物理路径 ID（sent_id），
  从句子中提取的实体节点继承该 sent_id，
  保证底层图中每个节点都有精确的物理坐标。

支持格式：
  .txt  — 纯文本
  .json — {"title": "...", "text": "..."} 或 [{"text": "..."}]
  .pdf  — 需安装 pypdf（可选依赖）

输出：
  List[TextUnit]，每个 TextUnit 对应一个段落，
  内含 List[SentenceUnit]，每个 SentenceUnit 对应一个句子。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ..pipeline_config import InputConfig


# ---------------------------------------------------------------------------
# 核心数据结构
# ---------------------------------------------------------------------------

@dataclass
class SentenceUnit:
    """
    句子级物理单元，是实体提取的最小粒度。

    Attributes
    ----------
    sent_id     : 全局唯一句子 ID，格式：{doc_id}-p{para_idx}-s{sent_idx}
                  例如：a3f2b1c4d5e6-p002-s001
    text        : 句子原始文本
    doc_id      : 所属文档 ID
    para_id     : 所属段落 ID，格式：{doc_id}-p{para_idx}
    sent_index  : 在段落内的句子序号（从 0 开始）
    para_index  : 段落在文档内的序号（从 0 开始）
    """
    sent_id: str
    text: str
    doc_id: str
    para_id: str
    sent_index: int
    para_index: int

    @property
    def physical_path(self) -> str:
        """返回人类可读的物理路径描述。"""
        return f"doc={self.doc_id} para={self.para_index} sent={self.sent_index}"


@dataclass
class TextUnit:
    """
    段落级文本单元，是 Pipeline 的基本处理容器。

    与旧版不同，TextUnit 现在对应一个段落（而非任意 chunk），
    内含该段落下所有句子的 SentenceUnit 列表。

    Attributes
    ----------
    chunk_id    : 段落 ID，格式：{doc_id}-p{para_idx}（即 para_id）
    text        : 段落完整文本
    doc_id      : 所属文档 ID
    doc_title   : 所属文档标题
    chunk_index : 段落在文档内的序号（从 0 开始）
    sentences   : 该段落下的所有 SentenceUnit 列表
    metadata    : 可扩展的附加元数据
    """
    chunk_id: str          # 即 para_id
    text: str
    doc_id: str
    doc_title: str
    chunk_index: int
    sentences: List[SentenceUnit] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def sentence_count(self) -> int:
        return len(self.sentences)


@dataclass
class Document:
    """原始文档，加载后立即切分为段落和句子。"""
    doc_id: str
    title: str
    raw_text: str
    source_path: str


# ---------------------------------------------------------------------------
# ID 生成
# ---------------------------------------------------------------------------

def _make_doc_id(path: str) -> str:
    """基于文件路径生成稳定的文档 ID（12位 hex）。"""
    return hashlib.md5(path.encode()).hexdigest()[:12]


def _make_para_id(doc_id: str, para_index: int) -> str:
    """段落 ID：{doc_id}-p{para_index:03d}"""
    return f"{doc_id}-p{para_index:03d}"


def _make_sent_id(para_id: str, sent_index: int) -> str:
    """句子 ID：{para_id}-s{sent_index:03d}"""
    return f"{para_id}-s{sent_index:03d}"


# ---------------------------------------------------------------------------
# 段落切分（不依赖 spaCy）
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str, min_chars: int = 20) -> List[str]:
    """
    按空行切分段落。
    过短的段落（< min_chars）与下一段合并，避免碎片化。
    """
    raw = re.split(r"\n\s*\n", text.strip())
    paragraphs: List[str] = []
    buffer = ""

    for para in raw:
        para = para.strip()
        if not para:
            continue
        buffer = (buffer + "\n\n" + para).strip() if buffer else para
        if len(buffer) >= min_chars:
            paragraphs.append(buffer)
            buffer = ""

    if buffer:
        paragraphs.append(buffer)

    return paragraphs if paragraphs else [text.strip()]


# ---------------------------------------------------------------------------
# 句子切分（spaCy）
# ---------------------------------------------------------------------------

_spacy_nlp = None  # 全局缓存，避免重复加载


def _get_spacy_nlp(model_name: str = "en_core_web_sm"):
    """懒加载 spaCy 模型，全局缓存。"""
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load(model_name, disable=["ner", "lemmatizer"])
        except ImportError:
            raise ImportError(
                "句子切分需要安装 spaCy：\n"
                "  pip install spacy\n"
                f"  python -m spacy download {model_name}"
            )
        except OSError:
            raise OSError(
                f"spaCy 模型 '{model_name}' 未找到，请运行：\n"
                f"  python -m spacy download {model_name}"
            )
    return _spacy_nlp


def _split_sentences_spacy(text: str, model_name: str = "en_core_web_sm") -> List[str]:
    """
    使用 spaCy 依存句法树切分句子。
    比正则切分更准确，能正确处理缩写、引号内标点、数字小数点等边界歧义。
    """
    nlp = _get_spacy_nlp(model_name)
    doc = nlp(text[:50000])  # spaCy 有长度限制，截断超长段落
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences if sentences else [text.strip()]


def _split_sentences_fallback(text: str) -> List[str]:
    """
    spaCy 不可用时的降级方案：正则切分。
    仅作为兜底，精度低于 spaCy。
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in parts if s.strip()] or [text.strip()]


# ---------------------------------------------------------------------------
# 文档 → TextUnit 转换（核心逻辑）
# ---------------------------------------------------------------------------

def document_to_text_units(
    doc: Document,
    spacy_model: str = "en_core_web_sm",
    use_spacy_sentences: bool = True,
) -> List[TextUnit]:
    """
    将单篇文档切分为段落级 TextUnit 列表，每个 TextUnit 内含句子级 SentenceUnit 列表。

    物理 ID 层次：
        doc_id  = hash(file_path)
        para_id = {doc_id}-p{para_index:03d}
        sent_id = {para_id}-s{sent_index:03d}

    Parameters
    ----------
    doc : Document
        已加载的文档
    spacy_model : str
        spaCy 模型名称，用于句子切分
    use_spacy_sentences : bool
        是否使用 spaCy 切分句子（False 时降级为正则）

    Returns
    -------
    List[TextUnit]
        段落级 TextUnit 列表，每个 TextUnit 内含 SentenceUnit 列表
    """
    paragraphs = _split_paragraphs(doc.raw_text)
    units: List[TextUnit] = []

    for para_idx, para_text in enumerate(paragraphs):
        para_id = _make_para_id(doc.doc_id, para_idx)

        # 切分句子
        if use_spacy_sentences:
            try:
                raw_sentences = _split_sentences_spacy(para_text, spacy_model)
            except (ImportError, OSError):
                raw_sentences = _split_sentences_fallback(para_text)
        else:
            raw_sentences = _split_sentences_fallback(para_text)

        # 构建 SentenceUnit 列表
        sentence_units: List[SentenceUnit] = []
        for sent_idx, sent_text in enumerate(raw_sentences):
            sent_text = sent_text.strip()
            if not sent_text:
                continue
            sent_id = _make_sent_id(para_id, sent_idx)
            sentence_units.append(SentenceUnit(
                sent_id=sent_id,
                text=sent_text,
                doc_id=doc.doc_id,
                para_id=para_id,
                sent_index=sent_idx,
                para_index=para_idx,
            ))

        if not sentence_units:
            continue

        units.append(TextUnit(
            chunk_id=para_id,
            text=para_text,
            doc_id=doc.doc_id,
            doc_title=doc.title,
            chunk_index=para_idx,
            sentences=sentence_units,
            metadata={"source_path": doc.source_path},
        ))

    return units


# ---------------------------------------------------------------------------
# 文件读取器
# ---------------------------------------------------------------------------

def _read_txt(path: str, encoding: str) -> List[Document]:
    with open(path, "r", encoding=encoding, errors="replace") as f:
        text = f.read()
    title = Path(path).stem
    doc_id = _make_doc_id(path)
    return [Document(doc_id=doc_id, title=title, raw_text=text, source_path=path)]


def _read_json(path: str, encoding: str) -> List[Document]:
    with open(path, "r", encoding=encoding) as f:
        data = json.load(f)

    docs: List[Document] = []

    if isinstance(data, dict):
        data = [data]
    elif isinstance(data, str):
        data = [{"text": data}]

    for i, item in enumerate(data):
        if isinstance(item, str):
            item = {"text": item}
        text = item.get("text") or item.get("content") or item.get("body") or ""
        title = item.get("title") or item.get("name") or f"{Path(path).stem}_{i}"
        if not text.strip():
            continue
        doc_id = _make_doc_id(f"{path}_{i}")
        docs.append(Document(doc_id=doc_id, title=title, raw_text=text, source_path=path))

    return docs


def _read_pdf(path: str) -> List[Document]:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "读取 PDF 需要安装 pypdf：pip install pypdf\n"
                f"文件：{path}"
            )

    reader = PdfReader(path)
    pages_text = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            pages_text.append(t)

    full_text = "\n\n".join(pages_text)
    title = Path(path).stem
    doc_id = _make_doc_id(path)
    return [Document(doc_id=doc_id, title=title, raw_text=full_text, source_path=path)]


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def load_documents(config: InputConfig) -> List[Document]:
    """
    从 config.data_dir 扫描并加载所有支持格式的文档。
    """
    data_dir = Path(config.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在：{data_dir}")

    docs: List[Document] = []
    supported = {".txt", ".json", ".pdf"}

    for file_path in sorted(data_dir.rglob("*")):
        if file_path.suffix.lower() not in supported:
            continue
        if file_path.name.startswith("."):
            continue

        path_str = str(file_path)
        try:
            if file_path.suffix.lower() == ".txt":
                docs.extend(_read_txt(path_str, config.encoding))
            elif file_path.suffix.lower() == ".json":
                docs.extend(_read_json(path_str, config.encoding))
            elif file_path.suffix.lower() == ".pdf":
                docs.extend(_read_pdf(path_str))
        except Exception as e:
            print(f"  [警告] 跳过文件 {file_path.name}：{e}")

    return docs


def documents_to_text_units(
    documents: List[Document],
    config: InputConfig,
) -> List[TextUnit]:
    """
    将文档列表切分为段落级 TextUnit 列表（每个 TextUnit 内含句子级 SentenceUnit）。

    Parameters
    ----------
    documents : List[Document]
        已加载的文档列表
    config : InputConfig
        输入配置

    Returns
    -------
    List[TextUnit]
        所有文档的 TextUnit 列表，ID 全局唯一
    """
    units: List[TextUnit] = []
    use_spacy = config.chunk_strategy != "sentence_regex"  # 默认使用 spaCy

    for doc in documents:
        doc_units = document_to_text_units(
            doc,
            spacy_model="en_core_web_sm",
            use_spacy_sentences=use_spacy,
        )
        units.extend(doc_units)

    return units


def ingest(config: InputConfig) -> List[TextUnit]:
    """
    一步完成文档加载 + 三级切分，返回 TextUnit 列表（含 SentenceUnit）。
    这是外部模块调用的主入口。
    """
    docs = load_documents(config)
    return documents_to_text_units(docs, config)


# ---------------------------------------------------------------------------
# 便捷工具：从 TextUnit 列表提取所有 SentenceUnit
# ---------------------------------------------------------------------------

def get_all_sentences(text_units: List[TextUnit]) -> List[SentenceUnit]:
    """从 TextUnit 列表中提取所有 SentenceUnit，展平为一维列表。"""
    sentences: List[SentenceUnit] = []
    for unit in text_units:
        sentences.extend(unit.sentences)
    return sentences


def build_sent_index(text_units: List[TextUnit]) -> dict:
    """
    构建 sent_id → SentenceUnit 的快速索引。

    Returns
    -------
    dict
        {sent_id: SentenceUnit}
    """
    index = {}
    for unit in text_units:
        for sent in unit.sentences:
            index[sent.sent_id] = sent
    return index


def build_para_index(text_units: List[TextUnit]) -> dict:
    """
    构建 chunk_id (para_id) → TextUnit 的快速索引。

    Returns
    -------
    dict
        {para_id: TextUnit}
    """
    return {unit.chunk_id: unit for unit in text_units}
