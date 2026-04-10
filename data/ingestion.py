"""
data/ingestion.py
-----------------
数据输入模块：从文件系统读取文档，执行分块，输出标准化的 TextUnit 列表。

支持格式：
  .txt  — 纯文本
  .json — {"title": "...", "text": "..."} 或 [{"text": "..."}]
  .pdf  — 需安装 pypdf（可选依赖）

分块策略：
  paragraph — 按空行分段（推荐，保留物理结构）
  sentence  — 按句号/换行分句
  token     — 按 token 数固定窗口（需安装 tiktoken，否则退化为字符数）

输出：
  List[TextUnit]，每个 TextUnit 携带唯一 chunk_id、原始文本、来源文件信息
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
class TextUnit:
    """
    文本块，是整个 Pipeline 的基本处理单元。

    Attributes
    ----------
    chunk_id   : 基于内容 hash 生成的唯一 ID，跨运行稳定
    text       : chunk 的原始文本内容
    doc_id     : 来源文档的唯一 ID
    doc_title  : 来源文档的标题（通常为文件名）
    chunk_index: 在文档内的顺序编号（从 0 开始）
    metadata   : 可扩展的附加元数据
    """
    chunk_id: str
    text: str
    doc_id: str
    doc_title: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class Document:
    """原始文档，加载后立即分块为 TextUnit 列表。"""
    doc_id: str
    title: str
    raw_text: str
    source_path: str


# ---------------------------------------------------------------------------
# 分块策略
# ---------------------------------------------------------------------------

def _chunk_by_paragraph(text: str, min_chars: int = 50) -> List[str]:
    """
    按空行分段。连续空行视为段落分隔符。
    过短的段落（< min_chars）会与下一段合并，避免碎片化。
    """
    raw_paragraphs = re.split(r"\n\s*\n", text.strip())
    chunks: List[str] = []
    buffer = ""

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        buffer = (buffer + "\n\n" + para).strip() if buffer else para
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""

    if buffer:
        chunks.append(buffer)

    return chunks if chunks else [text.strip()]


def _chunk_by_sentence(text: str, max_sentences: int = 5) -> List[str]:
    """
    按句子分块，每块最多 max_sentences 句。
    句子边界：句号/问号/感叹号后跟空格或换行。
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    buffer: List[str] = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        buffer.append(sent)
        if len(buffer) >= max_sentences:
            chunks.append(" ".join(buffer))
            buffer = []

    if buffer:
        chunks.append(" ".join(buffer))

    return chunks if chunks else [text.strip()]


def _chunk_by_token(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    按 token 数固定窗口分块。
    优先使用 tiktoken，不可用时退化为按字符数估算（1 token ≈ 4 字符）。
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(enc.decode(chunk_tokens))
            start += chunk_size - overlap
        return chunks if chunks else [text]
    except ImportError:
        # 退化：按字符数估算
        char_size = chunk_size * 4
        char_overlap = overlap * 4
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + char_size, len(text))
            chunks.append(text[start:end])
            start += char_size - char_overlap
        return chunks if chunks else [text]


def chunk_text(text: str, config: InputConfig) -> List[str]:
    """根据配置选择分块策略。"""
    strategy = config.chunk_strategy
    if strategy == "paragraph":
        return _chunk_by_paragraph(text)
    elif strategy == "sentence":
        return _chunk_by_sentence(text)
    elif strategy == "token":
        return _chunk_by_token(text, config.chunk_size, config.chunk_overlap)
    else:
        return _chunk_by_paragraph(text)


# ---------------------------------------------------------------------------
# 文件读取器
# ---------------------------------------------------------------------------

def _make_doc_id(path: str) -> str:
    """基于文件路径生成稳定的文档 ID。"""
    return hashlib.md5(path.encode()).hexdigest()[:12]


def _make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """基于内容生成稳定的 chunk ID。"""
    content = f"{doc_id}_{chunk_index}_{text[:64]}"
    return hashlib.md5(content.encode()).hexdigest()[:16]


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

    # 支持三种 JSON 格式：
    # 1. 单个对象 {"title": "...", "text": "..."}
    # 2. 对象列表 [{"title": "...", "text": "..."}, ...]
    # 3. 纯字符串列表 ["text1", "text2", ...]
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

    Parameters
    ----------
    config : InputConfig
        输入配置

    Returns
    -------
    List[Document]
        加载的文档列表
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
    将文档列表分块，生成 TextUnit 列表。

    Parameters
    ----------
    documents : List[Document]
        已加载的文档列表
    config : InputConfig
        输入配置（决定分块策略）

    Returns
    -------
    List[TextUnit]
        所有文档的 TextUnit 列表，chunk_id 全局唯一
    """
    units: List[TextUnit] = []

    for doc in documents:
        chunks = chunk_text(doc.raw_text, config)
        for idx, chunk_text_content in enumerate(chunks):
            chunk_text_content = chunk_text_content.strip()
            if not chunk_text_content:
                continue
            chunk_id = _make_chunk_id(doc.doc_id, idx, chunk_text_content)
            units.append(TextUnit(
                chunk_id=chunk_id,
                text=chunk_text_content,
                doc_id=doc.doc_id,
                doc_title=doc.title,
                chunk_index=idx,
                metadata={"source_path": doc.source_path},
            ))

    return units


def ingest(config: InputConfig) -> List[TextUnit]:
    """
    一步完成文档加载 + 分块，返回 TextUnit 列表。
    这是外部模块调用的主入口。
    """
    docs = load_documents(config)
    return documents_to_text_units(docs, config)
