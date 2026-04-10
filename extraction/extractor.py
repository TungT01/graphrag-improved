"""
extraction/extractor.py
-----------------------
实体与关系抽取模块。

支持两种后端：
  rule  — 纯规则抽取（无需任何模型，开箱即用）
          基于大写词/专有名词模式识别实体，
          基于共现窗口构建关系，适合快速原型验证。

  spacy — spaCy NER 抽取（需安装 spacy 及对应模型）
          使用 spaCy 的命名实体识别，精度更高，
          支持 PERSON / ORG / GPE / PRODUCT 等标准类型。

输出：
  entities DataFrame      : id, title, type, description, text_unit_ids
  relationships DataFrame : id, source, target, weight, description
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from ..data.ingestion import TextUnit
from ..pipeline_config import ExtractionConfig


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    title: str
    entity_type: str
    chunk_ids: List[str] = field(default_factory=list)
    description: str = ""

    @property
    def entity_id(self) -> str:
        return hashlib.md5(self.title.lower().encode()).hexdigest()[:12]


@dataclass
class Relation:
    source: str
    target: str
    weight: float = 1.0
    description: str = ""

    @property
    def relation_id(self) -> str:
        key = f"{self.source}|{self.target}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 规则后端
# ---------------------------------------------------------------------------

# 常见停用词，避免将普通词误识别为实体
_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "we", "our", "they", "their", "he", "she", "his", "her",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "also", "however", "therefore", "thus", "hence",
    "figure", "table", "section", "chapter", "appendix",
    "et", "al", "fig", "eq", "ref",
}

# 科研文献中常见的实体模式
_ENTITY_PATTERNS = [
    # 全大写缩写（如 RAG, LLM, NLP）
    (re.compile(r"\b[A-Z]{2,8}\b"), "ACRONYM"),
    # 首字母大写的专有名词序列（如 GraphRAG, Leiden Algorithm）
    (re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"), "CONCEPT"),
    # 驼峰命名（如 GraphRAG, TextUnit）
    (re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"), "TECHNICAL"),
]


def _extract_entities_rule(
    text_units: List[TextUnit],
    min_freq: int = 1,
) -> Dict[str, Entity]:
    """
    基于规则从文本块中抽取实体。

    Returns
    -------
    Dict[str, Entity]
        {entity_title: Entity} 映射
    """
    # 第一遍：收集所有候选实体及其出现的 chunk_id
    candidate_chunks: Dict[str, List[str]] = defaultdict(list)
    candidate_types: Dict[str, str] = {}

    for unit in text_units:
        seen_in_chunk: Set[str] = set()
        for pattern, etype in _ENTITY_PATTERNS:
            for match in pattern.finditer(unit.text):
                title = match.group().strip()
                # 过滤停用词和过短词
                if title.lower() in _STOPWORDS or len(title) < 2:
                    continue
                # 过滤纯数字
                if title.isdigit():
                    continue
                if title not in seen_in_chunk:
                    candidate_chunks[title].append(unit.chunk_id)
                    seen_in_chunk.add(title)
                if title not in candidate_types:
                    candidate_types[title] = etype

    # 第二遍：过滤低频实体
    entities: Dict[str, Entity] = {}
    for title, chunk_ids in candidate_chunks.items():
        if len(set(chunk_ids)) >= min_freq:
            entities[title] = Entity(
                title=title,
                entity_type=candidate_types.get(title, "UNKNOWN"),
                chunk_ids=list(set(chunk_ids)),
                description=f"{title} (出现于 {len(set(chunk_ids))} 个文本块)",
            )

    return entities


def _extract_relations_cooccurrence(
    text_units: List[TextUnit],
    entities: Dict[str, Entity],
    window: int = 3,
) -> List[Relation]:
    """
    基于共现窗口构建实体关系。

    在同一个 chunk 内，若两个实体在 window 句以内共现，
    则建立关系，权重为共现次数。

    Parameters
    ----------
    window : int
        共现窗口大小（句子数）
    """
    entity_titles = set(entities.keys())
    cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

    for unit in text_units:
        # 将 chunk 分句
        sentences = re.split(r"(?<=[.!?])\s+|\n", unit.text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # 在窗口内查找共现实体对
        for i, sent in enumerate(sentences):
            window_sents = sentences[i: i + window]
            window_text = " ".join(window_sents)

            # 找出窗口内出现的所有实体
            present: List[str] = []
            for title in entity_titles:
                if title in window_text:
                    present.append(title)

            # 两两建立共现关系
            for j in range(len(present)):
                for k in range(j + 1, len(present)):
                    a, b = sorted([present[j], present[k]])
                    cooccurrence[(a, b)] += 1

    relations: List[Relation] = []
    for (src, tgt), count in cooccurrence.items():
        relations.append(Relation(
            source=src,
            target=tgt,
            weight=float(count),
            description=f"{src} 与 {tgt} 共现 {count} 次",
        ))

    return relations


# ---------------------------------------------------------------------------
# spaCy 后端
# ---------------------------------------------------------------------------

def _extract_entities_spacy(
    text_units: List[TextUnit],
    model_name: str,
    min_freq: int = 1,
) -> Dict[str, Entity]:
    """使用 spaCy NER 抽取实体。"""
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "spaCy 后端需要安装 spacy：\n"
            "  pip install spacy\n"
            f"  python -m spacy download {model_name}"
        )

    try:
        nlp = spacy.load(model_name)
    except OSError:
        raise OSError(
            f"spaCy 模型 '{model_name}' 未找到，请运行：\n"
            f"  python -m spacy download {model_name}"
        )

    # spaCy 支持的实体类型白名单（科研场景相关）
    ALLOWED_TYPES = {
        "PERSON", "ORG", "GPE", "PRODUCT", "WORK_OF_ART",
        "LAW", "LANGUAGE", "EVENT", "FAC", "LOC",
    }

    candidate_chunks: Dict[str, List[str]] = defaultdict(list)
    candidate_types: Dict[str, str] = {}

    for unit in text_units:
        doc = nlp(unit.text[:10000])  # spaCy 有长度限制
        seen: Set[str] = set()
        for ent in doc.ents:
            title = ent.text.strip()
            if ent.label_ not in ALLOWED_TYPES:
                continue
            if len(title) < 2 or title.lower() in _STOPWORDS:
                continue
            if title not in seen:
                candidate_chunks[title].append(unit.chunk_id)
                seen.add(title)
            if title not in candidate_types:
                candidate_types[title] = ent.label_

    entities: Dict[str, Entity] = {}
    for title, chunk_ids in candidate_chunks.items():
        if len(set(chunk_ids)) >= min_freq:
            entities[title] = Entity(
                title=title,
                entity_type=candidate_types.get(title, "UNKNOWN"),
                chunk_ids=list(set(chunk_ids)),
                description=f"{title} ({candidate_types.get(title, 'UNKNOWN')})",
            )

    return entities


# ---------------------------------------------------------------------------
# DataFrame 转换
# ---------------------------------------------------------------------------

def entities_to_dataframe(entities: Dict[str, Entity]) -> pd.DataFrame:
    """将实体字典转换为 GraphRAG 兼容的 entities DataFrame。"""
    records = []
    for title, ent in entities.items():
        records.append({
            "id": ent.entity_id,
            "title": ent.title,
            "type": ent.entity_type,
            "description": ent.description,
            "text_unit_ids": ent.chunk_ids,
        })
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["id", "title", "type", "description", "text_unit_ids"]
    )


def relations_to_dataframe(relations: List[Relation]) -> pd.DataFrame:
    """将关系列表转换为 GraphRAG 兼容的 relationships DataFrame。"""
    records = []
    for rel in relations:
        records.append({
            "id": rel.relation_id,
            "source": rel.source,
            "target": rel.target,
            "weight": rel.weight,
            "description": rel.description,
        })
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["id", "source", "target", "weight", "description"]
    )


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def extract(
    text_units: List[TextUnit],
    config: ExtractionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 TextUnit 列表中抽取实体和关系，返回两个 DataFrame。

    Parameters
    ----------
    text_units : List[TextUnit]
        分块后的文本单元列表
    config : ExtractionConfig
        抽取配置

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (entities_df, relationships_df)
        格式与 GraphRAG 的 entities.parquet / relationships.parquet 兼容
    """
    if not text_units:
        return (
            pd.DataFrame(columns=["id", "title", "type", "description", "text_unit_ids"]),
            pd.DataFrame(columns=["id", "source", "target", "weight", "description"]),
        )

    # 实体抽取
    if config.backend == "spacy":
        entities = _extract_entities_spacy(
            text_units, config.spacy_model, config.min_entity_freq
        )
    else:
        entities = _extract_entities_rule(text_units, config.min_entity_freq)

    # 关系抽取（两种后端都使用共现方法）
    relations = _extract_relations_cooccurrence(
        text_units, entities, config.cooccurrence_window
    )

    entities_df = entities_to_dataframe(entities)
    relationships_df = relations_to_dataframe(relations)

    return entities_df, relationships_df
