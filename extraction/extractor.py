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

# ---------------------------------------------------------------------------
# 实体消歧（Entity Disambiguation）
# ---------------------------------------------------------------------------

def _normalize_entity_title(title: str) -> str:
    """
    标准化实体名称，用于消歧。
    例如："Graph RAG" 和 "GraphRAG" 应被视为同一实体。
    """
    # 去除空格、连字符，统一小写
    return re.sub(r"[\s\-_]+", "", title).lower()


def _disambiguate_entities(
    entities: Dict[str, "Entity"],
) -> Dict[str, "Entity"]:
    """
    实体消歧：将写法不同但指向同一实体的名称合并。

    策略：
    1. 标准化所有实体名称（去空格、小写）
    2. 将标准化后相同的实体合并，保留出现频率最高的写法作为规范名
    3. 合并 chunk_ids（取并集）

    Parameters
    ----------
    entities : Dict[str, Entity]
        原始实体字典

    Returns
    -------
    Dict[str, Entity]
        消歧后的实体字典
    """
    # 按标准化名称分组
    groups: Dict[str, List[str]] = defaultdict(list)
    for title in entities:
        norm = _normalize_entity_title(title)
        groups[norm].append(title)

    merged: Dict[str, "Entity"] = {}
    for norm, titles in groups.items():
        if len(titles) == 1:
            merged[titles[0]] = entities[titles[0]]
            continue

        # 选出现 chunk 数最多的写法作为规范名
        canonical = max(titles, key=lambda t: len(entities[t].chunk_ids))
        canonical_entity = entities[canonical]

        # 合并所有写法的 chunk_ids
        all_chunks: Set[str] = set(canonical_entity.chunk_ids)
        for title in titles:
            if title != canonical:
                all_chunks.update(entities[title].chunk_ids)

        merged[canonical] = Entity(
            title=canonical,
            entity_type=canonical_entity.entity_type,
            chunk_ids=list(all_chunks),
            description=(
                f"{canonical}（含别名：{', '.join(t for t in titles if t != canonical)}）"
                f" 出现于 {len(all_chunks)} 个文本块"
                if len(titles) > 1
                else canonical_entity.description
            ),
        )

    return merged

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

    # 第三遍：实体消歧
    entities = _disambiguate_entities(entities)
    return entities


def _extract_relations_cooccurrence(
    text_units: List[TextUnit],
    entities: Dict[str, Entity],
    window: int = 3,
) -> List[Relation]:
    """
    基于共现窗口构建实体关系（倒排索引优化版）。

    策略：先在 chunk 级别过滤出现的实体子集，再在句子窗口内做共现统计，
    避免 O(|entities| × |sentences|) 的暴力遍历。

    Parameters
    ----------
    window : int
        共现窗口大小（句子数）
    """
    entity_titles = list(entities.keys())
    if not entity_titles:
        return []

    # 按长度降序排列，优先匹配更长的实体名（避免短名遮蔽长名）
    entity_titles_sorted = sorted(entity_titles, key=len, reverse=True)

    cooccurrence: Dict[Tuple[str, str], int] = defaultdict(int)

    for unit in text_units:
        # 先过滤出在整个 chunk 中出现的实体（大幅减少后续逐句匹配的候选集）
        chunk_text = unit.text
        chunk_entities = [t for t in entity_titles_sorted if t in chunk_text]

        if len(chunk_entities) < 2:
            continue

        # 将 chunk 分句
        sentences = re.split(r"(?<=[.!?])\s+|\n", chunk_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            continue

        # 逐句窗口统计共现
        for i in range(len(sentences)):
            window_text = " ".join(sentences[i: i + window])
            present: List[str] = [t for t in chunk_entities if t in window_text]

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
        entities = _disambiguate_entities(entities)
    else:
        entities = _extract_entities_rule(text_units, config.min_entity_freq)
        # rule 后端已在内部完成消歧，此处无需重复

    # 关系抽取（两种后端都使用共现方法）
    relations = _extract_relations_cooccurrence(
        text_units, entities, config.cooccurrence_window
    )

    entities_df = entities_to_dataframe(entities)
    relationships_df = relations_to_dataframe(relations)

    return entities_df, relationships_df
