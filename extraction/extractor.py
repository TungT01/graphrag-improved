"""
extraction/extractor.py
-----------------------
实体与关系抽取模块（v2：物理优先架构）。

核心变更（相比 v1）：
  - 移除实体消解（_disambiguate_entities）：不再合并同名实体
  - 移除共现窗口关系抽取：改用 spaCy 依存句法三元组提取
  - 实体节点 ID 包含完整物理路径：{sent_id}-{entity_name_normalized}
  - 同一实体在不同句子里出现 = 不同节点，底层图保持物理纯净

代词处理策略（方案一）：
  主语或宾语为代词（PRP / PRP$）时直接跳过，不生成节点。
  备选升级（方案二）：若召回率不理想，切换至共指消解（coreferee/neuralcoref）。

三元组提取逻辑：
  使用 spaCy 依存句法分析，从每个句子中提取主谓宾结构：
    - 主语（nsubj / nsubjpass）
    - 谓词（动词 token，即主语的 head）
    - 宾语（dobj / pobj / attr / xcomp）
  主宾必须在同一句子内，天然保证物理路径一致。

输出：
  entities DataFrame  : id, title, type, description, sent_id, para_id, doc_id
  relations DataFrame : id, source, target, weight, predicate, description
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from ..data.ingestion import SentenceUnit, TextUnit, get_all_sentences
from ..pipeline_config import ExtractionConfig


# ---------------------------------------------------------------------------
# 内部数据结构
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """
    带物理坐标的实体实例节点。

    与 v1 的根本区别：
      v1 的 Entity 是"概念节点"，一个实体名对应一个节点，携带多个 chunk_id。
      v2 的 Entity 是"实例节点"，同一实体名在不同句子里 = 不同节点，
      节点 ID 包含完整物理路径，保证底层图的物理纯净性。
    """
    title: str           # 实体名称（原始文本，保留大小写）
    entity_type: str     # 实体类型（spaCy NER 标签 或 规则推断类型）
    sent_id: str         # 所在句子 ID：{doc_id}-p{para_idx}-s{sent_idx}
    para_id: str         # 所在段落 ID：{doc_id}-p{para_idx}
    doc_id: str          # 所在文档 ID
    description: str = ""

    @property
    def node_id(self) -> str:
        """
        节点唯一 ID：{sent_id}-{title_normalized}
        包含完整物理路径，同名实体在不同句子里 ID 不同。
        """
        title_norm = re.sub(r"[^a-z0-9]", "_", self.title.lower())[:32]
        return f"{self.sent_id}-{title_norm}"

    @property
    def entity_id(self) -> str:
        """node_id 的别名，兼容旧接口。"""
        return self.node_id


@dataclass
class Relation:
    """
    实体间的有向关系（来自三元组提取）。

    source / target 均为 Entity.node_id（包含物理路径），
    保证关系天然是 chunk 内部的。
    """
    source_node_id: str   # 主语实体的 node_id
    target_node_id: str   # 宾语实体的 node_id
    predicate: str        # 谓词（动词原形或文本）
    weight: float = 1.0
    description: str = ""
    sent_id: str = ""     # 来源句子 ID

    @property
    def relation_id(self) -> str:
        key = f"{self.source_node_id}|{self.predicate}|{self.target_node_id}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# spaCy 依存句法三元组提取
# ---------------------------------------------------------------------------

# 代词词性标签（方案一：跳过代词节点）
_PRONOUN_TAGS = {"PRP", "PRP$", "WP", "WP$"}

# 主语依存关系标签
_SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"}

# 宾语依存关系标签
_OBJECT_DEPS = {"dobj", "pobj", "iobj", "attr", "xcomp", "acomp", "oprd"}

# 停用词（避免将普通词误识别为实体）
_STOPWORDS = {
    "the", "a", "an", "this", "that", "these", "those", "it", "its",
    "we", "our", "they", "their", "he", "she", "his", "her",
    "i", "me", "my", "you", "your", "us",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    "also", "however", "therefore", "thus", "hence",
    "figure", "table", "section", "chapter",
}


def _is_valid_entity_span(span_text: str, pos_tag: str) -> bool:
    """
    判断一个 span 是否是有效的实体候选。
    过滤代词、停用词、过短词、纯数字。
    """
    text = span_text.strip()
    if not text or len(text) < 2:
        return False
    if pos_tag in _PRONOUN_TAGS:
        return False
    if text.lower() in _STOPWORDS:
        return False
    if text.isdigit():
        return False
    return True


def _get_entity_span(token, doc) -> str:
    """
    获取 token 对应的实体 span 文本。
    优先使用 noun chunk（名词短语），否则使用 token 本身。
    """
    # 尝试找到包含该 token 的 noun chunk
    for chunk in doc.noun_chunks:
        if chunk.root == token:
            return chunk.text.strip()
    return token.text.strip()


def _extract_triples_from_sentence(
    sent,
    doc,
    sent_unit: SentenceUnit,
) -> Tuple[List[Entity], List[Relation]]:
    """
    从单个 spaCy 句子对象中提取三元组，生成实体节点和关系边。

    策略：
    1. 遍历句子中的所有动词 token（作为谓词候选）
    2. 找到该动词的主语（nsubj 等）和宾语（dobj 等）
    3. 过滤代词、停用词
    4. 生成实体节点（继承 sent_id）和关系边

    Parameters
    ----------
    sent : spaCy Span
        句子对象
    doc : spaCy Doc
        完整文档对象（用于 noun chunk 查找）
    sent_unit : SentenceUnit
        对应的物理句子单元

    Returns
    -------
    Tuple[List[Entity], List[Relation]]
        (实体列表, 关系列表)
    """
    entities: Dict[str, Entity] = {}   # node_id → Entity（去重）
    relations: List[Relation] = []

    for token in sent:
        # 只处理动词作为谓词
        if token.pos_ not in ("VERB", "AUX"):
            continue

        # 找主语
        subjects = []
        for child in token.children:
            if child.dep_ in _SUBJECT_DEPS:
                span_text = _get_entity_span(child, doc)
                if _is_valid_entity_span(span_text, child.tag_):
                    subjects.append((child, span_text))

        # 找宾语
        objects = []
        for child in token.children:
            if child.dep_ in _OBJECT_DEPS:
                span_text = _get_entity_span(child, doc)
                if _is_valid_entity_span(span_text, child.tag_):
                    objects.append((child, span_text))

        if not subjects or not objects:
            continue

        # 谓词文本：优先使用 lemma（动词原形），保留语义
        predicate = token.lemma_ if token.lemma_ != "-PRON-" else token.text

        # 为每对 (主语, 宾语) 生成三元组
        for subj_token, subj_text in subjects:
            for obj_token, obj_text in objects:
                # 推断实体类型（优先 spaCy NER，其次词性）
                subj_type = _infer_entity_type(subj_token, doc)
                obj_type = _infer_entity_type(obj_token, doc)

                # 创建主语实体节点
                subj_entity = Entity(
                    title=subj_text,
                    entity_type=subj_type,
                    sent_id=sent_unit.sent_id,
                    para_id=sent_unit.para_id,
                    doc_id=sent_unit.doc_id,
                    description=f"{subj_text} [{subj_type}] in: {sent_unit.text[:80]}",
                )
                if subj_entity.node_id not in entities:
                    entities[subj_entity.node_id] = subj_entity

                # 创建宾语实体节点
                obj_entity = Entity(
                    title=obj_text,
                    entity_type=obj_type,
                    sent_id=sent_unit.sent_id,
                    para_id=sent_unit.para_id,
                    doc_id=sent_unit.doc_id,
                    description=f"{obj_text} [{obj_type}] in: {sent_unit.text[:80]}",
                )
                if obj_entity.node_id not in entities:
                    entities[obj_entity.node_id] = obj_entity

                # 创建语义关系边
                rel = Relation(
                    source_node_id=subj_entity.node_id,
                    target_node_id=obj_entity.node_id,
                    predicate=predicate,
                    weight=1.0,
                    description=f"{subj_text} --{predicate}--> {obj_text}",
                    sent_id=sent_unit.sent_id,
                )
                relations.append(rel)

    return list(entities.values()), relations


def _infer_entity_type(token, doc) -> str:
    """
    推断实体类型：优先使用 spaCy NER 标签，其次根据词性推断。
    """
    # 检查 token 是否在 NER 结果中
    for ent in doc.ents:
        if ent.start <= token.i < ent.end:
            return ent.label_

    # 根据词性推断
    if token.pos_ == "PROPN":
        return "PROPER_NOUN"
    elif token.pos_ == "NOUN":
        return "NOUN"
    else:
        return "UNKNOWN"


def _extract_physical_structure_edges(
    entities: List[Entity],
) -> List[Relation]:
    """
    生成物理结构边：同一句子内的所有实体节点两两相连。

    这些边不携带语义信息，只反映物理共现关系，
    权重 1.0，帮助 Leiden 聚类识别同句实体的物理亲密性。

    Parameters
    ----------
    entities : List[Entity]
        同一句子内的实体列表

    Returns
    -------
    List[Relation]
        物理结构边列表
    """
    edges: List[Relation] = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            e1, e2 = entities[i], entities[j]
            if e1.sent_id != e2.sent_id:
                continue  # 严格限制在同句内
            edges.append(Relation(
                source_node_id=e1.node_id,
                target_node_id=e2.node_id,
                predicate="co_occurs",
                weight=1.0,
                description=f"{e1.title} co-occurs with {e2.title}",
                sent_id=e1.sent_id,
            ))
    return edges


# ---------------------------------------------------------------------------
# 主提取函数
# ---------------------------------------------------------------------------

def _extract_with_spacy(
    text_units: List[TextUnit],
    spacy_model: str = "en_core_web_sm",
    min_entity_freq: int = 1,
) -> Tuple[Dict[str, Entity], List[Relation]]:
    """
    使用 spaCy 依存句法分析从所有句子中提取实体和关系。

    注意：不做实体消解，同名实体在不同句子里是不同节点。

    Parameters
    ----------
    text_units : List[TextUnit]
        段落级 TextUnit 列表（含 SentenceUnit）
    spacy_model : str
        spaCy 模型名称
    min_entity_freq : int
        实体最低出现次数（按 node_id 计，v2 中通常为 1）

    Returns
    -------
    Tuple[Dict[str, Entity], List[Relation]]
        ({node_id: Entity}, [Relation])
    """
    try:
        import spacy
    except ImportError:
        raise ImportError(
            "三元组提取需要安装 spaCy：\n"
            "  pip install spacy\n"
            f"  python -m spacy download {spacy_model}"
        )

    try:
        nlp = spacy.load(spacy_model)
    except OSError:
        raise OSError(
            f"spaCy 模型 '{spacy_model}' 未找到，请运行：\n"
            f"  python -m spacy download {spacy_model}"
        )

    all_entities: Dict[str, Entity] = {}
    all_relations: List[Relation] = []

    # 收集所有句子
    all_sentences = get_all_sentences(text_units)

    for sent_unit in all_sentences:
        if not sent_unit.text.strip():
            continue

        # spaCy 解析
        doc = nlp(sent_unit.text[:5000])

        # 提取三元组（语义边）
        sent_entities: List[Entity] = []
        sent_relations: List[Relation] = []

        for spacy_sent in doc.sents:
            s_ents, s_rels = _extract_triples_from_sentence(
                spacy_sent, doc, sent_unit
            )
            sent_entities.extend(s_ents)
            sent_relations.extend(s_rels)

        # 去重合并实体（同一句子内同名实体只保留一个节点）
        for ent in sent_entities:
            if ent.node_id not in all_entities:
                all_entities[ent.node_id] = ent

        # 添加语义边
        all_relations.extend(sent_relations)

        # 添加物理结构边（同句内所有实体两两相连）
        unique_sent_entities = [
            all_entities[eid]
            for eid in {e.node_id for e in sent_entities}
            if eid in all_entities
        ]
        if len(unique_sent_entities) >= 2:
            phys_edges = _extract_physical_structure_edges(unique_sent_entities)
            all_relations.extend(phys_edges)

    # 合并重复关系（相同 source-predicate-target 的边权重累加）
    merged_relations = _merge_relations(all_relations)

    return all_entities, merged_relations


def _merge_relations(relations: List[Relation]) -> List[Relation]:
    """
    合并重复关系：相同 (source, predicate, target) 的边权重累加。
    """
    merged: Dict[str, Relation] = {}
    for rel in relations:
        key = f"{rel.source_node_id}|{rel.predicate}|{rel.target_node_id}"
        if key in merged:
            merged[key].weight += rel.weight
        else:
            merged[key] = Relation(
                source_node_id=rel.source_node_id,
                target_node_id=rel.target_node_id,
                predicate=rel.predicate,
                weight=rel.weight,
                description=rel.description,
                sent_id=rel.sent_id,
            )
    return list(merged.values())


# ---------------------------------------------------------------------------
# DataFrame 转换
# ---------------------------------------------------------------------------

def entities_to_dataframe(entities: Dict[str, Entity]) -> pd.DataFrame:
    """
    将实体字典转换为 DataFrame。

    列说明：
      id          : node_id（含物理路径）
      title       : 实体名称
      type        : 实体类型
      description : 描述
      sent_id     : 句子 ID（物理主锚点，精确到句子级）
      para_id     : 段落 ID
      doc_id      : 文档 ID
      text_unit_ids : [sent_id]（兼容旧接口，单元素列表）
      primary_chunk_id : sent_id（兼容旧接口）
    """
    records = []
    for node_id, ent in entities.items():
        records.append({
            "id": node_id,
            "title": ent.title,
            "type": ent.entity_type,
            "description": ent.description,
            "sent_id": ent.sent_id,
            "para_id": ent.para_id,
            "doc_id": ent.doc_id,
            # 兼容旧接口
            "text_unit_ids": [ent.sent_id],
            "primary_chunk_id": ent.sent_id,
        })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "id", "title", "type", "description",
        "sent_id", "para_id", "doc_id",
        "text_unit_ids", "primary_chunk_id",
    ])


def relations_to_dataframe(relations: List[Relation]) -> pd.DataFrame:
    """
    将关系列表转换为 DataFrame。

    列说明：
      id          : relation_id
      source      : 主语实体 node_id
      target      : 宾语实体 node_id
      weight      : 关系权重（重复出现累加）
      predicate   : 谓词（动词原形）
      description : 描述
      sent_id     : 来源句子 ID
    """
    records = []
    for rel in relations:
        records.append({
            "id": rel.relation_id,
            "source": rel.source_node_id,
            "target": rel.target_node_id,
            "weight": rel.weight,
            "predicate": rel.predicate,
            "description": rel.description,
            "sent_id": rel.sent_id,
        })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "id", "source", "target", "weight", "predicate", "description", "sent_id",
    ])


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def extract(
    text_units: List[TextUnit],
    config: ExtractionConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从 TextUnit 列表中提取实体和关系，返回两个 DataFrame。

    v2 变更：
      - 统一使用 spaCy 依存句法三元组提取（移除规则后端）
      - 不做实体消解，节点 ID 包含完整物理路径
      - 输出 entities_df 新增 sent_id / para_id / doc_id 列

    Parameters
    ----------
    text_units : List[TextUnit]
        段落级 TextUnit 列表（含 SentenceUnit，由 ingestion 模块生成）
    config : ExtractionConfig
        抽取配置

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (entities_df, relationships_df)
    """
    if not text_units:
        return (
            pd.DataFrame(columns=[
                "id", "title", "type", "description",
                "sent_id", "para_id", "doc_id",
                "text_unit_ids", "primary_chunk_id",
            ]),
            pd.DataFrame(columns=[
                "id", "source", "target", "weight",
                "predicate", "description", "sent_id",
            ]),
        )

    # 检查 TextUnit 是否包含 SentenceUnit（v2 格式）
    has_sentences = any(len(unit.sentences) > 0 for unit in text_units)
    if not has_sentences:
        raise ValueError(
            "TextUnit 列表不包含 SentenceUnit，请使用 v2 版本的 ingestion 模块生成数据。\n"
            "确保 data/ingestion.py 已更新为三级 ID 切分版本。"
        )

    spacy_model = config.spacy_model if hasattr(config, "spacy_model") else "en_core_web_sm"

    entities, relations = _extract_with_spacy(
        text_units,
        spacy_model=spacy_model,
        min_entity_freq=config.min_entity_freq,
    )

    entities_df = entities_to_dataframe(entities)
    relationships_df = relations_to_dataframe(relations)

    return entities_df, relationships_df
