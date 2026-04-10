"""
proposition/transformer.py
--------------------------
命题转换（Proposition Transfer）预处理模块。

核心功能：
  1. 共指消解（Coreference Resolution）
     将代词和指代表达替换为其所指的实体，
     使每个命题在脱离上下文后仍能独立理解。

  2. 命题原子化（Proposition Atomization）
     将复合句拆解为多个原子命题，
     每个命题只表达一个独立的事实。

实现策略：
  - 共指消解：基于规则的简单策略（代词替换为上一句主语）
    可选：spaCy neuralcoref 或 fastcoref（需额外安装）
  - 命题原子化：基于句法规则（连词分割、从句提取）
    可选：基于 LLM 的高质量拆解

设计原则：
  - 无 LLM 依赖的默认实现，保证开箱即用
  - 可插拔的后端接口，支持升级到更高质量的实现
  - 处理结果保留原始 chunk_id，维护物理锚点追踪
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class Proposition:
    """
    原子命题：一个独立的、可验证的事实陈述。

    Attributes
    ----------
    text        : 命题文本（已完成共指消解）
    source_text : 原始句子文本（未处理）
    chunk_id    : 来源文本块的 ID（物理锚点）
    prop_index  : 在原始 chunk 中的命题序号
    """
    text: str
    source_text: str
    chunk_id: str
    prop_index: int = 0
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 共指消解（规则后端）
# ---------------------------------------------------------------------------

# 主格代词 → 宾格/所有格映射（用于识别代词）
_SUBJECT_PRONOUNS = {
    "he", "she", "it", "they", "we", "i", "you",
    "his", "her", "its", "their", "our", "my", "your",
    "him", "them", "us", "me",
    "this", "that", "these", "those",
    "which", "who", "whom",
}

# 句子分割正则
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# 主语提取：匹配句首的名词短语（简化版）
_SUBJECT_PATTERN = re.compile(
    r"^(?:The |A |An |This |That |These |Those )?([A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*)"
)


def _extract_subject(sentence: str) -> Optional[str]:
    """
    从句子中提取主语（简化规则：句首大写名词短语）。

    Parameters
    ----------
    sentence : str
        输入句子

    Returns
    -------
    Optional[str]
        提取到的主语，若无法提取则返回 None
    """
    sentence = sentence.strip()
    match = _SUBJECT_PATTERN.match(sentence)
    if match:
        subject = match.group(1).strip()
        # 过滤过短或全大写的（可能是缩写）
        if len(subject) > 2 and not subject.isupper():
            return subject
    return None


def resolve_coreferences_rule(text: str) -> str:
    """
    基于规则的共指消解：将代词替换为上一句的主语。

    策略：
    - 遍历句子序列
    - 维护一个"当前主语"变量
    - 遇到代词时，替换为当前主语
    - 遇到新的命名实体时，更新当前主语

    Parameters
    ----------
    text : str
        输入文本（可包含多个句子）

    Returns
    -------
    str
        共指消解后的文本
    """
    sentences = _SENT_SPLIT.split(text.strip())
    resolved_sentences = []
    current_subject: Optional[str] = None

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # 尝试提取当前句子的主语，更新 current_subject
        new_subject = _extract_subject(sent)
        if new_subject:
            current_subject = new_subject

        # 若有当前主语，替换句中的代词
        if current_subject:
            resolved = _replace_pronouns(sent, current_subject)
        else:
            resolved = sent

        resolved_sentences.append(resolved)

    return " ".join(resolved_sentences)


def _replace_pronouns(sentence: str, subject: str) -> str:
    """
    将句子中的代词替换为指定主语。

    只替换句子中间位置的代词（不替换句首，避免破坏句子结构）。
    """
    words = sentence.split()
    if len(words) <= 1:
        return sentence

    result = [words[0]]  # 保留句首词不变
    for word in words[1:]:
        # 去除标点后检查是否为代词
        clean = word.rstrip(".,;:!?").lower()
        if clean in _SUBJECT_PRONOUNS and len(clean) <= 4:
            # 保留原词的标点
            punct = word[len(clean):]
            result.append(subject + punct)
        else:
            result.append(word)

    return " ".join(result)


def resolve_coreferences_spacy(text: str, nlp=None) -> str:
    """
    使用 spaCy + neuralcoref 进行共指消解（高质量版本）。

    需要安装：
        pip install spacy
        pip install https://github.com/huggingface/neuralcoref/archive/master.zip

    Parameters
    ----------
    text : str
        输入文本
    nlp : spacy.Language, optional
        已加载的 spaCy 模型（含 neuralcoref 组件）

    Returns
    -------
    str
        共指消解后的文本，若 neuralcoref 不可用则降级到规则方法
    """
    try:
        import spacy
        if nlp is None:
            nlp = spacy.load("en_core_web_sm")

        # 检查是否有 neuralcoref
        if not nlp.has_pipe("neuralcoref"):
            try:
                import neuralcoref
                neuralcoref.add_to_pipe(nlp)
            except ImportError:
                return resolve_coreferences_rule(text)

        doc = nlp(text)
        if doc._.has_coref:
            return doc._.coref_resolved
        return text

    except (ImportError, OSError):
        return resolve_coreferences_rule(text)


# ---------------------------------------------------------------------------
# 命题原子化（规则后端）
# ---------------------------------------------------------------------------

# 并列连词，用于拆分复合句
_COORD_CONJUNCTIONS = re.compile(
    r"\s+(?:and|but|or|nor|yet|so|however|therefore|thus|moreover|furthermore|additionally)\s+",
    re.IGNORECASE,
)

# 从句引导词，用于识别从句边界
_SUBORDINATE_MARKERS = re.compile(
    r"\s+(?:which|that|who|whom|whose|where|when|because|since|although|though|while|if|unless)\s+",
    re.IGNORECASE,
)

# 括号内容（通常是补充说明，可单独成命题）
_PARENTHETICAL = re.compile(r"\(([^)]+)\)")


def atomize_sentence_rule(sentence: str) -> List[str]:
    """
    基于规则将复合句拆解为原子命题列表。

    策略：
    1. 提取括号内的补充说明作为独立命题
    2. 按并列连词分割
    3. 按从句边界分割
    4. 过滤过短的片段（< 10 个字符）

    Parameters
    ----------
    sentence : str
        输入句子

    Returns
    -------
    List[str]
        原子命题列表（至少包含原句本身）
    """
    sentence = sentence.strip()
    if not sentence:
        return []

    propositions = []

    # Step 1：提取括号内容
    parentheticals = _PARENTHETICAL.findall(sentence)
    main_sentence = _PARENTHETICAL.sub("", sentence).strip()

    # Step 2：按并列连词分割主句
    parts = _COORD_CONJUNCTIONS.split(main_sentence)

    # Step 3：对每个部分，尝试按从句边界进一步分割
    for part in parts:
        part = part.strip()
        if not part:
            continue

        sub_parts = _SUBORDINATE_MARKERS.split(part)
        for sub in sub_parts:
            sub = sub.strip().rstrip(".,;:")
            if len(sub) >= 10:  # 过滤过短片段
                propositions.append(sub)

    # Step 4：添加括号内容作为独立命题
    for p in parentheticals:
        p = p.strip()
        if len(p) >= 10:
            propositions.append(p)

    # 若拆解结果为空，返回原句
    return propositions if propositions else [sentence]


# ---------------------------------------------------------------------------
# 主转换器
# ---------------------------------------------------------------------------

class PropositionTransformer:
    """
    命题转换器：将文本块转换为原子命题列表。

    Pipeline：
      原始文本 → 共指消解 → 句子分割 → 命题原子化 → Proposition 列表

    Parameters
    ----------
    coref_backend : str
        共指消解后端，"rule"（默认）或 "spacy"
    atomize_backend : str
        命题原子化后端，"rule"（默认）或 "llm"（需自定义 llm_fn）
    llm_fn : Callable, optional
        自定义 LLM 函数，签名为 (sentence: str) -> List[str]
        当 atomize_backend="llm" 时使用
    spacy_model : str
        spaCy 模型名称，当 coref_backend="spacy" 时使用
    """

    def __init__(
        self,
        coref_backend: str = "rule",
        atomize_backend: str = "rule",
        llm_fn: Optional[Callable[[str], List[str]]] = None,
        spacy_model: str = "en_core_web_sm",
    ):
        self.coref_backend = coref_backend
        self.atomize_backend = atomize_backend
        self.llm_fn = llm_fn
        self._spacy_nlp = None

        if coref_backend == "spacy":
            try:
                import spacy
                self._spacy_nlp = spacy.load(spacy_model)
            except (ImportError, OSError):
                print(f"  [警告] spaCy 模型 '{spacy_model}' 未找到，降级到规则共指消解")
                self.coref_backend = "rule"

    def transform(self, text: str, chunk_id: str) -> List[Proposition]:
        """
        将文本块转换为原子命题列表。

        Parameters
        ----------
        text : str
            输入文本块
        chunk_id : str
            来源文本块的 ID（物理锚点）

        Returns
        -------
        List[Proposition]
            原子命题列表
        """
        if not text.strip():
            return []

        # Step 1：共指消解
        if self.coref_backend == "spacy":
            resolved_text = resolve_coreferences_spacy(text, self._spacy_nlp)
        else:
            resolved_text = resolve_coreferences_rule(text)

        # Step 2：句子分割
        sentences = _SENT_SPLIT.split(resolved_text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        # Step 3：命题原子化
        propositions: List[Proposition] = []
        prop_idx = 0

        for sent in sentences:
            if self.atomize_backend == "llm" and self.llm_fn:
                atomic_props = self.llm_fn(sent)
            else:
                atomic_props = atomize_sentence_rule(sent)

            for prop_text in atomic_props:
                prop_text = prop_text.strip()
                if not prop_text:
                    continue
                propositions.append(Proposition(
                    text=prop_text,
                    source_text=sent,
                    chunk_id=chunk_id,
                    prop_index=prop_idx,
                ))
                prop_idx += 1

        return propositions

    def transform_batch(
        self,
        text_units: List[Dict],
    ) -> List[Proposition]:
        """
        批量处理文本块列表。

        Parameters
        ----------
        text_units : List[Dict]
            文本块列表，每个元素包含 chunk_id 和 text 字段

        Returns
        -------
        List[Proposition]
            所有文本块的原子命题列表
        """
        all_propositions: List[Proposition] = []
        for unit in text_units:
            chunk_id = unit.get("chunk_id", "")
            text = unit.get("text", "")
            props = self.transform(text, chunk_id)
            all_propositions.extend(props)
        return all_propositions

    def propositions_to_text_units(
        self,
        propositions: List[Proposition],
    ) -> List[Dict]:
        """
        将命题列表转换回 TextUnit 格式，供后续抽取模块使用。

        每个命题成为一个新的 TextUnit，但保留原始 chunk_id 作为物理锚点。

        Parameters
        ----------
        propositions : List[Proposition]
            原子命题列表

        Returns
        -------
        List[Dict]
            TextUnit 格式的字典列表
        """
        import hashlib
        units = []
        for prop in propositions:
            prop_id = hashlib.md5(
                f"{prop.chunk_id}_{prop.prop_index}_{prop.text[:32]}".encode()
            ).hexdigest()[:16]
            units.append({
                "chunk_id": prop_id,
                "text": prop.text,
                "source_chunk_id": prop.chunk_id,  # 保留物理锚点
                "prop_index": prop.prop_index,
            })
        return units
