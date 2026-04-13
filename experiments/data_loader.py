"""
experiments/data_loader.py
--------------------------
MultiHop-RAG 数据集加载器。

数据集结构：
  MultiHopRAG.json  — QA 对，每条含 query / answer / question_type / supporting_evidence
  corpus.json       — 知识库，每篇文章含 id / title / body / source / published_at

supporting_evidence 存储的是文章标题字符串，通过标题在 corpus 中定位对应文档。

下载方式（二选一）：
  1. HuggingFace: huggingface-cli download yixuantt/MultiHopRAG --repo-type dataset
  2. GitHub:      git clone https://github.com/yixuantt/MultiHop-RAG
                  数据在 dataset/ 目录下
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class CorpusDoc:
    """知识库中的一篇文章。"""
    doc_id: str
    title: str
    body: str
    source: str = ""
    published_at: str = ""


@dataclass
class QAPair:
    """一条 QA 对。"""
    query: str
    answer: str
    question_type: str
    supporting_titles: List[str] = field(default_factory=list)   # 文章标题列表
    supporting_doc_ids: List[str] = field(default_factory=list)  # 对应的 doc_id（加载后填充）


@dataclass
class MultiHopDataset:
    """完整的 MultiHop-RAG 数据集。"""
    corpus: List[CorpusDoc]
    qa_pairs: List[QAPair]
    title_to_doc: Dict[str, CorpusDoc] = field(default_factory=dict)

    def __post_init__(self):
        # 构建标题 → 文档的索引（用于 supporting_evidence 解析）
        self.title_to_doc = {doc.title: doc for doc in self.corpus}
        # 填充 supporting_doc_ids（doc_id 即 title）
        for qa in self.qa_pairs:
            qa.supporting_doc_ids = [
                t  # doc_id == title
                for t in qa.supporting_titles
                if t in self.title_to_doc
            ]

    @property
    def num_docs(self) -> int:
        return len(self.corpus)

    @property
    def num_qa(self) -> int:
        return len(self.qa_pairs)

    def filter_by_type(self, question_type: str) -> "MultiHopDataset":
        """按问题类型过滤 QA 对。"""
        filtered = [qa for qa in self.qa_pairs if qa.question_type == question_type]
        return MultiHopDataset(corpus=self.corpus, qa_pairs=filtered)

    def subset(self, n: int, seed: int = 42) -> "MultiHopDataset":
        """随机取 n 条 QA 对（用于快速实验）。"""
        rng = random.Random(seed)
        sampled = rng.sample(self.qa_pairs, min(n, len(self.qa_pairs)))
        return MultiHopDataset(corpus=self.corpus, qa_pairs=sampled)


# ---------------------------------------------------------------------------
# 加载函数
# ---------------------------------------------------------------------------

def _load_corpus(corpus_path: str) -> List[CorpusDoc]:
    """加载 corpus.json。"""
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for i, item in enumerate(raw):
        body = item.get("body") or item.get("text") or item.get("content") or ""
        title = item.get("title") or item.get("name") or f"doc_{i}"
        # 用 title 作为 doc_id（因为 QA 的 evidence_list 通过 title 关联）
        doc_id = str(title)
        docs.append(CorpusDoc(
            doc_id=doc_id,
            title=str(title),
            body=str(body),
            source=str(item.get("source", "")),
            published_at=str(item.get("published_at", "")),
        ))
    return docs


def _load_qa(qa_path: str) -> List[QAPair]:
    """加载 MultiHopRAG.json。"""
    with open(qa_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    qa_pairs = []
    for item in raw:
        # 真实字段名为 evidence_list，每条是含 title 的 dict
        evidence_raw = (
            item.get("evidence_list")
            or item.get("supporting_evidence")
            or item.get("required_evidence")
            or item.get("evidence")
            or []
        )
        # 兼容两种格式：字符串列表 或 dict 列表（含 title 字段）
        if isinstance(evidence_raw, str):
            import ast
            try:
                evidence_raw = ast.literal_eval(evidence_raw)
            except Exception:
                evidence_raw = []

        supporting_titles = []
        for ev in evidence_raw:
            if isinstance(ev, dict):
                t = ev.get("title", "")
                if t:
                    supporting_titles.append(str(t))
            elif isinstance(ev, str):
                supporting_titles.append(ev)

        qa_pairs.append(QAPair(
            query=str(item.get("query", "")),
            answer=str(item.get("answer", "")),
            question_type=str(item.get("question_type", "unknown")),
            supporting_titles=supporting_titles,
        ))
    return qa_pairs


def load_multihop_dataset(
    data_dir: str,
    corpus_filename: str = "corpus.json",
    qa_filename: str = "MultiHopRAG.json",
) -> MultiHopDataset:
    """
    从本地目录加载 MultiHop-RAG 数据集。

    Parameters
    ----------
    data_dir : str
        数据集目录（包含 corpus.json 和 MultiHopRAG.json）
    """
    data_dir = Path(data_dir)
    corpus_path = data_dir / corpus_filename
    qa_path = data_dir / qa_filename

    if not corpus_path.exists():
        raise FileNotFoundError(
            f"找不到 corpus 文件：{corpus_path}\n"
            "请先下载 MultiHop-RAG 数据集：\n"
            "  git clone https://github.com/yixuantt/MultiHop-RAG\n"
            "  数据在 MultiHop-RAG/dataset/ 目录下"
        )
    if not qa_path.exists():
        raise FileNotFoundError(f"找不到 QA 文件：{qa_path}")

    print(f"  加载 corpus: {corpus_path}")
    corpus = _load_corpus(str(corpus_path))
    print(f"  加载 QA 对: {qa_path}")
    qa_pairs = _load_qa(str(qa_path))

    dataset = MultiHopDataset(corpus=corpus, qa_pairs=qa_pairs)
    print(f"  ✓ 数据集加载完成：{dataset.num_docs} 篇文章，{dataset.num_qa} 条 QA 对")
    return dataset


def try_download_dataset(target_dir: str) -> bool:
    """
    尝试通过 HuggingFace datasets 库下载数据集。
    若下载成功，将文件保存到 target_dir，返回 True；否则返回 False。
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
        print("  通过 HuggingFace 下载 MultiHop-RAG...")

        # 下载 QA 对
        ds = load_dataset("yixuantt/MultiHopRAG", split="test", trust_remote_code=True)
        qa_records = [dict(row) for row in ds]
        qa_path = target / "MultiHopRAG.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa_records, f, ensure_ascii=False, indent=2)
        print(f"  ✓ QA 对已保存：{qa_path}（{len(qa_records)} 条）")

        # corpus 需要单独下载
        try:
            corpus_ds = load_dataset(
                "yixuantt/MultiHopRAG",
                data_files="corpus.json",
                trust_remote_code=True,
            )
            corpus_records = [dict(row) for row in corpus_ds["train"]]
            corpus_path = target / "corpus.json"
            with open(corpus_path, "w", encoding="utf-8") as f:
                json.dump(corpus_records, f, ensure_ascii=False, indent=2)
            print(f"  ✓ Corpus 已保存：{corpus_path}（{len(corpus_records)} 篇）")
        except Exception as e:
            print(f"  [提示] corpus 下载失败（{e}），请手动从 GitHub 获取")
            return False

        return True

    except ImportError:
        print("  [提示] 未安装 datasets 库，请运行：pip install datasets")
        return False
    except Exception as e:
        print(f"  [提示] HuggingFace 下载失败：{e}")
        return False


def corpus_to_text_units(corpus: List[CorpusDoc]) -> List[dict]:
    """
    将 CorpusDoc 列表转换为 TextUnit 格式（供 URetriever 使用）。
    每篇文章作为一个 TextUnit，doc_id 作为 chunk_id。
    """
    return [
        {
            "chunk_id": doc.doc_id,
            "text": doc.body,
            "doc_title": doc.title,
            "source": doc.source,
            "published_at": doc.published_at,
        }
        for doc in corpus
    ]
