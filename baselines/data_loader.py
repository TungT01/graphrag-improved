"""
baselines/data_loader.py
--------------------------
MultiHop-RAG 数据集加载器（baselines 专用版本）。

这是对 graphrag_improved/experiments/data_loader.py 的轻量封装，
使 baselines 目录可以独立使用，无需依赖项目内部模块。

数据集结构：
  MultiHopRAG.json  — QA 对，每条含 query / answer / question_type / evidence_list
  corpus.json       — 知识库，每篇文章含 title / body / source / published_at

下载方式（二选一）：
  1. HuggingFace: huggingface-cli download yixuantt/MultiHopRAG --repo-type dataset
  2. GitHub:      git clone https://github.com/yixuantt/MultiHop-RAG
                  数据在 MultiHop-RAG/dataset/ 目录下

本项目已在以下路径预置数据集：
  /Users/ttung/Desktop/个人学习/data/multihop_rag/
  /Users/ttung/Desktop/个人学习/data/MultiHop-RAG/dataset/
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# 优先使用项目内的 data_loader
# ---------------------------------------------------------------------------

def _try_import_project_loader():
    """尝试导入项目内的 data_loader。"""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    try:
        from graphrag_improved.experiments.data_loader import (
            load_multihop_dataset,
            MultiHopDataset,
            CorpusDoc,
            QAPair,
            corpus_to_text_units,
            try_download_dataset,
        )
        return load_multihop_dataset, MultiHopDataset, CorpusDoc, QAPair, corpus_to_text_units, try_download_dataset
    except ImportError:
        return None


_project_loader = _try_import_project_loader()

if _project_loader is not None:
    # 直接复用项目内的实现
    (
        load_multihop_dataset,
        MultiHopDataset,
        CorpusDoc,
        QAPair,
        corpus_to_text_units,
        try_download_dataset,
    ) = _project_loader
else:
    # 独立实现（当项目模块不可用时）

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
        supporting_titles: List[str] = field(default_factory=list)
        supporting_doc_ids: List[str] = field(default_factory=list)

    @dataclass
    class MultiHopDataset:
        """完整的 MultiHop-RAG 数据集。"""
        corpus: List[CorpusDoc]
        qa_pairs: List[QAPair]
        title_to_doc: Dict[str, CorpusDoc] = field(default_factory=dict)

        def __post_init__(self):
            self.title_to_doc = {doc.title: doc for doc in self.corpus}
            for qa in self.qa_pairs:
                qa.supporting_doc_ids = [
                    t for t in qa.supporting_titles if t in self.title_to_doc
                ]

        @property
        def num_docs(self) -> int:
            return len(self.corpus)

        @property
        def num_qa(self) -> int:
            return len(self.qa_pairs)

        def filter_by_type(self, question_type: str) -> "MultiHopDataset":
            filtered = [qa for qa in self.qa_pairs if qa.question_type == question_type]
            return MultiHopDataset(corpus=self.corpus, qa_pairs=filtered)

        def subset(self, n: int, seed: int = 42) -> "MultiHopDataset":
            rng = random.Random(seed)
            sampled = rng.sample(self.qa_pairs, min(n, len(self.qa_pairs)))
            return MultiHopDataset(corpus=self.corpus, qa_pairs=sampled)

    def _load_corpus(corpus_path: str) -> List[CorpusDoc]:
        with open(corpus_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        docs = []
        for i, item in enumerate(raw):
            body = item.get("body") or item.get("text") or item.get("content") or ""
            title = item.get("title") or item.get("name") or f"doc_{i}"
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
        with open(qa_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        qa_pairs = []
        for item in raw:
            evidence_raw = (
                item.get("evidence_list")
                or item.get("supporting_evidence")
                or item.get("required_evidence")
                or item.get("evidence")
                or []
            )
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
        """从本地目录加载 MultiHop-RAG 数据集。"""
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

    def corpus_to_text_units(corpus: List[CorpusDoc]) -> List[dict]:
        """将 CorpusDoc 列表转换为 TextUnit 格式。"""
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

    def try_download_dataset(target_dir: str) -> bool:
        """尝试通过 HuggingFace datasets 库下载数据集。"""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        try:
            from datasets import load_dataset
            print("  通过 HuggingFace 下载 MultiHop-RAG...")
            ds = load_dataset("yixuantt/MultiHopRAG", split="test", trust_remote_code=True)
            qa_records = [dict(row) for row in ds]
            qa_path = target / "MultiHopRAG.json"
            with open(qa_path, "w", encoding="utf-8") as f:
                json.dump(qa_records, f, ensure_ascii=False, indent=2)
            print(f"  ✓ QA 对已保存：{qa_path}（{len(qa_records)} 条）")
            return True
        except ImportError:
            print("  [提示] 未安装 datasets 库，请运行：pip install datasets")
            return False
        except Exception as e:
            print(f"  [提示] HuggingFace 下载失败：{e}")
            return False


# ---------------------------------------------------------------------------
# 数据集路径自动发现
# ---------------------------------------------------------------------------

# 项目内预置的数据集路径（按优先级排序）
_DEFAULT_DATA_PATHS = [
    Path("/Users/ttung/Desktop/个人学习/data/multihop_rag"),
    Path("/Users/ttung/Desktop/个人学习/data/MultiHop-RAG/dataset"),
    Path(__file__).parent.parent / "data" / "multihop_rag",
    Path(__file__).parent.parent.parent / "data" / "multihop_rag",
]


def find_default_data_dir() -> Optional[str]:
    """
    自动发现 MultiHop-RAG 数据集目录。

    Returns
    -------
    str or None
        找到的数据集目录路径，若未找到则返回 None
    """
    for path in _DEFAULT_DATA_PATHS:
        corpus = path / "corpus.json"
        qa = path / "MultiHopRAG.json"
        if corpus.exists() and qa.exists():
            return str(path)
    return None


def load_default_dataset(num_samples: Optional[int] = None, seed: int = 42) -> "MultiHopDataset":
    """
    加载默认数据集（自动发现路径）。

    Parameters
    ----------
    num_samples : int, optional
        随机采样数量
    seed : int
        随机种子

    Returns
    -------
    MultiHopDataset
        数据集对象
    """
    data_dir = find_default_data_dir()
    if data_dir is None:
        raise FileNotFoundError(
            "未找到 MultiHop-RAG 数据集。\n"
            "请将数据集放在以下路径之一：\n"
            + "\n".join(f"  - {p}" for p in _DEFAULT_DATA_PATHS)
            + "\n\n下载方式：\n"
            "  git clone https://github.com/yixuantt/MultiHop-RAG\n"
            "  cp -r MultiHop-RAG/dataset/ /Users/ttung/Desktop/个人学习/data/multihop_rag/"
        )

    dataset = load_multihop_dataset(data_dir)

    if num_samples is not None and num_samples < dataset.num_qa:
        dataset = dataset.subset(num_samples, seed=seed)

    return dataset


# ---------------------------------------------------------------------------
# CLI 入口（数据集检查）
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MultiHop-RAG 数据集检查工具")
    parser.add_argument("--data_dir", type=str, default=None, help="数据集目录")
    parser.add_argument("--show_sample", action="store_true", help="显示样本数据")
    args = parser.parse_args()

    if args.data_dir:
        data_dir = args.data_dir
    else:
        data_dir = find_default_data_dir()
        if data_dir:
            print(f"自动发现数据集：{data_dir}")
        else:
            print("未找到数据集，请指定 --data_dir")
            sys.exit(1)

    try:
        dataset = load_multihop_dataset(data_dir)
        print(f"\n数据集统计：")
        print(f"  文档数：{dataset.num_docs}")
        print(f"  QA 对数：{dataset.num_qa}")

        # 问题类型分布
        from collections import Counter
        type_counter = Counter(qa.question_type for qa in dataset.qa_pairs)
        print(f"\n问题类型分布：")
        for qtype, count in sorted(type_counter.items(), key=lambda x: -x[1]):
            print(f"  {qtype}: {count}")

        if args.show_sample and dataset.qa_pairs:
            qa = dataset.qa_pairs[0]
            print(f"\n样本 QA 对：")
            print(f"  查询：{qa.query}")
            print(f"  答案：{qa.answer[:100]}...")
            print(f"  类型：{qa.question_type}")
            print(f"  相关文档：{qa.supporting_titles[:3]}")

    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)
