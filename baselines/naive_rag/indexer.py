"""
baselines/naive_rag/indexer.py
-------------------------------
Naive RAG 基线：文档索引器。

功能：
  1. 读取 .txt 文档（或 corpus.json）
  2. 用 nltk.sent_tokenize 切分为句子
  3. 用 SentenceTransformer('all-MiniLM-L6-v2') 编码为向量
  4. 保存为 numpy 数组 + 句子列表（pickle）

用法：
  python indexer.py --input_dir /path/to/docs --index_path ./naive_rag_index.pkl
  python indexer.py --corpus_json /path/to/corpus.json --index_path ./naive_rag_index.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 文档加载
# ---------------------------------------------------------------------------

def load_txt_files(input_dir: str) -> List[Tuple[str, str]]:
    """
    从目录中加载所有 .txt 文件。

    Returns
    -------
    List[Tuple[str, str]]
        (doc_id, text) 列表
    """
    docs = []
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"目录不存在：{input_dir}")

    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        logger.warning(f"目录 {input_dir} 中没有找到 .txt 文件")
        return docs

    for fpath in txt_files:
        try:
            text = fpath.read_text(encoding="utf-8")
            docs.append((fpath.stem, text))
            logger.info(f"  加载文档：{fpath.name}（{len(text)} 字符）")
        except Exception as e:
            logger.warning(f"  跳过 {fpath.name}：{e}")

    return docs


def load_corpus_json(corpus_path: str) -> List[Tuple[str, str]]:
    """
    从 corpus.json 加载文档（MultiHop-RAG 格式）。

    Returns
    -------
    List[Tuple[str, str]]
        (doc_id, text) 列表，doc_id 为文章标题
    """
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for i, item in enumerate(raw):
        body = item.get("body") or item.get("text") or item.get("content") or ""
        title = item.get("title") or item.get("name") or f"doc_{i}"
        doc_id = str(title)
        docs.append((doc_id, str(body)))

    logger.info(f"  从 corpus.json 加载了 {len(docs)} 篇文档")
    return docs


# ---------------------------------------------------------------------------
# 句子切分
# ---------------------------------------------------------------------------

def split_into_sentences(docs: List[Tuple[str, str]]) -> Tuple[List[str], List[str]]:
    """
    将文档列表切分为句子列表。

    Returns
    -------
    sentences : List[str]
        所有句子文本
    sentence_doc_ids : List[str]
        每个句子对应的 doc_id
    """
    try:
        import nltk
        # 确保 punkt 分词器已下载
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            logger.info("  下载 NLTK punkt 分词器...")
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        sent_tokenize = nltk.sent_tokenize
    except ImportError:
        logger.warning("  nltk 未安装，使用简单句号切分")
        def sent_tokenize(text):
            import re
            return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    sentences = []
    sentence_doc_ids = []

    for doc_id, text in docs:
        if not text.strip():
            continue
        try:
            sents = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"  切分 {doc_id} 失败：{e}，使用简单切分")
            import re
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

        # 过滤过短的句子（少于 10 个字符）
        valid_sents = [s for s in sents if len(s.strip()) >= 10]
        sentences.extend(valid_sents)
        sentence_doc_ids.extend([doc_id] * len(valid_sents))

    logger.info(f"  共切分出 {len(sentences)} 个句子")
    return sentences, sentence_doc_ids


# ---------------------------------------------------------------------------
# 向量化
# ---------------------------------------------------------------------------

def encode_sentences(
    sentences: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    用 SentenceTransformer 将句子编码为向量。

    Parameters
    ----------
    sentences : List[str]
        句子列表
    model_name : str
        模型名称，默认 all-MiniLM-L6-v2
    batch_size : int
        批处理大小
    show_progress : bool
        是否显示进度条

    Returns
    -------
    np.ndarray
        形状为 (N, D) 的向量矩阵
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "请先安装 sentence-transformers：\n"
            "  pip install sentence-transformers"
        )

    logger.info(f"  加载模型：{model_name}")
    model = SentenceTransformer(model_name)

    logger.info(f"  编码 {len(sentences)} 个句子（batch_size={batch_size}）...")
    embeddings = model.encode(
        sentences,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2 归一化，方便余弦相似度计算
    )

    logger.info(f"  向量维度：{embeddings.shape}")
    return embeddings


# ---------------------------------------------------------------------------
# 保存/加载索引
# ---------------------------------------------------------------------------

def save_index(
    index_path: str,
    sentences: List[str],
    sentence_doc_ids: List[str],
    embeddings: np.ndarray,
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """
    将索引保存为 pickle 文件。

    索引结构：
    {
        "sentences": List[str],          # 句子文本
        "sentence_doc_ids": List[str],   # 每个句子的 doc_id
        "embeddings": np.ndarray,        # 向量矩阵 (N, D)
        "model_name": str,               # 使用的模型名称
        "num_sentences": int,            # 句子总数
    }
    """
    index = {
        "sentences": sentences,
        "sentence_doc_ids": sentence_doc_ids,
        "embeddings": embeddings,
        "model_name": model_name,
        "num_sentences": len(sentences),
    }

    index_path = Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    with open(index_path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = index_path.stat().st_size / (1024 * 1024)
    logger.info(f"  ✓ 索引已保存：{index_path}（{size_mb:.1f} MB）")


def load_index(index_path: str) -> dict:
    """
    加载已保存的索引。

    Returns
    -------
    dict
        包含 sentences, sentence_doc_ids, embeddings, model_name 的字典
    """
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"索引文件不存在：{index_path}")

    with open(index_path, "rb") as f:
        index = pickle.load(f)

    logger.info(
        f"  ✓ 索引加载完成：{index['num_sentences']} 个句子，"
        f"向量维度 {index['embeddings'].shape[1]}"
    )
    return index


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_index(
    input_dir: str = None,
    corpus_json: str = None,
    index_path: str = "./naive_rag_index.pkl",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> dict:
    """
    构建 Naive RAG 索引的主函数。

    Parameters
    ----------
    input_dir : str, optional
        包含 .txt 文件的目录
    corpus_json : str, optional
        corpus.json 文件路径（MultiHop-RAG 格式）
    index_path : str
        索引保存路径
    model_name : str
        SentenceTransformer 模型名称
    batch_size : int
        编码批处理大小

    Returns
    -------
    dict
        构建好的索引
    """
    if input_dir is None and corpus_json is None:
        raise ValueError("必须提供 input_dir 或 corpus_json 之一")

    logger.info("=" * 50)
    logger.info("Naive RAG 索引构建")
    logger.info("=" * 50)

    # 1. 加载文档
    logger.info("步骤 1/4：加载文档")
    docs = []
    if corpus_json:
        docs.extend(load_corpus_json(corpus_json))
    if input_dir:
        docs.extend(load_txt_files(input_dir))

    if not docs:
        raise ValueError("没有找到任何文档，请检查输入路径")

    logger.info(f"  共加载 {len(docs)} 篇文档")

    # 2. 切分句子
    logger.info("步骤 2/4：切分句子")
    sentences, sentence_doc_ids = split_into_sentences(docs)

    if not sentences:
        raise ValueError("切分后没有有效句子")

    # 3. 向量化
    logger.info("步骤 3/4：向量化")
    embeddings = encode_sentences(sentences, model_name=model_name, batch_size=batch_size)

    # 4. 保存索引
    logger.info("步骤 4/4：保存索引")
    save_index(index_path, sentences, sentence_doc_ids, embeddings, model_name)

    logger.info("=" * 50)
    logger.info("✓ 索引构建完成")
    logger.info("=" * 50)

    return load_index(index_path)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Naive RAG 索引构建器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="包含 .txt 文件的目录",
    )
    parser.add_argument(
        "--corpus_json",
        type=str,
        default=None,
        help="corpus.json 文件路径（MultiHop-RAG 格式）",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default="./naive_rag_index.pkl",
        help="索引保存路径",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer 模型名称",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="编码批处理大小",
    )

    args = parser.parse_args()

    if args.input_dir is None and args.corpus_json is None:
        parser.error("必须提供 --input_dir 或 --corpus_json 之一")

    try:
        build_index(
            input_dir=args.input_dir,
            corpus_json=args.corpus_json,
            index_path=args.index_path,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
    except Exception as e:
        logger.error(f"索引构建失败：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
