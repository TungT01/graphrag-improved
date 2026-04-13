"""
experiments/download_data.py
-----------------------------
MultiHop-RAG 数据集下载脚本。

支持两种方式：
  1. HuggingFace datasets（自动）
  2. 手动 git clone 后转换格式

用法：
  python -m graphrag_improved.experiments.download_data
  python -m graphrag_improved.experiments.download_data --target-dir ./data/multihop_rag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def download_via_huggingface(target_dir: str) -> bool:
    """通过 HuggingFace datasets 下载。"""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError:
        print("  [错误] 未安装 datasets 库，请运行：pip install datasets huggingface_hub")
        return False

    # ---- 下载 QA 对 ----
    print("  下载 QA 对（MultiHopRAG.json）...")
    try:
        ds = load_dataset(
            "yixuantt/MultiHopRAG",
            split="test",
            trust_remote_code=True,
        )
        qa_records = []
        for row in ds:
            qa_records.append({
                "query": row.get("query", ""),
                "answer": row.get("answer", ""),
                "question_type": row.get("question_type", ""),
                "supporting_evidence": row.get("supporting_evidence", []),
            })
        qa_path = target / "MultiHopRAG.json"
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa_records, f, ensure_ascii=False, indent=2)
        print(f"  ✓ QA 对：{len(qa_records)} 条 → {qa_path}")
    except Exception as e:
        print(f"  [错误] QA 下载失败：{e}")
        return False

    # ---- 下载 corpus ----
    print("  下载知识库（corpus.json）...")
    try:
        # 尝试不同的 split/data_files 组合
        corpus_loaded = False
        for split_name in ["train", "corpus", "validation"]:
            try:
                corpus_ds = load_dataset(
                    "yixuantt/MultiHopRAG",
                    split=split_name,
                    trust_remote_code=True,
                )
                corpus_records = []
                for row in corpus_ds:
                    corpus_records.append({
                        "id": row.get("id", ""),
                        "title": row.get("title", ""),
                        "body": row.get("body", row.get("text", "")),
                        "source": row.get("source", ""),
                        "published_at": row.get("published_at", ""),
                    })
                if corpus_records:
                    corpus_path = target / "corpus.json"
                    with open(corpus_path, "w", encoding="utf-8") as f:
                        json.dump(corpus_records, f, ensure_ascii=False, indent=2)
                    print(f"  ✓ Corpus：{len(corpus_records)} 篇 → {corpus_path}")
                    corpus_loaded = True
                    break
            except Exception:
                continue

        if not corpus_loaded:
            print("  [提示] corpus 无法通过 HuggingFace 自动下载")
            print("         请手动执行：")
            print("           git clone https://github.com/yixuantt/MultiHop-RAG")
            print(f"          cp MultiHop-RAG/dataset/corpus.json {target}/")
            return False

    except Exception as e:
        print(f"  [错误] corpus 下载失败：{e}")
        return False

    return True


def verify_dataset(data_dir: str) -> bool:
    """验证数据集文件完整性。"""
    data_dir = Path(data_dir)
    corpus_path = data_dir / "corpus.json"
    qa_path = data_dir / "MultiHopRAG.json"

    ok = True
    for p in [corpus_path, qa_path]:
        if p.exists():
            size_mb = p.stat().st_size / 1024 / 1024
            print(f"  ✓ {p.name}  ({size_mb:.1f} MB)")
        else:
            print(f"  ✗ {p.name}  [缺失]")
            ok = False

    if ok:
        # 快速检查字段
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_sample = json.load(f)
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_sample = json.load(f)

        print(f"\n  QA 对数量：{len(qa_sample)}")
        print(f"  Corpus 文章数：{len(corpus_sample)}")

        if qa_sample:
            sample = qa_sample[0]
            print(f"\n  QA 样例字段：{list(sample.keys())}")
            print(f"  问题：{sample.get('query', '')[:80]}...")
            print(f"  答案：{sample.get('answer', '')[:60]}")
            evidence = sample.get("supporting_evidence") or sample.get("required_evidence") or []
            print(f"  证据文章数：{len(evidence)}")

        if corpus_sample:
            sample = corpus_sample[0]
            print(f"\n  Corpus 样例字段：{list(sample.keys())}")
            print(f"  标题：{sample.get('title', '')[:60]}")
            body = sample.get("body", sample.get("text", ""))
            print(f"  正文长度：{len(body)} 字符")

    return ok


def main():
    parser = argparse.ArgumentParser(description="下载 MultiHop-RAG 数据集")
    parser.add_argument(
        "--target-dir",
        default="./data/multihop_rag",
        help="数据集保存目录",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验证已有数据集，不下载",
    )
    args = parser.parse_args()

    target = Path(args.target_dir)

    print("\n" + "="*55)
    print("  MultiHop-RAG 数据集下载工具")
    print("="*55)

    if args.verify_only:
        print(f"\n[验证] 检查 {target}...")
        verify_dataset(str(target))
        return

    # 检查是否已存在
    if (target / "corpus.json").exists() and (target / "MultiHopRAG.json").exists():
        print(f"\n[检测] 数据集已存在于 {target}")
        verify_dataset(str(target))
        return

    print(f"\n[下载] 目标目录：{target}")
    print("  方式：HuggingFace datasets\n")

    success = download_via_huggingface(str(target))

    if success:
        print("\n[验证] 检查下载结果...")
        verify_dataset(str(target))
        print("\n✅ 数据集准备完成！")
        print(f"\n运行实验：")
        print(f"  python -m graphrag_improved.experiments.run_experiment \\")
        print(f"      --data-dir {target} --n-qa 200")
    else:
        print("\n[手动下载说明]")
        print("  1. git clone https://github.com/yixuantt/MultiHop-RAG")
        print(f"  2. mkdir -p {target}")
        print(f"  3. cp MultiHop-RAG/dataset/corpus.json {target}/")
        print(f"  4. cp MultiHop-RAG/dataset/MultiHopRAG.json {target}/")
        print(f"  5. python -m graphrag_improved.experiments.download_data --verify-only --target-dir {target}")


if __name__ == "__main__":
    main()
