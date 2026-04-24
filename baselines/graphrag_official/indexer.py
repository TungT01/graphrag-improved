"""
baselines/graphrag_official/indexer.py
----------------------------------------
Microsoft GraphRAG 官方基线：文档索引器。

功能：
  1. 将输入文档（.txt 文件或 corpus.json）复制到 graphrag_workspace/input/
  2. 调用 python -m graphrag index --root ./graphrag_workspace
  3. 等待索引完成，检查输出

前置条件：
  - 已运行 setup.sh 安装 graphrag
  - 已在 graphrag_workspace/settings.yaml 中配置 LLM API key
  - 设置环境变量：export GRAPHRAG_API_KEY='your-api-key'

用法：
  python indexer.py --input_dir /path/to/docs
  python indexer.py --corpus_json /path/to/corpus.json --max_docs 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# GraphRAG 工作目录（相对于本文件）
WORKSPACE_DIR = Path(__file__).parent / "graphrag_workspace"


# ---------------------------------------------------------------------------
# 文档准备
# ---------------------------------------------------------------------------

def prepare_txt_files(input_dir: str, workspace_input_dir: Path) -> int:
    """
    将 .txt 文件复制到 graphrag_workspace/input/。

    Returns
    -------
    int
        复制的文件数量
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"目录不存在：{input_dir}")

    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        logger.warning(f"目录 {input_dir} 中没有找到 .txt 文件")
        return 0

    workspace_input_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for fpath in txt_files:
        dest = workspace_input_dir / fpath.name
        shutil.copy2(fpath, dest)
        logger.info(f"  复制：{fpath.name}")
        count += 1

    logger.info(f"  共复制 {count} 个 .txt 文件到 {workspace_input_dir}")
    return count


def prepare_corpus_json(
    corpus_json: str,
    workspace_input_dir: Path,
    max_docs: Optional[int] = None,
) -> int:
    """
    将 corpus.json 中的文章转换为 .txt 文件，放入 graphrag_workspace/input/。

    每篇文章保存为一个独立的 .txt 文件，文件名为 doc_{i}.txt。

    Returns
    -------
    int
        生成的文件数量
    """
    with open(corpus_json, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if max_docs is not None:
        raw = raw[:max_docs]
        logger.info(f"  限制文档数量：{max_docs}")

    workspace_input_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, item in enumerate(raw):
        body = item.get("body") or item.get("text") or item.get("content") or ""
        title = item.get("title") or item.get("name") or f"doc_{i}"

        if not body.strip():
            continue

        # 文件名：使用索引避免特殊字符问题
        filename = f"doc_{i:05d}.txt"
        dest = workspace_input_dir / filename

        # 写入标题 + 正文
        content = f"Title: {title}\n\n{body}"
        dest.write_text(content, encoding="utf-8")
        count += 1

    logger.info(f"  共生成 {count} 个 .txt 文件到 {workspace_input_dir}")
    return count


# ---------------------------------------------------------------------------
# 索引构建
# ---------------------------------------------------------------------------

def check_api_key() -> bool:
    """检查 API key 是否已配置。"""
    api_key = os.environ.get("GRAPHRAG_API_KEY", "")
    if not api_key or api_key == "your-openai-api-key":
        return False
    return True


def run_graphrag_index(
    workspace_dir: Path,
    timeout: int = 3600,
    verbose: bool = True,
) -> bool:
    """
    调用 graphrag index 命令构建索引。

    Parameters
    ----------
    workspace_dir : Path
        GraphRAG 工作目录
    timeout : int
        超时时间（秒），默认 1 小时
    verbose : bool
        是否显示详细输出

    Returns
    -------
    bool
        是否成功
    """
    if not workspace_dir.exists():
        raise FileNotFoundError(f"工作目录不存在：{workspace_dir}")

    settings_file = workspace_dir / "settings.yaml"
    if not settings_file.exists():
        raise FileNotFoundError(
            f"配置文件不存在：{settings_file}\n"
            "请先运行 setup.sh 初始化工作目录"
        )

    input_dir = workspace_dir / "input"
    if not input_dir.exists() or not list(input_dir.glob("*.txt")):
        raise ValueError(
            f"输入目录为空：{input_dir}\n"
            "请先将文档放入 input 目录"
        )

    logger.info(f"  工作目录：{workspace_dir}")
    logger.info(f"  输入文件数：{len(list(input_dir.glob('*.txt')))}")

    cmd = [
        sys.executable, "-m", "graphrag", "index",
        "--root", str(workspace_dir),
    ]

    if verbose:
        cmd.append("--verbose")

    logger.info(f"  执行命令：{' '.join(cmd)}")
    logger.info(f"  超时设置：{timeout} 秒")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=not verbose,
            text=True,
            env={**os.environ},
        )

        elapsed = time.time() - start_time
        logger.info(f"  索引耗时：{elapsed:.1f} 秒")

        if result.returncode != 0:
            if not verbose and result.stderr:
                logger.error(f"  错误输出：{result.stderr[-2000:]}")
            logger.error(f"  graphrag index 失败（返回码：{result.returncode}）")
            return False

        logger.info("  ✓ graphrag index 完成")
        return True

    except subprocess.TimeoutExpired:
        logger.error(f"  graphrag index 超时（{timeout} 秒）")
        return False
    except FileNotFoundError:
        logger.error(
            "  graphrag 命令未找到，请先安装：\n"
            "    pip install graphrag"
        )
        return False


def check_index_output(workspace_dir: Path) -> dict:
    """
    检查索引输出文件是否存在。

    Returns
    -------
    dict
        输出文件状态
    """
    output_dir = workspace_dir / "output"
    status = {
        "output_dir_exists": output_dir.exists(),
        "parquet_files": [],
        "report_files": [],
    }

    if output_dir.exists():
        # 查找 parquet 文件（GraphRAG 的主要输出格式）
        parquet_files = list(output_dir.rglob("*.parquet"))
        status["parquet_files"] = [str(f.relative_to(workspace_dir)) for f in parquet_files]

        # 查找报告文件
        report_files = list(output_dir.rglob("*.json"))
        status["report_files"] = [str(f.relative_to(workspace_dir)) for f in report_files[:10]]

    return status


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_index(
    input_dir: Optional[str] = None,
    corpus_json: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    max_docs: Optional[int] = None,
    timeout: int = 3600,
    skip_if_exists: bool = True,
) -> bool:
    """
    构建 GraphRAG 官方索引的主函数。

    Parameters
    ----------
    input_dir : str, optional
        包含 .txt 文件的目录
    corpus_json : str, optional
        corpus.json 文件路径
    workspace_dir : str, optional
        GraphRAG 工作目录（默认为 ./graphrag_workspace）
    max_docs : int, optional
        最大文档数量（用于快速测试）
    timeout : int
        索引超时时间（秒）
    skip_if_exists : bool
        若索引已存在则跳过

    Returns
    -------
    bool
        是否成功
    """
    if input_dir is None and corpus_json is None:
        raise ValueError("必须提供 input_dir 或 corpus_json 之一")

    ws_dir = Path(workspace_dir) if workspace_dir else WORKSPACE_DIR
    ws_input_dir = ws_dir / "input"

    logger.info("=" * 50)
    logger.info("Microsoft GraphRAG 官方基线 - 索引构建")
    logger.info("=" * 50)

    # 检查 API key
    if not check_api_key():
        logger.warning(
            "  [警告] 未检测到 GRAPHRAG_API_KEY 环境变量\n"
            "  请设置：export GRAPHRAG_API_KEY='your-openai-api-key'\n"
            "  或在 settings.yaml 中直接填写 api_key"
        )

    # 检查是否已有索引
    if skip_if_exists:
        output_dir = ws_dir / "output"
        if output_dir.exists() and list(output_dir.rglob("*.parquet")):
            logger.info("  检测到已有索引，跳过重建（使用 skip_if_exists=False 强制重建）")
            return True

    # 1. 准备输入文档
    logger.info("步骤 1/2：准备输入文档")

    # 清空旧的输入文件
    if ws_input_dir.exists():
        for f in ws_input_dir.glob("*.txt"):
            f.unlink()
        logger.info(f"  清空旧输入文件：{ws_input_dir}")

    count = 0
    if corpus_json:
        count += prepare_corpus_json(corpus_json, ws_input_dir, max_docs=max_docs)
    if input_dir:
        count += prepare_txt_files(input_dir, ws_input_dir)

    if count == 0:
        raise ValueError("没有找到任何文档")

    logger.info(f"  共准备 {count} 个文档")

    # 2. 运行索引
    logger.info("步骤 2/2：运行 graphrag index")
    success = run_graphrag_index(ws_dir, timeout=timeout)

    if success:
        # 检查输出
        status = check_index_output(ws_dir)
        logger.info(f"  输出 parquet 文件数：{len(status['parquet_files'])}")
        for f in status["parquet_files"][:5]:
            logger.info(f"    - {f}")

        logger.info("=" * 50)
        logger.info("✓ 索引构建完成")
        logger.info("=" * 50)
    else:
        logger.error("索引构建失败")

    return success


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Microsoft GraphRAG 官方基线 - 索引构建器",
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
        "--workspace_dir",
        type=str,
        default=None,
        help="GraphRAG 工作目录（默认为 ./graphrag_workspace）",
    )
    parser.add_argument(
        "--max_docs",
        type=int,
        default=None,
        help="最大文档数量（用于快速测试）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="索引超时时间（秒）",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="强制重建索引（即使已存在）",
    )

    args = parser.parse_args()

    if args.input_dir is None and args.corpus_json is None:
        parser.error("必须提供 --input_dir 或 --corpus_json 之一")

    try:
        success = build_index(
            input_dir=args.input_dir,
            corpus_json=args.corpus_json,
            workspace_dir=args.workspace_dir,
            max_docs=args.max_docs,
            timeout=args.timeout,
            skip_if_exists=not args.force_rebuild,
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"索引构建失败：{e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
