"""
main.py
-------
命令行入口（CLI）。

用法：
  # 使用默认配置（当前目录下的 config.yaml）
  python main.py

  # 指定配置文件
  python main.py --config path/to/config.yaml

  # 快速运行（指定数据目录，其余使用默认配置）
  python main.py --data-dir ./my_papers

  # 仅运行到抽取阶段，不执行聚类（用于调试）
  python main.py --dry-run

  # 调整关键参数（无需修改 config.yaml）
  python main.py --lambda-init 500 --schedule cosine --output-dir ./results
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graphrag-improved",
        description="带结构熵惩罚的层次化 Leiden 聚类 Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main.py                              # 使用 config.yaml 默认配置
  python main.py --data-dir ./papers          # 指定输入目录
  python main.py --lambda-init 500            # 调整物理约束强度
  python main.py --schedule linear            # 使用线性退火
  python main.py --dry-run                    # 仅抽取，不聚类
        """,
    )

    # 配置文件
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="YAML 配置文件路径（默认：./config.yaml）",
    )

    # 输入覆盖
    input_group = parser.add_argument_group("输入参数（覆盖 config.yaml）")
    input_group.add_argument(
        "--data-dir", "-d",
        type=str,
        default=None,
        help="输入数据目录（支持 .txt / .json / .pdf）",
    )
    input_group.add_argument(
        "--chunk-strategy",
        choices=["paragraph", "sentence", "token"],
        default=None,
        help="分块策略（默认：paragraph）",
    )

    # 抽取覆盖
    extract_group = parser.add_argument_group("抽取参数（覆盖 config.yaml）")
    extract_group.add_argument(
        "--backend",
        choices=["rule", "spacy"],
        default=None,
        help="实体抽取后端（默认：rule）",
    )

    # 聚类覆盖
    cluster_group = parser.add_argument_group("聚类参数（覆盖 config.yaml）")
    cluster_group.add_argument(
        "--lambda-init",
        type=float,
        default=None,
        help="初始 λ 值（默认：1000.0）",
    )
    cluster_group.add_argument(
        "--lambda-min",
        type=float,
        default=None,
        help="最小 λ 值（默认：0.0）",
    )
    cluster_group.add_argument(
        "--schedule",
        choices=["exponential", "linear", "cosine", "step"],
        default=None,
        help="退火曲线类型（默认：exponential）",
    )
    cluster_group.add_argument(
        "--max-cluster-size",
        type=int,
        default=None,
        help="社区最大节点数（默认：10）",
    )
    cluster_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（默认：42）",
    )

    # 输出覆盖
    output_group = parser.add_argument_group("输出参数（覆盖 config.yaml）")
    output_group.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="输出目录（默认：./output）",
    )
    output_group.add_argument(
        "--no-html",
        action="store_true",
        help="不生成 HTML 报告",
    )
    output_group.add_argument(
        "--no-parquet",
        action="store_true",
        help="不导出 Parquet 文件",
    )

    # 运行模式
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅执行数据摄入和抽取，不运行聚类（用于调试输入数据）",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式，不打印进度信息",
    )

    return parser


def apply_cli_overrides(config, args: argparse.Namespace):
    """将 CLI 参数覆盖到配置对象上。"""
    if args.data_dir:
        config.input.data_dir = str(Path(args.data_dir).resolve())
    if args.chunk_strategy:
        config.input.chunk_strategy = args.chunk_strategy
    if args.backend:
        config.extraction.backend = args.backend
    if args.lambda_init is not None:
        config.clustering.lambda_init = args.lambda_init
    if args.lambda_min is not None:
        config.clustering.lambda_min = args.lambda_min
    if args.schedule:
        config.clustering.annealing_schedule = args.schedule
    if args.max_cluster_size is not None:
        config.clustering.max_cluster_size = args.max_cluster_size
    if args.seed is not None:
        config.clustering.seed = args.seed
    if args.output_dir:
        config.output.output_dir = str(Path(args.output_dir).resolve())
    if args.no_html:
        config.output.html_report = False
    if args.no_parquet:
        config.output.parquet_export = False
    return config


def main():
    parser = build_parser()
    args = parser.parse_args()

    # 将项目根目录加入 sys.path，支持直接运行 main.py
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from graphrag_improved.pipeline_config import load_config
        from graphrag_improved.run import run_pipeline

        # 加载配置
        config = load_config(args.config)

        # 应用 CLI 覆盖参数
        config = apply_cli_overrides(config, args)

        verbose = not args.quiet

        # dry-run 模式：仅执行摄入和抽取
        if args.dry_run:
            from graphrag_improved.data.ingestion import ingest
            from graphrag_improved.extraction.extractor import extract

            print("\n[dry-run 模式] 仅执行数据摄入和实体抽取\n")
            text_units = ingest(config.input)
            print(f"✓ 加载 {len(text_units)} 个文本块")

            entities_df, relationships_df = extract(text_units, config.extraction)
            print(f"✓ 抽取实体 {len(entities_df)} 个，关系 {len(relationships_df)} 条")

            if not entities_df.empty:
                print(f"\n实体样本（前 10 条）：")
                print(entities_df[["title", "type", "description"]].head(10).to_string(index=False))
            return

        # 正常运行完整 Pipeline
        run_pipeline(config=config, verbose=verbose)

    except FileNotFoundError as e:
        print(f"\n❌ 文件未找到：{e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ 配置错误：{e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断运行", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ 运行出错：{e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
