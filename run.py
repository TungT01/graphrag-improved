"""
run.py
------
Pipeline 编排核心：将各模块串联为完整的端到端流程。

流程：
  1. 加载配置
  2. 数据摄入（文档读取 + 分块）
  3. 实体与关系抽取
  4. 约束 Leiden 社区检测
  5. 结果输出（CSV / Parquet / HTML 报告）

外部调用方式：
  from graphrag_improved.run import run_pipeline
  result = run_pipeline("path/to/config.yaml")
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .constrained_leiden.annealing import AnnealingConfig, AnnealingSchedule
from .constrained_leiden.graphrag_workflow import run_constrained_community_detection
from .data.ingestion import ingest
from .extraction.extractor import extract
from .output.reporter import PipelineResult, save_results
from .pipeline_config import PipelineConfig, load_config


def run_pipeline(
    config_path: Optional[str] = None,
    config: Optional[PipelineConfig] = None,
    verbose: bool = True,
) -> PipelineResult:
    """
    执行完整的 Pipeline。

    Parameters
    ----------
    config_path : str, optional
        YAML 配置文件路径。与 config 参数二选一。
    config : PipelineConfig, optional
        直接传入配置对象（用于编程调用或测试）。
    verbose : bool
        是否打印进度信息，默认 True。

    Returns
    -------
    PipelineResult
        包含社区、实体、关系 DataFrame 及运行统计的结果对象。
    """
    t_start = time.time()

    # ------------------------------------------------------------------
    # Step 0: 加载配置
    # ------------------------------------------------------------------
    if config is None:
        config = load_config(config_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  GraphRAG Improved Pipeline 启动")
        print(f"{'='*60}")
        print(f"  数据目录  : {config.input.data_dir}")
        print(f"  输出目录  : {config.output.output_dir}")
        print(f"  抽取后端  : {config.extraction.backend}")
        print(f"  退火曲线  : {config.clustering.annealing_schedule}")
        print(f"  λ 初始值  : {config.clustering.lambda_init}")
        print()

    # ------------------------------------------------------------------
    # Step 1: 数据摄入
    # ------------------------------------------------------------------
    if verbose:
        print("📥 Step 1/4  数据摄入中...")

    t1 = time.time()
    text_units = ingest(config.input)

    if verbose:
        print(f"   ✓ 加载 {len(text_units)} 个文本块  ({time.time()-t1:.2f}s)")

    if not text_units:
        raise ValueError(
            f"数据目录为空或无支持格式的文件：{config.input.data_dir}\n"
            "支持格式：.txt / .json / .pdf"
        )

    # ------------------------------------------------------------------
    # Step 2: 实体与关系抽取
    # ------------------------------------------------------------------
    if verbose:
        print("🔍 Step 2/4  实体与关系抽取中...")

    t2 = time.time()
    entities_df, relationships_df = extract(text_units, config.extraction)

    if verbose:
        print(f"   ✓ 抽取实体 {len(entities_df)} 个，关系 {len(relationships_df)} 条  ({time.time()-t2:.2f}s)")

    if entities_df.empty:
        print("  [警告] 未抽取到任何实体，请检查输入文本或调整 min_entity_freq 参数。")

    # ------------------------------------------------------------------
    # Step 3: 约束 Leiden 社区检测
    # ------------------------------------------------------------------
    if verbose:
        print("🔗 Step 3/4  约束 Leiden 社区检测中...")

    t3 = time.time()

    # 将配置转换为 AnnealingConfig
    annealing_config = AnnealingConfig(
        lambda_init=config.clustering.lambda_init,
        lambda_min=config.clustering.lambda_min,
        schedule=AnnealingSchedule(config.clustering.annealing_schedule),
        decay_rate=config.clustering.decay_rate,
        max_level=config.clustering.max_level,
    )

    communities_df = run_constrained_community_detection(
        entities=entities_df,
        relationships=relationships_df,
        annealing_config=annealing_config,
        max_cluster_size=config.clustering.max_cluster_size,
        max_iterations=config.clustering.max_iterations,
        seed=config.clustering.seed,
        use_lcc=config.clustering.use_lcc,
    )

    if verbose:
        num_levels = communities_df["level"].nunique() if not communities_df.empty else 0
        print(f"   ✓ 生成 {len(communities_df)} 个社区，{num_levels} 个层次  ({time.time()-t3:.2f}s)")

    # ------------------------------------------------------------------
    # Step 4: 结果输出
    # ------------------------------------------------------------------
    if verbose:
        print("💾 Step 4/4  结果输出中...")

    t4 = time.time()
    elapsed = time.time() - t_start

    result = PipelineResult(
        communities_df=communities_df,
        entities_df=entities_df,
        relationships_df=relationships_df,
        run_stats={
            "num_text_units": len(text_units),
            "num_entities": len(entities_df),
            "num_relationships": len(relationships_df),
            "num_communities": len(communities_df),
            "elapsed_seconds": elapsed,
        },
        config_snapshot={
            "lambda_init": config.clustering.lambda_init,
            "annealing_schedule": config.clustering.annealing_schedule,
            "max_cluster_size": config.clustering.max_cluster_size,
            "extraction_backend": config.extraction.backend,
        },
    )

    saved_files = save_results(result, config.output)

    if verbose:
        print(f"   ✓ 输出完成  ({time.time()-t4:.2f}s)")
        print(f"\n📁 输出文件：")
        for name, path in saved_files.items():
            print(f"   {name:<25} → {path}")
        print(f"\n✅ Pipeline 完成，总耗时 {elapsed:.2f} 秒\n")

    return result
