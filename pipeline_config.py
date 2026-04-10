"""
pipeline_config.py
------------------
配置加载与验证模块。将 config.yaml 解析为强类型的 Python 数据类，
供各模块统一引用，避免散落的字符串 key 访问。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


# ---------------------------------------------------------------------------
# 配置数据类
# ---------------------------------------------------------------------------

@dataclass
class InputConfig:
    data_dir: str = "./sample_data"
    encoding: str = "utf-8"
    chunk_strategy: str = "paragraph"   # token | sentence | paragraph
    chunk_size: int = 512
    chunk_overlap: int = 64


@dataclass
class ExtractionConfig:
    backend: str = "rule"               # spacy | rule
    spacy_model: str = "en_core_web_sm"
    cooccurrence_window: int = 3
    min_entity_freq: int = 1


@dataclass
class ClusteringConfig:
    lambda_init: float = 1000.0
    lambda_min: float = 0.0
    annealing_schedule: str = "exponential"
    decay_rate: float = 0.5
    max_level: int = 10
    max_cluster_size: int = 10
    max_iterations: int = 10
    seed: int = 42
    use_lcc: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "./output"
    html_report: bool = True
    csv_export: bool = True
    parquet_export: bool = True
    console_summary: bool = True


@dataclass
class PipelineConfig:
    input: InputConfig = field(default_factory=InputConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # 运行时派生属性（不在 yaml 中配置）
    project_root: Path = field(default_factory=Path.cwd)

    def resolve_paths(self) -> None:
        """将相对路径解析为基于 project_root 的绝对路径。"""
        root = self.project_root
        self.input.data_dir = str(root / self.input.data_dir)
        self.output.output_dir = str(root / self.output.output_dir)

    def validate(self) -> None:
        """校验配置合法性，不合法时抛出 ValueError。"""
        valid_strategies = {"token", "sentence", "paragraph"}
        if self.input.chunk_strategy not in valid_strategies:
            raise ValueError(
                f"chunk_strategy 必须是 {valid_strategies}，"
                f"当前值：{self.input.chunk_strategy}"
            )

        valid_backends = {"spacy", "rule"}
        if self.extraction.backend not in valid_backends:
            raise ValueError(
                f"extraction.backend 必须是 {valid_backends}，"
                f"当前值：{self.extraction.backend}"
            )

        valid_schedules = {"exponential", "linear", "cosine", "step"}
        if self.clustering.annealing_schedule not in valid_schedules:
            raise ValueError(
                f"annealing_schedule 必须是 {valid_schedules}，"
                f"当前值：{self.clustering.annealing_schedule}"
            )

        if self.clustering.lambda_init < 0:
            raise ValueError("lambda_init 必须 >= 0")
        if self.clustering.lambda_min > self.clustering.lambda_init:
            raise ValueError("lambda_min 不能大于 lambda_init")


# ---------------------------------------------------------------------------
# 加载函数
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    从 YAML 文件加载配置，若文件不存在则使用全部默认值。

    Parameters
    ----------
    config_path : str, optional
        YAML 配置文件路径。默认查找当前目录下的 config.yaml。

    Returns
    -------
    PipelineConfig
        已验证的配置对象
    """
    # 确定配置文件路径
    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    cfg = PipelineConfig(project_root=Path(config_path).parent)

    # 尝试加载 YAML
    if _YAML_AVAILABLE and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # 解析各节
        if "input" in raw:
            inp = raw["input"]
            cfg.input = InputConfig(
                data_dir=inp.get("data_dir", cfg.input.data_dir),
                encoding=inp.get("encoding", cfg.input.encoding),
                chunk_strategy=inp.get("chunk_strategy", cfg.input.chunk_strategy),
                chunk_size=inp.get("chunk_size", cfg.input.chunk_size),
                chunk_overlap=inp.get("chunk_overlap", cfg.input.chunk_overlap),
            )

        if "extraction" in raw:
            ext = raw["extraction"]
            cfg.extraction = ExtractionConfig(
                backend=ext.get("backend", cfg.extraction.backend),
                spacy_model=ext.get("spacy_model", cfg.extraction.spacy_model),
                cooccurrence_window=ext.get("cooccurrence_window", cfg.extraction.cooccurrence_window),
                min_entity_freq=ext.get("min_entity_freq", cfg.extraction.min_entity_freq),
            )

        if "clustering" in raw:
            cl = raw["clustering"]
            cfg.clustering = ClusteringConfig(
                lambda_init=cl.get("lambda_init", cfg.clustering.lambda_init),
                lambda_min=cl.get("lambda_min", cfg.clustering.lambda_min),
                annealing_schedule=cl.get("annealing_schedule", cfg.clustering.annealing_schedule),
                decay_rate=cl.get("decay_rate", cfg.clustering.decay_rate),
                max_level=cl.get("max_level", cfg.clustering.max_level),
                max_cluster_size=cl.get("max_cluster_size", cfg.clustering.max_cluster_size),
                max_iterations=cl.get("max_iterations", cfg.clustering.max_iterations),
                seed=cl.get("seed", cfg.clustering.seed),
                use_lcc=cl.get("use_lcc", cfg.clustering.use_lcc),
            )

        if "output" in raw:
            out = raw["output"]
            cfg.output = OutputConfig(
                output_dir=out.get("output_dir", cfg.output.output_dir),
                html_report=out.get("html_report", cfg.output.html_report),
                csv_export=out.get("csv_export", cfg.output.csv_export),
                parquet_export=out.get("parquet_export", cfg.output.parquet_export),
                console_summary=out.get("console_summary", cfg.output.console_summary),
            )

    cfg.resolve_paths()
    cfg.validate()
    return cfg
