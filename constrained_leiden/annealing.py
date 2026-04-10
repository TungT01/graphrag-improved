"""
annealing.py
------------
λ 退火机制模块。

核心思想：
- 底层（level=0）：λ 极大，物理约束近似为硬约束，死守物理边界
- 高层（level→max）：λ 趋近于 0，释放语义融合能力，允许跨文档聚合

退火函数设计原则：
1. 单调递减：λ 随层级升高严格递减
2. 底层极大：level=0 时 λ 足够大，能压制任何跨边界合并
3. 高层趋零：level=max_level 时 λ 接近 0，不干扰语义聚类
4. 可配置：衰减速率可调，适应不同数据集特性
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable


class AnnealingSchedule(str, Enum):
    """
    退火曲线类型。

    EXPONENTIAL : 指数衰减，衰减速度快，适合层次较深的图
    LINEAR      : 线性衰减，衰减均匀，适合层次较浅的图
    COSINE      : 余弦退火，衰减先慢后快再慢，适合需要平滑过渡的场景
    STEP        : 阶梯衰减，在指定层级突变，适合需要明确边界的场景
    """
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    COSINE = "cosine"
    STEP = "step"


@dataclass
class AnnealingConfig:
    """
    退火配置参数。

    Attributes
    ----------
    lambda_init : float
        初始 λ 值（level=0 时使用）。
        需要足够大以压制跨物理边界合并。
        经验值：设为图中最大可能 ΔQ_leiden 的 10-100 倍。
        默认 1000.0（适合大多数科研文献场景）。

    lambda_min : float
        最小 λ 值（最高层级时使用）。
        设为 0 表示高层完全不受物理约束。
        默认 0.0。

    max_level : int
        预期的最大层次级别。用于归一化退火进度。
        实际层次可能少于此值，不影响正确性。
        默认 10。

    decay_rate : float
        衰减速率，仅对 EXPONENTIAL 和 STEP 有效。
        EXPONENTIAL：λ(l) = λ_init * exp(-decay_rate * l)
        STEP：在 decay_rate * max_level 层级处发生阶梯衰减
        默认 0.5。

    schedule : AnnealingSchedule
        退火曲线类型，默认指数衰减。
    """
    lambda_init: float = 1000.0
    lambda_min: float = 0.0
    max_level: int = 10
    decay_rate: float = 0.5
    schedule: AnnealingSchedule = AnnealingSchedule.EXPONENTIAL

    def __post_init__(self):
        if self.lambda_init < 0:
            raise ValueError(f"lambda_init 必须 >= 0，当前值：{self.lambda_init}")
        if self.lambda_min < 0:
            raise ValueError(f"lambda_min 必须 >= 0，当前值：{self.lambda_min}")
        if self.lambda_min > self.lambda_init:
            raise ValueError("lambda_min 不能大于 lambda_init")
        if self.max_level < 1:
            raise ValueError(f"max_level 必须 >= 1，当前值：{self.max_level}")
        if self.decay_rate <= 0:
            raise ValueError(f"decay_rate 必须 > 0，当前值：{self.decay_rate}")


def get_lambda(level: int, config: AnnealingConfig) -> float:
    """
    根据当前层级和退火配置，计算当前 λ 值。

    Parameters
    ----------
    level : int
        当前聚类层级（0 = 底层叶节点层）
    config : AnnealingConfig
        退火配置

    Returns
    -------
    float
        当前层级对应的 λ 值，范围 [lambda_min, lambda_init]

    Examples
    --------
    >>> cfg = AnnealingConfig(lambda_init=1000.0, lambda_min=0.0,
    ...                       max_level=5, decay_rate=0.5,
    ...                       schedule=AnnealingSchedule.EXPONENTIAL)
    >>> get_lambda(0, cfg)   # 底层，λ 极大
    1000.0
    >>> get_lambda(5, cfg)   # 顶层，λ 趋近于 0
    # 约 82.1（指数衰减）
    """
    if level <= 0:
        return config.lambda_init

    # 归一化进度 t ∈ [0, 1]
    t = min(level / config.max_level, 1.0)
    lambda_range = config.lambda_init - config.lambda_min

    if config.schedule == AnnealingSchedule.EXPONENTIAL:
        # λ(t) = λ_min + λ_range * exp(-decay_rate * level)
        raw = lambda_range * math.exp(-config.decay_rate * level)

    elif config.schedule == AnnealingSchedule.LINEAR:
        # λ(t) = λ_init * (1 - t) + λ_min * t
        raw = lambda_range * (1.0 - t)

    elif config.schedule == AnnealingSchedule.COSINE:
        # λ(t) = λ_min + λ_range * 0.5 * (1 + cos(π * t))
        raw = lambda_range * 0.5 * (1.0 + math.cos(math.pi * t))

    elif config.schedule == AnnealingSchedule.STEP:
        # 在 decay_rate * max_level 层级处阶梯衰减到 lambda_min
        step_level = config.decay_rate * config.max_level
        raw = lambda_range if level < step_level else 0.0

    else:
        raise ValueError(f"未知的退火曲线类型：{config.schedule}")

    return max(config.lambda_min, config.lambda_min + raw)


def build_annealing_schedule(config: AnnealingConfig) -> list[float]:
    """
    预计算所有层级的 λ 值序列，便于调试和可视化。

    Parameters
    ----------
    config : AnnealingConfig
        退火配置

    Returns
    -------
    list[float]
        长度为 max_level+1 的 λ 值列表，索引对应层级
    """
    return [get_lambda(level, config) for level in range(config.max_level + 1)]
