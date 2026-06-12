"""
reflection_trigger.py - 自适应反思触发器

基于残差能量自动判断何时需要学习。
核心机制：
1. 监控残差能量的滑动平均值
2. 当残差能量超过自适应阈值时触发反思
3. 动态调整阈值以维持目标触发率（15%）

自适应阈值策略：
- min_energy_ratio: 初始最小能量比
- value_gain_threshold: 触发阈值
- target_trigger_rate: 目标15%触发率
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TriggerResult:
    """触发判断结果"""
    triggered: bool               # 是否触发反思
    energy_ratio: float           # 当前残差能量比率
    threshold: float              # 当前使用的阈值
    trigger_rate: float           # 当前实际触发率（滑动窗口）
    reason: str                   # 触发/不触发的原因


class ReflectionTrigger:
    """
    自适应反思触发器

    基于残差能量的滑动窗口统计，动态调整阈值以维持目标触发率。

    Args:
        min_energy_ratio: 最小能量比阈值（初始值）
        value_gain_threshold: 价值增益触发阈值
        target_trigger_rate: 目标触发率（默认15%）
        window_size: 滑动窗口大小
        adaptation_rate: 阈值自适应调整速率
    """

    def __init__(
        self,
        min_energy_ratio: float = 0.10,
        value_gain_threshold: float = 0.20,
        target_trigger_rate: float = 0.15,
        window_size: int = 100,
        adaptation_rate: float = 0.05,
    ):
        self.min_energy_ratio = min_energy_ratio
        self.value_gain_threshold = value_gain_threshold
        self.target_trigger_rate = target_trigger_rate
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate

        # 运行时状态
        self._current_threshold = min_energy_ratio
        self._energy_history: deque = deque(maxlen=window_size)
        self._trigger_history: deque = deque(maxlen=window_size)
        self._total_evaluations = 0
        self._total_triggers = 0

    def evaluate(
        self,
        residual_ratio: float,
        novelty_score: float,
        value_gain: Optional[float] = None,
    ) -> TriggerResult:
        """
        评估是否应该触发反思

        Args:
            residual_ratio: 残差能量比率（来自ResidualPyramid）
            novelty_score: 新颖性评分（来自ResidualPyramid）
            value_gain: 可选的价值增益信号

        Returns:
            TriggerResult: 触发判断结果
        """
        self._total_evaluations += 1
        self._energy_history.append(residual_ratio)

        # 综合触发分数 = 残差能量 + 新颖性奖励 + 价值增益奖励
        trigger_score = residual_ratio

        # 新颖性奖励：新颖性高时提高触发概率
        trigger_score += novelty_score * 0.3

        # 价值增益奖励：如果提供了价值增益且超过阈值
        if value_gain is not None and value_gain > self.value_gain_threshold:
            trigger_score += (value_gain - self.value_gain_threshold) * 0.5

        # 判断是否触发
        triggered = trigger_score > self._current_threshold

        self._trigger_history.append(1 if triggered else 0)
        if triggered:
            self._total_triggers += 1

        # 自适应调整阈值
        self._adapt_threshold(triggered)

        # 计算当前触发率
        current_rate = (
            sum(self._trigger_history) / len(self._trigger_history)
            if self._trigger_history
            else 0.0
        )

        reason = self._get_reason(triggered, trigger_score, residual_ratio, novelty_score, value_gain)

        return TriggerResult(
            triggered=triggered,
            energy_ratio=residual_ratio,
            threshold=self._current_threshold,
            trigger_rate=current_rate,
            reason=reason,
        )

    def _adapt_threshold(self, was_triggered: bool) -> None:
        """
        基于目标触发率自适应调整阈值

        如果实际触发率高于目标触发率 → 提高阈值（减少触发）
        如果实际触发率低于目标触发率 → 降低阈值（增加触发）
        """
        if len(self._trigger_history) < 20:
            return  # 数据太少，暂不调整

        current_rate = sum(self._trigger_history) / len(self._trigger_history)
        error = current_rate - self.target_trigger_rate

        # 使用PID风格的简单自适应
        adjustment = self.adaptation_rate * error
        self._current_threshold += adjustment

        # 限制阈值范围 [min_energy_ratio * 0.5, min_energy_ratio * 3.0]
        min_thresh = self.min_energy_ratio * 0.5
        max_thresh = self.min_energy_ratio * 3.0
        self._current_threshold = max(min_thresh, min(max_thresh, self._current_threshold))

    def _get_reason(
        self,
        triggered: bool,
        score: float,
        residual_ratio: float,
        novelty: float,
        value_gain: Optional[float],
    ) -> str:
        """生成触发/不触发的原因描述"""
        if triggered:
            parts = [
                f"综合分数 {score:.3f} 超过阈值 {self._current_threshold:.3f}",
                f"(残差={residual_ratio:.3f}, 新颖性={novelty:.3f}",
            ]
            if value_gain is not None:
                parts.append(f", 价值增益={value_gain:.3f}")
            parts.append(")")
            return "触发反思: " + "".join(parts)
        else:
            parts = [
                f"综合分数 {score:.3f} 未达阈值 {self._current_threshold:.3f}",
                f"(残差={residual_ratio:.3f}, 新颖性={novelty:.3f}",
            ]
            if value_gain is not None:
                parts.append(f", 价值增益={value_gain:.3f}")
            parts.append(")")
            return "跳过反思: " + "".join(parts)

    def get_trigger_rate(self) -> float:
        """获取当前滑动窗口内的触发率"""
        if not self._trigger_history:
            return 0.0
        return sum(self._trigger_history) / len(self._trigger_history)

    def get_overall_trigger_rate(self) -> float:
        """获取总体触发率"""
        if self._total_evaluations == 0:
            return 0.0
        return self._total_triggers / self._total_evaluations

    def get_statistics(self) -> dict:
        """获取触发器统计信息"""
        return {
            "current_threshold": self._current_threshold,
            "min_energy_ratio": self.min_energy_ratio,
            "target_trigger_rate": self.target_trigger_rate,
            "current_trigger_rate": self.get_trigger_rate(),
            "overall_trigger_rate": self.get_overall_trigger_rate(),
            "total_evaluations": self._total_evaluations,
            "total_triggers": self._total_triggers,
            "window_size": self.window_size,
        }

    def reset(self) -> None:
        """重置触发器状态"""
        self._current_threshold = self.min_energy_ratio
        self._energy_history.clear()
        self._trigger_history.clear()
        self._total_evaluations = 0
        self._total_triggers = 0
