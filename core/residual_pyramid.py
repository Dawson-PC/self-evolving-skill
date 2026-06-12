"""
residual_pyramid.py - 残差金字塔（SVD分解）

基于预测编码理论的残差分解模块，将知识表征逐层分解为：
1. 高层抽象（POLICY）— 策略级知识
2. 中层模式（SUB_SKILL）— 子技能级知识
3. 底层细节（PREDICATE）— 原子谓词级知识

核心思想：认知缺口 = 当前嵌入无法被已有金字塔有效重构的残差能量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class AbstractionLevel(str, Enum):
    """抽象层级枚举，对应三层跃迁规则"""
    POLICY = "POLICY"          # >80% 覆盖率，调整策略权重
    SUB_SKILL = "SUB_SKILL"    # 40-80% 覆盖率，生成子Skill
    PREDICATE = "PREDICATE"    # <40% 覆盖率，归纳新谓词


@dataclass
class DecompositionResult:
    """残差金字塔分解结果"""
    residual_ratio: float          # 残差能量比率 (0~1)，越小表示越能被现有知识覆盖
    suggested_abstraction: AbstractionLevel  # 建议的抽象层级
    novelty_score: float           # 综合新颖性评分 (0~1)
    layer_residuals: List[float]   # 每层的残差能量
    layer_explained: List[float]   # 每层解释的方差比例
    n_layers_used: int             # 实际使用的层数


class ResidualPyramid:
    """
    残差金字塔分解器

    将高维嵌入向量逐层分解，每层提取一个主成分方向，
    剩余残差传递给下一层，形成层级化的知识表征。

    Args:
        max_layers: 金字塔最大层数
        use_pca: 是否使用PCA风格的方差解释
        energy_threshold: 残差能量截断阈值（低于此值停止分解）
    """

    def __init__(
        self,
        max_layers: int = 5,
        use_pca: bool = True,
        energy_threshold: float = 0.01,
    ):
        self.max_layers = max_layers
        self.use_pca = use_pca
        self.energy_threshold = energy_threshold
        # 存储已学习的基向量（每层一个主方向）
        self.basis_vectors: List[np.ndarray] = []
        # 存储每层的特征值（方差解释量）
        self.eigenvalues: List[float] = []

    def _compute_layer(self, residual: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """
        对当前残差计算一个主成分方向

        Returns:
            basis: 本层基向量（单位向量）
            explained_variance: 本层解释的方差
            residual_energy: 分解后的残差能量
        """
        # 使用当前残差的最大方差方向作为基向量
        if np.linalg.norm(residual) < 1e-10:
            return np.zeros_like(residual), 0.0, 0.0

        # 归一化得到基向量
        basis = residual / np.linalg.norm(residual)

        # 计算投影长度（特征值）
        projection = np.dot(residual, basis)
        explained_variance = float(projection ** 2)

        # 更新残差：减去在主方向上的投影
        new_residual = residual - projection * basis
        residual_energy = float(np.linalg.norm(new_residual) ** 2)

        return basis, explained_variance, residual_energy

    def decompose(self, embedding: np.ndarray) -> DecompositionResult:
        """
        对嵌入向量进行残差金字塔分解

        Args:
            embedding: 输入的高维嵌入向量

        Returns:
            DecompositionResult: 分解结果
        """
        embedding = np.asarray(embedding, dtype=np.float64).flatten()
        total_energy = float(np.linalg.norm(embedding) ** 2)

        if total_energy < 1e-12:
            return DecompositionResult(
                residual_ratio=1.0,
                suggested_abstraction=AbstractionLevel.PREDICATE,
                novelty_score=1.0,
                layer_residuals=[],
                layer_explained=[],
                n_layers_used=0,
            )

        residual = embedding.copy()
        layer_residuals: List[float] = []
        layer_explained: List[float] = []
        n_used = 0

        for layer_idx in range(self.max_layers):
            # 如果已学习的基向量不够，从当前残差学习新基
            if layer_idx >= len(self.basis_vectors):
                new_basis, exp_var, res_energy = self._compute_layer(residual)
                if np.linalg.norm(new_basis) > 1e-10:
                    self.basis_vectors.append(new_basis)
                    self.eigenvalues.append(exp_var)
                else:
                    break
            else:
                # 使用已有基向量进行分解
                basis = self.basis_vectors[layer_idx]
                projection = np.dot(residual, basis)
                exp_var = float(projection ** 2)
                residual = residual - projection * basis
                res_energy = float(np.linalg.norm(residual) ** 2)

            explained_ratio = exp_var / total_energy if total_energy > 0 else 0
            residual_ratio = res_energy / total_energy if total_energy > 0 else 1.0

            layer_explained.append(explained_ratio)
            layer_residuals.append(residual_ratio)
            n_used += 1

            # 如果残差能量低于阈值，提前停止
            if residual_ratio < self.energy_threshold:
                break

        # 计算最终残差比率
        final_residual_ratio = layer_residuals[-1] if layer_residuals else 1.0

        # 根据覆盖率确定抽象层级
        coverage = 1.0 - final_residual_ratio
        if coverage > 0.80:
            suggested = AbstractionLevel.POLICY
        elif coverage > 0.40:
            suggested = AbstractionLevel.SUB_SKILL
        else:
            suggested = AbstractionLevel.PREDICATE

        # 综合新颖性评分 = 残差比率 * (1 + 衰减因子)
        novelty_score = final_residual_ratio * (1.0 + 0.1 * (self.max_layers - n_used))
        novelty_score = min(1.0, max(0.0, novelty_score))

        return DecompositionResult(
            residual_ratio=final_residual_ratio,
            suggested_abstraction=suggested,
            novelty_score=novelty_score,
            layer_residuals=layer_residuals,
            layer_explained=layer_explained,
            n_layers_used=n_used,
        )

    def get_coverage(self, embedding: np.ndarray) -> float:
        """计算当前金字塔对给定嵌入的覆盖率（1 - 残差比率）"""
        result = self.decompose(embedding)
        return 1.0 - result.residual_ratio

    def get_basis_matrix(self) -> np.ndarray:
        """获取基向量矩阵（每行一个基向量）"""
        if not self.basis_vectors:
            return np.array([])
        return np.stack(self.basis_vectors, axis=0)

    def reset(self) -> None:
        """重置金字塔（清空已学习的基向量）"""
        self.basis_vectors.clear()
        self.eigenvalues.clear()
