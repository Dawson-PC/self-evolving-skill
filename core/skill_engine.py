"""
skill_engine.py - 核心引擎 + ValueGate（价值门控）

核心引擎将残差金字塔、自适应触发器、经验回放、价值门控
整合为统一的自我演化循环。

价值门控机制：只有提升长期价值的变异才被接受，
确保系统朝着更有用的方向演化。
"""

from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .residual_pyramid import ResidualPyramid, AbstractionLevel
from .reflection_trigger import ReflectionTrigger
from .experience_replay import ExperienceReplay

logger = logging.getLogger(__name__)


class SkillStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    EVOLVING = "evolving"
    STABLE = "stable"
    DEPRECATED = "deprecated"


@dataclass
class Skill:
    """单个Skill的数据结构"""
    id: str
    name: str
    description: str
    status: SkillStatus = SkillStatus.CREATED
    embedding: Optional[List[float]] = None
    value_score: float = 0.0
    evolution_count: int = 0
    parent_id: Optional[str] = None
    created_at: float = 0.0
    last_used_at: float = 0.0
    abstraction_level: AbstractionLevel = AbstractionLevel.PREDICATE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvolutionProposal:
    """演化提案：描述一个可能的变异"""
    skill_id: str
    mutation_type: str          # "refine", "merge", "split", "prune"
    expected_value_gain: float  # 预期的价值增益
    description: str
    priority: float             # 优先级


@dataclass
class ValueGateResult:
    """价值门控评估结果"""
    accepted: bool               # 是否接受该变异
    value_gain: float            # 实际价值增益
    confidence: float            # 评估置信度
    reason: str                  # 决策原因


class ValueGate:
    """
    价值门控

    只有提升长期价值的变异才被接受。
    使用指数移动平均（EMA）跟踪价值基线。
    """

    def __init__(
        self,
        min_value_gain: float = 0.10,
        ema_alpha: float = 0.3,
        exploration_rate: float = 0.10,
    ):
        self.min_value_gain = min_value_gain
        self.ema_alpha = ema_alpha
        self.exploration_rate = exploration_rate
        self._value_baseline: Dict[str, float] = {}  # skill_id -> EMA value
        self._acceptance_history: List[Tuple[str, bool, float, float]] = []

    def evaluate(
        self,
        skill_id: str,
        current_value: float,
        proposed_value: float,
        confidence: float = 0.5,
    ) -> ValueGateResult:
        """
        评估是否接受变异

        Args:
            skill_id: 目标Skill ID
            current_value: 当前价值
            proposed_value: 变异后的预期价值
            confidence: 评估置信度 (0~1)

        Returns:
            ValueGateResult
        """
        value_gain = proposed_value - current_value

        # 更新基线
        if skill_id not in self._value_baseline:
            self._value_baseline[skill_id] = current_value
        baseline = self._value_baseline[skill_id]

        # 判断是否接受
        if value_gain > self.min_value_gain and confidence > 0.3:
            accepted = True
            reason = f"价值增益 {value_gain:.3f} 超过阈值 {self.min_value_gain}"
        elif value_gain > 0 and confidence > 0.7:
            accepted = True
            reason = f"正向增益 {value_gain:.3f} 且置信度高 {confidence:.2f}"
        elif np.random.random() < self.exploration_rate:
            accepted = True
            reason = f"探索性接受（探索率 {self.exploration_rate:.2f}）"
        else:
            accepted = False
            reason = f"价值增益 {value_gain:.3f} 未达阈值，拒绝变异"

        # 更新EMA基线
        if accepted:
            new_baseline = self.ema_alpha * proposed_value + (1 - self.ema_alpha) * baseline
            self._value_baseline[skill_id] = new_baseline

        self._acceptance_history.append((skill_id, accepted, value_gain, time.time()))

        return ValueGateResult(
            accepted=accepted,
            value_gain=value_gain,
            confidence=confidence,
            reason=reason,
        )

    def get_statistics(self) -> dict:
        """获取门控统计信息"""
        total = len(self._acceptance_history)
        accepted = sum(1 for _, a, _, _ in self._acceptance_history if a)
        return {
            "total_evaluations": total,
            "total_accepted": accepted,
            "acceptance_rate": accepted / total if total > 0 else 0.0,
            "min_value_gain": self.min_value_gain,
            "exploration_rate": self.exploration_rate,
            "tracked_skills": len(self._value_baseline),
        }


class SelfEvolvingSkillEngine:
    """
    自我演化核心引擎

    整合残差金字塔、自适应触发器、经验回放和价值门控，
    形成完整的元认知自学习循环。
    """

    def __init__(
        self,
        pyramid: Optional[ResidualPyramid] = None,
        trigger: Optional[ReflectionTrigger] = None,
        replay: Optional[ExperienceReplay] = None,
        value_gate: Optional[ValueGate] = None,
    ):
        self.pyramid = pyramid or ResidualPyramid(max_layers=5, use_pca=True)
        self.trigger = trigger or ReflectionTrigger(
            min_energy_ratio=0.10,
            value_gain_threshold=0.20,
            target_trigger_rate=0.15,
        )
        self.replay = replay or ExperienceReplay(capacity=1000)
        self.value_gate = value_gate or ValueGate(
            min_value_gain=0.10,
            ema_alpha=0.3,
            exploration_rate=0.10,
        )

        self._skills: Dict[str, Skill] = {}
        self._initialized = False

    async def init(self) -> None:
        """初始化引擎"""
        logger.info("Initializing SelfEvolvingSkillEngine...")
        self._initialized = True

    def create_skill(
        self,
        name: str,
        description: str = "",
        embedding: Optional[List[float]] = None,
    ) -> Skill:
        """
        创建新Skill

        Args:
            name: Skill名称
            description: 描述
            embedding: 初始嵌入向量（可选）

        Returns:
            创建的Skill对象
        """
        skill_id = str(uuid.uuid4())[:8]
        skill = Skill(
            id=skill_id,
            name=name,
            description=description,
            embedding=embedding or [],
            created_at=time.time(),
            last_used_at=time.time(),
        )
        self._skills[skill_id] = skill
        logger.info(f"Created skill: {name} (id={skill_id})")
        return skill

    async def execute(
        self,
        skill_id: str,
        context: Dict[str, Any],
        success: bool = True,
        value: float = 0.0,
    ) -> Dict[str, Any]:
        """
        执行Skill并触发学习循环

        Args:
            skill_id: 目标Skill ID
            context: 执行上下文
            success: 是否执行成功
            value: 执行价值评分

        Returns:
            执行结果
        """
        if skill_id not in self._skills:
            raise ValueError(f"Skill {skill_id} not found")

        skill = self._skills[skill_id]
        skill.last_used_at = time.time()
        skill.status = SkillStatus.ACTIVE

        # 如果成功且有价值，更新Skill评分
        if success and value > 0:
            skill.value_score = self.value_gate.ema_alpha * value + (1 - self.value_gate.ema_alpha) * skill.value_score

        # 构建上下文嵌入（如果有上下文特征）
        embedding = self._context_to_embedding(context)

        # 残差金字塔分解
        decomposition = self.pyramid.decompose(embedding)
        skill.abstraction_level = decomposition.suggested_abstraction

        # 自适应触发检查
        trigger_result = self.trigger.evaluate(
            residual_ratio=decomposition.residual_ratio,
            novelty_score=decomposition.novelty_score,
            value_gain=value,
        )

        result = {
            "skill_id": skill_id,
            "success": success,
            "value": value,
            "decomposition": {
                "residual_ratio": decomposition.residual_ratio,
                "novelty_score": decomposition.novelty_score,
                "suggested_abstraction": decomposition.suggested_abstraction.value,
            },
            "trigger": {
                "triggered": trigger_result.triggered,
                "reason": trigger_result.reason,
            },
        }

        # 将经验加入回放缓存
        pattern = f"skill:{skill.name}/value:{value:.2f}"
        self.replay.add(
            embedding=embedding.tolist(),
            pattern=pattern,
            skill_id=skill_id,
            value_score=value,
            novelty=decomposition.novelty_score,
        )

        # 如果需要反思
        if trigger_result.triggered:
            evolution = await self._evolve(skill, decomposition)
            result["evolution"] = evolution

        return result

    async def _evolve(
        self,
        skill: Skill,
        decomposition: DecompositionResult,
    ) -> Optional[Dict[str, Any]]:
        """执行演化循环"""
        proposal = self._generate_proposal(skill, decomposition)

        if proposal is None:
            return None

        # 价值门控评估
        gate_result = self.value_gate.evaluate(
            skill_id=skill.id,
            current_value=skill.value_score,
            proposed_value=skill.value_score + proposal.expected_value_gain,
            confidence=0.5 + decomposition.novelty_score * 0.5,
        )

        if gate_result.accepted:
            skill.evolution_count += 1
            skill.status = SkillStatus.EVOLVING
            logger.info(f"Evolution accepted: {skill.name} -> {proposal.mutation_type}")

            return {
                "evolution_accepted": True,
                "mutation_type": proposal.mutation_type,
                "value_gain": gate_result.value_gain,
                "description": proposal.description,
                "new_skill_id": None,  # 预留：可在此生成子Skill
            }
        else:
            logger.info(f"Evolution rejected: {skill.name} - {gate_result.reason}")
            return {
                "evolution_accepted": False,
                "reason": gate_result.reason,
            }

    def _generate_proposal(
        self,
        skill: Skill,
        decomposition: DecompositionResult,
    ) -> Optional[EvolutionProposal]:
        """根据分解结果生成演化提案"""
        level = decomposition.suggested_abstraction

        if level == AbstractionLevel.POLICY:
            return EvolutionProposal(
                skill_id=skill.id,
                mutation_type="refine",
                expected_value_gain=0.15,
                description="覆盖率>80%，调整策略权重以优化高层行为",
                priority=0.3,
            )
        elif level == AbstractionLevel.SUB_SKILL:
            return EvolutionProposal(
                skill_id=skill.id,
                mutation_type="split",
                expected_value_gain=0.25,
                description="覆盖率40-80%，可以拆分为子Skill以细化专业知识",
                priority=0.6,
            )
        else:
            return EvolutionProposal(
                skill_id=skill.id,
                mutation_type="prune",
                expected_value_gain=0.10,
                description="覆盖率<40%，归纳新谓词模式",
                priority=0.8,
            )

    def _context_to_embedding(self, context: Dict[str, Any]) -> np.ndarray:
        """将上下文转换为嵌入向量"""
        features = []
        for key, value in sorted(context.items()):
            if isinstance(value, (int, float)):
                features.append(float(value))
            elif isinstance(value, str):
                features.append(float(hash(value) % 1000) / 1000.0)
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                features.extend([float(v) if isinstance(v, (int, float)) else 0.0 for v in value[:10]])

        # 填充或截断到固定维度
        target_dim = 64
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        elif len(features) > target_dim:
            features = features[:target_dim]

        return np.array(features, dtype=np.float64)

    def analyze(self, embedding: List[float]) -> Dict[str, Any]:
        """分析嵌入向量"""
        emb = np.asarray(embedding, dtype=np.float64)
        decomposition = self.pyramid.decompose(emb)

        return {
            "residual_ratio": decomposition.residual_ratio,
            "novelty_score": decomposition.novelty_score,
            "suggested_abstraction": decomposition.suggested_abstraction.value,
            "coverage": 1.0 - decomposition.residual_ratio,
        }

    def list_skills(self) -> List[Dict[str, Any]]:
        """列出所有Skills"""
        return [
            {
                "id": s.id,
                "name": s.name,
                "description": s.description,
                "status": s.status.value,
                "value_score": s.value_score,
                "evolution_count": s.evolution_count,
                "abstraction_level": s.abstraction_level.value,
            }
            for s in self._skills.values()
        ]

    def stats(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            "skills": {
                "total": len(self._skills),
                "by_status": {
                    status.value: sum(1 for s in self._skills.values() if s.status == status)
                    for status in SkillStatus
                },
                "avg_value": sum(s.value_score for s in self._skills.values()) / len(self._skills) if self._skills else 0.0,
            },
            "pyramid": {
                "max_layers": self.pyramid.max_layers,
                "basis_vectors": len(self.pyramid.basis_vectors),
            },
            "trigger": self.trigger.get_statistics(),
            "replay": self.replay.get_statistics(),
            "value_gate": self.value_gate.get_statistics(),
        }
