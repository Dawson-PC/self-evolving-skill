"""
experience_replay.py - 经验回放缓存

缓存已学模式，降低重复触发，提高学习效率。
核心机制：
1. 存储已处理的经验（嵌入向量、结果、价值评估）
2. 基于相似度去重，避免重复学习
3. 支持优先采样（按价值或新颖性排序）
4. 基于容量的自动淘汰策略
"""

from __future__ import annotations

import time
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Experience:
    """单条经验记录"""
    id: str
    embedding: List[float]          # 嵌入向量
    skill_id: Optional[str]         # 关联的Skill ID
    pattern: str                    # 模式摘要
    value_score: float              # 价值评分
    novelty_at_creation: float      # 创建时的新颖性
    timestamp: float                # 时间戳
    access_count: int = 0           # 被采样次数
    last_accessed: float = 0.0      # 最后访问时间
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExperienceReplay:
    """
    经验回放缓存

    Args:
        capacity: 最大缓存容量
        similarity_threshold: 相似度去重阈值（cosine距离）
        priority_alpha: 优先采样指数（0=均匀采样, 1=完全按优先级）
    """

    def __init__(
        self,
        capacity: int = 1000,
        similarity_threshold: float = 0.85,
        priority_alpha: float = 0.6,
    ):
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.priority_alpha = priority_alpha
        self._experiences: OrderedDict[str, Experience] = OrderedDict()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._hit_count = 0
        self._miss_count = 0

    def add(
        self,
        embedding: List[float],
        pattern: str,
        skill_id: Optional[str] = None,
        value_score: float = 0.0,
        novelty: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """
        添加新经验（自动去重）

        Returns:
            (experience_id, 是否是新添加的)
        """
        # 去重检查
        existing_id = self._find_similar(embedding)
        if existing_id:
            self._hit_count += 1
            # 更新已有经验的访问计数
            exp = self._experiences[existing_id]
            exp.access_count += 1
            exp.last_accessed = time.time()
            return existing_id, False

        exp_id = f"exp_{int(time.time() * 1000)}_{len(self._experiences)}"
        experience = Experience(
            id=exp_id,
            embedding=embedding,
            skill_id=skill_id,
            pattern=pattern,
            value_score=value_score,
            novelty_at_creation=novelty,
            timestamp=time.time(),
            metadata=metadata or {},
        )

        # 缓存满时淘汰最旧的
        if len(self._experiences) >= self.capacity:
            self._evict_oldest()

        self._experiences[exp_id] = experience
        self._embedding_cache[exp_id] = np.asarray(embedding, dtype=np.float64)
        self._miss_count += 1

        return exp_id, True

    def sample(self, batch_size: int = 1, use_priority: bool = True) -> List[Experience]:
        """
        采样一批经验

        Args:
            batch_size: 采样数量
            use_priority: 是否使用优先采样

        Returns:
            采样的经验列表
        """
        if not self._experiences:
            return []

        batch_size = min(batch_size, len(self._experiences))

        if use_priority:
            return self._priority_sample(batch_size)
        else:
            return self._uniform_sample(batch_size)

    def _priority_sample(self, batch_size: int) -> List[Experience]:
        """基于优先级采样（价值越高、新颖性越高、访问越少 → 优先级越高）"""
        experiences = list(self._experiences.values())

        # 计算优先级分数
        priorities = []
        for exp in experiences:
            # 价值分数
            value_score = exp.value_score + 0.1  # 避免零概率
            # 新颖性衰减 (越久没被访问过越重要)
            recency_bonus = 1.0 / (1.0 + exp.access_count)
            # 综合优先级
            priority = (value_score ** self.priority_alpha) * recency_bonus
            priorities.append(priority)

        # 归一化
        total = sum(priorities)
        if total == 0:
            probabilities = [1.0 / len(priorities)] * len(priorities)
        else:
            probabilities = [p / total for p in priorities]

        # 加权采样
        indices = random.choices(range(len(experiences)), weights=probabilities, k=batch_size)
        sampled = [experiences[i] for i in indices]

        # 更新访问计数
        for exp in sampled:
            exp.access_count += 1
            exp.last_accessed = time.time()

        return sampled

    def _uniform_sample(self, batch_size: int) -> List[Experience]:
        """均匀随机采样"""
        experiences = list(self._experiences.values())
        sampled = random.sample(experiences, min(batch_size, len(experiences)))

        for exp in sampled:
            exp.access_count += 1
            exp.last_accessed = time.time()

        return sampled

    def _find_similar(self, embedding: List[float]) -> Optional[str]:
        """
        查找与给定嵌入相似的经验

        Returns:
            匹配的经验ID，如果没有找到返回None
        """
        query = np.asarray(embedding, dtype=np.float64)
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-12:
            return None

        for exp_id, cached in self._embedding_cache.items():
            similarity = self._cosine_similarity(query, cached)
            if similarity >= self.similarity_threshold:
                return exp_id

        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _evict_oldest(self) -> None:
        """淘汰最旧的（或最不常用的）经验"""
        if not self._experiences:
            return

        # 选择淘汰策略：综合考虑时间和访问频率
        scores = []
        for exp_id, exp in self._experiences.items():
            # 分数 = 访问次数 * 衰减因子（时间越久分数越低）
            age = time.time() - exp.timestamp
            score = exp.access_count / (1.0 + age / 3600.0)
            scores.append((score, exp_id))

        # 淘汰分数最低的
        scores.sort()
        victim_id = scores[0][1]
        self._embedding_cache.pop(victim_id, None)
        self._experiences.pop(victim_id, None)

    def get_similarity_to(self, embedding: List[float]) -> float:
        """计算输入嵌入与缓存中最相似经验的最高相似度"""
        query = np.asarray(embedding, dtype=np.float64)
        max_sim = 0.0
        for cached in self._embedding_cache.values():
            sim = self._cosine_similarity(query, cached)
            max_sim = max(max_sim, sim)
        return max_sim

    def update_value(self, exp_id: str, value_score: float) -> bool:
        """更新指定经验的价值评分"""
        if exp_id in self._experiences:
            self._experiences[exp_id].value_score = value_score
            return True
        return False

    def get_statistics(self) -> dict:
        """获取回放缓存统计信息"""
        return {
            "size": len(self._experiences),
            "capacity": self.capacity,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / (self._hit_count + self._miss_count) if (self._hit_count + self._miss_count) > 0 else 0.0,
            "avg_value": sum(e.value_score for e in self._experiences.values()) / len(self._experiences) if self._experiences else 0.0,
        }

    def to_dict(self) -> dict:
        """序列化为字典（用于持久化）"""
        return {
            "capacity": self.capacity,
            "similarity_threshold": self.similarity_threshold,
            "priority_alpha": self.priority_alpha,
            "experiences": {
                eid: {
                    "id": e.id,
                    "embedding": e.embedding,
                    "skill_id": e.skill_id,
                    "pattern": e.pattern,
                    "value_score": e.value_score,
                    "novelty_at_creation": e.novelty_at_creation,
                    "timestamp": e.timestamp,
                    "access_count": e.access_count,
                    "last_accessed": e.last_accessed,
                    "metadata": e.metadata,
                }
                for eid, e in self._experiences.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceReplay":
        """从字典反序列化"""
        replay = cls(
            capacity=data.get("capacity", 1000),
            similarity_threshold=data.get("similarity_threshold", 0.85),
            priority_alpha=data.get("priority_alpha", 0.6),
        )
        for eid, edata in data.get("experiences", {}).items():
            exp = Experience(**edata)
            replay._experiences[eid] = exp
            replay._embedding_cache[eid] = np.asarray(exp.embedding, dtype=np.float64)
        return replay

    def __len__(self) -> int:
        return len(self._experiences)
