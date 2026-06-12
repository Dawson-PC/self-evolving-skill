"""
storage.py - 持久化模块

经验自动保存/加载，基于JSON文件存储。
支持增量保存和版本管理。
"""

from __future__ import annotations

import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .skill_engine import SelfEvolvingSkillEngine, SkillStatus
from .experience_replay import ExperienceReplay

logger = logging.getLogger(__name__)


class Storage:
    """
    持久化存储

    将引擎状态保存到本地JSON文件，
    支持自动快照和恢复。

    Args:
        storage_dir: 存储目录
        auto_save_interval: 自动保存间隔（秒）
        max_versions: 保留的最大版本数
    """

    def __init__(
        self,
        storage_dir: str = "~/.openclaw/workspace/self-evolving-skill/storage",
        auto_save_interval: int = 300,
        max_versions: int = 10,
    ):
        self.storage_dir = os.path.expanduser(storage_dir)
        self.auto_save_interval = auto_save_interval
        self.max_versions = max_versions
        self._last_save_time: float = 0
        self._engine: Optional[SelfEvolvingSkillEngine] = None

        # 确保存储目录存在
        Path(self.storage_dir).mkdir(parents=True, exist_ok=True)

    def bind_engine(self, engine: SelfEvolvingSkillEngine) -> None:
        """绑定引擎实例"""
        self._engine = engine

    def save(self, label: str = "") -> str:
        """
        保存当前状态

        Args:
            label: 可选的标签

        Returns:
            保存的文件路径
        """
        if self._engine is None:
            raise RuntimeError("No engine bound to storage")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        version = self._get_next_version()
        filename = f"state_v{version}_{timestamp}"
        if label:
            filename += f"_{label}"
        filename += ".json"

        filepath = os.path.join(self.storage_dir, filename)

        state = self._serialize_engine()
        state["_meta"] = {
            "version": version,
            "timestamp": time.time(),
            "label": label,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        self._last_save_time = time.time()
        self._cleanup_old_versions()

        logger.info(f"State saved to {filepath}")
        return filepath

    def load(self, filepath: Optional[str] = None) -> SelfEvolvingSkillEngine:
        """
        加载状态

        Args:
            filepath: 文件路径，不指定则加载最新版本

        Returns:
            恢复的引擎实例
        """
        if filepath is None:
            filepath = self._get_latest_file()
            if filepath is None:
                logger.warning("No saved state found, creating new engine")
                return SelfEvolvingSkillEngine()

        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        engine = self._deserialize_engine(state)
        self._engine = engine
        logger.info(f"State loaded from {filepath}")
        return engine

    def check_auto_save(self) -> Optional[str]:
        """检查是否需要自动保存"""
        if self._engine is None:
            return None
        if time.time() - self._last_save_time > self.auto_save_interval:
            return self.save(label="auto")
        return None

    def list_snapshots(self) -> list:
        """列出所有保存的快照"""
        if not os.path.isdir(self.storage_dir):
            return []
        files = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
        snapshots = []
        for f in sorted(files, reverse=True):
            fp = os.path.join(self.storage_dir, f)
            snapshots.append({
                "filename": f,
                "path": fp,
                "size_bytes": os.path.getsize(fp),
                "modified": os.path.getmtime(fp),
            })
        return snapshots

    def _serialize_engine(self) -> dict:
        """序列化引擎状态"""
        engine = self._engine
        skills_data = {}
        for sid, skill in engine._skills.items():
            skills_data[sid] = {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "status": skill.status.value,
                "embedding": skill.embedding,
                "value_score": skill.value_score,
                "evolution_count": skill.evolution_count,
                "parent_id": skill.parent_id,
                "created_at": skill.created_at,
                "last_used_at": skill.last_used_at,
                "abstraction_level": skill.abstraction_level.value,
                "metadata": skill.metadata,
            }

        return {
            "skills": skills_data,
            "pyramid_basis_vectors": [v.tolist() for v in engine.pyramid.basis_vectors],
            "pyramid_eigenvalues": engine.pyramid.eigenvalues,
            "replay": engine.replay.to_dict(),
            "trigger_threshold": engine.trigger._current_threshold,
        }

    def _deserialize_engine(self, state: dict) -> SelfEvolvingSkillEngine:
        """反序列化引擎状态"""
        from .residual_pyramid import AbstractionLevel, ResidualPyramid
        from .reflection_trigger import ReflectionTrigger
        from .experience_replay import ExperienceReplay

        engine = SelfEvolvingSkillEngine()

        # 恢复金字塔
        engine.pyramid.basis_vectors = [np.array(v) for v in state.get("pyramid_basis_vectors", [])]
        engine.pyramid.eigenvalues = state.get("pyramid_eigenvalues", [])

        # 恢复触发器
        engine.trigger._current_threshold = state.get("trigger_threshold", 0.10)

        # 恢复回放缓存
        if "replay" in state:
            engine.replay = ExperienceReplay.from_dict(state["replay"])

        # 恢复Skills
        from .residual_pyramid import AbstractionLevel
        for sid, sdata in state.get("skills", {}).items():
            skill = Skill(
                id=sdata["id"],
                name=sdata.get("name", ""),
                description=sdata.get("description", ""),
                status=SkillStatus(sdata.get("status", "created")),
                embedding=sdata.get("embedding"),
                value_score=sdata.get("value_score", 0.0),
                evolution_count=sdata.get("evolution_count", 0),
                parent_id=sdata.get("parent_id"),
                created_at=sdata.get("created_at", 0.0),
                last_used_at=sdata.get("last_used_at", 0.0),
                abstraction_level=AbstractionLevel(sdata.get("abstraction_level", "PREDICATE")),
                metadata=sdata.get("metadata", {}),
            )
            engine._skills[sid] = skill

        return engine

    def _get_next_version(self) -> int:
        """获取下一个版本号"""
        if not os.path.isdir(self.storage_dir):
            return 1
        existing = [f for f in os.listdir(self.storage_dir) if f.endswith(".json")]
        versions = []
        for f in existing:
            parts = f.split("_")
            if parts and parts[0].startswith("state"):
                ver_str = parts[0].replace("state", "").replace("v", "")
                try:
                    versions.append(int(ver_str))
                except ValueError:
                    continue
        return max(versions) + 1 if versions else 1

    def _get_latest_file(self) -> Optional[str]:
        """获取最新的保存文件"""
        snapshots = self.list_snapshots()
        if not snapshots:
            return None
        return snapshots[0]["path"]

    def _cleanup_old_versions(self) -> None:
        """清理旧版本"""
        snapshots = self.list_snapshots()
        if len(snapshots) > self.max_versions:
            for old in snapshots[self.max_versions:]:
                try:
                    os.remove(old["path"])
                    logger.info(f"Removed old snapshot: {old['filename']}")
                except OSError:
                    pass
