"""
阶段192: 预测模型持久化与热加载

第一性原理:
1. 状态持久化：学习到的模型状态应可保存和恢复
2. 热加载：系统运行时可切换模型，无需重启
3. 版本管理：模型版本可追溯，支持回滚
4. 崩溃恢复：系统崩溃后可从最近检查点恢复

设计原则:
- JSON序列化：人类可读，跨平台兼容
- 增量保存：只保存可序列化的状态（参数、精度、权重）
- 热加载安全：加载失败时保持原状态不变
- 向后兼容：模型格式版本化，支持升级

使用方式:
    # 保存模型
    ModelPersistence.save(engine, "model_v1.json")

    # 加载模型
    snapshot = ModelPersistence.load("model_v1.json")

    # 热加载到运行中的引擎
    HotModelLoader.swap(engine, "model_v2.json")
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 模型快照
# ============================================================

@dataclass
class ModelSnapshot:
    """
    模型快照（可序列化的引擎状态）

    第一性原理：快照是引擎在某一时刻的完整状态投影
    """
    engine_id: str = ""
    timestamp: float = 0.0
    version: str = "1.0"

    # 核心状态
    cycle_count: int = 0
    total_surprise: float = 0.0

    # 贝叶斯参数（参数名 → [均值, 方差]）
    param_priors: Dict[str, List[float]] = field(default_factory=dict)

    # 层状态
    layers: List[Dict[str, Any]] = field(default_factory=list)

    # 阶段184：自上而下权重
    top_down_weights: Dict[str, Any] = field(default_factory=dict)

    # 阶段184：置信度统计
    confidence_stats: Dict[str, float] = field(default_factory=dict)

    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "engine_id": self.engine_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "cycle_count": self.cycle_count,
            "total_surprise": self.total_surprise,
            "param_priors": self.param_priors,
            "layers": self.layers,
            "top_down_weights": self.top_down_weights,
            "confidence_stats": self.confidence_stats,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelSnapshot":
        """从字典创建"""
        return cls(
            engine_id=data.get("engine_id", ""),
            timestamp=data.get("timestamp", 0.0),
            version=data.get("version", "1.0"),
            cycle_count=data.get("cycle_count", 0),
            total_surprise=data.get("total_surprise", 0.0),
            param_priors=data.get("param_priors", {}),
            layers=data.get("layers", []),
            top_down_weights=data.get("top_down_weights", {}),
            confidence_stats=data.get("confidence_stats", {}),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """转为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ModelSnapshot":
        """从JSON字符串创建"""
        return cls.from_dict(json.loads(json_str))


# ============================================================
# 模型持久化
# ============================================================

class ModelPersistence:
    """
    模型持久化（保存/加载）

    第一性原理：
    - 保存：从引擎提取状态 → 序列化 → 写入文件
    - 加载：读取文件 → 反序列化 → 构建快照
    """

    MODEL_VERSION = "1.0"

    @staticmethod
    def save(engine: Any, path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存引擎状态到文件

        Args:
            engine: PredictiveCodingEngine实例
            path: 文件路径
            metadata: 额外元数据

        Returns:
            True if saved successfully
        """
        try:
            snapshot = ModelPersistence.extract_snapshot(engine, metadata)
            json_data = snapshot.to_json()

            # 确保目录存在
            dir_path = os.path.dirname(path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            with open(path, "w", encoding="utf-8") as f:
                f.write(json_data)

            return True
        except Exception:
            return False

    @staticmethod
    def load(path: str) -> Optional[ModelSnapshot]:
        """
        从文件加载模型快照

        Args:
            path: 文件路径

        Returns:
            ModelSnapshot or None if loading fails
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                json_data = f.read()
            return ModelSnapshot.from_json(json_data)
        except Exception:
            return None

    @staticmethod
    def extract_snapshot(
        engine: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelSnapshot:
        """
        从引擎提取状态快照

        第一性原理：快照是引擎状态的完整投影
        """
        stats = engine.get_stats()

        # 提取贝叶斯参数
        param_priors = {}
        try:
            bayesian_params = engine.get_bayesian_params()
            for name, params in bayesian_params.items():
                param_priors[name] = [params["mean"], params["variance"]]
        except Exception:
            pass

        # 提取层状态
        layers = []
        try:
            layer_stats = engine.get_layer_stats()
            for ls in layer_stats:
                layers.append({
                    "layer_id": ls.get("layer_id", 0),
                    "layer_name": ls.get("layer_name", ""),
                    "precision": ls.get("precision", 0.5),
                    "history_size": ls.get("history_size", 0),
                    "parameter_count": ls.get("parameter_count", 0),
                })
        except Exception:
            pass

        # 提取自上而下权重（阶段184）
        top_down_weights = {}
        try:
            top_down_weights = engine.get_top_down_stats()
        except Exception:
            pass

        # 提取置信度统计（阶段184）
        confidence_stats = {}
        try:
            confidence_stats = engine.get_confidence_stats()
        except Exception:
            pass

        return ModelSnapshot(
            engine_id=getattr(engine, "engine_id", "unknown"),
            timestamp=time.time(),
            version=ModelPersistence.MODEL_VERSION,
            cycle_count=stats.get("cycle_count", 0),
            total_surprise=stats.get("free_energy_estimate", 0.0),
            param_priors=param_priors,
            layers=layers,
            top_down_weights=top_down_weights,
            confidence_stats=confidence_stats,
            metadata=metadata or {},
        )

    @staticmethod
    def apply_snapshot(engine: Any, snapshot: ModelSnapshot) -> bool:
        """
        将快照应用到引擎（恢复状态）

        第一性原理：将快照中的状态写回引擎
        - 贝叶斯参数：通过内部字典恢复
        - 层精度：通过层对象恢复
        - 统计计数：通过内部变量恢复

        Args:
            engine: PredictiveCodingEngine实例
            snapshot: 要应用的快照

        Returns:
            True if applied successfully
        """
        try:
            # 恢复贝叶斯参数
            with engine._lock:
                engine._param_priors.clear()
                for name, params in snapshot.param_priors.items():
                    if len(params) >= 2:
                        engine._param_priors[name] = (float(params[0]), float(params[1]))

                # 恢复层精度
                for layer_data in snapshot.layers:
                    layer_id = layer_data.get("layer_id", -1)
                    if 0 <= layer_id < len(engine._multi_layer.layers):
                        layer = engine._multi_layer.layers[layer_id]
                        layer.precision = float(layer_data.get("precision", 0.5))

                # 恢复统计
                engine._cycle_count = snapshot.cycle_count
                engine._total_surprise = snapshot.total_surprise

            return True
        except Exception:
            return False


# ============================================================
# 热加载器
# ============================================================

class HotModelLoader:
    """
    热加载器（运行时模型切换）

    第一性原理：
    - 热加载：系统运行时切换模型，无需重启
    - 原子性：加载失败时保持原状态不变
    - 可回滚：保存切换前状态，支持回滚

    线程安全。
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._last_snapshot: Optional[ModelSnapshot] = None
        self._history: List[Tuple[float, str, ModelSnapshot]] = []

    def swap(
        self,
        engine: Any,
        path: str,
        backup: bool = True,
    ) -> bool:
        """
        热交换模型

        第一性原理：
        1. 保存当前状态（备份）
        2. 加载新模型
        3. 应用新模型
        4. 如果失败，回滚到备份

        Args:
            engine: 引擎实例
            path: 模型文件路径
            backup: 是否备份当前状态

        Returns:
            True if swap succeeded
        """
        with self._lock:
            # 步骤1：备份当前状态
            if backup:
                self._last_snapshot = ModelPersistence.extract_snapshot(engine)

            # 步骤2：加载新模型
            new_snapshot = ModelPersistence.load(path)
            if new_snapshot is None:
                return False  # 加载失败，保持原状态

            # 步骤3：应用新模型
            success = ModelPersistence.apply_snapshot(engine, new_snapshot)

            # 步骤4：失败时回滚
            if not success and backup and self._last_snapshot is not None:
                ModelPersistence.apply_snapshot(engine, self._last_snapshot)
                return False

            # 记录历史
            if success:
                self._history.append((time.time(), path, new_snapshot))
                # 限制历史长度
                if len(self._history) > 20:
                    self._history.pop(0)

            return success

    def rollback(self, engine: Any) -> bool:
        """
        回滚到上一个状态

        Returns:
            True if rollback succeeded
        """
        with self._lock:
            if self._last_snapshot is None:
                return False

            success = ModelPersistence.apply_snapshot(engine, self._last_snapshot)
            if success:
                self._last_snapshot = None  # 清除备份
            return success

    def get_last_snapshot(self) -> Optional[ModelSnapshot]:
        """获取上一个状态快照"""
        with self._lock:
            return self._last_snapshot

    def get_history(self) -> List[Tuple[float, str]]:
        """获取加载历史"""
        with self._lock:
            return [(ts, path) for ts, path, _ in self._history]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "has_backup": self._last_snapshot is not None,
                "history_count": len(self._history),
                "last_load_time": self._history[-1][0] if self._history else None,
                "last_load_path": self._history[-1][1] if self._history else None,
            }


# ============================================================
# 模型版本管理器
# ============================================================

class ModelVersionManager:
    """
    模型版本管理器

    第一性原理：
    - 版本追踪：每次保存创建新版本
    - 版本比较：比较两个版本的状态差异
    - 版本回滚：恢复到任意历史版本
    """

    def __init__(self, base_dir: str = ".model_versions"):
        self._base_dir = base_dir
        self._lock = threading.RLock()
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """确保版本目录存在"""
        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir, exist_ok=True)

    def save_version(
        self,
        engine: Any,
        version_tag: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        保存模型版本

        Args:
            engine: 引擎实例
            version_tag: 版本标签
            metadata: 额外元数据

        Returns:
            版本文件路径，失败返回None
        """
        with self._lock:
            timestamp = time.time()
            filename = f"{version_tag}_{int(timestamp)}.json"
            path = os.path.join(self._base_dir, filename)

            version_metadata = {
                **(metadata or {}),
                "version_tag": version_tag,
                "save_time": timestamp,
            }

            if ModelPersistence.save(engine, path, version_metadata):
                return path
            return None

    def load_version(self, version_tag: str) -> Optional[ModelSnapshot]:
        """
        加载指定版本的模型

        Args:
            version_tag: 版本标签

        Returns:
            ModelSnapshot or None
        """
        with self._lock:
            # 查找匹配的版本文件
            versions = self.list_versions()
            matching = [v for v in versions if v["tag"] == version_tag]
            if not matching:
                return None

            # 加载最新的匹配版本
            latest = matching[-1]
            return ModelPersistence.load(latest["path"])

    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本"""
        with self._lock:
            versions = []
            if not os.path.exists(self._base_dir):
                return versions

            for filename in sorted(os.listdir(self._base_dir)):
                if not filename.endswith(".json"):
                    continue
                path = os.path.join(self._base_dir, filename)

                # 从文件名解析标签和时间戳
                parts = filename.rsplit("_", 1)
                if len(parts) == 2:
                    tag = parts[0]
                    try:
                        timestamp = float(parts[1].replace(".json", ""))
                    except ValueError:
                        timestamp = 0.0
                else:
                    tag = filename
                    timestamp = 0.0

                versions.append({
                    "tag": tag,
                    "path": path,
                    "filename": filename,
                    "timestamp": timestamp,
                })

            return versions

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            versions = self.list_versions()
            return {
                "total_versions": len(versions),
                "base_dir": self._base_dir,
                "latest_version": versions[-1]["tag"] if versions else None,
            }


# ============================================================
# 便捷函数
# ============================================================

def save_model(engine: Any, path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """保存模型"""
    return ModelPersistence.save(engine, path, metadata)


def load_model(path: str) -> Optional[ModelSnapshot]:
    """加载模型"""
    return ModelPersistence.load(path)


def create_hot_loader() -> HotModelLoader:
    """创建热加载器"""
    return HotModelLoader()


def create_version_manager(base_dir: str = ".model_versions") -> ModelVersionManager:
    """创建版本管理器"""
    return ModelVersionManager(base_dir)
