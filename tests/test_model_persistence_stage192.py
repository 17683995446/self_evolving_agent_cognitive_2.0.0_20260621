"""
阶段192测试：预测模型持久化与热加载

验证内容:
1. ModelSnapshot: 模型快照数据结构
2. ModelPersistence: 保存/加载模型
3. HotModelLoader: 热加载与回滚
4. ModelVersionManager: 版本管理
5. 端到端：保存→加载→应用完整流程
6. 第一性原理验证
"""

import pytest
import os
import tempfile
import time
from self_evolving_agent.core.model_persistence import (
    ModelSnapshot,
    ModelPersistence,
    HotModelLoader,
    ModelVersionManager,
    save_model,
    load_model,
    create_hot_loader,
    create_version_manager,
)
from self_evolving_agent.core.predictive_coding_engine import (
    PredictiveCodingEngine,
    create_predictive_coding_engine,
)


# ============================================================
# ModelSnapshot 测试
# ============================================================

class TestModelSnapshot:
    """模型快照数据结构测试"""

    def test_snapshot_defaults(self):
        """测试快照默认值"""
        snapshot = ModelSnapshot()
        assert snapshot.engine_id == ""
        assert snapshot.timestamp == 0.0
        assert snapshot.version == "1.0"
        assert snapshot.cycle_count == 0
        assert snapshot.total_surprise == 0.0
        assert snapshot.param_priors == {}
        assert snapshot.layers == []
        assert snapshot.top_down_weights == {}
        assert snapshot.confidence_stats == {}

    def test_snapshot_to_dict(self):
        """测试快照转字典"""
        snapshot = ModelSnapshot(
            engine_id="test_engine",
            cycle_count=10,
            total_surprise=1.5,
        )
        d = snapshot.to_dict()
        assert d["engine_id"] == "test_engine"
        assert d["cycle_count"] == 10
        assert d["total_surprise"] == 1.5

    def test_snapshot_from_dict(self):
        """测试从字典创建快照"""
        data = {
            "engine_id": "test",
            "cycle_count": 5,
            "total_surprise": 2.0,
            "param_priors": {"p1": [0.5, 0.1]},
        }
        snapshot = ModelSnapshot.from_dict(data)
        assert snapshot.engine_id == "test"
        assert snapshot.cycle_count == 5
        assert snapshot.total_surprise == 2.0
        assert snapshot.param_priors == {"p1": [0.5, 0.1]}

    def test_snapshot_json_roundtrip(self):
        """测试JSON序列化往返"""
        original = ModelSnapshot(
            engine_id="roundtrip",
            cycle_count=42,
            total_surprise=3.14,
            param_priors={"x": [1.0, 0.5]},
            layers=[{"layer_id": 0, "precision": 0.8}],
        )
        json_str = original.to_json()
        restored = ModelSnapshot.from_json(json_str)

        assert restored.engine_id == "roundtrip"
        assert restored.cycle_count == 42
        assert restored.total_surprise == 3.14
        assert restored.param_priors == {"x": [1.0, 0.5]}
        assert len(restored.layers) == 1


# ============================================================
# ModelPersistence 测试
# ============================================================

class TestModelPersistence:
    """模型持久化测试"""

    def test_save_and_load(self, tmp_path):
        """测试保存和加载"""
        engine = create_predictive_coding_engine("test_persist")
        # 运行一些周期以产生状态
        engine.run_cycle("module1", {"v": 0.5}, {"v": 0.6})

        path = str(tmp_path / "model.json")
        success = ModelPersistence.save(engine, path)
        assert success is True
        assert os.path.exists(path)

        snapshot = ModelPersistence.load(path)
        assert snapshot is not None
        assert snapshot.engine_id == "test_persist"
        assert snapshot.cycle_count >= 1

    def test_save_creates_directory(self, tmp_path):
        """测试保存时创建目录"""
        engine = create_predictive_coding_engine("test_dir")
        path = str(tmp_path / "subdir" / "model.json")

        success = ModelPersistence.save(engine, path)
        assert success is True
        assert os.path.exists(path)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        snapshot = ModelPersistence.load("/nonexistent/path/model.json")
        assert snapshot is None

    def test_extract_snapshot(self):
        """测试提取快照"""
        engine = create_predictive_coding_engine("test_extract")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})

        snapshot = ModelPersistence.extract_snapshot(engine)
        assert snapshot.engine_id == "test_extract"
        assert snapshot.cycle_count >= 1
        assert snapshot.timestamp > 0
        assert snapshot.version == "1.0"

    def test_extract_snapshot_with_metadata(self):
        """测试带元数据提取快照"""
        engine = create_predictive_coding_engine("test_meta")
        snapshot = ModelPersistence.extract_snapshot(
            engine,
            metadata={"author": "test", "description": "test model"},
        )
        assert snapshot.metadata["author"] == "test"
        assert snapshot.metadata["description"] == "test model"

    def test_apply_snapshot(self):
        """测试应用快照"""
        engine1 = create_predictive_coding_engine("engine1")
        engine1.run_cycle("m1", {"v": 0.3}, {"v": 0.7})
        engine1.run_cycle("m1", {"v": 0.4}, {"v": 0.8})

        # 提取快照
        snapshot = ModelPersistence.extract_snapshot(engine1)

        # 应用到新引擎
        engine2 = create_predictive_coding_engine("engine2")
        success = ModelPersistence.apply_snapshot(engine2, snapshot)
        assert success is True

        # 验证状态恢复
        assert engine2._cycle_count == snapshot.cycle_count
        assert engine2._total_surprise == snapshot.total_surprise

    def test_apply_snapshot_restores_bayesian_params(self):
        """测试应用快照恢复贝叶斯参数"""
        engine1 = create_predictive_coding_engine("engine1")
        # 设置一些贝叶斯参数
        engine1._param_priors["param_a"] = (0.5, 0.1)
        engine1._param_priors["param_b"] = (0.8, 0.2)

        snapshot = ModelPersistence.extract_snapshot(engine1)

        engine2 = create_predictive_coding_engine("engine2")
        ModelPersistence.apply_snapshot(engine2, snapshot)

        assert "param_a" in engine2._param_priors
        assert engine2._param_priors["param_a"] == (0.5, 0.1)
        assert engine2._param_priors["param_b"] == (0.8, 0.2)

    def test_apply_snapshot_restores_layer_precision(self):
        """测试应用快照恢复层精度"""
        engine1 = create_predictive_coding_engine("engine1")
        # 修改层精度
        if engine1._multi_layer.layers:
            engine1._multi_layer.layers[0].precision = 0.9

        snapshot = ModelPersistence.extract_snapshot(engine1)

        engine2 = create_predictive_coding_engine("engine2")
        ModelPersistence.apply_snapshot(engine2, snapshot)

        if engine2._multi_layer.layers:
            assert engine2._multi_layer.layers[0].precision == 0.9

    def test_save_load_apply_roundtrip(self, tmp_path):
        """测试保存→加载→应用完整往返"""
        engine1 = create_predictive_coding_engine("roundtrip1")
        engine1.run_cycle("m1", {"v": 0.3}, {"v": 0.7})
        engine1.run_cycle("m1", {"v": 0.4}, {"v": 0.6})
        engine1._param_priors["test_param"] = (0.5, 0.2)

        # 保存
        path = str(tmp_path / "roundtrip.json")
        ModelPersistence.save(engine1, path)

        # 加载
        snapshot = ModelPersistence.load(path)
        assert snapshot is not None

        # 应用到新引擎
        engine2 = create_predictive_coding_engine("roundtrip2")
        ModelPersistence.apply_snapshot(engine2, snapshot)

        # 验证
        assert engine2._cycle_count == engine1._cycle_count
        assert engine2._total_surprise == engine1._total_surprise
        assert "test_param" in engine2._param_priors


# ============================================================
# HotModelLoader 测试
# ============================================================

class TestHotModelLoader:
    """热加载器测试"""

    def test_swap_success(self, tmp_path):
        """测试热交换成功"""
        engine = create_predictive_coding_engine("hot_swap")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})

        # 保存当前状态
        path = str(tmp_path / "model.json")
        ModelPersistence.save(engine, path)

        # 修改引擎状态
        engine.run_cycle("m1", {"v": 0.5}, {"v": 0.9})
        original_cycle = engine._cycle_count

        # 热交换回保存的状态
        loader = HotModelLoader()
        success = loader.swap(engine, path)
        assert success is True

        # 引擎应恢复到保存的状态
        assert engine._cycle_count < original_cycle

    def test_swap_with_backup(self, tmp_path):
        """测试热交换时备份"""
        engine = create_predictive_coding_engine("backup_test")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})

        path = str(tmp_path / "model.json")
        ModelPersistence.save(engine, path)

        loader = HotModelLoader()
        loader.swap(engine, path, backup=True)

        # 应有备份
        assert loader.get_last_snapshot() is not None

    def test_swap_without_backup(self, tmp_path):
        """测试热交换不备份"""
        engine = create_predictive_coding_engine("no_backup")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})

        path = str(tmp_path / "model.json")
        ModelPersistence.save(engine, path)

        loader = HotModelLoader()
        loader.swap(engine, path, backup=False)

        # 不应有备份
        assert loader.get_last_snapshot() is None

    def test_swap_nonexistent_file(self):
        """测试热交换不存在的文件"""
        engine = create_predictive_coding_engine("nonexist")
        loader = HotModelLoader()

        success = loader.swap(engine, "/nonexistent/model.json")
        assert success is False

    def test_rollback(self, tmp_path):
        """测试回滚"""
        engine = create_predictive_coding_engine("rollback")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})
        original_surprise = engine._total_surprise

        path = str(tmp_path / "model.json")
        ModelPersistence.save(engine, path)

        # 修改状态
        engine.run_cycle("m1", {"v": 0.1}, {"v": 0.9})
        assert engine._total_surprise > original_surprise

        # 热交换（带备份）
        loader = HotModelLoader()
        loader.swap(engine, path, backup=True)

        # 回滚
        success = loader.rollback(engine)
        assert success is True

    def test_rollback_without_backup(self):
        """测试无备份时回滚失败"""
        engine = create_predictive_coding_engine("no_rollback")
        loader = HotModelLoader()

        success = loader.rollback(engine)
        assert success is False

    def test_get_history(self, tmp_path):
        """测试获取加载历史"""
        engine = create_predictive_coding_engine("history")
        path = str(tmp_path / "model.json")
        ModelPersistence.save(engine, path)

        loader = HotModelLoader()
        loader.swap(engine, path)
        loader.swap(engine, path)

        history = loader.get_history()
        assert len(history) == 2

    def test_get_stats(self, tmp_path):
        """测试获取统计"""
        engine = create_predictive_coding_engine("stats")
        path = str(tmp_path / "model.json")
        ModelPersistence.save(engine, path)

        loader = HotModelLoader()
        loader.swap(engine, path)

        stats = loader.get_stats()
        assert "has_backup" in stats
        assert "history_count" in stats
        assert stats["history_count"] == 1


# ============================================================
# ModelVersionManager 测试
# ============================================================

class TestModelVersionManager:
    """模型版本管理器测试"""

    def test_save_version(self, tmp_path):
        """测试保存版本"""
        manager = ModelVersionManager(str(tmp_path / "versions"))
        engine = create_predictive_coding_engine("version_test")

        path = manager.save_version(engine, "v1.0")
        assert path is not None
        assert os.path.exists(path)

    def test_load_version(self, tmp_path):
        """测试加载版本"""
        manager = ModelVersionManager(str(tmp_path / "versions"))
        engine = create_predictive_coding_engine("load_version")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})

        manager.save_version(engine, "v1.0")

        snapshot = manager.load_version("v1.0")
        assert snapshot is not None
        assert snapshot.engine_id == "load_version"

    def test_load_nonexistent_version(self, tmp_path):
        """测试加载不存在的版本"""
        manager = ModelVersionManager(str(tmp_path / "versions"))
        snapshot = manager.load_version("nonexistent")
        assert snapshot is None

    def test_list_versions(self, tmp_path):
        """测试列出版本"""
        manager = ModelVersionManager(str(tmp_path / "versions"))
        engine = create_predictive_coding_engine("list_test")

        manager.save_version(engine, "v1.0")
        time.sleep(0.01)
        manager.save_version(engine, "v1.1")
        time.sleep(0.01)
        manager.save_version(engine, "v2.0")

        versions = manager.list_versions()
        assert len(versions) == 3
        tags = [v["tag"] for v in versions]
        assert "v1.0" in tags
        assert "v1.1" in tags
        assert "v2.0" in tags

    def test_get_stats(self, tmp_path):
        """测试获取统计"""
        manager = ModelVersionManager(str(tmp_path / "versions"))
        engine = create_predictive_coding_engine("stats_test")

        manager.save_version(engine, "v1.0")
        manager.save_version(engine, "v1.1")

        stats = manager.get_stats()
        assert stats["total_versions"] == 2
        assert stats["latest_version"] == "v1.1"


# ============================================================
# 便捷函数测试
# ============================================================

class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_save_model(self, tmp_path):
        """测试save_model函数"""
        engine = create_predictive_coding_engine("conv_save")
        path = str(tmp_path / "model.json")

        success = save_model(engine, path)
        assert success is True

    def test_load_model(self, tmp_path):
        """测试load_model函数"""
        engine = create_predictive_coding_engine("conv_load")
        path = str(tmp_path / "model.json")
        save_model(engine, path)

        snapshot = load_model(path)
        assert snapshot is not None
        assert snapshot.engine_id == "conv_load"

    def test_create_hot_loader(self):
        """测试create_hot_loader函数"""
        loader = create_hot_loader()
        assert isinstance(loader, HotModelLoader)

    def test_create_version_manager(self, tmp_path):
        """测试create_version_manager函数"""
        manager = create_version_manager(str(tmp_path / "vm"))
        assert isinstance(manager, ModelVersionManager)


# ============================================================
# 第一性原理验证测试
# ============================================================

class TestFirstPrinciples:
    """第一性原理验证"""

    def test_state_persistence(self, tmp_path):
        """测试状态持久化（第一性原理）"""
        engine = create_predictive_coding_engine("persist_test")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})
        engine.run_cycle("m1", {"v": 0.4}, {"v": 0.6})
        engine._param_priors["learned_param"] = (0.7, 0.1)

        original_surprise = engine._total_surprise
        original_cycle = engine._cycle_count

        # 保存
        path = str(tmp_path / "persist.json")
        ModelPersistence.save(engine, path)

        # 创建新引擎并加载
        new_engine = create_predictive_coding_engine("new_engine")
        snapshot = ModelPersistence.load(path)
        ModelPersistence.apply_snapshot(new_engine, snapshot)

        # 验证状态恢复
        assert new_engine._total_surprise == original_surprise
        assert new_engine._cycle_count == original_cycle
        assert "learned_param" in new_engine._param_priors

    def test_hot_loading_safety(self, tmp_path):
        """测试热加载安全性（第一性原理：加载失败保持原状态）"""
        engine = create_predictive_coding_engine("safety_test")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})
        original_surprise = engine._total_surprise

        # 尝试加载不存在的文件
        loader = HotModelLoader()
        success = loader.swap(engine, "/nonexistent/model.json", backup=True)

        # 应失败，但原状态不变
        assert success is False
        assert engine._total_surprise == original_surprise

    def test_version_traceability(self, tmp_path):
        """测试版本可追溯性（第一性原理）"""
        manager = ModelVersionManager(str(tmp_path / "trace"))
        engine = create_predictive_coding_engine("trace_test")

        # 保存多个版本
        manager.save_version(engine, "v1.0")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})
        manager.save_version(engine, "v1.1")

        versions = manager.list_versions()
        assert len(versions) == 2

        # 可以加载任意版本
        v1_snapshot = manager.load_version("v1.0")
        v11_snapshot = manager.load_version("v1.1")
        assert v1_snapshot is not None
        assert v11_snapshot is not None

    def test_crash_recovery(self, tmp_path):
        """测试崩溃恢复（第一性原理）"""
        engine = create_predictive_coding_engine("crash_test")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})
        engine._param_priors["important_param"] = (0.9, 0.05)

        # 保存检查点
        path = str(tmp_path / "checkpoint.json")
        ModelPersistence.save(engine, path, metadata={"checkpoint": True})

        # 模拟崩溃：创建全新引擎
        recovered_engine = create_predictive_coding_engine("recovered")

        # 从检查点恢复
        snapshot = ModelPersistence.load(path)
        assert snapshot is not None
        assert snapshot.metadata.get("checkpoint") is True

        ModelPersistence.apply_snapshot(recovered_engine, snapshot)

        # 验证恢复
        assert "important_param" in recovered_engine._param_priors
        assert recovered_engine._param_priors["important_param"] == (0.9, 0.05)
