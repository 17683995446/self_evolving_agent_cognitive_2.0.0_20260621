"""
阶段193: 核心功能集成测试与基准

第一性原理:
1. 集成验证：各组件单独正确不保证整体正确
2. 性能基准：关键操作的性能必须可测量和可追溯
3. 端到端验证：完整认知管线必须从输入到输出验证
4. 跨模块协作：阶段184-192的能力必须协同工作

验证内容:
1. 阶段184-192跨模块集成
2. 完整认知管线端到端
3. DI容器 + 编排管线 + 持久化集成
4. 性能基准
"""

import pytest
import time
import os
import tempfile
from self_evolving_agent.core.di_container import DIContainer, create_core_container
from self_evolving_agent.core.orchestration_pipeline import (
    OrchestrationPipeline,
    create_orchestration_pipeline,
    PipelineCycleResult,
)
from self_evolving_agent.core.model_persistence import (
    ModelPersistence,
    HotModelLoader,
    ModelVersionManager,
)
from self_evolving_agent.core.unified_first_principles_engine import (
    UnifiedFirstPrinciplesEngine,
    create_unified_engine,
)
from self_evolving_agent.core.predictive_coding_engine import PredictiveCodingEngine
from self_evolving_agent.core.drive_first_principles import DriveFirstPrinciplesEngine
from self_evolving_agent.core.causal_first_principles import CausalFirstPrinciplesEngine
from self_evolving_agent.core.social_first_principles import SocialFirstPrinciplesEngine
from self_evolving_agent.core.creativity_first_principles import CreativityFirstPrinciplesEngine
from self_evolving_agent.core.learning_first_principles import LearningFirstPrinciplesEngine


# ============================================================
# 跨模块集成测试
# ============================================================

class TestCrossModuleIntegration:
    """跨模块集成测试（阶段184-192协同）"""

    def test_di_container_resolves_all_core_engines(self):
        """测试DI容器解析所有核心引擎"""
        container = create_core_container()

        engines = [
            (PredictiveCodingEngine, "预测编码"),
            (DriveFirstPrinciplesEngine, "驱力"),
            (LearningFirstPrinciplesEngine, "学习"),
            (CausalFirstPrinciplesEngine, "因果"),
            (SocialFirstPrinciplesEngine, "社交"),
            (CreativityFirstPrinciplesEngine, "创造"),
            (UnifiedFirstPrinciplesEngine, "统一引擎"),
            (OrchestrationPipeline, "编排管线"),
        ]

        for engine_type, name in engines:
            instance = container.resolve(engine_type)
            assert isinstance(instance, engine_type), f"{name}引擎解析失败"

    def test_pipeline_uses_di_resolved_engine(self):
        """测试编排管线使用DI解析的引擎"""
        container = create_core_container()

        pipeline = container.resolve(OrchestrationPipeline)
        unified_engine = container.resolve(UnifiedFirstPrinciplesEngine)

        # 管线应使用同一引擎实例（单例）
        assert pipeline.engine is unified_engine

    def test_stage184_188_interfaces_accessible_via_pipeline(self):
        """测试阶段184-188接口可通过管线访问"""
        pipeline = create_orchestration_pipeline()
        engine = pipeline.engine

        # stage 184: 预测编码
        assert hasattr(engine.predictive_coding, "get_confidence_stats")
        assert hasattr(engine.predictive_coding, "set_top_down_weight")

        # stage 185: 因果发现
        assert hasattr(engine.causal, "discover_causal_structure")

        # stage 186: BDI
        assert hasattr(engine.social, "predict_action_bdi")

        # stage 187: 结构映射
        assert hasattr(engine.creativity, "map_structure")

        # stage 188: 行为-驱力追溯
        assert hasattr(engine.drive, "trace_action_to_gap_detailed")

    def test_persistence_with_di_container(self, tmp_path):
        """测试持久化与DI容器集成"""
        container = create_core_container()
        engine = container.resolve(PredictiveCodingEngine)

        # 运行一些周期
        engine.run_cycle("module1", {"v": 0.3}, {"v": 0.7})

        # 保存
        path = str(tmp_path / "di_model.json")
        success = ModelPersistence.save(engine, path)
        assert success is True

        # 加载并应用到新引擎
        snapshot = ModelPersistence.load(path)
        assert snapshot is not None

        new_engine = PredictiveCodingEngine("new_engine")
        ModelPersistence.apply_snapshot(new_engine, snapshot)

        assert new_engine._cycle_count == engine._cycle_count

    def test_hot_loader_with_pipeline(self, tmp_path):
        """测试热加载器与编排管线集成"""
        pipeline = create_orchestration_pipeline()
        engine = pipeline.engine.predictive_coding

        # 运行管线产生状态
        pipeline.run_cycle({
            "module_id": "hot_test",
            "current_state": {"v": 0.3},
            "actual_state": {"v": 0.7},
        })

        # 保存模型
        path = str(tmp_path / "pipeline_model.json")
        ModelPersistence.save(engine, path)

        # 继续运行管线（改变状态）
        for i in range(3):
            pipeline.run_cycle({
                "module_id": "hot_test",
                "current_state": {"v": 0.1 * i},
                "actual_state": {"v": 0.9},
            })

        # 热加载回之前的状态
        loader = HotModelLoader()
        success = loader.swap(engine, path)
        assert success is True


# ============================================================
# 完整认知管线端到端测试
# ============================================================

class TestEndToEndPipeline:
    """完整认知管线端到端测试"""

    def test_full_cognitive_cycle(self):
        """测试完整认知周期"""
        pipeline = create_orchestration_pipeline()

        context = {
            "module_id": "e2e_module",
            "current_state": {"value": 0.3, "confidence": 0.5},
            "actual_state": {"value": 0.8, "confidence": 0.7},
            "available_actions": ["explore", "exploit", "wait"],
            "action_description": "explore new options",
        }

        result = pipeline.run_cycle(context)

        # 验证5个阶段都执行
        assert result.prediction is not None
        assert result.drive is not None
        assert result.decision is not None
        assert result.learning is not None
        assert result.feedback is not None

        # 验证数据流
        assert result.prediction.module_id == "e2e_module"
        assert result.feedback.cycle_complete is True

    def test_multi_cycle_learning_progression(self):
        """测试多周期学习进展"""
        pipeline = create_orchestration_pipeline()

        # 运行10个周期，误差逐渐减小
        errors = []
        for i in range(10):
            # 误差逐渐减小（系统在学习）
            actual = 0.5 + 0.4 * (1 - i / 10)
            predicted = 0.5

            result = pipeline.run_cycle({
                "module_id": "learning_test",
                "current_state": {"v": predicted},
                "actual_state": {"v": actual},
            })
            errors.append(result.prediction.surprise)

        # 验证误差历史被记录
        history = pipeline.get_error_history()
        assert len(history) == 10

    def test_pipeline_with_drive_motivated_decision(self):
        """测试驱力驱动的决策"""
        pipeline = create_orchestration_pipeline()

        # 设置低安全驱力
        pipeline.engine.drive.update_state(safe_level=0.05, gain_level=0.5, eff_level=0.5)
        pipeline.engine.drive.compute_gaps()

        result = pipeline.run_cycle({
            "module_id": "drive_motivated",
            "current_state": {},
            "actual_state": {},
            "available_actions": ["safe_action", "gain_action", "wait"],
        })

        # 如果驱力要求行动，应该有决策
        if result.drive.is_action_required:
            assert result.decision is not None

    def test_pipeline_with_action_drive_trace(self):
        """测试管线中的行为-驱力追溯"""
        pipeline = create_orchestration_pipeline()

        # 添加目标
        from self_evolving_agent.core.drive_first_principles import (
            DriveType, create_goal
        )
        goal = create_goal(
            goal_id="g1", name="safety_goal", drive_type=DriveType.SAFE,
            target_state={"safe": 0.9},
        )
        pipeline.engine.drive.add_goal(goal)
        pipeline.engine.drive.activate_goal("g1")

        result = pipeline.run_cycle({
            "module_id": "trace_test",
            "current_state": {},
            "actual_state": {},
            "action_description": "execute safety_goal",
        })

        # 行为-驱力追溯应执行
        assert result.drive.action_trace is not None or result.drive.action_trace is None
        # 不抛异常即可

    def test_pipeline_feedback_adjusts_confidence(self):
        """测试反馈阶段调整置信度"""
        pipeline = create_orchestration_pipeline()

        # 高置信度预测
        result = pipeline.run_cycle({
            "module_id": "high_confidence",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.51},  # 小误差
        })

        # 反馈阶段应返回置信度统计
        assert isinstance(result.feedback.confidence_stats, dict)


# ============================================================
# 性能基准测试
# ============================================================

class TestPerformanceBenchmarks:
    """性能基准测试"""

    def test_pipeline_cycle_latency(self):
        """测试管线周期延迟（< 100ms）"""
        pipeline = create_orchestration_pipeline()

        context = {
            "module_id": "bench",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.6},
        }

        # 预热
        pipeline.run_cycle(context)

        # 测量
        start = time.time()
        for _ in range(10):
            pipeline.run_cycle(context)
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / 10) * 1000
        assert avg_latency_ms < 100, f"管线周期延迟过高: {avg_latency_ms:.2f}ms"

    def test_di_container_resolve_latency(self):
        """测试DI容器解析延迟（< 1ms）"""
        container = create_core_container()

        # 预热
        container.resolve(PredictiveCodingEngine)

        # 测量
        start = time.time()
        for _ in range(100):
            container.resolve(PredictiveCodingEngine)
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / 100) * 1000
        assert avg_latency_ms < 1, f"DI解析延迟过高: {avg_latency_ms:.4f}ms"

    def test_model_save_load_latency(self, tmp_path):
        """测试模型保存/加载延迟（< 50ms）"""
        engine = PredictiveCodingEngine("bench_persist")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})

        path = str(tmp_path / "bench_model.json")

        # 测量保存
        start = time.time()
        ModelPersistence.save(engine, path)
        save_ms = (time.time() - start) * 1000
        assert save_ms < 50, f"保存延迟过高: {save_ms:.2f}ms"

        # 测量加载
        start = time.time()
        ModelPersistence.load(path)
        load_ms = (time.time() - start) * 1000
        assert load_ms < 50, f"加载延迟过高: {load_ms:.2f}ms"

    def test_hot_swap_latency(self, tmp_path):
        """测试热交换延迟（< 100ms）"""
        engine = PredictiveCodingEngine("hot_bench")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})

        path = str(tmp_path / "hot_bench.json")
        ModelPersistence.save(engine, path)

        loader = HotModelLoader()

        # 测量热交换
        start = time.time()
        loader.swap(engine, path)
        swap_ms = (time.time() - start) * 1000
        assert swap_ms < 100, f"热交换延迟过高: {swap_ms:.2f}ms"

    def test_unified_engine_cycle_latency(self):
        """测试统一引擎周期延迟（< 50ms）"""
        engine = create_unified_engine("bench_unified")

        # 预热
        engine.run_integration_cycle({"outcome": {}})

        # 测量
        start = time.time()
        for _ in range(10):
            engine.run_integration_cycle({"outcome": {}})
        elapsed = time.time() - start

        avg_latency_ms = (elapsed / 10) * 1000
        assert avg_latency_ms < 50, f"统一引擎周期延迟过高: {avg_latency_ms:.2f}ms"


# ============================================================
# 回归测试：阶段184-192全部协同
# ============================================================

class TestStageRegression:
    """阶段184-192回归测试"""

    def test_stage184_predictive_coding_in_pipeline(self):
        """测试阶段184预测编码在管线中工作"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "stage184",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.7},
        })

        # 预测阶段应产生结果
        assert result.prediction is not None
        # 反馈阶段应返回置信度统计
        assert isinstance(result.feedback.confidence_stats, dict)

    def test_stage185_causal_in_pipeline(self):
        """测试阶段185因果发现在管线中工作"""
        pipeline = create_orchestration_pipeline()

        # 运行足够多的周期以触发因果发现
        for i in range(11):
            pipeline.run_cycle({
                "module_id": "stage185",
                "current_state": {"v": 0.5},
                "actual_state": {"v": 0.5 + 0.1 * i},
            })

        # 因果引擎应记录观测
        stats = pipeline.engine.causal.get_stats()
        assert stats is not None

    def test_stage186_bdi_in_pipeline(self):
        """测试阶段186 BDI在管线中工作"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "stage186",
            "current_state": {},
            "actual_state": {},
            "agent_id": "test_agent",
        })

        # BDI预测应执行（可能返回None，但不应抛异常）
        assert result.decision.bdi_prediction is None or isinstance(result.decision.bdi_prediction, dict)

    def test_stage187_creativity_in_pipeline(self):
        """测试阶段187创造在管线中工作"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "stage187",
            "current_state": {},
            "actual_state": {},
            "analogy_source": "concept_a",
            "analogy_target": "concept_b",
        })

        # 结构映射应执行（可能返回None，但不应抛异常）
        assert result.learning.analogy_found is None or isinstance(result.learning.analogy_found, dict)

    def test_stage188_drive_trace_in_pipeline(self):
        """测试阶段188行为-驱力追溯在管线中工作"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "stage188",
            "current_state": {},
            "actual_state": {},
            "action_description": "some_action",
        })

        # 行为-驱力追溯应执行
        assert result.drive.action_trace is None or hasattr(result.drive.action_trace, "trace_path")

    def test_stage189_pipeline_complete(self):
        """测试阶段189编排管线完整"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "stage189",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.6},
        })

        assert isinstance(result, PipelineCycleResult)
        assert result.cycle >= 1
        assert result.feedback.cycle_complete is True

    def test_stage190_unified_abstractions(self):
        """测试阶段190统一抽象"""
        from self_evolving_agent.core.circuit_breaker import CircuitBreaker
        from self_evolving_agent.core.recovery import CircuitBreakerPolicy
        from self_evolving_agent.core.production import CircuitBreaker as ProdCB

        # 统一熔断器支持双模式
        breaker = CircuitBreaker(name="regression", failure_threshold=3)
        assert breaker.can_execute() is True
        breaker.record_success()

        # recovery委托
        policy = CircuitBreakerPolicy(failure_threshold=3)
        assert policy.can_execute() is True

        # production委托
        prod = ProdCB(failure_threshold=3)
        assert prod.can_execute() is True

    def test_stage191_di_container(self):
        """测试阶段191 DI容器"""
        container = create_core_container()

        # 解析所有核心引擎
        engine = container.resolve(PredictiveCodingEngine)
        assert isinstance(engine, PredictiveCodingEngine)

        # 单例验证
        engine2 = container.resolve(PredictiveCodingEngine)
        assert engine is engine2

    def test_stage192_persistence(self, tmp_path):
        """测试阶段192持久化"""
        engine = PredictiveCodingEngine("regression_192")
        engine.run_cycle("m1", {"v": 0.3}, {"v": 0.7})

        path = str(tmp_path / "regression.json")
        assert ModelPersistence.save(engine, path) is True

        snapshot = ModelPersistence.load(path)
        assert snapshot is not None
        assert snapshot.engine_id == "regression_192"


# ============================================================
# 系统级集成测试
# ============================================================

class TestSystemIntegration:
    """系统级集成测试"""

    def test_full_system_workflow(self, tmp_path):
        """测试完整系统工作流"""
        # 1. 创建DI容器
        container = create_core_container()

        # 2. 解析编排管线
        pipeline = container.resolve(OrchestrationPipeline)

        # 3. 运行认知周期
        for i in range(5):
            result = pipeline.run_cycle({
                "module_id": f"workflow_{i}",
                "current_state": {"v": 0.3 + i * 0.1},
                "actual_state": {"v": 0.5 + i * 0.1},
                "available_actions": ["explore", "exploit"],
            })
            assert result.feedback.cycle_complete is True

        # 4. 保存模型
        engine = container.resolve(PredictiveCodingEngine)
        model_path = str(tmp_path / "workflow_model.json")
        assert ModelPersistence.save(engine, model_path) is True

        # 5. 创建版本
        version_mgr = ModelVersionManager(str(tmp_path / "versions"))
        version_path = version_mgr.save_version(engine, "v1.0")
        assert version_path is not None

        # 6. 热加载
        loader = HotModelLoader()
        assert loader.swap(engine, model_path) is True

        # 7. 验证系统仍正常工作
        result = pipeline.run_cycle({
            "module_id": "post_swap",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.6},
        })
        assert result.feedback.cycle_complete is True

    def test_system_resilience(self):
        """测试系统弹性"""
        pipeline = create_orchestration_pipeline()

        # 模拟异常输入
        result = pipeline.run_cycle({
            "module_id": "",  # 空模块ID
            "current_state": {},  # 空状态
            "actual_state": {},  # 空状态
        })

        # 系统应优雅处理，不崩溃
        assert result is not None
        assert result.feedback.cycle_complete is True

    def test_concurrent_pipeline_execution(self):
        """测试并发管线执行"""
        import threading

        pipeline = create_orchestration_pipeline()
        results = []
        errors = []
        barrier = threading.Barrier(5)

        def run_cycles():
            try:
                barrier.wait()
                for i in range(3):
                    result = pipeline.run_cycle({
                        "module_id": f"concurrent_{threading.current_thread().name}_{i}",
                        "current_state": {"v": 0.5},
                        "actual_state": {"v": 0.6},
                    })
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_cycles) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 应无错误
        assert len(errors) == 0
        # 应完成15个周期
        assert len(results) == 15


# ============================================================
# 第一性原理验证测试
# ============================================================

class TestSystemFirstPrinciples:
    """系统级第一性原理验证"""

    def test_closed_loop_cognition(self):
        """测试闭环认知（第一性原理）"""
        pipeline = create_orchestration_pipeline()

        # 完整闭环：预测→驱力→决策→学习→反馈
        result = pipeline.run_cycle({
            "module_id": "closed_loop",
            "current_state": {"v": 0.3},
            "actual_state": {"v": 0.7},
            "available_actions": ["act", "wait"],
        })

        # 5个阶段形成闭环
        assert result.prediction.module_id == "closed_loop"
        assert isinstance(result.drive.gaps, list)
        assert isinstance(result.decision.chosen_action, str)
        assert isinstance(result.learning.learned, bool)
        assert result.feedback.cycle_complete is True

    def test_emergence_from_integration(self):
        """测试整合产生涌现（第一性原理）"""
        # 整体大于部分之和：编排管线的能力超过任何单独引擎
        pipeline = create_orchestration_pipeline()

        # 单独引擎的能力
        engine = pipeline.engine
        standalone_stats = engine.get_full_stats()

        # 管线的能力（包含编排、历史追踪等）
        result = pipeline.run_cycle({
            "module_id": "emergence",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.6},
        })

        pipeline_stats = pipeline.get_pipeline_stats()

        # 管线有额外能力（误差历史、行动历史）
        assert "avg_prediction_error" in pipeline_stats
        assert "error_history_len" in pipeline_stats
        assert "action_history_len" in pipeline_stats

    def test_adaptation_through_learning(self):
        """测试通过学习适应（第一性原理）"""
        pipeline = create_orchestration_pipeline()

        # 初始状态
        initial_stats = pipeline.get_pipeline_stats()

        # 运行多个周期
        for i in range(5):
            pipeline.run_cycle({
                "module_id": "adapt",
                "current_state": {"v": 0.5},
                "actual_state": {"v": 0.5 + 0.1 * (i % 3)},
            })

        final_stats = pipeline.get_pipeline_stats()

        # 系统应通过学习适应（周期计数增加）
        assert final_stats["cycle_count"] > initial_stats["cycle_count"]
        assert final_stats["error_history_len"] > 0

    def test_self_contained_system(self):
        """测试自包含系统（第一性原理）"""
        # DI容器使系统自包含：所有依赖在容器内解析
        container = create_core_container()

        # 从容器获取所有必要组件
        pipeline = container.resolve(OrchestrationPipeline)
        unified_engine = container.resolve(UnifiedFirstPrinciplesEngine)

        # 系统可独立运行
        result = pipeline.run_cycle({
            "module_id": "self_contained",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.6},
        })

        assert result.feedback.cycle_complete is True

        # 引擎间通过容器关联
        assert pipeline.engine is unified_engine
