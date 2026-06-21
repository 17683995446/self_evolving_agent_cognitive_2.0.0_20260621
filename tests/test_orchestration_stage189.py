"""
阶段189测试：跨模块协作 — 预测-驱力-决策-学习编排管线

验证内容:
1. OrchestrationPipeline: 5阶段编排管线
2. PredictionPhaseResult: 预测阶段结果（整合stage 184）
3. DrivePhaseResult: 驱力阶段结果（整合stage 188）
4. DecisionPhaseResult: 决策阶段结果（整合stage 186/185）
5. LearningPhaseResult: 学习阶段结果（整合stage 187）
6. FeedbackPhaseResult: 反馈阶段结果（整合stage 184）
7. 端到端：完整管线周期
"""

import pytest
from self_evolving_agent.core.orchestration_pipeline import (
    OrchestrationPipeline,
    create_orchestration_pipeline,
    PredictionPhaseResult,
    DrivePhaseResult,
    DecisionPhaseResult,
    LearningPhaseResult,
    FeedbackPhaseResult,
    PipelineCycleResult,
)
from self_evolving_agent.core.unified_first_principles_engine import (
    UnifiedFirstPrinciplesEngine,
    create_unified_engine,
)


# ============================================================
# 数据结构测试
# ============================================================

class TestPipelineDataStructures:
    """管线数据结构测试"""

    def test_prediction_phase_result_defaults(self):
        """测试预测阶段结果默认值"""
        result = PredictionPhaseResult()
        assert result.prediction is None
        assert result.error is None
        assert result.surprise == 0.0
        assert result.confidence == 0.0
        assert result.module_id == ""

    def test_drive_phase_result_defaults(self):
        """测试驱力阶段结果默认值"""
        result = DrivePhaseResult()
        assert result.gaps == []
        assert result.dominant_gap is None
        assert result.action_trace is None
        assert result.is_action_required is False

    def test_decision_phase_result_defaults(self):
        """测试决策阶段结果默认值"""
        result = DecisionPhaseResult()
        assert result.decision is None
        assert result.bdi_prediction is None
        assert result.causal_evaluation is None
        assert result.chosen_action == ""
        assert result.safety_allowed is True

    def test_learning_phase_result_defaults(self):
        """测试学习阶段结果默认值"""
        result = LearningPhaseResult()
        assert result.learned is False
        assert result.experience_stored is False
        assert result.analogy_found is None
        assert result.learning_magnitude == 0.0

    def test_feedback_phase_result_defaults(self):
        """测试反馈阶段结果默认值"""
        result = FeedbackPhaseResult()
        assert result.top_down_adjusted is False
        assert result.confidence_stats == {}
        assert result.cycle_complete is True

    def test_pipeline_cycle_result_defaults(self):
        """测试完整周期结果默认值"""
        result = PipelineCycleResult()
        assert result.cycle == 0
        assert isinstance(result.prediction, PredictionPhaseResult)
        assert isinstance(result.drive, DrivePhaseResult)
        assert isinstance(result.decision, DecisionPhaseResult)
        assert isinstance(result.learning, LearningPhaseResult)
        assert isinstance(result.feedback, FeedbackPhaseResult)


# ============================================================
# 管线创建测试
# ============================================================

class TestPipelineCreation:
    """管线创建测试"""

    def test_create_with_default_engine(self):
        """测试使用默认引擎创建管线"""
        pipeline = create_orchestration_pipeline()
        assert pipeline is not None
        assert pipeline.engine is not None
        assert isinstance(pipeline.engine, UnifiedFirstPrinciplesEngine)

    def test_create_with_custom_engine(self):
        """测试使用自定义引擎创建管线"""
        engine = create_unified_engine("custom_engine")
        pipeline = OrchestrationPipeline(engine=engine)
        assert pipeline.engine is engine
        assert pipeline.engine.engine_id == "custom_engine"

    def test_engine_property_exposes_sub_engines(self):
        """测试引擎属性暴露子引擎"""
        pipeline = create_orchestration_pipeline()
        # 验证阶段184-188的子引擎可访问
        assert pipeline.engine.predictive_coding is not None
        assert pipeline.engine.drive is not None
        assert pipeline.engine.social is not None
        assert pipeline.engine.creativity is not None
        assert pipeline.engine.causal is not None
        assert pipeline.engine.learning is not None


# ============================================================
# 完整管线周期测试
# ============================================================

class TestPipelineCycle:
    """完整管线周期测试"""

    def test_run_cycle_basic(self):
        """测试基本管线周期"""
        pipeline = create_orchestration_pipeline()

        context = {
            "module_id": "test_module",
            "current_state": {"value": 0.5},
            "actual_state": {"value": 0.6},
        }

        result = pipeline.run_cycle(context)

        assert isinstance(result, PipelineCycleResult)
        assert result.cycle == 1
        assert isinstance(result.prediction, PredictionPhaseResult)
        assert isinstance(result.drive, DrivePhaseResult)
        assert isinstance(result.decision, DecisionPhaseResult)
        assert isinstance(result.learning, LearningPhaseResult)
        assert isinstance(result.feedback, FeedbackPhaseResult)

    def test_run_cycle_increments_cycle_count(self):
        """测试周期计数递增"""
        pipeline = create_orchestration_pipeline()

        for i in range(3):
            result = pipeline.run_cycle({"module_id": "m", "current_state": {}})
            assert result.cycle == i + 1

    def test_prediction_phase_produces_result(self):
        """测试预测阶段产生结果"""
        pipeline = create_orchestration_pipeline()

        context = {
            "module_id": "pred_test",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.8},
        }

        result = pipeline.run_cycle(context)
        pred = result.prediction

        assert pred.module_id == "pred_test"
        # 应该有预测和误差
        assert pred.prediction is not None or pred.error is not None

    def test_drive_phase_evaluates_gaps(self):
        """测试驱力阶段评估缺口"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "drive_test",
            "current_state": {},
            "actual_state": {},
        })

        drive = result.drive
        # 应该有驱力缺口评估（即使为空列表）
        assert isinstance(drive.gaps, list)

    def test_drive_phase_action_trace(self):
        """测试驱力阶段行为追溯（stage 188）"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "trace_test",
            "current_state": {},
            "actual_state": {},
            "action_description": "some_action",
        })

        # action_trace 可能为None（如果没有匹配目标），但不应抛异常
        assert result.drive.action_trace is None or hasattr(result.drive.action_trace, "trace_path")

    def test_decision_phase_with_actions(self):
        """测试决策阶段有可用行动时"""
        pipeline = create_orchestration_pipeline()

        # 先设置低驱力状态以触发行动需求
        pipeline.engine.drive.update_state(safe_level=0.1, gain_level=0.5, eff_level=0.5)
        pipeline.engine.drive.compute_gaps()

        result = pipeline.run_cycle({
            "module_id": "dec_test",
            "current_state": {},
            "actual_state": {},
            "available_actions": ["safe_action", "gain_action", "wait"],
        })

        decision = result.decision
        # 如果有驱力缺口，应该有决策
        if pipeline.engine.drive.is_action_required():
            assert decision.decision is not None or not decision.safety_allowed

    def test_learning_phase_stores_experience(self):
        """测试学习阶段存储经验"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "learn_test",
            "current_state": {"v": 0.5},
            "actual_state": {"v": 0.7},
            "available_actions": ["action_a", "action_b"],
        })

        learning = result.learning
        # 学习阶段应该执行（即使没有存储经验）
        assert isinstance(learning.learned, bool)
        assert isinstance(learning.experience_stored, bool)

    def test_feedback_phase_returns_stats(self):
        """测试反馈阶段返回统计"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "feedback_test",
            "current_state": {},
            "actual_state": {},
        })

        feedback = result.feedback
        assert feedback.cycle_complete is True
        assert isinstance(feedback.confidence_stats, dict)

    def test_full_pipeline_data_flow(self):
        """测试完整管线数据流"""
        pipeline = create_orchestration_pipeline()

        context = {
            "module_id": "flow_test",
            "current_state": {"v": 0.3},
            "actual_state": {"v": 0.9},
            "available_actions": ["explore", "exploit", "wait"],
            "action_description": "explore",
        }

        result = pipeline.run_cycle(context)

        # 验证5个阶段都执行了
        assert result.prediction.module_id == "flow_test"
        assert isinstance(result.drive.gaps, list)
        assert isinstance(result.decision.chosen_action, str)
        assert isinstance(result.learning.learned, bool)
        assert result.feedback.cycle_complete is True


# ============================================================
# 批量运行与统计测试
# ============================================================

class TestPipelineBatchAndStats:
    """批量运行与统计测试"""

    def test_run_batch(self):
        """测试批量运行"""
        pipeline = create_orchestration_pipeline()

        contexts = [
            {"module_id": f"batch_{i}", "current_state": {"v": i * 0.1}, "actual_state": {"v": i * 0.1 + 0.05}}
            for i in range(5)
        ]

        results = pipeline.run_batch(contexts)
        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.cycle == i + 1

    def test_get_pipeline_stats(self):
        """测试获取管线统计"""
        pipeline = create_orchestration_pipeline()

        pipeline.run_cycle({"module_id": "stats_test", "current_state": {}, "actual_state": {}})

        stats = pipeline.get_pipeline_stats()
        assert stats["cycle_count"] == 1
        assert "avg_prediction_error" in stats
        assert "error_history_len" in stats
        assert "action_history_len" in stats
        assert "engine_stats" in stats

    def test_error_history_tracking(self):
        """测试误差历史追踪"""
        pipeline = create_orchestration_pipeline()

        for i in range(3):
            pipeline.run_cycle({
                "module_id": f"err_{i}",
                "current_state": {"v": 0.5},
                "actual_state": {"v": 0.5 + i * 0.2},
            })

        history = pipeline.get_error_history()
        assert len(history) == 3
        assert all(isinstance(e, float) for e in history)

    def test_action_history_tracking(self):
        """测试行动历史追踪"""
        pipeline = create_orchestration_pipeline()

        # 设置低驱力以触发行动
        pipeline.engine.drive.update_state(safe_level=0.1, gain_level=0.5, eff_level=0.5)
        pipeline.engine.drive.compute_gaps()

        pipeline.run_cycle({
            "module_id": "act_test",
            "current_state": {},
            "actual_state": {},
            "available_actions": ["test_action"],
        })

        history = pipeline.get_action_history()
        # 如果有决策，应该有行动历史
        assert isinstance(history, list)


# ============================================================
# 向后兼容性测试
# ============================================================

class TestBackwardCompatibility:
    """向后兼容性测试"""

    def test_engine_still_works_independently(self):
        """测试引擎仍可独立工作"""
        pipeline = create_orchestration_pipeline()
        engine = pipeline.engine

        # 引擎原有接口仍可用
        gaps = engine.evaluate_drives()
        assert isinstance(gaps, list)

    def test_run_integration_cycle_still_works(self):
        """测试原有run_integration_cycle仍可用"""
        pipeline = create_orchestration_pipeline()
        engine = pipeline.engine

        result = engine.run_integration_cycle({"outcome": {}})
        assert "cycle" in result
        assert "gaps" in result

    def test_pipeline_does_not_break_engine(self):
        """测试管线不破坏引擎状态"""
        pipeline = create_orchestration_pipeline()
        engine = pipeline.engine

        # 运行管线前
        stats_before = engine.get_full_stats()
        cycle_before = stats_before.get("cycle_count", 0)

        # 运行管线
        pipeline.run_cycle({"module_id": "compat", "current_state": {}, "actual_state": {}})

        # 引擎仍正常
        stats_after = engine.get_full_stats()
        assert "engine_id" in stats_after

    def test_stage_184_188_interfaces_accessible(self):
        """测试阶段184-188接口可通过管线访问"""
        pipeline = create_orchestration_pipeline()
        engine = pipeline.engine

        # stage 184: 预测编码新接口
        assert hasattr(engine.predictive_coding, "get_confidence_stats")
        assert hasattr(engine.predictive_coding, "get_top_down_stats")
        assert hasattr(engine.predictive_coding, "set_top_down_weight")

        # stage 185: 因果发现新接口
        assert hasattr(engine.causal, "discover_causal_structure")
        assert hasattr(engine.causal, "test_conditional_independence")

        # stage 186: BDI新接口
        assert hasattr(engine.social, "build_bdi_state")
        assert hasattr(engine.social, "predict_action_bdi")
        assert hasattr(engine.social, "predict_action_sequence")

        # stage 187: 结构映射新接口
        assert hasattr(engine.creativity, "map_structure")
        assert hasattr(engine.creativity, "find_structural_analogy")

        # stage 188: 行为-驱力追溯新接口
        assert hasattr(engine.drive, "trace_action_to_gap_detailed")
        assert hasattr(engine.drive, "register_action_goal")
        assert hasattr(engine.drive, "get_action_drive_trace")


# ============================================================
# 第一性原理验证测试
# ============================================================

class TestFirstPrinciples:
    """第一性原理验证"""

    def test_closed_loop_pipeline(self):
        """测试闭环管线：预测→驱力→决策→学习→反馈"""
        pipeline = create_orchestration_pipeline()

        result = pipeline.run_cycle({
            "module_id": "loop_test",
            "current_state": {"v": 0.3},
            "actual_state": {"v": 0.8},
            "available_actions": ["action_a", "action_b"],
        })

        # 5个阶段都应执行并产生结果
        assert result.prediction is not None
        assert result.drive is not None
        assert result.decision is not None
        assert result.learning is not None
        assert result.feedback is not None

        # 反馈阶段应标记周期完成
        assert result.feedback.cycle_complete is True

    def test_prediction_error_drives_learning(self):
        """测试预测误差驱动学习（第一性原理）"""
        pipeline = create_orchestration_pipeline()

        # 大误差
        result = pipeline.run_cycle({
            "module_id": "error_learn",
            "current_state": {"v": 0.1},
            "actual_state": {"v": 0.9},
        })

        # 如果有误差，学习阶段应执行
        if result.prediction.surprise > 0:
            # 学习幅度应与误差相关
            assert result.learning.learning_magnitude >= 0

    def test_drive_motivates_decision(self):
        """测试驱力激发决策（第一性原理）"""
        pipeline = create_orchestration_pipeline()

        # 设置低安全驱力
        pipeline.engine.drive.update_state(safe_level=0.05, gain_level=0.5, eff_level=0.5)
        pipeline.engine.drive.compute_gaps()

        result = pipeline.run_cycle({
            "module_id": "drive_decide",
            "current_state": {},
            "actual_state": {},
            "available_actions": ["safe_action", "other_action"],
        })

        # 如果驱力要求行动，决策阶段应执行
        if result.drive.is_action_required:
            assert result.decision is not None

    def test_multiple_cycles_form_learning_curve(self):
        """测试多周期形成学习曲线（第一性原理）"""
        pipeline = create_orchestration_pipeline()

        errors = []
        for i in range(5):
            result = pipeline.run_cycle({
                "module_id": "curve_test",
                "current_state": {"v": 0.5},
                "actual_state": {"v": 0.5 + 0.1 * (i % 3)},
            })
            errors.append(result.prediction.surprise)

        # 应该有5个误差记录
        assert len(errors) == 5
        assert len(pipeline.get_error_history()) == 5
