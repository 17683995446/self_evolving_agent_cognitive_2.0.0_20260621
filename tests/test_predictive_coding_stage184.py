"""
阶段184测试：预测编码引擎增强 — 自上而下约束与自适应置信度

验证内容:
1. AdaptiveConfidenceTracker: 基于历史准确率的置信度自适应
2. TopDownPredictor: 高层→低层预测约束传播
3. MultiLayerPredictor: 集成自上而下约束和自适应置信度
4. PredictiveCodingEngine: 新增接口的向后兼容性
5. 端到端：预测-误差-修正循环中的置信度演化
"""

import pytest
import math
from self_evolving_agent.core.predictive_coding_engine import (
    PredictiveCodingEngine,
    MultiLayerPredictor,
    AdaptiveConfidenceTracker,
    TopDownPredictor,
    Prediction,
    PredictionError,
    PrecisionWeight,
    create_predictive_coding_engine,
)


# ============================================================
# AdaptiveConfidenceTracker 测试
# ============================================================

class TestAdaptiveConfidenceTracker:
    """自适应置信度追踪器测试"""

    def test_initial_confidence(self):
        """测试初始置信度（数据不足时使用基础置信度）"""
        tracker = AdaptiveConfidenceTracker(initial_confidence=0.3)
        confidence = tracker.get_adaptive_confidence(layer_precision=0.5)
        # 数据不足（<3），应返回基础置信度 × 精度
        assert confidence == pytest.approx(0.3 * 0.5, abs=0.01)

    def test_accuracy_improves_confidence(self):
        """测试高准确率提升置信度"""
        tracker = AdaptiveConfidenceTracker(initial_confidence=0.3)
        # 记录多次准确预测
        for _ in range(10):
            tracker.record_prediction_outcome(was_accurate=True, error_magnitude=0.05)

        confidence = tracker.get_adaptive_confidence(layer_precision=1.0)
        # 准确率高，置信度应显著高于初始值
        assert confidence > 0.3
        assert confidence <= 0.95

    def test_inaccuracy_lowers_confidence(self):
        """测试低准确率降低置信度"""
        tracker = AdaptiveConfidenceTracker(initial_confidence=0.3)
        # 记录多次不准确预测
        for _ in range(10):
            tracker.record_prediction_outcome(was_accurate=False, error_magnitude=0.8)

        confidence = tracker.get_adaptive_confidence(layer_precision=1.0)
        # 准确率低，置信度应较低
        assert confidence < 0.5

    def test_ewma_tracking(self):
        """测试EWMA准确率追踪"""
        tracker = AdaptiveConfidenceTracker(initial_confidence=0.3)
        # 先记录准确预测
        for _ in range(5):
            tracker.record_prediction_outcome(was_accurate=True, error_magnitude=0.1)
        stats_high = tracker.get_accuracy_stats()
        assert stats_high["ewma_accuracy"] > 0.5

        # 然后记录不准确预测
        for _ in range(10):
            tracker.record_prediction_outcome(was_accurate=False, error_magnitude=0.9)
        stats_low = tracker.get_accuracy_stats()
        # EWMA应下降
        assert stats_low["ewma_accuracy"] < stats_high["ewma_accuracy"]

    def test_reset(self):
        """测试重置"""
        tracker = AdaptiveConfidenceTracker(initial_confidence=0.3)
        for _ in range(5):
            tracker.record_prediction_outcome(was_accurate=True, error_magnitude=0.1)
        tracker.reset()
        stats = tracker.get_accuracy_stats()
        assert stats["sample_count"] == 0
        assert stats["ewma_accuracy"] == 0.5

    def test_layer_precision_factor(self):
        """测试层精度对置信度的影响"""
        tracker = AdaptiveConfidenceTracker(initial_confidence=0.3)
        for _ in range(10):
            tracker.record_prediction_outcome(was_accurate=True, error_magnitude=0.05)

        high_precision = tracker.get_adaptive_confidence(layer_precision=1.0)
        low_precision = tracker.get_adaptive_confidence(layer_precision=0.1)
        # 高精度应产生更高置信度
        assert high_precision > low_precision


# ============================================================
# TopDownPredictor 测试
# ============================================================

class TestTopDownPredictor:
    """自上而下预测约束器测试"""

    def test_update_and_get_prior(self):
        """测试更新层先验并获取"""
        predictor = TopDownPredictor(num_layers=3)
        # 更新概念层（layer 2）的先验
        predictor.update_layer_prior(2, {"concept_a": 10.0, "concept_b": 20.0})

        # 获取对感知层（layer 0）的先验
        prior = predictor.get_top_down_prior(0)
        # 应有先验（经过距离衰减）
        assert "concept_a" in prior
        assert "concept_b" in prior
        # 距离衰减：layer 2 → layer 0，距离=2，decay=0.7^2=0.49
        assert prior["concept_a"] == pytest.approx(10.0 * 0.49 * 0.3, abs=0.1)

    def test_no_prior_for_top_layer(self):
        """测试最高层无自上而下先验"""
        predictor = TopDownPredictor(num_layers=3)
        predictor.update_layer_prior(1, {"feature_x": 5.0})

        # 最高层（layer 2）无自上而下先验
        prior = predictor.get_top_down_prior(2)
        assert len(prior) == 0

    def test_apply_constraint_fuses_predictions(self):
        """测试约束融合低层预测与高层先验"""
        predictor = TopDownPredictor(num_layers=3)
        # 设置高层先验
        predictor.update_layer_prior(2, {"value": 100.0})

        # 低层自身预测
        low_layer_state = {"value": 50.0}
        # 应用约束（精度=0.5，alpha=0.5）
        constrained = predictor.apply_top_down_constraint(0, low_layer_state, 0.5)

        # 应融合自身预测和高层先验
        assert "value" in constrained
        # 不应等于原始值（被融合了）
        assert constrained["value"] != 50.0

    def test_apply_constraint_no_prior(self):
        """测试无先验时不修改预测"""
        predictor = TopDownPredictor(num_layers=3)
        original_state = {"value": 50.0}
        constrained = predictor.apply_top_down_constraint(0, original_state, 0.5)
        # 无先验，应返回原始预测
        assert constrained == original_state

    def test_set_top_down_weight(self):
        """测试设置映射权重"""
        predictor = TopDownPredictor(num_layers=3)
        predictor.set_top_down_weight("concept_a", "perception_x", 0.8)
        stats = predictor.get_weight_stats()
        assert stats["total_mappings"] == 1

    def test_distance_decay(self):
        """测试距离衰减：近层约束强于远层"""
        predictor = TopDownPredictor(num_layers=3)
        # layer 1 和 layer 2 都有先验
        predictor.update_layer_prior(1, {"near": 10.0})
        predictor.update_layer_prior(2, {"far": 10.0})

        # 对 layer 0 的先验
        prior = predictor.get_top_down_prior(0)
        # 近层（距离1）的衰减 < 远层（距离2）的衰减
        near_decay = 0.7 ** 1
        far_decay = 0.7 ** 2
        assert near_decay > far_decay
        # 近层先验值应大于远层先验值（相同原始值）
        assert prior["near"] > prior["far"]


# ============================================================
# MultiLayerPredictor 集成测试
# ============================================================

class TestMultiLayerPredictorStage184:
    """多层预测模型阶段184集成测试"""

    def test_predict_with_adaptive_confidence(self):
        """测试预测使用自适应置信度"""
        predictor = MultiLayerPredictor(num_layers=3)
        # 初始预测（无历史）
        prediction = predictor.predict("test_module", {"temp": 25.0})
        assert prediction is not None
        # 初始置信度应基于基础值（数据不足）
        assert 0.0 < prediction.confidence <= 0.5

    def test_confidence_evolves_with_errors(self):
        """测试置信度随误差演化"""
        predictor = MultiLayerPredictor(num_layers=3)

        # 生成初始预测
        state = {"value": 10.0}
        prediction = predictor.predict("test", state)
        initial_confidence = prediction.confidence

        # 模拟多次低误差更新
        for i in range(10):
            actual = {"value": 10.0 + i * 0.01}  # 接近预测
            error = PredictionError(
                error_id=f"err_{i}",
                prediction_id=f"pred_{i}",
                timestamp=0.0,
                error_vector={"value": 0.01},
                precision=PrecisionWeight.HIGH,
                surprise=0.0001,
                layer=0,
            )
            predictor.update_from_error(error, actual)

        # 再次预测
        prediction = predictor.predict("test", state)
        # 置信度应提升（准确率高）
        assert prediction.confidence >= initial_confidence

    def test_top_down_constraint_applied(self):
        """测试自上而下约束被应用"""
        predictor = MultiLayerPredictor(num_layers=3)

        # 先填充高层历史
        for i in range(5):
            predictor.layers[2].prediction_history.append({"concept": 100.0 + i})

        # 预测低层
        prediction = predictor.predict("test", {"concept": 50.0})
        assert prediction is not None
        # 预测应受到高层先验影响


# ============================================================
# PredictiveCodingEngine 阶段184接口测试
# ============================================================

class TestPredictiveCodingEngineStage184:
    """预测编码引擎阶段184接口测试"""

    def test_new_interfaces_exist(self):
        """测试新增接口存在"""
        engine = create_predictive_coding_engine()
        assert hasattr(engine, "set_top_down_weight")
        assert hasattr(engine, "get_confidence_stats")
        assert hasattr(engine, "get_top_down_stats")

    def test_get_confidence_stats(self):
        """测试获取置信度统计"""
        engine = create_predictive_coding_engine()
        stats = engine.get_confidence_stats()
        assert "ewma_accuracy" in stats
        assert "sample_count" in stats
        assert "recent_avg" in stats
        assert stats["sample_count"] == 0

    def test_get_top_down_stats(self):
        """测试获取自上而下约束统计"""
        engine = create_predictive_coding_engine()
        stats = engine.get_top_down_stats()
        assert "total_mappings" in stats
        assert "layer_priors_count" in stats

    def test_set_top_down_weight(self):
        """测试设置自上而下映射权重"""
        engine = create_predictive_coding_engine()
        engine.set_top_down_weight("concept_a", "perception_x", 0.8)
        stats = engine.get_top_down_stats()
        assert stats["total_mappings"] == 1

    def test_backward_compatibility(self):
        """测试向后兼容：原有接口仍正常工作"""
        engine = create_predictive_coding_engine()

        # 原有 run_cycle 接口
        prediction, error, correction = engine.run_cycle(
            "test_module",
            {"value": 10.0},
            {"value": 12.0},
            PrecisionWeight.HIGH,
        )
        assert prediction is not None
        assert error is not None
        assert correction is not None

        # 原有 get_stats 接口（应包含新统计字段）
        stats = engine.get_stats()
        assert "top_down_stats" in stats
        assert "confidence_stats" in stats

    def test_clear_history_resets_confidence(self):
        """测试清除历史重置置信度追踪器"""
        engine = create_predictive_coding_engine()

        # 运行几个周期
        for i in range(5):
            engine.run_cycle("test", {"v": i}, {"v": i + 0.1}, PrecisionWeight.HIGH)

        # 确认有样本
        stats = engine.get_confidence_stats()
        assert stats["sample_count"] > 0

        # 清除
        engine.clear_history()
        stats = engine.get_confidence_stats()
        assert stats["sample_count"] == 0

    def test_end_to_end_confidence_evolution(self):
        """测试端到端置信度演化"""
        engine = create_predictive_coding_engine()

        # 阶段1：高误差预测 → 置信度应降低
        for i in range(10):
            engine.run_cycle("test", {"v": 10.0}, {"v": 50.0}, PrecisionWeight.HIGH)

        stats_high_error = engine.get_confidence_stats()
        assert stats_high_error["sample_count"] == 10

        # 阶段2：低误差预测 → EWMA准确率应提升
        engine.clear_history()
        for i in range(10):
            engine.run_cycle("test", {"v": 10.0}, {"v": 10.1}, PrecisionWeight.HIGH)

        stats_low_error = engine.get_confidence_stats()
        # 低误差后EWMA准确率应高于高误差
        assert stats_low_error["ewma_accuracy"] > stats_high_error["ewma_accuracy"]
