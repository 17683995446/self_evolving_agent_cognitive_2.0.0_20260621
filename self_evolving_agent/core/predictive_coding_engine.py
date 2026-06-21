"""
阶段114: 预测编码能力强化 — 多层预测 + 贝叶斯更新
阶段184: 自上而下预测约束 + 自适应置信度

第一性原理:
1. 自由能最小化：大脑通过分层模型最小化预测误差
2. 贝叶斯更新：后验 ∝ 先验 × 似然，精度加权融合
3. 多层预测：高层预测约束低层，低层误差上传高层
4. 精度控制：高精度信号驱动强更新，低精度信号被抑制
5. 自上而下约束（阶段184）：高层概念预测作为低层感知的先验
6. 自适应置信度（阶段184）：置信度基于历史预测准确率动态调整

强化内容:
- MultiLayerPredictor: 多层预测模型（感知层→特征层→概念层）
- BayesianUpdater: 贝叶斯参数更新（精度加权）
- HierarchicalErrorPropagation: 层间误差传播
- PrecisionControl: 动态精度调节（注意力机制）
- EnhancedPredictiveCodingEngine: 整合以上能力
- TopDownPredictor（阶段184）: 高层→低层预测约束传播
- AdaptiveConfidenceTracker（阶段184）: 基于历史准确率的置信度自适应

向后兼容: 保留原有 PredictiveCodingEngine 接口
"""

import threading
import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
from enum import Enum
from collections import deque


# ============================================================
# 数据结构
# ============================================================

class PrecisionWeight(Enum):
    """精度权重级别"""
    HIGH = 1.0
    MEDIUM = 0.5
    LOW = 0.1
    IGNORE = 0.0


@dataclass
class Prediction:
    """预测结果"""
    prediction_id: str
    timestamp: float
    predicted_state: Dict[str, Any]
    confidence: float = 0.5
    time_horizon: float = 1.0
    source: str = "unknown"
    # 阶段114新增：层级和精度
    layer: int = 0  # 0=感知层, 1=特征层, 2=概念层
    precision_estimate: float = 0.5  # 预测的预期精度


@dataclass
class PredictionError:
    """预测误差"""
    error_id: str
    prediction_id: str
    timestamp: float
    error_vector: Dict[str, float]
    precision: PrecisionWeight = PrecisionWeight.MEDIUM
    surprise: float = 0.0
    # 阶段114新增：层间传播
    layer: int = 0
    propagated_to: List[str] = field(default_factory=list)

    @property
    def weighted_error(self) -> float:
        """加权误差"""
        if not self.error_vector:
            return 0.0
        avg_error = sum(abs(v) for v in self.error_vector.values()) / len(self.error_vector)
        return avg_error * self.precision.value


@dataclass
class Correction:
    """修正动作"""
    correction_id: str
    error_id: str
    timestamp: float
    parameter_updates: Dict[str, float]
    update_magnitude: float = 0.0
    target_module: str = "unknown"
    # 阶段114新增：贝叶斯更新信息
    prior_value: float = 0.0
    posterior_value: float = 0.0
    learning_rate: float = 0.1


# ============================================================
# 阶段184: 自适应置信度追踪器
# ============================================================

class AdaptiveConfidenceTracker:
    """
    自适应置信度追踪器（阶段184）

    第一性原理：置信度应反映模型的历史预测准确率
    - 准确率高 → 置信度提升
    - 准确率低 → 置信度降低
    - 使用指数加权移动平均（EWMA）追踪准确率趋势
    """

    def __init__(self, window_size: int = 50, initial_confidence: float = 0.3):
        self._window_size = window_size
        self._accuracy_history: deque = deque(maxlen=window_size)
        self._ewma_accuracy: float = 0.5  # 初始假设50%准确率
        self._alpha: float = 0.1  # EWMA平滑因子
        self._base_confidence: float = initial_confidence
        self._lock = threading.RLock()

    def record_prediction_outcome(self, was_accurate: bool, error_magnitude: float = 0.0) -> None:
        """
        记录预测结果

        Args:
            was_accurate: 预测是否准确（误差低于阈值）
            error_magnitude: 误差大小（0.0=完美，>0.5=不准确）
        """
        with self._lock:
            # 连续度量：准确=1.0，不准确=0.0，中间值基于误差
            accuracy_score = max(0.0, 1.0 - error_magnitude)
            self._accuracy_history.append(accuracy_score)
            # EWMA更新
            self._ewma_accuracy = (1 - self._alpha) * self._ewma_accuracy + self._alpha * accuracy_score

    def get_adaptive_confidence(self, layer_precision: float = 0.5) -> float:
        """
        获取自适应置信度

        第一性原理：置信度 = 基础置信度 × 历史准确率 × 层精度
        """
        with self._lock:
            if len(self._accuracy_history) < 3:
                # 数据不足时使用基础置信度
                return self._base_confidence * layer_precision

            # 自适应置信度 = 基础 + EWMA准确率的增益
            # 准确率高时置信度可达0.95，准确率低时降至0.1
            adaptive_component = self._ewma_accuracy * 0.65
            confidence = (self._base_confidence + adaptive_component) * layer_precision
            return max(0.05, min(0.95, confidence))

    def get_accuracy_stats(self) -> Dict[str, float]:
        """获取准确率统计"""
        with self._lock:
            if not self._accuracy_history:
                return {"ewma_accuracy": 0.5, "sample_count": 0, "recent_avg": 0.5}
            return {
                "ewma_accuracy": self._ewma_accuracy,
                "sample_count": len(self._accuracy_history),
                "recent_avg": sum(self._accuracy_history) / len(self._accuracy_history),
            }

    def reset(self) -> None:
        """重置追踪器"""
        with self._lock:
            self._accuracy_history.clear()
            self._ewma_accuracy = 0.5


# ============================================================
# 阶段184: 自上而下预测约束器
# ============================================================

class TopDownPredictor:
    """
    自上而下预测约束器（阶段184）

    第一性原理：在分层预测编码中，高层概念预测为低层感知提供先验约束
    - 高层（概念层）的预测通过降采样映射到低层（感知层）
    - 低层预测融合高层先验和自身历史，形成更稳健的预测
    - 高层精度越高，对低层的约束越强
    """

    def __init__(self, num_layers: int = 3):
        self._num_layers = num_layers
        # 高层→低层的映射权重：高层参数名 → 低层参数名 → 权重
        self._top_down_weights: Dict[str, Dict[str, float]] = {}
        # 各层最近的高层预测
        self._layer_priors: Dict[int, Dict[str, float]] = {}
        self._lock = threading.RLock()

    def set_top_down_weight(self, high_layer_param: str, low_layer_param: str, weight: float) -> None:
        """设置高层→低层的映射权重"""
        with self._lock:
            if high_layer_param not in self._top_down_weights:
                self._top_down_weights[high_layer_param] = {}
            self._top_down_weights[high_layer_param][low_layer_param] = max(-1.0, min(1.0, weight))

    def update_layer_prior(self, layer_id: int, prediction_state: Dict[str, Any]) -> None:
        """
        更新层先验（高层预测完成后调用）

        第一性原理：高层预测作为低层的先验约束
        """
        with self._lock:
            # 将预测状态中的数值提取为先验
            prior: Dict[str, float] = {}
            for key, value in prediction_state.items():
                try:
                    prior[key] = float(value)
                except (TypeError, ValueError):
                    continue
            self._layer_priors[layer_id] = prior

    def get_top_down_prior(self, target_layer: int) -> Dict[str, float]:
        """
        获取对目标层的自上而下先验

        第一性原理：融合所有高于目标层的预测作为先验
        - 越高层的预测权重越小（距离越远，约束越弱）
        """
        with self._lock:
            combined_prior: Dict[str, float] = {}
            for source_layer in range(target_layer + 1, self._num_layers):
                layer_prior = self._layer_priors.get(source_layer, {})
                # 距离衰减：每层距离衰减因子0.7
                distance = source_layer - target_layer
                decay = 0.7 ** distance
                for key, value in layer_prior.items():
                    if key in combined_prior:
                        combined_prior[key] = combined_prior[key] * (1 - decay * 0.3) + value * decay * 0.3
                    else:
                        combined_prior[key] = value * decay * 0.3
            return combined_prior

    def apply_top_down_constraint(
        self,
        layer_id: int,
        predicted_state: Dict[str, Any],
        layer_precision: float,
    ) -> Dict[str, Any]:
        """
        将自上而下先验应用到低层预测

        第一性原理：低层预测 = α × 自身预测 + (1-α) × 高层先验
        - α = 层精度（精度高→更信任自身预测）
        """
        with self._lock:
            top_down_prior = self.get_top_down_prior(layer_id)
            if not top_down_prior:
                return predicted_state

            # 融合自身预测与高层先验
            alpha = max(0.3, min(0.9, layer_precision))  # 自身权重
            fused_state: Dict[str, Any] = dict(predicted_state)

            for key, prior_value in top_down_prior.items():
                if key in fused_state:
                    try:
                        own_value = float(fused_state[key])
                        fused_state[key] = alpha * own_value + (1 - alpha) * prior_value
                    except (TypeError, ValueError):
                        pass
                else:
                    # 自身无此键时直接采用先验
                    fused_state[key] = prior_value

            return fused_state

    def get_weight_stats(self) -> Dict[str, Any]:
        """获取映射权重统计"""
        with self._lock:
            return {
                "total_mappings": sum(len(v) for v in self._top_down_weights.values()),
                "layer_priors_count": len(self._layer_priors),
                "source_params": list(self._top_down_weights.keys()),
            }


# ============================================================
# 多层预测模型
# ============================================================

@dataclass
class PredictionLayer:
    """
    预测层

    第一性原理：每层维护自己的生成模型，高层约束低层
    """
    layer_id: int
    layer_name: str
    # 该层的参数：参数名 -> (均值, 方差) 的贝叶斯先验
    parameters: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    # 该层的预测历史（用于趋势分析）
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=100))
    # 该层的精度估计
    precision: float = 0.5


class MultiLayerPredictor:
    """
    多层预测模型

    层级结构（自底向上）:
    - Layer 0 (感知层): 原始感官输入的预测
    - Layer 1 (特征层): 抽象特征的预测
    - Layer 2 (概念层): 高层概念的预测

    第一性原理：
    - 高层预测为低层提供先验（阶段184：TopDownPredictor）
    - 低层误差上传到高层更新高层模型
    - 精度决定信息流方向（精度加权）
    - 置信度基于历史准确率自适应（阶段184：AdaptiveConfidenceTracker）
    """

    def __init__(self, num_layers: int = 3):
        self.layers: List[PredictionLayer] = [
            PredictionLayer(
                layer_id=i,
                layer_name=["perception", "feature", "concept"][i] if i < 3 else f"layer_{i}",
            )
            for i in range(num_layers)
        ]
        self._lock = threading.RLock()
        # 阶段184新增：自上而下预测约束器
        self._top_down = TopDownPredictor(num_layers=num_layers)
        # 阶段184新增：自适应置信度追踪器
        self._confidence_tracker = AdaptiveConfidenceTracker()

    def predict(self, module_id: str, current_state: Dict[str, Any]) -> Optional[Prediction]:
        """
        生成多层预测（阶段184增强：自上而下约束）

        第一性原理：先从高层生成预测，再传播到低层作为先验
        """
        with self._lock:
            # 阶段184：先从最高层开始预测，建立自上而下先验
            for layer in reversed(self.layers):
                prediction = self._predict_layer(layer, module_id, current_state)
                if prediction is not None:
                    # 更新该层的先验（供低层使用）
                    self._top_down.update_layer_prior(layer.layer_id, prediction.predicted_state)

            # 从感知层开始，逐层向上生成最终预测
            best_prediction = None
            best_confidence = 0.0

            for layer in self.layers:
                prediction = self._predict_layer(layer, module_id, current_state)
                if prediction and prediction.confidence > best_confidence:
                    best_prediction = prediction
                    best_confidence = prediction.confidence

            return best_prediction

    def _predict_layer(
        self,
        layer: PredictionLayer,
        module_id: str,
        current_state: Dict[str, Any],
    ) -> Optional[Prediction]:
        """在单层生成预测（阶段184增强：自适应置信度 + 自上而下约束）"""
        if not layer.parameters and not layer.prediction_history:
            # 无参数时，使用当前状态作为预测（零阶预测）
            # 阶段184：应用自上而下约束
            constrained_state = self._top_down.apply_top_down_constraint(
                layer.layer_id, dict(current_state), layer.precision
            )
            # 阶段184：自适应置信度替代固定0.3
            confidence = self._confidence_tracker.get_adaptive_confidence(layer.precision)
            return Prediction(
                prediction_id=f"pred_{module_id}_L{layer.layer_id}_{int(time.time() * 1000000)}",
                timestamp=time.time(),
                predicted_state=constrained_state,
                confidence=confidence,
                source=module_id,
                layer=layer.layer_id,
                precision_estimate=layer.precision,
            )

        # 有历史时，使用指数加权移动平均预测
        if layer.prediction_history:
            history = list(layer.prediction_history)
            weights = [math.exp(-0.1 * (len(history) - i)) for i in range(len(history))]
            total_weight = sum(weights)
            predicted_state: Dict[str, Any] = {}

            # 对每个状态键做加权平均
            all_keys = set()
            for h in history:
                all_keys.update(h.keys())

            for key in all_keys:
                weighted_sum = 0.0
                weight_sum = 0.0
                for h, w in zip(history, weights):
                    if key in h:
                        try:
                            weighted_sum += float(h[key]) * w
                            weight_sum += w
                        except (TypeError, ValueError):
                            pass
                if weight_sum > 0:
                    predicted_state[key] = weighted_sum / weight_sum

            # 阶段184：应用自上而下约束
            predicted_state = self._top_down.apply_top_down_constraint(
                layer.layer_id, predicted_state, layer.precision
            )

            # 阶段184：置信度基于历史长度、精度和自适应准确率
            base_confidence = min(0.95, 0.3 + len(history) * 0.05)
            adaptive_confidence = self._confidence_tracker.get_adaptive_confidence(layer.precision)
            confidence = max(base_confidence, adaptive_confidence) * layer.precision

            return Prediction(
                prediction_id=f"pred_{module_id}_L{layer.layer_id}_{int(time.time() * 1000000)}",
                timestamp=time.time(),
                predicted_state=predicted_state,
                confidence=confidence,
                source=module_id,
                layer=layer.layer_id,
                precision_estimate=layer.precision,
            )

        return None

    def update_from_error(self, error: PredictionError, actual_state: Dict[str, Any]) -> None:
        """
        误差驱动的层间更新

        第一性原理：低层误差上传到高层，更新高层模型
        阶段184：同时更新自适应置信度追踪器
        """
        with self._lock:
            # 更新产生误差的层
            if error.layer < len(self.layers):
                layer = self.layers[error.layer]
                # 记录实际状态到历史
                layer.prediction_history.append(dict(actual_state))

                # 根据误差调整精度（误差大→降低精度，误差小→提高精度）
                error_magnitude = error.weighted_error
                if error_magnitude > 0.5:
                    layer.precision = max(0.1, layer.precision * 0.9)
                else:
                    layer.precision = min(1.0, layer.precision * 1.05)

            # 误差上传到更高层
            for upper_layer_idx in range(error.layer + 1, len(self.layers)):
                upper_layer = self.layers[upper_layer_idx]
                # 高层也记录实际状态（抽象化）
                upper_layer.prediction_history.append(dict(actual_state))

            # 阶段184：记录预测结果到自适应置信度追踪器
            error_magnitude = error.weighted_error
            self._confidence_tracker.record_prediction_outcome(
                was_accurate=error_magnitude < 0.3,
                error_magnitude=error_magnitude,
            )


# ============================================================
# 贝叶斯更新器
# ============================================================

class BayesianUpdater:
    """
    贝叶斯参数更新器

    第一性原理：后验 ∝ 先验 × 似然
    - 参数建模为高斯分布 N(μ, σ²)
    - 观测提供似然 N(x, τ²)
    - 后验：μ' = (μ/σ² + x/τ²) / (1/σ² + 1/τ²)
    - 后验：σ'² = 1 / (1/σ² + 1/τ²)
    """

    @staticmethod
    def update(
        prior_mean: float,
        prior_var: float,
        observation: float,
        observation_var: float,
    ) -> Tuple[float, float]:
        """
        高斯贝叶斯更新

        Returns: (posterior_mean, posterior_var)
        """
        # 防御：方差必须为正
        prior_var = max(1e-10, prior_var)
        observation_var = max(1e-10, observation_var)

        # 防御：NaN/Inf 处理
        if math.isnan(prior_mean) or math.isnan(observation):
            return prior_mean, prior_var
        if math.isinf(prior_mean) or math.isinf(observation):
            return observation, observation_var

        posterior_var = 1.0 / (1.0 / prior_var + 1.0 / observation_var)
        posterior_mean = posterior_var * (prior_mean / prior_var + observation / observation_var)

        return posterior_mean, posterior_var

    @staticmethod
    def precision_weighted_update(
        prior_mean: float,
        prior_precision: float,  # 精度 = 1/方差
        observation: float,
        observation_precision: float,
    ) -> Tuple[float, float]:
        """
        精度形式的贝叶斯更新

        Returns: (posterior_mean, posterior_precision)
        """
        prior_precision = max(1e-10, prior_precision)
        observation_precision = max(1e-10, observation_precision)

        if math.isnan(prior_mean) or math.isnan(observation):
            return prior_mean, prior_precision

        posterior_precision = prior_precision + observation_precision
        posterior_mean = (prior_mean * prior_precision + observation * observation_precision) / posterior_precision

        return posterior_mean, posterior_precision


# ============================================================
# 精度控制器（注意力机制）
# ============================================================

class PrecisionController:
    """
    精度控制器

    第一性原理：精度 = 信号可信度的倒数
    - 注意力 = 提升特定通道的精度
    - 高精度信号驱动强更新
    - 低精度信号被抑制（噪声过滤）
    """

    def __init__(self):
        self._channel_precision: Dict[str, float] = {}
        self._attention_boost: Dict[str, float] = {}
        self._lock = threading.RLock()

    def set_base_precision(self, channel: str, precision: float) -> None:
        """设置通道基础精度"""
        with self._lock:
            self._channel_precision[channel] = max(0.0, min(1.0, precision))

    def boost_attention(self, channel: str, boost: float) -> None:
        """提升通道注意力（精度增强）"""
        with self._lock:
            self._attention_boost[channel] = max(0.0, min(2.0, boost))

    def get_effective_precision(self, channel: str) -> float:
        """获取有效精度 = 基础精度 × 注意力增强"""
        with self._lock:
            base = self._channel_precision.get(channel, 0.5)
            boost = self._attention_boost.get(channel, 1.0)
            return max(0.0, min(1.0, base * boost))

    def decay_attention(self, decay_rate: float = 0.95) -> None:
        """注意力衰减（防止过度聚焦）"""
        with self._lock:
            for ch in self._attention_boost:
                self._attention_boost[ch] *= decay_rate
                if self._attention_boost[ch] < 0.01:
                    self._attention_boost[ch] = 1.0  # 回归默认

    def auto_boost_surprising_channels(
        self,
        errors: Dict[str, PredictionError],
        threshold: float = 0.5,
    ) -> None:
        """自动提升高惊奇度通道的注意力"""
        with self._lock:
            for ch, err in errors.items():
                if err.surprise > threshold:
                    boost = min(2.0, 1.0 + err.surprise)
                    self._attention_boost[ch] = boost


# ============================================================
# 增强预测编码引擎
# ============================================================

class PredictiveCodingEngine:
    """
    预测编码引擎（阶段114增强版）

    新增能力:
    - 多层预测模型（MultiLayerPredictor）
    - 贝叶斯参数更新（BayesianUpdater）
    - 精度控制/注意力机制（PrecisionController）
    - 层间误差传播

    向后兼容: 保留原有 register_predictor / run_cycle 接口
    """

    def __init__(self, engine_id: str = "predictive_coding", num_layers: int = 3):
        self.engine_id = engine_id
        self._predictions: Dict[str, Prediction] = {}
        self._errors: Dict[str, PredictionError] = {}
        self._corrections: Dict[str, Correction] = {}
        self._predictors: Dict[str, Callable[[Dict[str, Any]], Prediction]] = {}
        self._correctors: Dict[str, Callable[[PredictionError], Correction]] = {}
        self._lock = threading.RLock()
        self._cycle_count = 0
        self._total_surprise = 0.0

        # 阶段114新增能力
        self._multi_layer = MultiLayerPredictor(num_layers=num_layers)
        self._bayesian = BayesianUpdater()
        self._precision_ctrl = PrecisionController()
        # 参数的贝叶斯先验：参数名 -> (均值, 方差)
        self._param_priors: Dict[str, Tuple[float, float]] = {}

    # ============================================================
    # 原有接口（向后兼容）
    # ============================================================

    def register_predictor(
        self,
        module_id: str,
        predictor: Callable[[Dict[str, Any]], Prediction],
    ) -> None:
        """注册预测器"""
        with self._lock:
            self._predictors[module_id] = predictor

    def register_corrector(
        self,
        module_id: str,
        corrector: Callable[[PredictionError], Correction],
    ) -> None:
        """注册修正器"""
        with self._lock:
            self._correctors[module_id] = corrector

    def _normalize_module_id(self, module_id: Any) -> str:
        """规范化 module_id 为可哈希字符串"""
        if not isinstance(module_id, (str, int, float, tuple, type(None))):
            module_id = str(module_id)
        if module_id is None:
            module_id = "default"
        return str(module_id)

    def generate_prediction(
        self,
        module_id: str,
        current_state: Dict[str, Any],
    ) -> Optional[Prediction]:
        """
        生成预测

        优先级：
        1. 注册的自定义预测器
        2. 多层预测模型
        """
        module_id = self._normalize_module_id(module_id)

        # 1. 尝试注册的预测器
        with self._lock:
            predictor = self._predictors.get(module_id)
        if predictor is not None:
            prediction = predictor(current_state)
            if prediction is not None:
                with self._lock:
                    self._predictions[prediction.prediction_id] = prediction
                return prediction

        # 2. 使用多层预测模型
        prediction = self._multi_layer.predict(module_id, current_state)
        if prediction is not None:
            with self._lock:
                self._predictions[prediction.prediction_id] = prediction
            return prediction

        return None

    def compute_error(
        self,
        prediction: Prediction,
        actual_state: Dict[str, Any],
        precision: PrecisionWeight = PrecisionWeight.MEDIUM,
    ) -> PredictionError:
        """计算预测误差"""
        error_vector: Dict[str, float] = {}
        for key, predicted in prediction.predicted_state.items():
            actual = actual_state.get(key)
            if actual is not None:
                try:
                    error_vector[key] = float(actual) - float(predicted)
                except (TypeError, ValueError):
                    error_vector[key] = 0.0 if actual == predicted else 1.0

        # 惊奇度 = 误差的 L2 范数
        surprise = sum(e ** 2 for e in error_vector.values()) ** 0.5
        # 防御 NaN
        if math.isnan(surprise) or math.isinf(surprise):
            surprise = 0.0

        # 精度控制：根据通道调整精度
        effective_precision = precision
        for key in error_vector:
            channel_precision = self._precision_ctrl.get_effective_precision(key)
            if channel_precision < 0.1:
                effective_precision = PrecisionWeight.IGNORE
                break

        error = PredictionError(
            error_id=f"err_{prediction.prediction_id}_{int(time.time() * 1000000)}_{self._cycle_count}",
            prediction_id=prediction.prediction_id,
            timestamp=time.time(),
            error_vector=error_vector,
            precision=effective_precision,
            surprise=surprise,
            layer=prediction.layer,
        )

        with self._lock:
            self._errors[error.error_id] = error
            self._total_surprise += surprise
            # 防止累积值溢出
            if math.isinf(self._total_surprise):
                self._total_surprise = surprise

        # 误差驱动多层模型更新
        self._multi_layer.update_from_error(error, actual_state)

        # 自动注意力提升
        self._precision_ctrl.auto_boost_surprising_channels({prediction.source: error})

        return error

    def generate_correction(
        self,
        module_id: str,
        error: PredictionError,
    ) -> Optional[Correction]:
        """生成修正（贝叶斯更新）"""
        module_id = self._normalize_module_id(module_id)

        # 1. 尝试注册的修正器
        with self._lock:
            corrector = self._correctors.get(module_id)
        if corrector is not None:
            correction = corrector(error)
            if correction is not None:
                with self._lock:
                    self._corrections[correction.correction_id] = correction
                    self._cycle_count += 1
                return correction

        # 2. 使用贝叶斯更新生成修正
        parameter_updates: Dict[str, float] = {}
        prior_value = 0.0
        posterior_value = 0.0

        # 获取原始预测（用于重建实际值作为观测）
        with self._lock:
            original_prediction = self._predictions.get(error.prediction_id)

        for key, err in error.error_vector.items():
            param_name = f"{module_id}_{key}"
            # 获取先验
            with self._lock:
                prior_mean, prior_var = self._param_priors.get(param_name, (0.0, 1.0))

            # 观测 = 实际值 = 预测值 + 误差
            # 第一性原理：贝叶斯参数应追踪真值，而非偏差
            observation = prior_mean + err  # 默认回退
            if original_prediction is not None:
                predicted_val = original_prediction.predicted_state.get(key)
                if predicted_val is not None:
                    try:
                        observation = float(predicted_val) + err
                    except (TypeError, ValueError):
                        observation = prior_mean + err

            # 观测方差 = 1/精度
            obs_var = 1.0 / max(0.01, error.precision.value)

            # 贝叶斯更新
            post_mean, post_var = self._bayesian.update(
                prior_mean, prior_var, observation, obs_var
            )

            # 学习率 = 精度加权
            learning_rate = error.precision.value * 0.1

            parameter_updates[param_name] = post_mean - prior_mean
            prior_value = prior_mean
            posterior_value = post_mean

            # 更新先验
            with self._lock:
                self._param_priors[param_name] = (post_mean, post_var)

        update_magnitude = sum(abs(v) for v in parameter_updates.values())

        correction = Correction(
            correction_id=f"corr_{error.error_id}_{int(time.time() * 1000000)}",
            error_id=error.error_id,
            timestamp=time.time(),
            parameter_updates=parameter_updates,
            update_magnitude=update_magnitude,
            target_module=module_id,
            prior_value=prior_value,
            posterior_value=posterior_value,
            learning_rate=error.precision.value * 0.1,
        )

        with self._lock:
            self._corrections[correction.correction_id] = correction
            self._cycle_count += 1

        return correction

    def run_cycle(
        self,
        module_id: str,
        current_state: Dict[str, Any],
        actual_state: Dict[str, Any],
        precision: PrecisionWeight = PrecisionWeight.MEDIUM,
    ) -> Tuple[Optional[Prediction], Optional[PredictionError], Optional[Correction]]:
        """执行完整预测-修正循环"""
        prediction = self.generate_prediction(module_id, current_state)
        if prediction is None:
            return None, None, None

        error = self.compute_error(prediction, actual_state, precision)
        correction = self.generate_correction(module_id, error)

        return prediction, error, correction

    # ============================================================
    # 新增能力接口
    # ============================================================

    def set_channel_precision(self, channel: str, precision: float) -> None:
        """设置通道精度（注意力机制）"""
        self._precision_ctrl.set_base_precision(channel, precision)

    def boost_attention(self, channel: str, boost: float) -> None:
        """提升通道注意力"""
        self._precision_ctrl.boost_attention(channel, boost)

    def decay_attention(self, decay_rate: float = 0.95) -> None:
        """注意力衰减"""
        self._precision_ctrl.decay_attention(decay_rate)

    def get_layer_stats(self) -> List[Dict[str, Any]]:
        """获取各层统计"""
        with self._lock:
            return [
                {
                    "layer_id": layer.layer_id,
                    "layer_name": layer.layer_name,
                    "precision": layer.precision,
                    "history_size": len(layer.prediction_history),
                    "parameter_count": len(layer.parameters),
                }
                for layer in self._multi_layer.layers
            ]

    def get_bayesian_params(self) -> Dict[str, Dict[str, float]]:
        """获取贝叶斯参数"""
        with self._lock:
            return {
                name: {"mean": mean, "variance": var}
                for name, (mean, var) in self._param_priors.items()
            }

    def get_free_energy_estimate(self) -> float:
        """估计自由能 = 累积惊奇度"""
        with self._lock:
            return self._total_surprise

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "engine_id": self.engine_id,
                "predictions_count": len(self._predictions),
                "errors_count": len(self._errors),
                "corrections_count": len(self._corrections),
                "cycle_count": self._cycle_count,
                "free_energy_estimate": self._total_surprise,
                "registered_predictors": list(self._predictors.keys()),
                "registered_correctors": list(self._correctors.keys()),
                # 阶段114新增统计
                "num_layers": len(self._multi_layer.layers),
                "bayesian_params_count": len(self._param_priors),
                "layer_stats": self.get_layer_stats(),
                # 阶段184新增统计
                "top_down_stats": self._multi_layer._top_down.get_weight_stats(),
                "confidence_stats": self._multi_layer._confidence_tracker.get_accuracy_stats(),
            }

    def clear_history(self) -> None:
        """清除历史"""
        with self._lock:
            self._predictions.clear()
            self._errors.clear()
            self._corrections.clear()
            self._total_surprise = 0.0
            self._param_priors.clear()
            for layer in self._multi_layer.layers:
                layer.prediction_history.clear()
                layer.precision = 0.5
            # 阶段184：重置置信度追踪器
            self._multi_layer._confidence_tracker.reset()

    # ============================================================
    # 阶段184新增接口：自上而下约束与自适应置信度
    # ============================================================

    def set_top_down_weight(self, high_layer_param: str, low_layer_param: str, weight: float) -> None:
        """
        设置自上而下映射权重

        第一性原理：高层概念到低层感知的映射关系
        """
        self._multi_layer._top_down.set_top_down_weight(high_layer_param, low_layer_param, weight)

    def get_confidence_stats(self) -> Dict[str, float]:
        """
        获取自适应置信度统计

        Returns:
            ewma_accuracy: EWMA准确率
            sample_count: 样本数
            recent_avg: 最近平均准确率
        """
        return self._multi_layer._confidence_tracker.get_accuracy_stats()

    def get_top_down_stats(self) -> Dict[str, Any]:
        """获取自上而下约束统计"""
        return self._multi_layer._top_down.get_weight_stats()


# ============================================================
# 便捷函数
# ============================================================

def create_predictive_coding_engine(engine_id: str = "predictive_coding", num_layers: int = 3) -> PredictiveCodingEngine:
    """创建预测编码引擎"""
    return PredictiveCodingEngine(engine_id=engine_id, num_layers=num_layers)


def create_prediction(
    prediction_id: str,
    predicted_state: Dict[str, Any],
    confidence: float = 0.5,
    time_horizon: float = 1.0,
    source: str = "unknown",
) -> Prediction:
    """创建预测"""
    return Prediction(
        prediction_id=prediction_id,
        timestamp=time.time(),
        predicted_state=predicted_state,
        confidence=confidence,
        time_horizon=time_horizon,
        source=source,
    )


def create_prediction_error(
    prediction_id: str,
    error_vector: Dict[str, float],
    surprise: float = 0.0,
    precision: PrecisionWeight = PrecisionWeight.MEDIUM,
) -> PredictionError:
    """创建预测误差"""
    return PredictionError(
        error_id=f"err_{prediction_id}_{int(time.time() * 1000000)}",
        prediction_id=prediction_id,
        timestamp=time.time(),
        error_vector=error_vector,
        precision=precision,
        surprise=surprise,
    )
