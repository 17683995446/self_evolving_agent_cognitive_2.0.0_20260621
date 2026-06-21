"""
阶段189: 跨模块协作 — 预测-驱力-决策-学习编排管线

第一性原理:
1. 认知系统是闭环：预测误差驱动学习，驱力激发决策，决策产生行动，行动产生新观测
2. 每个阶段产生结构化输出，作为下一阶段的输入（数据流驱动）
3. 编排管线整合阶段184-188的新能力：
   - 阶段184: 预测编码的自上而下约束与自适应置信度
   - 阶段185: 因果发现（PC算法、条件独立性检验）
   - 阶段186: BDI行为预测
   - 阶段187: 结构映射类比推理
   - 阶段188: 行为-驱力因果追溯
4. 向后兼容：不修改 UnifiedFirstPrinciplesEngine，在其上构建编排层

管线结构:
    PredictionPhase  → 预测 + 误差 + 自适应置信度调整 (stage 184)
    DrivePhase       → 驱力评估 + 行为-驱力追溯 (stage 188)
    DecisionPhase    → BDI预测 (stage 186) + 因果评估 (stage 185) + 决策
    LearningPhase    → 误差学习 + 结构映射类比 (stage 187) + 经验存储
    FeedbackPhase    → 自上而下权重更新 + 下一周期准备
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from .unified_first_principles_engine import UnifiedFirstPrinciplesEngine, create_unified_engine
    from .drive_first_principles import DriveGap, DriveType, ActionDriveTrace
    from .free_will_first_principles import ChoiceSet
except ImportError:
    from unified_first_principles_engine import UnifiedFirstPrinciplesEngine, create_unified_engine
    from drive_first_principles import DriveGap, DriveType, ActionDriveTrace
    from free_will_first_principles import ChoiceSet


# ============================================================
# 管线阶段结果数据结构
# ============================================================

@dataclass
class PredictionPhaseResult:
    """预测阶段结果"""
    prediction: Any = None
    error: Any = None
    correction: Any = None
    surprise: float = 0.0
    confidence: float = 0.0
    module_id: str = ""


@dataclass
class DrivePhaseResult:
    """驱力阶段结果"""
    gaps: List[DriveGap] = field(default_factory=list)
    dominant_gap: Optional[DriveGap] = None
    action_trace: Optional[ActionDriveTrace] = None
    is_action_required: bool = False


@dataclass
class DecisionPhaseResult:
    """决策阶段结果"""
    decision: Any = None
    bdi_prediction: Optional[Dict[str, Any]] = None
    causal_evaluation: Optional[Dict[str, Any]] = None
    chosen_action: str = ""
    safety_allowed: bool = True


@dataclass
class LearningPhaseResult:
    """学习阶段结果"""
    learned: bool = False
    experience_stored: bool = False
    analogy_found: Optional[Dict[str, Any]] = None
    learning_magnitude: float = 0.0


@dataclass
class FeedbackPhaseResult:
    """反馈阶段结果"""
    top_down_adjusted: bool = False
    confidence_stats: Dict[str, float] = field(default_factory=dict)
    cycle_complete: bool = True


@dataclass
class PipelineCycleResult:
    """完整管线周期结果"""
    cycle: int = 0
    prediction: PredictionPhaseResult = field(default_factory=PredictionPhaseResult)
    drive: DrivePhaseResult = field(default_factory=DrivePhaseResult)
    decision: DecisionPhaseResult = field(default_factory=DecisionPhaseResult)
    learning: LearningPhaseResult = field(default_factory=LearningPhaseResult)
    feedback: FeedbackPhaseResult = field(default_factory=FeedbackPhaseResult)
    timestamp: float = field(default_factory=time.time)


# ============================================================
# 编排管线
# ============================================================

class OrchestrationPipeline:
    """
    预测-驱力-决策-学习编排管线

    第一性原理：
    - 闭环认知架构：预测→误差→驱力→决策→行动→学习→反馈→预测
    - 数据流驱动：每阶段输出结构化数据，作为下阶段输入
    - 整合阶段184-188新能力，形成连贯的认知循环

    使用方式:
        pipeline = OrchestrationPipeline(engine)
        result = pipeline.run_cycle(context)
    """

    def __init__(self, engine: Optional[UnifiedFirstPrinciplesEngine] = None):
        self._engine = engine or create_unified_engine("orchestration_pipeline")
        self._lock = threading.RLock()
        self._cycle_count = 0

        # 管线历史（用于因果发现和结构映射）
        self._prediction_errors: List[float] = []
        self._action_history: List[str] = []
        self._max_history = 100

    @property
    def engine(self) -> UnifiedFirstPrinciplesEngine:
        return self._engine

    # ============================================================
    # 阶段1: 预测阶段（整合 stage 184 自适应置信度）
    # ============================================================

    def _run_prediction_phase(
        self,
        module_id: str,
        current_state: Dict[str, Any],
        actual_state: Dict[str, Any],
    ) -> PredictionPhaseResult:
        """
        预测阶段

        第一性原理：
        - 预测编码：自上而下约束 + 自适应置信度（stage 184）
        - 误差 = 实际 - 预测，驱动后续学习和驱力
        """
        result = self._engine.perceive_and_predict(module_id, current_state, actual_state)

        prediction = result.get("prediction")
        error = result.get("error")
        correction = result.get("correction")

        surprise = 0.0
        confidence = 0.0
        if error is not None:
            surprise = getattr(error, "surprise", 0.0)
        if prediction is not None:
            confidence = getattr(prediction, "confidence", 0.0)

        # 记录预测误差历史（供因果发现使用）
        with self._lock:
            self._prediction_errors.append(surprise)
            if len(self._prediction_errors) > self._max_history:
                self._prediction_errors.pop(0)

        return PredictionPhaseResult(
            prediction=prediction,
            error=error,
            correction=correction,
            surprise=surprise,
            confidence=confidence,
            module_id=module_id,
        )

    # ============================================================
    # 阶段2: 驱力阶段（整合 stage 188 行为-驱力追溯）
    # ============================================================

    def _run_drive_phase(
        self,
        action_description: str = "",
    ) -> DrivePhaseResult:
        """
        驱力阶段

        第一性原理：
        - 驱力缺口激发行动（stage 188）
        - 行为-驱力追溯：行为 → 目标 → 驱力类型 → 缺口
        """
        gaps = self._engine.evaluate_drives()
        dominant = self._engine.drive.get_dominant_gap()
        is_required = self._engine.drive.is_action_required()

        # 行为-驱力追溯（stage 188）
        action_trace = None
        if action_description:
            try:
                action_trace = self._engine.drive.trace_action_to_gap_detailed(action_description)
            except Exception:
                action_trace = None

        return DrivePhaseResult(
            gaps=gaps,
            dominant_gap=dominant,
            action_trace=action_trace,
            is_action_required=is_required,
        )

    # ============================================================
    # 阶段3: 决策阶段（整合 stage 186 BDI + stage 185 因果）
    # ============================================================

    def _run_decision_phase(
        self,
        context: Dict[str, Any],
        drive_result: DrivePhaseResult,
    ) -> DecisionPhaseResult:
        """
        决策阶段

        第一性原理：
        - BDI预测：基于信念-欲望-意图预测行为（stage 186）
        - 因果评估：利用因果结构评估行动后果（stage 185）
        - 决策 = f(驱力, 预测, 安全, 能效)
        """
        bdi_prediction = None
        causal_evaluation = None
        decision = None
        chosen_action = ""
        safety_allowed = True

        # BDI行为预测（stage 186）
        agent_id = context.get("agent_id", "self")
        try:
            bdi_prediction = self._engine.social.predict_action_bdi(agent_id)
        except Exception:
            bdi_prediction = None

        # 因果结构发现（stage 185）— 周期性执行
        if self._cycle_count % 10 == 0 and len(self._prediction_errors) >= 5:
            try:
                causal_evaluation = self._engine.causal.discover_causal_structure(
                    max_condition_set_size=1
                )
            except Exception:
                causal_evaluation = None

        # 决策（如果有驱力缺口）
        if drive_result.is_action_required:
            options = context.get("available_actions", ["wait", "explore", "exploit"])
            choices = ChoiceSet(
                choice_id=f"pipeline_cycle_{self._cycle_count}",
                options=options,
                context=context,
            )
            try:
                decision_result = self._engine.make_consolidated_decision(choices)
                decision = decision_result.get("decision")
                if decision is not None:
                    chosen_action = getattr(decision, "chosen_action", "")
                safety_check = decision_result.get("safety_check", {})
                safety_allowed = safety_check.get("allowed", True)
            except Exception:
                decision = None

        return DecisionPhaseResult(
            decision=decision,
            bdi_prediction=bdi_prediction,
            causal_evaluation=causal_evaluation,
            chosen_action=chosen_action,
            safety_allowed=safety_allowed,
        )

    # ============================================================
    # 阶段4: 学习阶段（整合 stage 187 结构映射）
    # ============================================================

    def _run_learning_phase(
        self,
        prediction_result: PredictionPhaseResult,
        decision_result: DecisionPhaseResult,
        context: Dict[str, Any],
    ) -> LearningPhaseResult:
        """
        学习阶段

        第一性原理：
        - 误差驱动学习：预测误差 → 参数更新
        - 结构映射类比：从相似经验中迁移知识（stage 187）
        - 经验回放：存储经验供后续学习
        """
        learned = False
        learning_magnitude = 0.0
        experience_stored = False
        analogy_found = None

        # 误差驱动学习
        if prediction_result.error is not None and prediction_result.module_id:
            try:
                self._engine.learning.learn_from_error(
                    parameter_name=f"{prediction_result.module_id}_params",
                    error_magnitude=prediction_result.surprise,
                    delta=-0.1 * min(1.0, prediction_result.surprise),
                    error_source=prediction_result.module_id,
                )
                learned = True
                learning_magnitude = prediction_result.surprise
            except Exception:
                pass

        # 结构映射类比（stage 187）— 当有概念上下文时
        source_id = context.get("analogy_source")
        target_id = context.get("analogy_target")
        if source_id and target_id:
            try:
                analogy_found = self._engine.creativity.map_structure(source_id, target_id)
            except Exception:
                analogy_found = None

        # 经验存储
        if decision_result.chosen_action:
            try:
                state = context.get("current_state", {})
                next_state = context.get("next_state", {})
                reward = context.get("reward", 1.0 - prediction_result.surprise)
                self._engine.learning.add_experience(
                    state=state if isinstance(state, dict) else {},
                    action=decision_result.chosen_action,
                    reward=reward,
                    next_state=next_state if isinstance(next_state, dict) else {},
                    td_error=prediction_result.surprise,
                )
                experience_stored = True

                with self._lock:
                    self._action_history.append(decision_result.chosen_action)
                    if len(self._action_history) > self._max_history:
                        self._action_history.pop(0)
            except Exception:
                pass

        return LearningPhaseResult(
            learned=learned,
            experience_stored=experience_stored,
            analogy_found=analogy_found,
            learning_magnitude=learning_magnitude,
        )

    # ============================================================
    # 阶段5: 反馈阶段（整合 stage 184 自上而下权重调整）
    # ============================================================

    def _run_feedback_phase(
        self,
        prediction_result: PredictionPhaseResult,
    ) -> FeedbackPhaseResult:
        """
        反馈阶段

        第一性原理：
        - 自上而下权重根据预测精度调整（stage 184）
        - 高精度预测 → 增强自上而下约束
        - 低精度预测 → 减弱自上而下约束，增加自下而上影响
        """
        top_down_adjusted = False
        confidence_stats = {}

        # 获取自适应置信度统计（stage 184）
        try:
            confidence_stats = self._engine.predictive_coding.get_confidence_stats()
        except Exception:
            confidence_stats = {}

        # 根据预测精度调整自上而下权重
        if prediction_result.confidence > 0 and prediction_result.module_id:
            try:
                # 高置信度时增强自上而下约束
                if prediction_result.confidence > 0.7:
                    self._engine.predictive_coding.set_top_down_weight(
                        "high_layer", "low_layer", 0.8
                    )
                    top_down_adjusted = True
                elif prediction_result.confidence < 0.3:
                    self._engine.predictive_coding.set_top_down_weight(
                        "high_layer", "low_layer", 0.3
                    )
                    top_down_adjusted = True
            except Exception:
                pass

        return FeedbackPhaseResult(
            top_down_adjusted=top_down_adjusted,
            confidence_stats=confidence_stats,
            cycle_complete=True,
        )

    # ============================================================
    # 完整管线周期
    # ============================================================

    def run_cycle(self, context: Dict[str, Any]) -> PipelineCycleResult:
        """
        运行完整编排管线周期

        管线流程:
        1. 预测阶段: 感知 → 预测 → 误差
        2. 驱力阶段: 评估缺口 → 行为追溯
        3. 决策阶段: BDI预测 → 因果评估 → 决策
        4. 学习阶段: 误差学习 → 类比迁移 → 经验存储
        5. 反馈阶段: 权重调整 → 下一周期准备

        Args:
            context: 包含 module_id, current_state, actual_state,
                     available_actions, agent_id 等

        Returns:
            PipelineCycleResult: 完整周期结果
        """
        with self._lock:
            self._cycle_count += 1
            cycle_num = self._cycle_count

        module_id = context.get("module_id", "default")
        current_state = context.get("current_state", {})
        actual_state = context.get("actual_state", current_state)
        action_description = context.get("action_description", "")

        # 阶段1: 预测
        prediction_result = self._run_prediction_phase(module_id, current_state, actual_state)

        # 阶段2: 驱力
        drive_result = self._run_drive_phase(action_description)

        # 阶段3: 决策
        decision_result = self._run_decision_phase(context, drive_result)

        # 阶段4: 学习
        learning_result = self._run_learning_phase(prediction_result, decision_result, context)

        # 阶段5: 反馈
        feedback_result = self._run_feedback_phase(prediction_result)

        return PipelineCycleResult(
            cycle=cycle_num,
            prediction=prediction_result,
            drive=drive_result,
            decision=decision_result,
            learning=learning_result,
            feedback=feedback_result,
        )

    # ============================================================
    # 批量运行与统计
    # ============================================================

    def run_batch(
        self,
        contexts: List[Dict[str, Any]],
    ) -> List[PipelineCycleResult]:
        """批量运行管线周期"""
        results = []
        for ctx in contexts:
            results.append(self.run_cycle(ctx))
        return results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """获取管线统计"""
        with self._lock:
            avg_error = (
                sum(self._prediction_errors) / len(self._prediction_errors)
                if self._prediction_errors else 0.0
            )
            return {
                "cycle_count": self._cycle_count,
                "avg_prediction_error": avg_error,
                "error_history_len": len(self._prediction_errors),
                "action_history_len": len(self._action_history),
                "engine_stats": self._engine.get_full_stats(),
            }

    def get_error_history(self) -> List[float]:
        """获取预测误差历史（供因果发现使用）"""
        with self._lock:
            return list(self._prediction_errors)

    def get_action_history(self) -> List[str]:
        """获取行动历史"""
        with self._lock:
            return list(self._action_history)


# ============================================================
# 便捷函数
# ============================================================

def create_orchestration_pipeline(
    engine: Optional[UnifiedFirstPrinciplesEngine] = None,
) -> OrchestrationPipeline:
    """创建编排管线"""
    return OrchestrationPipeline(engine=engine)
