"""
阶段186测试：社会认知增强 — BDI框架与行为预测

验证内容:
1. BDIState: BDI状态构建与一致性计算
2. BDIReasoner: BDI组合推理
3. ActionPredictor: 基于BDI的行为预测
4. TheoryOfMind.predict_action: 增强的BDI组合预测
5. SocialFirstPrinciplesEngine: BDI接口集成
"""

import pytest
from self_evolving_agent.core.social_first_principles import (
    SocialFirstPrinciplesEngine,
    TheoryOfMind,
    BDIState,
    BDIReasoner,
    ActionPredictor,
    MentalStateType,
    create_social_engine,
    create_theory_of_mind,
)


# ============================================================
# BDIState 测试
# ============================================================

class TestBDIState:
    """BDI状态测试"""

    def test_default_state(self):
        """测试默认BDI状态"""
        state = BDIState(agent_id="agent_1")
        assert state.agent_id == "agent_1"
        assert state.beliefs == []
        assert state.desires == []
        assert state.intentions == []
        assert state.belief_confidence == 0.5
        assert state.desire_strength == 0.5
        assert state.intention_commitment == 0.5

    def test_coherence_high(self):
        """测试高一致性"""
        state = BDIState(
            agent_id="a",
            belief_confidence=0.9,
            desire_strength=0.9,
            intention_commitment=0.9,
        )
        assert state.coherence == pytest.approx(0.9, abs=0.01)

    def test_coherence_low(self):
        """测试低一致性"""
        state = BDIState(
            agent_id="a",
            belief_confidence=0.1,
            desire_strength=0.9,
            intention_commitment=0.9,
        )
        # 几何平均，低值拉低整体
        assert state.coherence < 0.6

    def test_coherence_zero(self):
        """测试零一致性"""
        state = BDIState(
            agent_id="a",
            belief_confidence=0.0,
            desire_strength=0.9,
            intention_commitment=0.9,
        )
        assert state.coherence == 0.0


# ============================================================
# BDIReasoner 测试
# ============================================================

class TestBDIReasoner:
    """BDI组合推理器测试"""

    def test_reason_with_all_components(self):
        """测试完整BDI推理"""
        reasoner = BDIReasoner()
        result = reasoner.reason(
            beliefs=["task_is_feasible"],
            desires=["complete_task"],
            intentions=["execute_step_1"],
            belief_confidence=0.8,
            desire_strength=0.9,
            intention_commitment=0.7,
        )
        assert result["recommended_action"] == "execute_step_1"
        assert result["bdi_confidence"] > 0.0
        assert result["coherence"] > 0.5
        assert len(result["reasoning_trace"]) > 0

    def test_reason_no_intention(self):
        """测试无意图时无法预测"""
        reasoner = BDIReasoner()
        result = reasoner.reason(
            beliefs=["some_belief"],
            desires=["some_desire"],
            intentions=[],
        )
        assert result["recommended_action"] is None
        assert result["bdi_confidence"] == 0.0

    def test_reason_intention_only(self):
        """测试仅有意图"""
        reasoner = BDIReasoner()
        result = reasoner.reason(
            beliefs=[],
            desires=[],
            intentions=["act"],
            intention_commitment=0.8,
        )
        assert result["recommended_action"] == "act"
        # 无信念约束和欲望导向，置信度 = 意图承诺度
        assert result["bdi_confidence"] == pytest.approx(0.8, abs=0.01)

    def test_belief_reduces_confidence(self):
        """测试低信念置信度降低行为置信度"""
        reasoner = BDIReasoner()
        high_belief = reasoner.reason(
            beliefs=["b"], desires=["d"], intentions=["i"],
            belief_confidence=0.9, desire_strength=0.9, intention_commitment=0.9,
        )
        low_belief = reasoner.reason(
            beliefs=["b"], desires=["d"], intentions=["i"],
            belief_confidence=0.1, desire_strength=0.9, intention_commitment=0.9,
        )
        assert high_belief["bdi_confidence"] > low_belief["bdi_confidence"]

    def test_evaluate_action(self):
        """测试行动评估"""
        reasoner = BDIReasoner()
        bdi_state = BDIState(
            agent_id="a",
            beliefs=["feasible_action"],
            desires=["desired_action"],
            intentions=["intended_action"],
            belief_confidence=0.8,
            desire_strength=0.8,
            intention_commitment=0.8,
        )

        # 评估意图中的行动（高适合度）
        eval_intended = reasoner.evaluate_action("intended_action", bdi_state)
        assert eval_intended["intention_alignment"] == 1.0

        # 评估不在意图中的行动（低适合度）
        eval_other = reasoner.evaluate_action("other_action", bdi_state)
        assert eval_other["intention_alignment"] == 0.3
        assert eval_intended["overall_suitability"] > eval_other["overall_suitability"]


# ============================================================
# ActionPredictor 测试
# ============================================================

class TestActionPredictor:
    """行为预测器测试"""

    def test_predict_no_candidates(self):
        """测试无候选行动时使用意图"""
        predictor = ActionPredictor()
        bdi_state = BDIState(
            agent_id="a",
            beliefs=["b"],
            desires=["d"],
            intentions=["intended_action"],
            belief_confidence=0.8,
            desire_strength=0.8,
            intention_commitment=0.8,
        )
        result = predictor.predict(bdi_state)
        assert result["predicted_action"] == "intended_action"
        assert result["confidence"] > 0.0
        assert "reasoning" in result

    def test_predict_with_candidates(self):
        """测试有候选行动时评估选择"""
        predictor = ActionPredictor()
        bdi_state = BDIState(
            agent_id="a",
            beliefs=["action_a"],
            desires=["action_b"],
            intentions=["action_a"],
            belief_confidence=0.8,
            desire_strength=0.8,
            intention_commitment=0.8,
        )
        result = predictor.predict(bdi_state, candidate_actions=["action_a", "action_c"])
        # action_a在意图中，应被选为最佳
        assert result["predicted_action"] == "action_a"
        assert len(result["alternatives"]) == 2
        # 备选按适合度排序
        assert result["alternatives"][0]["overall_suitability"] >= result["alternatives"][1]["overall_suitability"]

    def test_predict_sequence(self):
        """测试多步预测"""
        predictor = ActionPredictor()
        bdi_state = BDIState(
            agent_id="a",
            intentions=["step1", "step2", "step3"],
            intention_commitment=0.9,
        )
        predictions = predictor.predict_sequence(bdi_state, steps=3)
        assert len(predictions) == 3
        # 置信度应随步数衰减
        assert predictions[0]["confidence"] >= predictions[1]["confidence"]
        assert predictions[1]["confidence"] >= predictions[2]["confidence"]

    def test_predict_sequence_empty_intentions(self):
        """测试空意图时序列预测为空"""
        predictor = ActionPredictor()
        bdi_state = BDIState(agent_id="a", intentions=[])
        predictions = predictor.predict_sequence(bdi_state, steps=3)
        assert len(predictions) == 0


# ============================================================
# TheoryOfMind.predict_action 增强测试
# ============================================================

class TestTheoryOfMindPredictActionStage186:
    """理论心智行为预测增强测试"""

    def test_predict_action_intention_only_backward_compat(self):
        """测试仅有意图时向后兼容（返回意图内容）"""
        tom = TheoryOfMind()
        tom.model_intention("agent_1", "go_to_store")
        result = tom.predict_action("agent_1")
        # 仅有意图时退化为旧行为
        assert result == "go_to_store"

    def test_predict_action_with_bdi(self):
        """测试BDI组合预测"""
        tom = TheoryOfMind()
        tom.model_belief("agent_1", "store_is_open", confidence=0.9)
        tom.model_desire("agent_1", "buy_milk", confidence=0.8)
        tom.model_intention("agent_1", "go_to_store", confidence=0.7)

        result = tom.predict_action("agent_1")
        # 应返回BDI组合预测字典
        assert isinstance(result, dict)
        assert result["predicted_action"] == "go_to_store"
        assert "bdi_confidence" in result
        assert "belief_context" in result
        assert "desire" in result

    def test_predict_action_no_intention(self):
        """测试无意图时返回None"""
        tom = TheoryOfMind()
        tom.model_belief("agent_1", "some_belief")
        result = tom.predict_action("agent_1")
        assert result is None


# ============================================================
# SocialFirstPrinciplesEngine BDI接口测试
# ============================================================

class TestSocialEngineStage186:
    """社会引擎阶段186接口测试"""

    def test_new_interfaces_exist(self):
        """测试新增接口存在"""
        engine = create_social_engine()
        assert hasattr(engine, "build_bdi_state")
        assert hasattr(engine, "predict_action_bdi")
        assert hasattr(engine, "predict_action_sequence")

    def test_build_bdi_state(self):
        """测试构建BDI状态"""
        engine = create_social_engine()
        engine.model_belief("agent_1", "world_is_safe", confidence=0.8)
        engine.model_desire("agent_1", "explore", confidence=0.7)
        engine.model_intention("agent_1", "go_explore", confidence=0.9)

        bdi_state = engine.build_bdi_state("agent_1")
        assert bdi_state.agent_id == "agent_1"
        assert "world_is_safe" in bdi_state.beliefs
        assert "explore" in bdi_state.desires
        assert "go_explore" in bdi_state.intentions
        assert bdi_state.belief_confidence == pytest.approx(0.8, abs=0.01)
        assert bdi_state.desire_strength == pytest.approx(0.7, abs=0.01)
        assert bdi_state.intention_commitment == pytest.approx(0.9, abs=0.01)

    def test_predict_action_bdi(self):
        """测试BDI行为预测"""
        engine = create_social_engine()
        engine.model_belief("agent_1", "task_feasible", confidence=0.8)
        engine.model_desire("agent_1", "complete_task", confidence=0.9)
        engine.model_intention("agent_1", "execute_step", confidence=0.7)

        result = engine.predict_action_bdi("agent_1")
        assert result["predicted_action"] == "execute_step"
        assert result["confidence"] > 0.0
        assert "reasoning" in result

    def test_predict_action_bdi_with_candidates(self):
        """测试带候选行动的BDI行为预测"""
        engine = create_social_engine()
        engine.model_intention("agent_1", "action_a", confidence=0.9)

        result = engine.predict_action_bdi("agent_1", candidate_actions=["action_a", "action_b"])
        assert result["predicted_action"] == "action_a"
        assert len(result["alternatives"]) == 2

    def test_predict_action_sequence(self):
        """测试行动序列预测"""
        engine = create_social_engine()
        engine.model_intention("agent_1", "step1", confidence=0.9)
        engine.model_intention("agent_1", "step2", confidence=0.8)
        engine.model_intention("agent_1", "step3", confidence=0.7)

        predictions = engine.predict_action_sequence("agent_1", steps=3)
        assert len(predictions) == 3

    def test_backward_compatibility(self):
        """测试向后兼容：原有接口仍正常工作"""
        engine = create_social_engine()
        engine.register_peer("peer_1", ["capability_a"])
        engine.model_belief("peer_1", "some_belief")
        engine.model_intention("peer_1", "some_action")

        # 原有 predict_action
        result = engine.predict_action("peer_1")
        assert result is not None

        # 原有 send_message / receive_messages
        from self_evolving_agent.core.social_first_principles import MessageType
        engine.send_message("self", "peer_1", MessageType.QUERY, {"type": "test"})
        assert isinstance(engine.receive_messages("peer_1"), list)
