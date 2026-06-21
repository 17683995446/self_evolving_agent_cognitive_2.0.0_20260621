"""
阶段188测试：驱力系统增强 — 行为-驱力追溯链路

验证内容:
1. ActionDriveTracer: 行为-驱力因果追溯
2. ActionDriveTrace: 追溯结果结构
3. DriveFirstPrinciplesEngine.trace_action_to_gap: 增强的追溯
4. 端到端：行为→目标→驱力缺口完整链路
"""

import pytest
from self_evolving_agent.core.drive_first_principles import (
    DriveFirstPrinciplesEngine,
    DriveType,
    DriveGap,
    DriveState,
    Goal,
    GoalPriority,
    GoalManager,
    ActionDriveTracer,
    ActionDriveTrace,
    create_drive_engine,
    create_goal,
)


# ============================================================
# ActionDriveTracer 测试
# ============================================================

class TestActionDriveTracer:
    """行为-驱力追溯器测试"""

    def test_trace_with_registered_mapping(self):
        """测试通过注册映射追溯"""
        tracer = ActionDriveTracer()
        goal_manager = GoalManager()

        # 创建并激活目标
        goal = create_goal(
            goal_id="g1", name="improve_safety", drive_type=DriveType.SAFE,
            target_state={"safe": 0.9},
        )
        goal_manager.add_goal(goal)
        goal_manager.activate_goal("g1")

        # 注册行为-目标映射
        tracer.register_action_goal("secure", "g1")

        # 追溯
        gaps = {DriveType.SAFE: DriveGap(gap_id="gap1", drive_type=DriveType.SAFE, current_level=0.3, target_level=0.9)}
        trace = tracer.trace("secure the system", goal_manager, gaps)

        assert trace.matched_goal is not None
        assert trace.matched_goal.goal_id == "g1"
        assert trace.drive_type == DriveType.SAFE
        assert trace.matched_gap is not None
        assert trace.confidence > 0.5
        assert len(trace.trace_path) >= 3

    def test_trace_by_goal_name(self):
        """测试通过目标名称追溯"""
        tracer = ActionDriveTracer()
        goal_manager = GoalManager()

        goal = create_goal(
            goal_id="g1", name="optimize_performance", drive_type=DriveType.EFF,
            target_state={"eff": 0.9},
        )
        goal_manager.add_goal(goal)
        goal_manager.activate_goal("g1")

        gaps = {DriveType.EFF: DriveGap(gap_id="gap1", drive_type=DriveType.EFF, current_level=0.4, target_level=0.9)}
        trace = tracer.trace("execute optimize_performance task", goal_manager, gaps)

        assert trace.matched_goal is not None
        assert trace.drive_type == DriveType.EFF

    def test_trace_no_match(self):
        """测试无匹配目标"""
        tracer = ActionDriveTracer()
        goal_manager = GoalManager()

        goal = create_goal(
            goal_id="g1", name="some_goal", drive_type=DriveType.SAFE,
            target_state={"safe": 0.9},
        )
        goal_manager.add_goal(goal)
        goal_manager.activate_goal("g1")

        gaps = {}
        trace = tracer.trace("unrelated action", goal_manager, gaps)

        assert trace.matched_goal is None
        assert trace.confidence == 0.0
        assert "无匹配目标" in trace.trace_path[-1]

    def test_trace_no_gap_for_drive_type(self):
        """测试驱力类型无对应缺口"""
        tracer = ActionDriveTracer()
        goal_manager = GoalManager()

        goal = create_goal(
            goal_id="g1", name="safety_goal", drive_type=DriveType.SAFE,
            target_state={"safe": 0.9},
        )
        goal_manager.add_goal(goal)
        goal_manager.activate_goal("g1")

        # 无SAFE类型缺口
        gaps = {DriveType.GAIN: DriveGap(gap_id="gap1", drive_type=DriveType.GAIN, current_level=0.3, target_level=0.9)}
        trace = tracer.trace("execute safety_goal", goal_manager, gaps)

        assert trace.matched_goal is not None
        assert trace.drive_type == DriveType.SAFE
        assert trace.matched_gap is None  # 无对应缺口

    def test_get_registered_mappings(self):
        """测试获取已注册映射"""
        tracer = ActionDriveTracer()
        tracer.register_action_goal("secure", "g1")
        tracer.register_action_goal("optimize", "g2")
        mappings = tracer.get_registered_mappings()
        assert mappings["secure"] == "g1"
        assert mappings["optimize"] == "g2"


# ============================================================
# DriveFirstPrinciplesEngine 阶段188接口测试
# ============================================================

class TestDriveEngineStage188:
    """驱力引擎阶段188接口测试"""

    def test_new_interfaces_exist(self):
        """测试新增接口存在"""
        engine = create_drive_engine()
        assert hasattr(engine, "trace_action_to_gap_detailed")
        assert hasattr(engine, "register_action_goal")
        assert hasattr(engine, "get_action_drive_trace")

    def test_trace_action_to_gap_detailed(self):
        """测试详细行为-驱力追溯"""
        engine = create_drive_engine()

        # 设置驱力状态（低安全）
        engine.update_state(safe_level=0.2, gain_level=0.5, eff_level=0.5)
        engine.compute_gaps()

        # 添加并激活目标
        goal = create_goal(
            goal_id="g1", name="improve_safety", drive_type=DriveType.SAFE,
            target_state={"safe": 0.9},
        )
        engine.add_goal(goal)
        engine.activate_goal("g1")

        # 追溯
        trace = engine.trace_action_to_gap_detailed("execute improve_safety task")

        assert isinstance(trace, ActionDriveTrace)
        assert trace.action_description == "execute improve_safety task"
        # 应匹配到目标
        assert trace.matched_goal is not None
        assert trace.drive_type == DriveType.SAFE

    def test_register_and_trace(self):
        """测试注册映射后追溯"""
        engine = create_drive_engine()

        # 设置驱力状态
        engine.update_state(safe_level=0.2, gain_level=0.5, eff_level=0.5)
        engine.compute_gaps()

        # 添加目标
        goal = create_goal(
            goal_id="g1", name="safety_goal", drive_type=DriveType.SAFE,
            target_state={"safe": 0.9},
        )
        engine.add_goal(goal)
        engine.activate_goal("g1")

        # 注册行为-目标映射
        engine.register_action_goal("secure", "g1")

        # 追溯
        trace = engine.get_action_drive_trace("secure the perimeter")

        assert trace.matched_goal is not None
        assert trace.matched_goal.goal_id == "g1"
        assert trace.confidence > 0.5

    def test_trace_no_match(self):
        """测试无匹配时返回低置信度"""
        engine = create_drive_engine()
        engine.update_state(safe_level=0.5, gain_level=0.5, eff_level=0.5)
        engine.compute_gaps()

        trace = engine.get_action_drive_trace("unrelated action")
        assert trace.matched_goal is None
        assert trace.confidence == 0.0

    def test_backward_compatibility(self):
        """测试向后兼容：原有trace_action_to_gap仍正常工作"""
        engine = create_drive_engine()
        engine.update_state(safe_level=0.2, gain_level=0.5, eff_level=0.5)
        engine.compute_gaps()

        # 原有 trace_action_to_gap
        gap = engine.trace_action_to_gap("some action")
        # 应返回dominant gap（向后兼容）
        assert gap is not None

    def test_trace_path_completeness(self):
        """测试追溯路径完整性"""
        engine = create_drive_engine()
        engine.update_state(safe_level=0.1, gain_level=0.5, eff_level=0.5)
        engine.compute_gaps()

        goal = create_goal(
            goal_id="g1", name="safety_improvement", drive_type=DriveType.SAFE,
            target_state={"safe": 0.9},
        )
        engine.add_goal(goal)
        engine.activate_goal("g1")

        trace = engine.trace_action_to_gap_detailed("run safety_improvement")

        # 追溯路径应包含：行为 → 目标 → 驱力类型 → 驱力缺口
        assert len(trace.trace_path) >= 3
        assert "行为" in trace.trace_path[0]
