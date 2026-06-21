"""
阶段115: 驱力系统强化 — 动态目标 + 优先级仲裁
阶段188: 行为-驱力追溯链路增强

第一性原理:
1. 动机源于差距：没有差距就没有行动的必要
2. 驱力是行为的唯一根源：所有行为必须服务于消除某个缺口
3. 三大驱力是生存的基础：安全、收益、能效
4. 目标是动态的：随环境变化、有时间约束、有优先级
5. 资源有限：必须仲裁多个竞争目标，选择最优行动
6. 行为-驱力追溯（阶段188）：每个行为可追溯到其服务的驱力缺口
7. 因果链路（阶段188）：行为→目标→驱力缺口的完整追溯

强化内容:
- Goal: 动态目标（带截止时间、优先级、依赖、验收标准）
- GoalManager: 目标管理（增删改查、过期清理、完成检测）
- PriorityArbitrator: 多因子优先级仲裁
  因子: 紧急度 × 重要性 × 可行性 / 资源成本
- EnhancedDriveEngine: 整合动态目标与仲裁
- ActionDriveTracer（阶段188）: 行为-驱力因果追溯器

向后兼容: 保留原有 DriveFirstPrinciplesEngine 接口
"""

import threading
import time
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from enum import Enum
from collections import deque


# ============================================================
# 基础枚举与数据结构（向后兼容）
# ============================================================

class DriveType(Enum):
    """驱力类型"""
    SAFE = "safe"      # 安全驱力：避免危险、保持稳定
    GAIN = "gain"      # 收益驱力：获取资源、增长能力
    EFF = "eff"        # 能效驱力：节约能量、提高效率


class GoalStatus(Enum):
    """目标状态"""
    PENDING = "pending"      # 待启动
    ACTIVE = "active"        # 进行中
    PAUSED = "paused"        # 暂停
    COMPLETED = "completed"  # 已完成
    FAILED = "failed"        # 失败
    EXPIRED = "expired"      # 过期


class GoalPriority(Enum):
    """目标优先级"""
    CRITICAL = 1.0   # 关键：必须立即处理
    HIGH = 0.8       # 高：优先处理
    MEDIUM = 0.5     # 中：正常处理
    LOW = 0.2        # 低：空闲时处理
    BACKGROUND = 0.05  # 后台：不影响其他目标


@dataclass
class DriveGap:
    """驱力缺口（向后兼容）"""
    gap_id: str
    drive_type: DriveType
    current_level: float = 0.0
    target_level: float = 1.0
    gap_magnitude: float = 0.0
    urgency: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self.gap_magnitude = max(0.0, self.target_level - self.current_level)
        base_urgency = min(1.0, self.gap_magnitude / max(0.01, self.target_level))
        if self.drive_type == DriveType.SAFE:
            self.urgency = min(1.0, base_urgency * 1.5)
        else:
            self.urgency = base_urgency


@dataclass
class DriveState:
    """驱力状态（向后兼容）"""
    safe_level: float = 0.5
    gain_level: float = 0.5
    eff_level: float = 0.5
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, float]:
        return {
            "safe": self.safe_level,
            "gain": self.gain_level,
            "eff": self.eff_level,
        }


# ============================================================
# 阶段115新增：动态目标
# ============================================================

@dataclass
class Goal:
    """
    动态目标

    第一性原理：
    - 目标是系统期望达到的状态
    - 目标有时间约束（截止时间）
    - 目标有优先级（资源有限时的仲裁依据）
    - 目标有验收标准（可判定完成）
    - 目标可有依赖（前置目标必须先完成）
    """
    goal_id: str
    name: str
    drive_type: DriveType
    target_state: Dict[str, float]      # 期望状态键值对
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None     # 截止时间戳（None=无截止）
    progress: float = 0.0                # 完成进度 [0, 1]
    resource_cost: float = 0.5           # 预估资源成本 [0, 1]
    feasibility: float = 0.8             # 可行性 [0, 1]
    dependencies: Set[str] = field(default_factory=set)  # 前置目标ID
    acceptance_criteria: Optional[Callable[[Dict[str, float]], bool]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 运行时统计
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    effort_spent: float = 0.0            # 已花费努力

    @property
    def time_remaining(self) -> Optional[float]:
        """剩余时间（秒），None表示无截止"""
        if self.deadline is None:
            return None
        return max(0.0, self.deadline - time.time())

    @property
    def time_pressure(self) -> float:
        """
        时间压力 [0, 1]

        第一性原理：剩余时间越少，压力越大
        无截止时间时压力为0
        """
        remaining = self.time_remaining
        if remaining is None:
            return 0.0
        if remaining <= 0:
            return 1.0
        # 指数衰减：剩余时间 < 60s 时压力快速上升
        total_duration = self.deadline - self.created_at
        if total_duration <= 0:
            return 1.0
        ratio = remaining / total_duration
        return max(0.0, min(1.0, 1.0 - ratio))

    @property
    def is_expired(self) -> bool:
        """是否过期"""
        if self.deadline is None:
            return False
        return time.time() > self.deadline and self.status not in (
            GoalStatus.COMPLETED, GoalStatus.FAILED
        )

    @property
    def is_blocked(self) -> bool:
        """是否被依赖阻塞（需外部查询依赖状态）"""
        return len(self.dependencies) > 0

    def check_completion(self, current_state: Dict[str, float]) -> bool:
        """
        检查目标是否完成

        第一性原理：满足验收标准或所有目标键达到目标值
        """
        if self.acceptance_criteria is not None:
            try:
                return bool(self.acceptance_criteria(current_state))
            except Exception:
                pass
        # 默认：所有目标键达到目标值（误差 < 0.05）
        for key, target_val in self.target_state.items():
            current_val = current_state.get(key)
            if current_val is None:
                return False
            try:
                if abs(float(current_val) - float(target_val)) > 0.05:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    def update_progress(self, current_state: Dict[str, float]) -> float:
        """更新并返回进度"""
        if not self.target_state:
            return self.progress
        total = 0.0
        count = 0
        for key, target_val in self.target_state.items():
            current_val = current_state.get(key, 0.0)
            try:
                ratio = float(current_val) / max(0.01, float(target_val))
                total += min(1.0, max(0.0, ratio))
                count += 1
            except (TypeError, ValueError, ZeroDivisionError):
                count += 1
        self.progress = total / count if count > 0 else 0.0
        return self.progress


# ============================================================
# 阶段115新增：目标管理器
# ============================================================

class GoalManager:
    """
    目标管理器

    第一性原理：
    - 目标生命周期管理：创建→激活→完成/失败/过期
    - 依赖解析：前置目标完成后才能激活
    - 过期清理：自动标记过期目标
    - 状态追踪：记录目标进度
    """

    def __init__(self):
        self._goals: Dict[str, Goal] = {}
        self._lock = threading.RLock()
        self._cycle_count = 0

    def add_goal(self, goal: Goal) -> bool:
        """添加目标"""
        with self._lock:
            if goal.goal_id in self._goals:
                return False  # ID冲突
            self._goals[goal.goal_id] = goal
            return True

    def remove_goal(self, goal_id: str) -> bool:
        """移除目标"""
        with self._lock:
            if goal_id in self._goals:
                del self._goals[goal_id]
                return True
            return False

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """获取目标"""
        with self._lock:
            return self._goals.get(goal_id)

    def get_goals_by_status(self, status: GoalStatus) -> List[Goal]:
        """按状态获取目标"""
        with self._lock:
            return [g for g in self._goals.values() if g.status == status]

    def get_goals_by_drive(self, drive_type: DriveType) -> List[Goal]:
        """按驱力类型获取目标"""
        with self._lock:
            return [g for g in self._goals.values() if g.drive_type == drive_type]

    def activate_goal(self, goal_id: str) -> bool:
        """
        激活目标

        第一性原理：前置目标必须全部完成
        """
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            if goal.status != GoalStatus.PENDING:
                return False
            # 检查依赖
            for dep_id in goal.dependencies:
                dep = self._goals.get(dep_id)
                if dep is None or dep.status != GoalStatus.COMPLETED:
                    return False  # 依赖未满足
            goal.status = GoalStatus.ACTIVE
            goal.started_at = time.time()
            return True

    def complete_goal(self, goal_id: str) -> bool:
        """完成目标"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            goal.status = GoalStatus.COMPLETED
            goal.completed_at = time.time()
            goal.progress = 1.0
            return True

    def fail_goal(self, goal_id: str) -> bool:
        """标记目标失败"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            goal.status = GoalStatus.FAILED
            return True

    def pause_goal(self, goal_id: str) -> bool:
        """暂停目标"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            if goal.status == GoalStatus.ACTIVE:
                goal.status = GoalStatus.PAUSED
                return True
            return False

    def resume_goal(self, goal_id: str) -> bool:
        """恢复暂停的目标"""
        with self._lock:
            goal = self._goals.get(goal_id)
            if goal is None:
                return False
            if goal.status == GoalStatus.PAUSED:
                goal.status = GoalStatus.ACTIVE
                return True
            return False

    def update_progress(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """
        更新所有活跃目标的进度

        Returns: {goal_id: progress}
        """
        results: Dict[str, float] = {}
        with self._lock:
            for goal in self._goals.values():
                if goal.status == GoalStatus.ACTIVE:
                    progress = goal.update_progress(current_state)
                    results[goal.goal_id] = progress
                    # 自动完成检测
                    if goal.check_completion(current_state):
                        goal.status = GoalStatus.COMPLETED
                        goal.completed_at = time.time()
                        goal.progress = 1.0
            self._cycle_count += 1
        return results

    def cleanup_expired(self) -> List[str]:
        """
        清理过期目标

        Returns: 已过期的目标ID列表
        """
        expired_ids: List[str] = []
        with self._lock:
            for goal in self._goals.values():
                if goal.is_expired and goal.status not in (
                    GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.EXPIRED
                ):
                    goal.status = GoalStatus.EXPIRED
                    expired_ids.append(goal.goal_id)
        return expired_ids

    def get_active_goals(self) -> List[Goal]:
        """获取所有活跃目标"""
        with self._lock:
            return [g for g in self._goals.values() if g.status == GoalStatus.ACTIVE]

    def get_pending_goals(self) -> List[Goal]:
        """获取所有待启动目标"""
        with self._lock:
            return [g for g in self._goals.values() if g.status == GoalStatus.PENDING]

    def get_all_goals(self) -> List[Goal]:
        """获取所有目标"""
        with self._lock:
            return list(self._goals.values())

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            status_counts: Dict[str, int] = {}
            for goal in self._goals.values():
                status_counts[goal.status.value] = status_counts.get(goal.status.value, 0) + 1
            return {
                "total_goals": len(self._goals),
                "status_counts": status_counts,
                "cycle_count": self._cycle_count,
            }


# ============================================================
# 阶段115新增：优先级仲裁器
# ============================================================

class PriorityArbitrator:
    """
    优先级仲裁器

    第一性原理：
    - 资源有限，必须选择最优目标
    - 优先级 = f(紧急度, 重要性, 可行性, 资源成本)
    - 多目标竞争时，选择期望价值最高的
    - 期望价值 = 重要性 × 可行性 × 紧急度 / 资源成本

    仲裁公式:
    score = priority × urgency × feasibility × time_pressure / (resource_cost + ε)

    其中：
    - priority: 目标优先级权重 [0, 1]
    - urgency: 驱力缺口紧急度 [0, 1]
    - feasibility: 可行性 [0, 1]
    - time_pressure: 时间压力 [0, 1]
    - resource_cost: 资源成本 [0, 1]
    """

    def __init__(self):
        self._lock = threading.RLock()
        # 权重配置（可调）
        self._weights: Dict[str, float] = {
            "priority": 1.0,
            "urgency": 1.0,
            "feasibility": 0.8,
            "time_pressure": 1.2,
            "resource_cost": 0.6,
        }

    def set_weight(self, factor: str, weight: float) -> None:
        """设置因子权重"""
        with self._lock:
            if factor in self._weights:
                self._weights[factor] = max(0.0, weight)

    def compute_score(
        self,
        goal: Goal,
        urgency: float = 0.5,
    ) -> float:
        """
        计算目标优先级分数

        第一性原理：期望价值 = 收益 × 概率 / 成本
        """
        with self._lock:
            w = dict(self._weights)

        # 各因子值
        priority_val = goal.priority.value
        feasibility = max(0.01, min(1.0, goal.feasibility))
        time_pressure = goal.time_pressure
        resource_cost = max(0.01, min(1.0, goal.resource_cost))
        urgency_val = max(0.0, min(1.0, urgency))

        # 分数 = (priority^w1 × urgency^w2 × feasibility^w3 × time_pressure^w4) / (cost^w5 + ε)
        # 使用加权和形式避免幂运算溢出
        numerator = (
            priority_val * w["priority"]
            + urgency_val * w["urgency"]
            + feasibility * w["feasibility"]
            + time_pressure * w["time_pressure"]
        )
        denominator = resource_cost * w["resource_cost"] + 0.1  # ε防止除零

        score = numerator / denominator

        # 防御 NaN/Inf
        if math.isnan(score) or math.isinf(score):
            return 0.0
        return score

    def arbitrate(
        self,
        goals: List[Goal],
        urgency_map: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[Goal, float]]:
        """
        仲裁多个目标，返回按分数降序排列的列表

        Returns: [(goal, score), ...] 按分数降序
        """
        if not goals:
            return []
        urgency_map = urgency_map or {}
        scored: List[Tuple[Goal, float]] = []
        for goal in goals:
            urgency = urgency_map.get(goal.goal_id, 0.5)
            score = self.compute_score(goal, urgency)
            scored.append((goal, score))
        # 降序排列
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def select_top(
        self,
        goals: List[Goal],
        urgency_map: Optional[Dict[str, float]] = None,
        k: int = 1,
    ) -> List[Goal]:
        """选择前k个最优目标"""
        ranked = self.arbitrate(goals, urgency_map)
        return [g for g, _ in ranked[:k]]

    def get_weights(self) -> Dict[str, float]:
        """获取当前权重"""
        with self._lock:
            return dict(self._weights)


# ============================================================
# 增强驱力引擎（向后兼容 + 新能力）
# ============================================================

class DriveFirstPrinciplesEngine:
    """
    驱力第一性原理引擎（阶段115增强版）

    新增能力:
    - 动态目标管理（GoalManager）
    - 多因子优先级仲裁（PriorityArbitrator）
    - 目标驱动的缺口计算
    - 目标进度追踪

    向后兼容: 保留原有 compute_gaps / get_dominant_gap / is_action_required 接口
    """

    def __init__(self, engine_id: str = "drive_engine"):
        self.engine_id = engine_id
        self._state = DriveState()
        self._gaps: Dict[str, DriveGap] = {}
        self._gap_history: List[DriveGap] = []
        self._lock = threading.RLock()
        self._cycle_count = 0

        # 目标水平（第一性原理：无限趋近但永不达到）
        self._target_safe = 1.0
        self._target_gain = 1.0
        self._target_eff = 1.0

        # 阶段115新增能力
        self._goal_manager = GoalManager()
        self._arbitrator = PriorityArbitrator()
        # 目标ID到紧急度的映射缓存
        self._urgency_cache: Dict[str, float] = {}

    # ============================================================
    # 原有接口（向后兼容）
    # ============================================================

    def update_state(
        self,
        safe_level: Optional[float] = None,
        gain_level: Optional[float] = None,
        eff_level: Optional[float] = None,
    ) -> None:
        """更新当前驱力水平"""
        with self._lock:
            if safe_level is not None:
                self._state.safe_level = max(0.0, min(1.0, safe_level))
            if gain_level is not None:
                self._state.gain_level = max(0.0, min(1.0, gain_level))
            if eff_level is not None:
                self._state.eff_level = max(0.0, min(1.0, eff_level))
            self._state.timestamp = time.time()

    def compute_gaps(self) -> List[DriveGap]:
        """
        计算所有驱力缺口

        第一性原理：缺口 = 目标 - 当前
        """
        with self._lock:
            gaps = [
                DriveGap(
                    gap_id=f"gap_safe_{int(time.time() * 1000)}_{self._cycle_count}",
                    drive_type=DriveType.SAFE,
                    current_level=self._state.safe_level,
                    target_level=self._target_safe,
                ),
                DriveGap(
                    gap_id=f"gap_gain_{int(time.time() * 1000)}_{self._cycle_count}",
                    drive_type=DriveType.GAIN,
                    current_level=self._state.gain_level,
                    target_level=self._target_gain,
                ),
                DriveGap(
                    gap_id=f"gap_eff_{int(time.time() * 1000)}_{self._cycle_count}",
                    drive_type=DriveType.EFF,
                    current_level=self._state.eff_level,
                    target_level=self._target_eff,
                ),
            ]

            for gap in gaps:
                self._gaps[gap.gap_id] = gap
                self._gap_history.append(gap)

            self._cycle_count += 1
            return gaps

    def get_dominant_gap(self) -> Optional[DriveGap]:
        """获取主导缺口（向后兼容）"""
        with self._lock:
            if not self._gaps:
                return None
            return max(self._gaps.values(), key=lambda g: g.urgency)

    def get_total_drive_pressure(self) -> float:
        """获取总驱力压力"""
        with self._lock:
            gaps = list(self._gaps.values())
        if not gaps:
            return 0.0
        return sum(g.gap_magnitude * g.urgency for g in gaps)

    def is_action_required(self) -> bool:
        """是否需要行动（向后兼容，无副作用）"""
        with self._lock:
            safe_gap = abs(self._target_safe - self._state.safe_level)
            gain_gap = abs(self._target_gain - self._state.gain_level)
            eff_gap = abs(self._target_eff - self._state.eff_level)
            return safe_gap > 0.01 or gain_gap > 0.01 or eff_gap > 0.01

    def trace_action_to_gap(self, action_description: str) -> Optional[DriveGap]:
        """
        追溯行为到驱力缺口（阶段188增强：因果链路追溯）

        第一性原理：
        - 阶段115：直接返回dominant gap（无真正追溯）
        - 阶段188：通过行为-目标-驱力因果链路追溯
          1. 行为 → 关联目标（通过action_description匹配）
          2. 目标 → 驱力类型（Goal.drive_type）
          3. 驱力类型 → 当前缺口
        """
        # 阶段188：通过目标追溯
        with self._lock:
            # 查找与行为描述相关的活跃目标
            active_goals = self._goal_manager.get_active_goals()
            matched_goal = None
            for goal in active_goals:
                # 简化匹配：行为描述包含目标ID或名称关键词
                if goal.goal_id in action_description or \
                   any(kw in action_description for kw in goal.name.split()):
                    matched_goal = goal
                    break

            if matched_goal is not None:
                # 通过目标的驱力类型找到对应缺口
                for gap in self._gaps.values():
                    if gap.drive_type == matched_goal.drive_type:
                        return gap

            # 退化为dominant gap（向后兼容）
            return self.get_dominant_gap()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            gaps = list(self._gaps.values())
            dominant = max(self._gaps.values(), key=lambda g: g.urgency) if self._gaps else None
            safe_gap = abs(self._target_safe - self._state.safe_level)
            gain_gap = abs(self._target_gain - self._state.gain_level)
            eff_gap = abs(self._target_eff - self._state.eff_level)
            action_required = safe_gap > 0.01 or gain_gap > 0.01 or eff_gap > 0.01
            total_pressure = sum(g.gap_magnitude * g.urgency for g in gaps) if gaps else 0.0
            return {
                "engine_id": self.engine_id,
                "state": self._state.to_dict(),
                "active_gaps": len(gaps),
                "total_gaps_history": len(self._gap_history),
                "cycle_count": self._cycle_count,
                "dominant_drive": dominant.drive_type.value if dominant else None,
                "total_pressure": total_pressure,
                "action_required": action_required,
                # 阶段115新增统计
                "goal_stats": self._goal_manager.get_stats(),
                "active_goals_count": len(self._goal_manager.get_active_goals()),
            }

    def clear_history(self) -> None:
        """清除历史"""
        with self._lock:
            self._gap_history.clear()
            self._gaps.clear()

    # ============================================================
    # 阶段115新增：动态目标接口
    # ============================================================

    def add_goal(self, goal: Goal) -> bool:
        """添加动态目标"""
        return self._goal_manager.add_goal(goal)

    def remove_goal(self, goal_id: str) -> bool:
        """移除目标"""
        return self._goal_manager.remove_goal(goal_id)

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """获取目标"""
        return self._goal_manager.get_goal(goal_id)

    def activate_goal(self, goal_id: str) -> bool:
        """激活目标（检查依赖）"""
        return self._goal_manager.activate_goal(goal_id)

    def complete_goal(self, goal_id: str) -> bool:
        """手动完成目标"""
        return self._goal_manager.complete_goal(goal_id)

    def fail_goal(self, goal_id: str) -> bool:
        """标记目标失败"""
        return self._goal_manager.fail_goal(goal_id)

    def pause_goal(self, goal_id: str) -> bool:
        """暂停目标"""
        return self._goal_manager.pause_goal(goal_id)

    def resume_goal(self, goal_id: str) -> bool:
        """恢复目标"""
        return self._goal_manager.resume_goal(goal_id)

    def get_active_goals(self) -> List[Goal]:
        """获取所有活跃目标"""
        return self._goal_manager.get_active_goals()

    def get_pending_goals(self) -> List[Goal]:
        """获取所有待启动目标"""
        return self._goal_manager.get_pending_goals()

    def update_goal_progress(self, current_state: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        更新所有活跃目标的进度

        第一性原理：用当前状态检测目标完成情况
        """
        if current_state is None:
            current_state = self._state.to_dict()
        # 清理过期目标
        self._goal_manager.cleanup_expired()
        # 更新进度
        return self._goal_manager.update_progress(current_state)

    # ============================================================
    # 阶段115新增：优先级仲裁接口
    # ============================================================

    def arbitrate_goals(
        self,
        goals: Optional[List[Goal]] = None,
    ) -> List[Tuple[Goal, float]]:
        """
        仲裁目标优先级

        第一性原理：资源有限，选择期望价值最高的目标
        score = priority × urgency × feasibility × time_pressure / resource_cost

        Returns: [(goal, score), ...] 按分数降序
        """
        if goals is None:
            goals = self._goal_manager.get_active_goals()
        if not goals:
            return []

        # 计算每个目标的紧急度（基于驱力缺口）
        urgency_map: Dict[str, float] = {}
        gaps = self.compute_gaps()
        gap_by_drive: Dict[DriveType, float] = {g.drive_type: g.urgency for g in gaps}

        for goal in goals:
            urgency_map[goal.goal_id] = gap_by_drive.get(goal.drive_type, 0.5)

        return self._arbitrator.arbitrate(goals, urgency_map)

    def select_top_goals(self, k: int = 1) -> List[Goal]:
        """选择前k个最优目标"""
        ranked = self.arbitrate_goals()
        return [g for g, _ in ranked[:k]]

    def get_top_goal(self) -> Optional[Goal]:
        """获取最高优先级目标"""
        top = self.select_top_goals(k=1)
        return top[0] if top else None

    def set_arbitration_weight(self, factor: str, weight: float) -> None:
        """设置仲裁因子权重"""
        self._arbitrator.set_weight(factor, weight)

    def get_arbitration_weights(self) -> Dict[str, float]:
        """获取仲裁因子权重"""
        return self._arbitrator.get_weights()

    def get_goal_manager(self) -> GoalManager:
        """获取目标管理器（高级用法）"""
        return self._goal_manager

    def get_arbitrator(self) -> PriorityArbitrator:
        """获取仲裁器（高级用法）"""
        return self._arbitrator

    # ============================================================
    # 阶段188新增：行为-驱力追溯接口
    # ============================================================

    def trace_action_to_gap_detailed(self, action_description: str) -> ActionDriveTrace:
        """
        详细的行为-驱力追溯（阶段188）

        第一性原理：返回完整的追溯链路，包括匹配的目标和驱力缺口
        """
        tracer = ActionDriveTracer()

        # 构建驱力类型→缺口的映射
        with self._lock:
            gaps_by_type: Dict[DriveType, DriveGap] = {
                gap.drive_type: gap for gap in self._gaps.values()
            }

        return tracer.trace(action_description, self._goal_manager, gaps_by_type)

    def register_action_goal(self, action_keyword: str, goal_id: str) -> None:
        """
        注册行为-目标关联（阶段188）

        第一性原理：显式注册行为与目标的关联，提高追溯精度
        """
        # 存储在引擎实例上
        if not hasattr(self, "_action_tracer"):
            self._action_tracer = ActionDriveTracer()
        self._action_tracer.register_action_goal(action_keyword, goal_id)

    def get_action_drive_trace(self, action_description: str) -> ActionDriveTrace:
        """
        获取行为-驱力追溯结果（阶段188）

        第一性原理：使用已注册的映射进行精确追溯
        """
        if not hasattr(self, "_action_tracer"):
            self._action_tracer = ActionDriveTracer()

        with self._lock:
            gaps_by_type: Dict[DriveType, DriveGap] = {
                gap.drive_type: gap for gap in self._gaps.values()
            }

        return self._action_tracer.trace(action_description, self._goal_manager, gaps_by_type)


# ============================================================
# 便捷函数（向后兼容 + 新增）
# ============================================================

def create_drive_engine(engine_id: str = "drive_engine") -> DriveFirstPrinciplesEngine:
    """创建驱力引擎"""
    return DriveFirstPrinciplesEngine(engine_id=engine_id)


def create_drive_gap(
    gap_id: str,
    drive_type: DriveType,
    current_level: float,
    target_level: float,
) -> DriveGap:
    """创建驱力缺口"""
    return DriveGap(
        gap_id=gap_id,
        drive_type=drive_type,
        current_level=current_level,
        target_level=target_level,
    )


def create_drive_state(
    safe_level: float = 0.5,
    gain_level: float = 0.5,
    eff_level: float = 0.5,
) -> DriveState:
    """创建驱力状态"""
    return DriveState(
        safe_level=safe_level,
        gain_level=gain_level,
        eff_level=eff_level,
    )


def create_goal(
    goal_id: str,
    name: str,
    drive_type: DriveType,
    target_state: Dict[str, float],
    priority: GoalPriority = GoalPriority.MEDIUM,
    deadline: Optional[float] = None,
    resource_cost: float = 0.5,
    feasibility: float = 0.8,
    dependencies: Optional[Set[str]] = None,
    acceptance_criteria: Optional[Callable[[Dict[str, float]], bool]] = None,
) -> Goal:
    """创建动态目标"""
    return Goal(
        goal_id=goal_id,
        name=name,
        drive_type=drive_type,
        target_state=target_state,
        priority=priority,
        deadline=deadline,
        resource_cost=resource_cost,
        feasibility=feasibility,
        dependencies=dependencies or set(),
        acceptance_criteria=acceptance_criteria,
    )


# ============================================================
# 阶段188: 行为-驱力因果追溯器
# ============================================================

@dataclass
class ActionDriveTrace:
    """
    行为-驱力追溯结果（阶段188）

    第一性原理：记录行为到驱力的完整因果链路
    """
    action_description: str
    matched_goal: Optional[Goal] = None
    matched_gap: Optional[DriveGap] = None
    drive_type: Optional[DriveType] = None
    trace_path: List[str] = field(default_factory=list)
    confidence: float = 0.0


class ActionDriveTracer:
    """
    行为-驱力因果追溯器（阶段188）

    第一性原理：
    - 每个行为都应能追溯到其服务的驱力缺口
    - 追溯链路：行为 → 目标 → 驱力类型 → 驱力缺口
    - 无追溯链路的行为是"无动机行为"（低优先级）
    """

    def __init__(self):
        self._lock = threading.RLock()
        # 行为-目标关联映射
        self._action_goal_map: Dict[str, str] = {}  # action_keyword -> goal_id

    def register_action_goal(self, action_keyword: str, goal_id: str) -> None:
        """注册行为-目标关联"""
        with self._lock:
            self._action_goal_map[action_keyword] = goal_id

    def trace(
        self,
        action_description: str,
        goal_manager: GoalManager,
        gaps: Dict[DriveType, DriveGap],
    ) -> ActionDriveTrace:
        """
        追溯行为到驱力缺口

        第一性原理：
        1. 行为描述 → 匹配目标（通过关键词或注册映射）
        2. 目标 → 驱力类型（Goal.drive_type）
        3. 驱力类型 → 驱力缺口

        Returns:
            ActionDriveTrace: 追溯结果
        """
        with self._lock:
            trace_path: List[str] = [f"行为: {action_description}"]
            confidence = 0.0

            # 步骤1：匹配目标
            matched_goal = None

            # 1a. 通过注册映射匹配
            for keyword, goal_id in self._action_goal_map.items():
                if keyword in action_description:
                    goal = goal_manager.get_goal(goal_id)
                    if goal is not None:
                        matched_goal = goal
                        trace_path.append(f"→ 匹配目标(注册): {goal.name}")
                        confidence = 0.9
                        break

            # 1b. 通过目标名称关键词匹配
            if matched_goal is None:
                active_goals = goal_manager.get_active_goals()
                for goal in active_goals:
                    if goal.goal_id in action_description or \
                       any(kw in action_description for kw in goal.name.split()):
                        matched_goal = goal
                        trace_path.append(f"→ 匹配目标(名称): {goal.name}")
                        confidence = 0.7
                        break

            # 1c. 通过目标名称子串匹配
            if matched_goal is None:
                active_goals = goal_manager.get_active_goals()
                for goal in active_goals:
                    if goal.name in action_description:
                        matched_goal = goal
                        trace_path.append(f"→ 匹配目标(名称子串): {goal.name}")
                        confidence = 0.6
                        break

            if matched_goal is None:
                trace_path.append("→ 无匹配目标（无动机行为）")
                return ActionDriveTrace(
                    action_description=action_description,
                    trace_path=trace_path,
                    confidence=0.0,
                )

            # 步骤2：目标 → 驱力类型
            drive_type = matched_goal.drive_type
            trace_path.append(f"→ 驱力类型: {drive_type.name}")

            # 步骤3：驱力类型 → 驱力缺口
            matched_gap = gaps.get(drive_type)
            if matched_gap is not None:
                trace_path.append(
                    f"→ 驱力缺口: {drive_type.name} "
                    f"(当前={matched_gap.current_level:.2f}, "
                    f"目标={matched_gap.target_level:.2f}, "
                    f"紧迫度={matched_gap.urgency:.2f})"
                )
                # 置信度受缺口紧迫度影响
                confidence *= (0.5 + 0.5 * matched_gap.urgency)
            else:
                trace_path.append(f"→ 无对应驱力缺口: {drive_type.name}")
                confidence *= 0.5

            return ActionDriveTrace(
                action_description=action_description,
                matched_goal=matched_goal,
                matched_gap=matched_gap,
                drive_type=drive_type,
                trace_path=trace_path,
                confidence=confidence,
            )

    def get_registered_mappings(self) -> Dict[str, str]:
        """获取已注册的行为-目标映射"""
        with self._lock:
            return dict(self._action_goal_map)
