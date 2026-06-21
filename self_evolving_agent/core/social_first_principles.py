"""
阶段123: 社交能力强化 — 理论心智 + 协作
阶段186: BDI框架增强 — 意图×信念×欲望组合行为预测

第一性原理:
1. 理论心智 (Theory of Mind) = 理解他人的信念、意图、欲望
2. 协作 = 多Agent共同达成目标（1+1 > 2）
3. 社会契约 = 信任、互惠
4. 视角采择 = 从他人角度看问题
5. 情感识别 = 理解他人情感
6. 信息交换加速进化
7. BDI框架（阶段186）：行为 = f(信念, 意图, 欲望)

强化内容:
- TheoryOfMind: 理论心智（建模他人心理状态）
- PerspectiveTaking: 视角采择
- Collaboration: 协作引擎
- TrustModel: 信任模型
- BDIReasoner（阶段186）: BDI组合推理器
- ActionPredictor（阶段186）: 基于BDI的行为预测器
- SocialFirstPrinciplesEngine: 增强

向后兼容: 保留原有 MessageType / AgentIdentity / InterAgentMessage / 接口
"""

import threading
import time
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum


# ============================================================
# 基础枚举与数据结构（向后兼容 + 增强）
# ============================================================

class MessageType(Enum):
    """消息类型"""
    KNOWLEDGE_SHARE = "knowledge_share"    # 知识共享
    TASK_COLLABORATION = "task_collab"     # 任务协作
    ERROR_REPORT = "error_report"          # 错误报告
    QUERY = "query"                        # 查询请求
    RESPONSE = "response"                  # 查询响应


class MentalStateType(Enum):
    """心理状态类型（阶段123新增）"""
    BELIEF = "belief"          # 信念
    DESIRE = "desire"          # 欲望
    INTENTION = "intention"    # 意图
    EMOTION = "emotion"        # 情感


class TaskStatus(Enum):
    """任务状态（阶段123新增）"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentIdentity:
    """Agent身份（增强）"""
    agent_id: str
    capabilities: List[str] = field(default_factory=list)
    trust_score: float = 0.5              # 信任分数 [0,1]
    last_seen: float = field(default_factory=time.time)


@dataclass
class InterAgentMessage:
    """Agent间消息"""
    message_id: str
    sender_id: str
    receiver_id: str                      # "*" 表示广播
    message_type: MessageType
    payload: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class MentalState:
    """心理状态（阶段123新增）"""
    agent_id: str
    state_type: MentalStateType
    content: Any
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


@dataclass
class CollaborationTask:
    """协作任务（阶段123新增）"""
    task_id: str
    description: str
    required_capabilities: List[str] = field(default_factory=list)
    assigned_agents: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# ============================================================
# 阶段123新增：理论心智
# ============================================================

class TheoryOfMind:
    """
    理论心智引擎

    第一性原理：
    - 理论心智 = 理解他人的信念、意图、欲望
    - 通过观察行为推断心理状态
    - 预测他人行动基于其心理状态
    """

    def __init__(self):
        self._mental_states: Dict[str, List[MentalState]] = {}  # agent_id -> states
        self._lock = threading.RLock()

    def model_belief(
        self,
        agent_id: str,
        belief: Any,
        confidence: float = 0.5,
    ) -> MentalState:
        """建模信念"""
        # 防御 confidence
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        state = MentalState(
            agent_id=agent_id,
            state_type=MentalStateType.BELIEF,
            content=belief,
            confidence=confidence,
        )
        with self._lock:
            if agent_id not in self._mental_states:
                self._mental_states[agent_id] = []
            self._mental_states[agent_id].append(state)
        return state

    def model_desire(
        self,
        agent_id: str,
        desire: Any,
        confidence: float = 0.5,
    ) -> MentalState:
        """建模欲望"""
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        state = MentalState(
            agent_id=agent_id,
            state_type=MentalStateType.DESIRE,
            content=desire,
            confidence=confidence,
        )
        with self._lock:
            if agent_id not in self._mental_states:
                self._mental_states[agent_id] = []
            self._mental_states[agent_id].append(state)
        return state

    def model_intention(
        self,
        agent_id: str,
        intention: Any,
        confidence: float = 0.5,
    ) -> MentalState:
        """建模意图"""
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        state = MentalState(
            agent_id=agent_id,
            state_type=MentalStateType.INTENTION,
            content=intention,
            confidence=confidence,
        )
        with self._lock:
            if agent_id not in self._mental_states:
                self._mental_states[agent_id] = []
            self._mental_states[agent_id].append(state)
        return state

    def model_emotion(
        self,
        agent_id: str,
        emotion: Any,
        confidence: float = 0.5,
    ) -> MentalState:
        """建模情感"""
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        if math.isnan(confidence) or math.isinf(confidence):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        state = MentalState(
            agent_id=agent_id,
            state_type=MentalStateType.EMOTION,
            content=emotion,
            confidence=confidence,
        )
        with self._lock:
            if agent_id not in self._mental_states:
                self._mental_states[agent_id] = []
            self._mental_states[agent_id].append(state)
        return state

    def get_mental_states(
        self,
        agent_id: str,
        state_type: Optional[MentalStateType] = None,
    ) -> List[MentalState]:
        """获取心理状态"""
        with self._lock:
            states = list(self._mental_states.get(agent_id, []))
        if state_type is not None:
            states = [s for s in states if s.state_type == state_type]
        return states

    def get_latest_belief(self, agent_id: str) -> Optional[MentalState]:
        """获取最新信念"""
        states = self.get_mental_states(agent_id, MentalStateType.BELIEF)
        return states[-1] if states else None

    def get_latest_desire(self, agent_id: str) -> Optional[MentalState]:
        """获取最新欲望"""
        states = self.get_mental_states(agent_id, MentalStateType.DESIRE)
        return states[-1] if states else None

    def get_latest_intention(self, agent_id: str) -> Optional[MentalState]:
        """获取最新意图"""
        states = self.get_mental_states(agent_id, MentalStateType.INTENTION)
        return states[-1] if states else None

    def predict_action(self, agent_id: str) -> Optional[Any]:
        """
        预测行动（阶段186增强：BDI组合预测）

        第一性原理：行动 = 意图 × 信念 × 欲望
        - 意图（Intention）：Agent计划做什么
        - 信念（Belief）：Agent认为世界是什么样的
        - 欲望（Desire）：Agent想要什么
        - 行为 = 意图在信念约束下追求欲望的结果

        阶段186改进：不再仅返回意图内容，而是组合BDI三要素
        """
        intention = self.get_latest_intention(agent_id)
        if intention is None:
            return None

        belief = self.get_latest_belief(agent_id)
        desire = self.get_latest_desire(agent_id)

        # 如果只有意图，退化为旧行为（向后兼容）
        if belief is None and desire is None:
            return intention.content

        # BDI组合预测
        # 意图是主要驱动力，信念提供约束，欲望提供方向
        predicted = {
            "intention": intention.content,
            "confidence": intention.confidence,
        }

        if belief is not None:
            predicted["belief_context"] = belief.content
            # 信念修正意图置信度：如果信念与意图冲突，降低置信度
            predicted["confidence"] *= (0.5 + 0.5 * belief.confidence)

        if desire is not None:
            predicted["desire"] = desire.content
            # 欲望增强意图：欲望置信度高时提升行为概率
            predicted["confidence"] = min(1.0, predicted["confidence"] * (0.7 + 0.3 * desire.confidence))

        # 综合预测：意图内容 + BDI置信度
        predicted["predicted_action"] = intention.content
        predicted["bdi_confidence"] = predicted.pop("confidence")

        return predicted

    def clear_agent(self, agent_id: str) -> None:
        """清除Agent的心理状态"""
        with self._lock:
            if agent_id in self._mental_states:
                del self._mental_states[agent_id]

    def clear_all(self) -> None:
        """清除所有"""
        with self._lock:
            self._mental_states.clear()


# ============================================================
# 阶段123新增：视角采择
# ============================================================

class PerspectiveTaking:
    """
    视角采择引擎

    第一性原理：
    - 视角采择 = 从他人角度看问题
    - 基于他人的心理状态推断其视角
    - 共情 = 理解并感受他人情感
    """

    def __init__(self, theory_of_mind: TheoryOfMind):
        self._tom = theory_of_mind
        self._lock = threading.RLock()

    def take_perspective(
        self,
        agent_id: str,
        situation: Any,
    ) -> Dict[str, Any]:
        """
        采取他人视角

        第一性原理：基于他人的信念和欲望理解情境
        """
        belief = self._tom.get_latest_belief(agent_id)
        desire = self._tom.get_latest_desire(agent_id)
        intention = self._tom.get_latest_intention(agent_id)

        perspective: Dict[str, Any] = {
            "agent_id": agent_id,
            "situation": situation,
            "interpreted_belief": belief.content if belief else None,
            "interpreted_desire": desire.content if desire else None,
            "interpreted_intention": intention.content if intention else None,
        }

        # 评估情境对Agent的相关性
        relevance = 0.5
        if desire and isinstance(situation, dict):
            # 简化：如果情境包含欲望相关内容，相关性高
            desire_str = str(desire.content).lower()
            for key in situation:
                if desire_str in str(key).lower() or str(key).lower() in desire_str:
                    relevance = 0.9
                    break
        perspective["relevance"] = relevance

        return perspective

    def infer_belief(
        self,
        agent_id: str,
        observation: Any,
    ) -> Optional[Any]:
        """
        从观察推断信念

        第一性原理：观察行为推断心理状态
        """
        # 简化：基于观察推断可能的信念
        if observation is None:
            return None
        # 如果观察是动作，推断其背后的信念
        if isinstance(observation, dict):
            action = observation.get("action")
            if action:
                return f"agent believes {action} is appropriate"
        return f"agent believes something about {observation}"

    def empathize(
        self,
        agent_id: str,
        emotion: Any,
    ) -> float:
        """
        共情

        第一性原理：共情 = 理解并感受他人情感
        Returns: 共情程度 [0, 1]
        """
        # 获取Agent当前情感状态
        emotions = self._tom.get_mental_states(agent_id, MentalStateType.EMOTION)
        if not emotions:
            return 0.3  # 默认低共情

        latest_emotion = emotions[-1]
        # 简化：如果情感匹配，共情程度高
        try:
            if str(latest_emotion.content) == str(emotion):
                return 0.9
            # 部分匹配
            emotion_str = str(emotion).lower()
            content_str = str(latest_emotion.content).lower()
            common = set(emotion_str.split()) & set(content_str.split())
            if common:
                return 0.6 + 0.3 * (len(common) / max(len(emotion_str.split()), 1))
        except Exception:
            pass

        return 0.4


# ============================================================
# 阶段123新增：信任模型
# ============================================================

class TrustModel:
    """
    信任模型

    第一性原理：
    - 信任基于历史交互的成功/失败
    - 信任影响信息权重
    - 信任可以建立和破坏
    """

    def __init__(self, initial_trust: float = 0.5):
        self._trust: Dict[str, float] = {}
        self._interactions: Dict[str, int] = {}  # agent_id -> interaction count
        self._initial_trust = max(0.0, min(1.0, initial_trust))
        self._lock = threading.RLock()

    def update_trust(self, agent_id: str, success: bool) -> float:
        """
        更新信任

        第一性原理：成功增加信任，失败减少信任
        """
        with self._lock:
            current = self._trust.get(agent_id, self._initial_trust)
            # 成功+0.1，失败-0.2（失败惩罚大于成功奖励）
            delta = 0.1 if success else -0.2
            new_trust = max(0.0, min(1.0, current + delta))
            self._trust[agent_id] = new_trust
            self._interactions[agent_id] = self._interactions.get(agent_id, 0) + 1
            return new_trust

    def get_trust(self, agent_id: str) -> float:
        """获取信任度"""
        with self._lock:
            return self._trust.get(agent_id, self._initial_trust)

    def get_trusted_agents(self, threshold: float = 0.7) -> List[str]:
        """获取可信Agent列表"""
        with self._lock:
            return [
                agent_id for agent_id, trust in self._trust.items()
                if trust >= threshold
            ]

    def get_interaction_count(self, agent_id: str) -> int:
        """获取交互次数"""
        with self._lock:
            return self._interactions.get(agent_id, 0)

    def reset_trust(self, agent_id: str) -> None:
        """重置信任"""
        with self._lock:
            if agent_id in self._trust:
                del self._trust[agent_id]
            if agent_id in self._interactions:
                del self._interactions[agent_id]

    def clear_all(self) -> None:
        """清除所有"""
        with self._lock:
            self._trust.clear()
            self._interactions.clear()


# ============================================================
# 阶段123新增：协作引擎
# ============================================================

class Collaboration:
    """
    协作引擎

    第一性原理：
    - 协作 = 多Agent共同达成目标
    - 任务分配基于能力匹配
    - 协作产生规模效应（1+1 > 2）
    """

    def __init__(self, trust_model: Optional[TrustModel] = None):
        self._tasks: Dict[str, CollaborationTask] = {}
        self._trust_model = trust_model or TrustModel()
        self._lock = threading.RLock()

    def create_task(
        self,
        task_id: str,
        description: str,
        required_capabilities: List[str],
    ) -> CollaborationTask:
        """创建协作任务"""
        task = CollaborationTask(
            task_id=task_id,
            description=description,
            required_capabilities=list(required_capabilities),
        )
        with self._lock:
            self._tasks[task_id] = task
        return task

    def assign_task(
        self,
        task_id: str,
        agent_ids: List[str],
    ) -> bool:
        """分配任务给Agent"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status != TaskStatus.PENDING:
                return False
            task.assigned_agents = list(agent_ids)
            task.status = TaskStatus.IN_PROGRESS
            return True

    def update_progress(
        self,
        task_id: str,
        progress: float,
    ) -> bool:
        """更新任务进度"""
        # 防御 progress
        try:
            progress = float(progress)
        except (TypeError, ValueError):
            return False
        if math.isnan(progress) or math.isinf(progress):
            return False
        progress = max(0.0, min(1.0, progress))

        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status != TaskStatus.IN_PROGRESS:
                return False
            task.progress = progress
            return True

    def complete_task(self, task_id: str) -> bool:
        """完成任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            if task.status != TaskStatus.IN_PROGRESS:
                return False
            task.status = TaskStatus.COMPLETED
            task.progress = 1.0
            task.completed_at = time.time()
            return True

    def fail_task(self, task_id: str) -> bool:
        """失败任务"""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            return True

    def get_task(self, task_id: str) -> Optional[CollaborationTask]:
        """获取任务"""
        with self._lock:
            return self._tasks.get(task_id)

    def get_tasks_by_status(self, status: TaskStatus) -> List[CollaborationTask]:
        """按状态获取任务"""
        with self._lock:
            return [
                task for task in self._tasks.values()
                if task.status == status
            ]

    def find_best_team(
        self,
        required_capabilities: List[str],
        candidates: Dict[str, List[str]],  # agent_id -> capabilities
    ) -> List[str]:
        """
        寻找最佳团队

        第一性原理：能力互补 + 信任度高
        """
        if not required_capabilities:
            return []

        # 找出能覆盖所有所需能力的Agent组合
        best_team: List[str] = []
        best_score = -1.0

        # 简化：贪心算法，按能力覆盖选择
        covered: Set[str] = set()
        team: List[str] = []

        # 按能力数量排序候选者
        sorted_candidates = sorted(
            candidates.items(),
            key=lambda x: len(set(x[1]) & set(required_capabilities)),
            reverse=True,
        )

        for agent_id, caps in sorted_candidates:
            new_caps = set(caps) & set(required_capabilities) - covered
            if new_caps:
                team.append(agent_id)
                covered |= new_caps
                if covered == set(required_capabilities):
                    break

        if covered == set(required_capabilities):
            # 计算团队信任分
            total_trust = sum(self._trust_model.get_trust(a) for a in team)
            avg_trust = total_trust / len(team) if team else 0
            best_team = team
            best_score = avg_trust

        return best_team

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "total_tasks": len(self._tasks),
                "pending": sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING),
                "in_progress": sum(1 for t in self._tasks.values() if t.status == TaskStatus.IN_PROGRESS),
                "completed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED),
                "failed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED),
            }

    def clear(self) -> None:
        """清除"""
        with self._lock:
            self._tasks.clear()


# ============================================================
# 社交第一性原理引擎（增强版）
# ============================================================

class SocialFirstPrinciplesEngine:
    """
    社交第一性原理引擎（阶段123增强版）

    新增能力:
    - TheoryOfMind: 理论心智
    - PerspectiveTaking: 视角采择
    - TrustModel: 信任模型
    - Collaboration: 协作引擎

    向后兼容: 保留原有 register_peer / send_message / share_knowledge 等接口
    """

    def __init__(self, engine_id: str = "social_engine"):
        self.engine_id = engine_id
        self._peers: Dict[str, AgentIdentity] = {}
        self._inbox: List[InterAgentMessage] = []
        self._outbox: List[InterAgentMessage] = []
        self._shared_knowledge: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._cycle_count = 0

        # 阶段123新增组件
        self._theory_of_mind = TheoryOfMind()
        self._trust_model = TrustModel()
        self._collaboration = Collaboration(self._trust_model)
        self._perspective_taking = PerspectiveTaking(self._theory_of_mind)

    # ============================================================
    # 原有接口（向后兼容）
    # ============================================================

    def register_peer(self, agent_id: str, capabilities: List[str] = None) -> AgentIdentity:
        """注册对等Agent（向后兼容）"""
        peer = AgentIdentity(
            agent_id=agent_id,
            capabilities=capabilities or [],
        )
        with self._lock:
            self._peers[agent_id] = peer
        return peer

    def update_trust(self, agent_id: str, delta: float) -> None:
        """更新信任分数（向后兼容）"""
        with self._lock:
            peer = self._peers.get(agent_id)
            if peer:
                # 防御 delta
                try:
                    delta = float(delta)
                except (TypeError, ValueError):
                    delta = 0.0
                if math.isnan(delta) or math.isinf(delta):
                    delta = 0.0
                peer.trust_score = max(0.0, min(1.0, peer.trust_score + delta))
                peer.last_seen = time.time()
                # 同步到信任模型
                self._trust_model._trust[agent_id] = peer.trust_score

    def send_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: MessageType,
        payload: Any,
    ) -> InterAgentMessage:
        """发送消息（向后兼容）"""
        msg = InterAgentMessage(
            message_id=f"msg_{int(time.time() * 1000)}_{self._cycle_count}",
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
        )
        with self._lock:
            self._outbox.append(msg)
            self._inbox.append(msg)  # 同时放入inbox供接收
            self._cycle_count += 1
        return msg

    def broadcast(
        self,
        sender_id: str,
        message_type: MessageType,
        payload: Any,
    ) -> InterAgentMessage:
        """广播消息（向后兼容）"""
        return self.send_message(sender_id, "*", message_type, payload)

    def receive_messages(self, agent_id: str) -> List[InterAgentMessage]:
        """接收发给指定Agent的消息（向后兼容）"""
        with self._lock:
            messages = [
                msg for msg in self._inbox
                if msg.receiver_id == agent_id or msg.receiver_id == "*"
            ]
            # 从inbox移除已接收的消息
            self._inbox = [
                msg for msg in self._inbox
                if msg.receiver_id != agent_id and msg.receiver_id != "*"
            ]
        return messages

    def share_knowledge(self, agent_id: str, key: str, value: Any) -> None:
        """共享知识（向后兼容）"""
        with self._lock:
            if key not in self._shared_knowledge:
                self._shared_knowledge[key] = {}
            self._shared_knowledge[key][agent_id] = {
                "value": value,
                "timestamp": time.time(),
            }

    def query_knowledge(self, key: str) -> Dict[str, Any]:
        """查询知识（向后兼容）"""
        with self._lock:
            entries = self._shared_knowledge.get(key, {})
            result = {}
            for agent_id, data in entries.items():
                peer = self._peers.get(agent_id)
                trust = peer.trust_score if peer else 0.5
                result[agent_id] = {
                    "value": data["value"],
                    "trust": trust,
                    "weighted_value": data["value"] * trust if isinstance(data["value"], (int, float)) else data["value"],
                }
            return result

    def find_collaborators(self, required_capability: str) -> List[AgentIdentity]:
        """寻找协作者（向后兼容）"""
        with self._lock:
            return [
                peer for peer in self._peers.values()
                if required_capability in peer.capabilities
            ]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            base_stats = {
                "engine_id": self.engine_id,
                "peers_count": len(self._peers),
                "inbox_size": len(self._inbox),
                "outbox_size": len(self._outbox),
                "shared_knowledge_topics": len(self._shared_knowledge),
                "cycle_count": self._cycle_count,
                "avg_trust": sum(p.trust_score for p in self._peers.values()) / len(self._peers) if self._peers else 0,
            }
            # 阶段123新增统计
            base_stats["collaboration_stats"] = self._collaboration.get_stats()
            return base_stats

    def clear(self) -> None:
        """清除"""
        with self._lock:
            self._inbox.clear()
            self._outbox.clear()
            self._shared_knowledge.clear()
            self._theory_of_mind.clear_all()
            self._trust_model.clear_all()
            self._collaboration.clear()

    # ============================================================
    # 阶段123新增：理论心智接口
    # ============================================================

    def model_belief(
        self,
        agent_id: str,
        belief: Any,
        confidence: float = 0.5,
    ) -> MentalState:
        """建模信念"""
        return self._theory_of_mind.model_belief(agent_id, belief, confidence)

    def model_desire(
        self,
        agent_id: str,
        desire: Any,
        confidence: float = 0.5,
    ) -> MentalState:
        """建模欲望"""
        return self._theory_of_mind.model_desire(agent_id, desire, confidence)

    def model_intention(
        self,
        agent_id: str,
        intention: Any,
        confidence: float = 0.5,
    ) -> MentalState:
        """建模意图"""
        return self._theory_of_mind.model_intention(agent_id, intention, confidence)

    def model_emotion(
        self,
        agent_id: str,
        emotion: Any,
        confidence: float = 0.5,
    ) -> MentalState:
        """建模情感"""
        return self._theory_of_mind.model_emotion(agent_id, emotion, confidence)

    def get_mental_states(
        self,
        agent_id: str,
        state_type: Optional[MentalStateType] = None,
    ) -> List[MentalState]:
        """获取心理状态"""
        return self._theory_of_mind.get_mental_states(agent_id, state_type)

    def predict_action(self, agent_id: str) -> Optional[Any]:
        """预测行动"""
        return self._theory_of_mind.predict_action(agent_id)

    def get_theory_of_mind(self) -> TheoryOfMind:
        """获取理论心智引擎"""
        return self._theory_of_mind

    # ============================================================
    # 阶段123新增：视角采择接口
    # ============================================================

    def take_perspective(
        self,
        agent_id: str,
        situation: Any,
    ) -> Dict[str, Any]:
        """采取他人视角"""
        return self._perspective_taking.take_perspective(agent_id, situation)

    def infer_belief(
        self,
        agent_id: str,
        observation: Any,
    ) -> Optional[Any]:
        """推断信念"""
        return self._perspective_taking.infer_belief(agent_id, observation)

    def empathize(
        self,
        agent_id: str,
        emotion: Any,
    ) -> float:
        """共情"""
        return self._perspective_taking.empathize(agent_id, emotion)

    def get_perspective_taking(self) -> PerspectiveTaking:
        """获取视角采择引擎"""
        return self._perspective_taking

    # ============================================================
    # 阶段123新增：信任模型接口
    # ============================================================

    def record_interaction(
        self,
        agent_id: str,
        success: bool,
    ) -> float:
        """记录交互结果，更新信任"""
        new_trust = self._trust_model.update_trust(agent_id, success)
        # 同步到peer
        with self._lock:
            peer = self._peers.get(agent_id)
            if peer:
                peer.trust_score = new_trust
                peer.last_seen = time.time()
        return new_trust

    def get_agent_trust(self, agent_id: str) -> float:
        """获取Agent信任度"""
        return self._trust_model.get_trust(agent_id)

    def get_trusted_agents(self, threshold: float = 0.7) -> List[str]:
        """获取可信Agent"""
        return self._trust_model.get_trusted_agents(threshold)

    def get_trust_model(self) -> TrustModel:
        """获取信任模型"""
        return self._trust_model

    # ============================================================
    # 阶段123新增：协作接口
    # ============================================================

    def create_collaboration_task(
        self,
        task_id: str,
        description: str,
        required_capabilities: List[str],
    ) -> CollaborationTask:
        """创建协作任务"""
        return self._collaboration.create_task(task_id, description, required_capabilities)

    def assign_collaboration_task(
        self,
        task_id: str,
        agent_ids: List[str],
    ) -> bool:
        """分配协作任务"""
        return self._collaboration.assign_task(task_id, agent_ids)

    def update_task_progress(
        self,
        task_id: str,
        progress: float,
    ) -> bool:
        """更新任务进度"""
        return self._collaboration.update_progress(task_id, progress)

    def complete_collaboration_task(self, task_id: str) -> bool:
        """完成协作任务"""
        return self._collaboration.complete_task(task_id)

    def fail_collaboration_task(self, task_id: str) -> bool:
        """失败协作任务"""
        return self._collaboration.fail_task(task_id)

    def get_collaboration_task(self, task_id: str) -> Optional[CollaborationTask]:
        """获取协作任务"""
        return self._collaboration.get_task(task_id)

    def find_best_team(
        self,
        required_capabilities: List[str],
    ) -> List[str]:
        """寻找最佳团队"""
        with self._lock:
            candidates = {
                peer.agent_id: peer.capabilities
                for peer in self._peers.values()
            }
        return self._collaboration.find_best_team(required_capabilities, candidates)

    def get_collaboration_engine(self) -> Collaboration:
        """获取协作引擎"""
        return self._collaboration

    # ============================================================
    # 阶段186新增：BDI框架接口
    # ============================================================

    def build_bdi_state(self, agent_id: str) -> BDIState:
        """
        从理论心智模型构建Agent的BDI状态

        第一性原理：将心理状态映射为BDI三组件
        - Belief ← 心智模型中的信念
        - Desire ← 心智模型中的欲望
        - Intention ← 心智模型中的意图
        """
        beliefs = [s.content for s in self._theory_of_mind.get_mental_states(agent_id, MentalStateType.BELIEF)]
        desires = [s.content for s in self._theory_of_mind.get_mental_states(agent_id, MentalStateType.DESIRE)]
        intentions = [s.content for s in self._theory_of_mind.get_mental_states(agent_id, MentalStateType.INTENTION)]

        # 从最新状态提取置信度
        belief_conf = 0.5
        desire_conf = 0.5
        intention_conf = 0.5

        latest_belief = self._theory_of_mind.get_latest_belief(agent_id)
        if latest_belief:
            belief_conf = latest_belief.confidence

        latest_desire = self._theory_of_mind.get_latest_desire(agent_id)
        if latest_desire:
            desire_conf = latest_desire.confidence

        latest_intention = self._theory_of_mind.get_latest_intention(agent_id)
        if latest_intention:
            intention_conf = latest_intention.confidence

        return BDIState(
            agent_id=agent_id,
            beliefs=beliefs,
            desires=desires,
            intentions=intentions,
            belief_confidence=belief_conf,
            desire_strength=desire_conf,
            intention_commitment=intention_conf,
        )

    def predict_action_bdi(
        self,
        agent_id: str,
        candidate_actions: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        基于BDI框架预测Agent行为

        第一性原理：组合信念、欲望、意图三要素预测行为
        - 比简单predict_action提供更丰富的推理过程
        - 支持候选行动评估
        """
        bdi_state = self.build_bdi_state(agent_id)
        predictor = ActionPredictor()
        return predictor.predict(bdi_state, candidate_actions)

    def predict_action_sequence(
        self,
        agent_id: str,
        steps: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        预测Agent的行动序列

        第一性原理：基于意图队列预测多步行动
        """
        bdi_state = self.build_bdi_state(agent_id)
        predictor = ActionPredictor()
        return predictor.predict_sequence(bdi_state, steps)


# ============================================================
# 便捷函数（向后兼容 + 新增）
# ============================================================

def create_social_engine(engine_id: str = "social_engine") -> SocialFirstPrinciplesEngine:
    """创建社交引擎"""
    return SocialFirstPrinciplesEngine(engine_id=engine_id)


def create_agent_identity(agent_id: str, capabilities: List[str] = None) -> AgentIdentity:
    """创建Agent身份"""
    return AgentIdentity(agent_id=agent_id, capabilities=capabilities or [])


def create_theory_of_mind() -> TheoryOfMind:
    """创建理论心智引擎"""
    return TheoryOfMind()


def create_trust_model(initial_trust: float = 0.5) -> TrustModel:
    """创建信任模型"""
    return TrustModel(initial_trust=initial_trust)


def create_collaboration(trust_model: Optional[TrustModel] = None) -> Collaboration:
    """创建协作引擎"""
    return Collaboration(trust_model=trust_model)


# ============================================================
# 阶段186: BDI组合推理器
# ============================================================

@dataclass
class BDIState:
    """
    BDI状态（阶段186）

    第一性原理：BDI模型将Agent心智分为三个组件
    - Belief：对世界状态的信念
    - Desire：期望达成的目标
    - Intention：承诺执行的行动
    """
    agent_id: str
    beliefs: List[Any] = field(default_factory=list)
    desires: List[Any] = field(default_factory=list)
    intentions: List[Any] = field(default_factory=list)
    belief_confidence: float = 0.5
    desire_strength: float = 0.5
    intention_commitment: float = 0.5

    @property
    def coherence(self) -> float:
        """
        BDI一致性：信念、欲望、意图之间的协调程度

        第一性原理：高一致性的BDI状态更可能产生可预测的行为
        - 一致性 = 三者置信度的几何平均
        """
        return (
            self.belief_confidence
            * self.desire_strength
            * self.intention_commitment
        ) ** (1.0 / 3.0)


class BDIReasoner:
    """
    BDI组合推理器（阶段186）

    第一性原理：
    - 行为 = 意图在信念约束下追求欲望的结果
    - 信念过滤可行行动（不可能的行动被排除）
    - 欲望评估行动价值（高价值的行动被优先选择）
    - 意图确定行动承诺（已承诺的行动被执行）
    """

    def __init__(self):
        self._lock = threading.RLock()

    def reason(
        self,
        beliefs: List[Any],
        desires: List[Any],
        intentions: List[Any],
        belief_confidence: float = 0.5,
        desire_strength: float = 0.5,
        intention_commitment: float = 0.5,
    ) -> Dict[str, Any]:
        """
        BDI组合推理

        第一性原理：
        1. 信念约束：从意图中排除与信念冲突的行动
        2. 欲望导向：选择最接近欲望的行动
        3. 意图承诺：执行已承诺的行动

        Returns:
            {
                "recommended_action": 推荐行动,
                "bdi_confidence": BDI置信度,
                "coherence": 一致性,
                "reasoning_trace": 推理过程,
            }
        """
        with self._lock:
            reasoning_trace: List[str] = []

            # 步骤1：意图是行为的主要驱动力
            if not intentions:
                reasoning_trace.append("无意图：无法预测行为")
                return {
                    "recommended_action": None,
                    "bdi_confidence": 0.0,
                    "coherence": 0.0,
                    "reasoning_trace": reasoning_trace,
                }

            primary_intention = intentions[-1]
            reasoning_trace.append(f"主要意图: {primary_intention}")

            # 步骤2：信念约束
            bdi_confidence = intention_commitment
            if beliefs:
                latest_belief = beliefs[-1]
                reasoning_trace.append(f"信念约束: {latest_belief}")
                # 信念置信度修正行为置信度
                bdi_confidence *= (0.5 + 0.5 * belief_confidence)
            else:
                reasoning_trace.append("无信念约束")

            # 步骤3：欲望导向
            if desires:
                latest_desire = desires[-1]
                reasoning_trace.append(f"欲望导向: {latest_desire}")
                # 欲望增强行为置信度
                bdi_confidence = min(1.0, bdi_confidence * (0.7 + 0.3 * desire_strength))
            else:
                reasoning_trace.append("无欲望导向")

            # 一致性
            coherence = (
                belief_confidence * desire_strength * intention_commitment
            ) ** (1.0 / 3.0)

            return {
                "recommended_action": primary_intention,
                "bdi_confidence": bdi_confidence,
                "coherence": coherence,
                "reasoning_trace": reasoning_trace,
            }

    def evaluate_action(
        self,
        action: Any,
        bdi_state: BDIState,
    ) -> Dict[str, Any]:
        """
        评估行动的BDI适合度

        第一性原理：
        - 行动与意图的一致性（承诺度）
        - 行动与信念的兼容性（可行性）
        - 行动与欲望的对齐度（价值）
        """
        with self._lock:
            # 意图一致性
            intention_alignment = 1.0 if action in bdi_state.intentions else 0.3

            # 信念兼容性（简化：检查行动是否在信念列表中）
            belief_compatibility = 0.8  # 默认高兼容
            if bdi_state.beliefs and action not in bdi_state.beliefs:
                belief_compatibility = 0.5

            # 欲望对齐
            desire_alignment = 1.0 if action in bdi_state.desires else 0.4

            # 综合适合度
            suitability = (
                intention_alignment * bdi_state.intention_commitment
                + belief_compatibility * bdi_state.belief_confidence
                + desire_alignment * bdi_state.desire_strength
            ) / 3.0

            return {
                "action": action,
                "intention_alignment": intention_alignment,
                "belief_compatibility": belief_compatibility,
                "desire_alignment": desire_alignment,
                "overall_suitability": suitability,
            }


class ActionPredictor:
    """
    基于BDI的行为预测器（阶段186）

    第一性原理：
    - 行为预测 = 基于BDI状态推断最可能的行动
    - 预测置信度 = BDI一致性 × 意图承诺度
    - 多步预测 = 基于当前BDI状态预测行动序列
    """

    def __init__(self, bdi_reasoner: Optional[BDIReasoner] = None):
        self._bdi_reasoner = bdi_reasoner or BDIReasoner()
        self._lock = threading.RLock()

    def predict(
        self,
        bdi_state: BDIState,
        candidate_actions: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        预测最可能的行动

        Args:
            bdi_state: BDI状态
            candidate_actions: 候选行动列表（None则使用意图作为候选）

        Returns:
            {
                "predicted_action": 预测行动,
                "confidence": 预测置信度,
                "alternatives": 备选行动及适合度,
            }
        """
        with self._lock:
            # BDI推理
            reasoning = self._bdi_reasoner.reason(
                beliefs=bdi_state.beliefs,
                desires=bdi_state.desires,
                intentions=bdi_state.intentions,
                belief_confidence=bdi_state.belief_confidence,
                desire_strength=bdi_state.desire_strength,
                intention_commitment=bdi_state.intention_commitment,
            )

            # 如果有候选行动，评估每个
            alternatives: List[Dict[str, Any]] = []
            if candidate_actions:
                for action in candidate_actions:
                    evaluation = self._bdi_reasoner.evaluate_action(action, bdi_state)
                    alternatives.append(evaluation)

                # 按适合度排序
                alternatives.sort(key=lambda x: x["overall_suitability"], reverse=True)

                # 最适合的行动
                if alternatives:
                    best = alternatives[0]
                    return {
                        "predicted_action": best["action"],
                        "confidence": best["overall_suitability"] * bdi_state.coherence,
                        "alternatives": alternatives,
                        "reasoning": reasoning,
                    }

            # 无候选行动时使用BDI推理结果
            return {
                "predicted_action": reasoning["recommended_action"],
                "confidence": reasoning["bdi_confidence"] * bdi_state.coherence,
                "alternatives": alternatives,
                "reasoning": reasoning,
            }

    def predict_sequence(
        self,
        bdi_state: BDIState,
        steps: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        预测行动序列

        第一性原理：基于当前BDI状态预测未来N步行动
        - 每步预测后，假设意图被执行并从意图列表移除
        - 置信度随步数衰减
        """
        with self._lock:
            predictions: List[Dict[str, Any]] = []
            current_intentions = list(bdi_state.intentions)

            for step in range(steps):
                if not current_intentions:
                    break

                # 创建临时BDI状态
                temp_state = BDIState(
                    agent_id=bdi_state.agent_id,
                    beliefs=bdi_state.beliefs,
                    desires=bdi_state.desires,
                    intentions=current_intentions,
                    belief_confidence=bdi_state.belief_confidence,
                    desire_strength=bdi_state.desire_strength,
                    intention_commitment=bdi_state.intention_commitment,
                )

                prediction = self.predict(temp_state)
                # 置信度随步数衰减
                prediction["step"] = step
                prediction["confidence"] *= (0.8 ** step)
                predictions.append(prediction)

                # 移除已执行的意图
                if current_intentions:
                    current_intentions.pop()

            return predictions
