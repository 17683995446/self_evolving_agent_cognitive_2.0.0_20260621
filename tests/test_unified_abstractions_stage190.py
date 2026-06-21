"""
阶段190测试：代码去重 — 统一熔断器与日志抽象

验证内容:
1. 统一熔断器：CircuitBreaker支持call()和can_execute()/record_*()双模式
2. recovery.py的CircuitBreakerPolicy委托给统一熔断器
3. production.py的CircuitBreaker委托给统一熔断器
4. observability.py的StructuredLogger委托给logging.py
5. 向后兼容性：原有接口仍正常工作
6. DRY原则：熔断逻辑和日志逻辑只有一份实现
"""

import pytest
import time
from self_evolving_agent.core.circuit_breaker import (
    CircuitBreaker as UnifiedCircuitBreaker,
    CircuitState as UnifiedCircuitState,
    CircuitBreakerAction,
    CircuitBreakerResult,
    create_circuit_breaker,
    circuit_breaker_protected,
)
from self_evolving_agent.core.recovery import (
    CircuitBreakerPolicy,
    CircuitState as RecoveryCircuitState,
)
from self_evolving_agent.core.production import (
    CircuitBreaker as ProductionCircuitBreaker,
    CircuitState as ProductionCircuitState,
)
from self_evolving_agent.core.observability import (
    StructuredLogger as ObservabilityLogger,
    LogLevel as ObservabilityLogLevel,
    LogRecord as ObservabilityLogRecord,
    LoggerRegistry,
    get_logger as get_observability_logger,
)
from self_evolving_agent.core.logging import (
    StructuredLogger as UnifiedLogger,
    LogLevel as UnifiedLogLevel,
    LogManager,
    get_logger as get_unified_logger,
)


# ============================================================
# 统一熔断器双模式测试
# ============================================================

class TestUnifiedCircuitBreakerDualMode:
    """统一熔断器双模式测试"""

    def test_call_mode_basic(self):
        """测试call()模式基本功能"""
        breaker = UnifiedCircuitBreaker(name="test", failure_threshold=3)

        def success_op():
            return "success"

        result = breaker.call(success_op)
        assert result.success is True
        assert result.result == "success"

    def test_call_mode_failure(self):
        """测试call()模式失败"""
        breaker = UnifiedCircuitBreaker(name="test", failure_threshold=2)

        def fail_op():
            raise ValueError("test error")

        result = breaker.call(fail_op)
        assert result.success is False
        assert isinstance(result.error, ValueError)

    def test_can_execute_mode_basic(self):
        """测试can_execute()模式基本功能"""
        breaker = UnifiedCircuitBreaker(name="test", failure_threshold=3)

        # 初始状态应允许执行
        assert breaker.can_execute() is True
        assert breaker.state == UnifiedCircuitState.CLOSED

    def test_can_execute_mode_with_success(self):
        """测试can_execute()模式记录成功"""
        breaker = UnifiedCircuitBreaker(name="test", failure_threshold=3)

        assert breaker.can_execute() is True
        breaker.record_success()
        assert breaker.state == UnifiedCircuitState.CLOSED

    def test_can_execute_mode_with_failure(self):
        """测试can_execute()模式记录失败后熔断"""
        breaker = UnifiedCircuitBreaker(name="test", failure_threshold=2)

        # 第一次失败
        assert breaker.can_execute() is True
        breaker.record_failure()

        # 第二次失败 → 达到阈值 → 熔断
        assert breaker.can_execute() is True
        breaker.record_failure()

        # 现在应该熔断
        assert breaker.state == UnifiedCircuitState.OPEN

    def test_both_modes_share_state(self):
        """测试双模式共享同一状态机"""
        breaker = UnifiedCircuitBreaker(name="test", failure_threshold=2)

        # 使用can_execute模式 — 第一次失败
        assert breaker.can_execute() is True
        breaker.record_failure()

        # 使用call模式 — 成功会重置失败计数（CLOSED状态下的正确行为）
        result = breaker.call(lambda: "ok")
        assert result.success is True

        # 再用can_execute模式 — 连续2次失败触发熔断
        assert breaker.can_execute() is True
        breaker.record_failure()

        assert breaker.can_execute() is True
        breaker.record_failure()

        # 应该熔断了
        assert breaker.state == UnifiedCircuitState.OPEN

    def test_get_stats_works_for_both_modes(self):
        """测试统计信息对双模式都有效"""
        breaker = UnifiedCircuitBreaker(name="test", failure_threshold=5)

        # call模式
        breaker.call(lambda: "ok")

        # can_execute模式
        breaker.can_execute()
        breaker.record_success()

        stats = breaker.get_stats()
        assert stats["total_calls"] >= 2
        assert stats["total_successes"] >= 2


# ============================================================
# recovery.py CircuitBreakerPolicy 委托测试
# ============================================================

class TestRecoveryCircuitBreakerPolicyDelegation:
    """recovery.py CircuitBreakerPolicy委托测试"""

    def test_policy_can_execute(self):
        """测试Policy的can_execute委托"""
        policy = CircuitBreakerPolicy(failure_threshold=3, recovery_timeout=1.0)

        assert policy.can_execute() is True
        assert policy.state == RecoveryCircuitState.CLOSED

    def test_policy_record_success(self):
        """测试Policy的record_success委托"""
        policy = CircuitBreakerPolicy(failure_threshold=3)

        policy.can_execute()
        policy.record_success()

        assert policy.state == RecoveryCircuitState.CLOSED

    def test_policy_record_failure_and_open(self):
        """测试Policy的record_failure委托后熔断"""
        policy = CircuitBreakerPolicy(failure_threshold=2, recovery_timeout=1.0)

        policy.can_execute()
        policy.record_failure()

        policy.can_execute()
        policy.record_failure()

        assert policy.state == RecoveryCircuitState.OPEN

    def test_policy_reset(self):
        """测试Policy的reset委托"""
        policy = CircuitBreakerPolicy(failure_threshold=2, recovery_timeout=1.0)

        # 触发熔断
        for _ in range(2):
            policy.can_execute()
            policy.record_failure()

        assert policy.state == RecoveryCircuitState.OPEN

        # 重置
        policy.reset()
        assert policy.state == RecoveryCircuitState.CLOSED

    def test_policy_get_stats(self):
        """测试Policy的get_stats委托"""
        policy = CircuitBreakerPolicy(failure_threshold=5, recovery_timeout=30.0)

        policy.can_execute()
        policy.record_success()

        stats = policy.get_stats()
        assert "state" in stats
        assert "failure_threshold" in stats
        assert "recovery_timeout" in stats
        assert stats["failure_threshold"] == 5
        assert stats["recovery_timeout"] == 30.0

    def test_policy_uses_unified_implementation(self):
        """测试Policy内部使用统一熔断器"""
        policy = CircuitBreakerPolicy(failure_threshold=3)
        assert hasattr(policy, "_breaker")
        assert isinstance(policy._breaker, UnifiedCircuitBreaker)


# ============================================================
# production.py CircuitBreaker 委托测试
# ============================================================

class TestProductionCircuitBreakerDelegation:
    """production.py CircuitBreaker委托测试"""

    def test_prod_can_execute(self):
        """测试Production CircuitBreaker的can_execute委托"""
        breaker = ProductionCircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

        assert breaker.can_execute() is True
        assert breaker.state == ProductionCircuitState.CLOSED

    def test_prod_record_success(self):
        """测试Production CircuitBreaker的record_success委托"""
        breaker = ProductionCircuitBreaker(failure_threshold=3)

        breaker.can_execute()
        breaker.record_success()

        assert breaker.state == ProductionCircuitState.CLOSED

    def test_prod_record_failure_and_open(self):
        """测试Production CircuitBreaker的record_failure委托后熔断"""
        breaker = ProductionCircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

        breaker.can_execute()
        breaker.record_failure()

        breaker.can_execute()
        breaker.record_failure()

        assert breaker.state == ProductionCircuitState.OPEN

    def test_prod_reset(self):
        """测试Production CircuitBreaker的reset委托"""
        breaker = ProductionCircuitBreaker(failure_threshold=2, recovery_timeout=60.0)

        for _ in range(2):
            breaker.can_execute()
            breaker.record_failure()

        assert breaker.state == ProductionCircuitState.OPEN

        breaker.reset()
        assert breaker.state == ProductionCircuitState.CLOSED

    def test_prod_stats(self):
        """测试Production CircuitBreaker的stats委托"""
        breaker = ProductionCircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

        breaker.can_execute()
        breaker.record_success()

        stats = breaker.stats()
        assert "state" in stats
        assert "failure_threshold" in stats
        assert "recovery_timeout" in stats

    def test_prod_get_stats_alias(self):
        """测试Production CircuitBreaker的get_stats别名"""
        breaker = ProductionCircuitBreaker(failure_threshold=5)

        stats1 = breaker.stats()
        stats2 = breaker.get_stats()
        assert stats1 == stats2

    def test_prod_uses_unified_implementation(self):
        """测试Production CircuitBreaker内部使用统一熔断器"""
        breaker = ProductionCircuitBreaker(failure_threshold=3)
        assert hasattr(breaker, "_breaker")
        assert isinstance(breaker._breaker, UnifiedCircuitBreaker)


# ============================================================
# 统一日志委托测试
# ============================================================

class TestUnifiedLoggingDelegation:
    """统一日志委托测试"""

    def test_observability_logger_delegates_to_unified(self):
        """测试observability日志器委托给统一日志器"""
        logger = ObservabilityLogger("test_source")

        assert hasattr(logger, "_unified_logger")
        assert isinstance(logger._unified_logger, UnifiedLogger)

    def test_observability_log_level_matches_unified(self):
        """测试observability日志级别与统一日志级别一致"""
        assert ObservabilityLogLevel.DEBUG == UnifiedLogLevel.DEBUG
        assert ObservabilityLogLevel.INFO == UnifiedLogLevel.INFO
        assert ObservabilityLogLevel.WARNING == UnifiedLogLevel.WARNING
        assert ObservabilityLogLevel.ERROR == UnifiedLogLevel.ERROR
        assert ObservabilityLogLevel.CRITICAL == UnifiedLogLevel.CRITICAL

    def test_observability_logger_info(self):
        """测试observability日志器info方法"""
        logger = ObservabilityLogger("test_info")
        logger.info("test message", key="value")

        records = logger.get_records()
        assert len(records) >= 1
        assert records[-1].message == "test message"
        assert records[-1].level == ObservabilityLogLevel.INFO

    def test_observability_logger_debug(self):
        """测试observability日志器debug方法"""
        logger = ObservabilityLogger("test_debug")
        logger.set_level(ObservabilityLogLevel.DEBUG)
        logger.debug("debug message")

        records = logger.get_records()
        assert len(records) >= 1
        assert records[-1].message == "debug message"

    def test_observability_logger_warning(self):
        """测试observability日志器warning方法"""
        logger = ObservabilityLogger("test_warning")
        logger.warning("warning message")

        records = logger.get_records()
        assert len(records) >= 1
        assert records[-1].level == ObservabilityLogLevel.WARNING

    def test_observability_logger_error(self):
        """测试observability日志器error方法"""
        logger = ObservabilityLogger("test_error")
        logger.error("error message", exception=ValueError("test"))

        records = logger.get_records()
        assert len(records) >= 1
        assert records[-1].level == ObservabilityLogLevel.ERROR

    def test_observability_logger_get_stats(self):
        """测试observability日志器get_stats方法"""
        logger = ObservabilityLogger("test_stats")
        logger.info("msg1")
        logger.warning("msg2")

        stats = logger.get_stats()
        assert stats["name"] == "test_stats"
        assert stats["total_records"] >= 2
        assert "INFO" in stats["counters"]
        assert "WARNING" in stats["counters"]

    def test_observability_logger_clear(self):
        """测试observability日志器clear方法"""
        logger = ObservabilityLogger("test_clear")
        logger.info("msg")
        assert len(logger.get_records()) >= 1

        logger.clear()
        assert len(logger.get_records()) == 0

    def test_observability_logger_add_handler(self):
        """测试observability日志器add_handler方法"""
        logger = ObservabilityLogger("test_handler")
        received = []
        logger.add_handler(lambda r: received.append(r))

        logger.info("handler test")

        assert len(received) >= 1
        assert received[0].message == "handler test"

    def test_observability_get_logger_returns_logger(self):
        """测试observability get_logger返回日志器"""
        logger = get_observability_logger("test_get")
        assert isinstance(logger, ObservabilityLogger)
        assert logger.name == "test_get"


# ============================================================
# DRY原则验证测试
# ============================================================

class TestDRYPrinciple:
    """DRY原则验证测试"""

    def test_circuit_breaker_single_implementation(self):
        """测试熔断逻辑只有一份实现"""
        # recovery.py的Policy内部委托给统一熔断器
        policy = CircuitBreakerPolicy(failure_threshold=3)
        assert isinstance(policy._breaker, UnifiedCircuitBreaker)

        # production.py的CircuitBreaker内部委托给统一熔断器
        prod_breaker = ProductionCircuitBreaker(failure_threshold=3)
        assert isinstance(prod_breaker._breaker, UnifiedCircuitBreaker)

    def test_logging_single_implementation(self):
        """测试日志逻辑只有一份实现"""
        # observability.py的Logger内部委托给统一日志器
        logger = ObservabilityLogger("dry_test")
        assert isinstance(logger._unified_logger, UnifiedLogger)

    def test_circuit_state_enums_compatible(self):
        """测试熔断状态枚举兼容"""
        # 所有CircuitState应来自同一枚举
        assert issubclass(RecoveryCircuitState, UnifiedCircuitState)
        assert issubclass(ProductionCircuitState, UnifiedCircuitState)

        # 值应一致
        assert RecoveryCircuitState.CLOSED == UnifiedCircuitState.CLOSED
        assert ProductionCircuitState.OPEN == UnifiedCircuitState.OPEN
        assert RecoveryCircuitState.HALF_OPEN == UnifiedCircuitState.HALF_OPEN

    def test_log_level_enums_compatible(self):
        """测试日志级别枚举兼容"""
        assert issubclass(ObservabilityLogLevel, UnifiedLogLevel)
        assert ObservabilityLogLevel.INFO == UnifiedLogLevel.INFO


# ============================================================
# 向后兼容性测试
# ============================================================

class TestBackwardCompatibility:
    """向后兼容性测试"""

    def test_unified_circuit_breaker_call_mode_preserved(self):
        """测试统一熔断器call()模式保留"""
        breaker = UnifiedCircuitBreaker(name="compat", failure_threshold=3)

        result = breaker.call(lambda x: x * 2, 5)
        assert result.success is True
        assert result.result == 10

    def test_unified_circuit_breaker_decorator_preserved(self):
        """测试熔断器装饰器保留"""
        @circuit_breaker_protected(failure_threshold=3)
        def my_function(x):
            return x + 1

        result = my_function(5)
        assert result == 6

    def test_recovery_policy_interface_preserved(self):
        """测试recovery Policy接口保留"""
        policy = CircuitBreakerPolicy(failure_threshold=3, recovery_timeout=30.0)

        # 原有接口仍可用
        assert policy.can_execute() is True
        policy.record_success()
        policy.record_failure()
        policy.reset()
        stats = policy.get_stats()
        assert isinstance(stats, dict)

    def test_production_breaker_interface_preserved(self):
        """测试production CircuitBreaker接口保留"""
        breaker = ProductionCircuitBreaker(failure_threshold=3, recovery_timeout=60.0)

        # 原有接口仍可用
        assert breaker.can_execute() is True
        breaker.record_success()
        breaker.record_failure()
        breaker.reset()
        stats = breaker.stats()
        assert isinstance(stats, dict)

    def test_observability_logger_interface_preserved(self):
        """测试observability日志器接口保留"""
        logger = ObservabilityLogger("compat_test")
        logger.set_level(ObservabilityLogLevel.DEBUG)

        # 原有接口仍可用
        logger.debug("d")
        logger.info("i")
        logger.warning("w")
        logger.error("e")
        logger.critical("c")

        records = logger.get_records()
        assert len(records) >= 5

        stats = logger.get_stats()
        assert "name" in stats

        logger.clear()


# ============================================================
# 第一性原理验证测试
# ============================================================

class TestFirstPrinciples:
    """第一性原理验证"""

    def test_circuit_breaker_prevents_cascade_failure(self):
        """测试熔断器阻止级联失败（第一性原理）"""
        breaker = UnifiedCircuitBreaker(name="cascade", failure_threshold=3)

        call_count = 0

        def failing_op():
            nonlocal call_count
            call_count += 1
            raise Exception("fail")

        # 前3次尝试（达到阈值）
        for _ in range(3):
            breaker.call(failing_op)

        # 熔断器应打开
        assert breaker.state == UnifiedCircuitState.OPEN

        # 后续调用应快速失败，不执行操作
        result = breaker.call(failing_op)
        assert result.success is False
        assert "OPEN" in str(result.error)

    def test_circuit_breaker_auto_recovery(self):
        """测试熔断器自动恢复（第一性原理）"""
        breaker = UnifiedCircuitBreaker(
            name="recovery",
            failure_threshold=2,
            recovery_timeout_ms=100,  # 100ms
            success_threshold=1,
        )

        # 触发熔断
        for _ in range(2):
            breaker.call(lambda: (_ for _ in ()).throw(Exception("fail")))

        assert breaker.state == UnifiedCircuitState.OPEN

        # 等待恢复
        time.sleep(0.15)

        # 应允许试探
        result = breaker.call(lambda: "recovered")
        assert result.success is True

    def test_logging_structured_fields(self):
        """测试结构化日志字段（第一性原理）"""
        logger = ObservabilityLogger("structured_test")

        logger.info(
            "user action",
            user_id=123,
            action="login",
            ip="192.168.1.1",
        )

        records = logger.get_records()
        assert len(records) >= 1
        assert records[-1].fields["user_id"] == 123
        assert records[-1].fields["action"] == "login"

    def test_single_source_of_truth(self):
        """测试单一数据源（第一性原理：DRY）"""
        # 熔断器：所有实现委托给同一个类
        unified = UnifiedCircuitBreaker(name="truth", failure_threshold=5)
        policy = CircuitBreakerPolicy(failure_threshold=5)
        prod = ProductionCircuitBreaker(failure_threshold=5)

        # 它们内部都使用UnifiedCircuitBreaker
        assert type(policy._breaker) == type(unified)
        assert type(prod._breaker) == type(unified)

        # 日志：observability委托给logging
        obs_logger = ObservabilityLogger("truth_test")
        assert isinstance(obs_logger._unified_logger, UnifiedLogger)
