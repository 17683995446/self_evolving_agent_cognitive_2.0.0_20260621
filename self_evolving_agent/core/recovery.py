"""
阶段140: 错误恢复策略 — 自动恢复

第一性原理:
1. 弹性 — 系统应能从错误中恢复，而非崩溃
2. 隔离性 — 错误应被隔离在最小范围内，不传播
3. 可观测性 — 错误和恢复过程应被记录
4. 渐进降级 — 系统在部分失效时应渐进降级而非完全失效
5. 自愈性 — 系统应能自动检测和修复常见问题

组件:
- ErrorContext — 错误上下文（记录错误详情）
- RecoveryAction — 恢复动作类型
- RetryPolicy — 重试策略（指数退避）
- FallbackHandler — 回退处理器
- CircuitBreakerPolicy — 熔断策略
- HealthMonitor — 健康监控（定期检查）
- RecoveryStrategy — 恢复策略（组合多种策略）
- RecoveryManager — 恢复管理器（协调所有策略）
"""

import os
import time
import threading
import traceback
from typing import Any, Dict, List, Optional, Set, Callable, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from collections import deque, defaultdict


# ============================================================
# 错误上下文
# ============================================================

class ErrorSeverity(Enum):
    """错误严重级别"""
    LOW = "low"           # 低 — 可忽略
    MEDIUM = "medium"     # 中 — 需关注
    HIGH = "high"         # 高 — 需立即处理
    CRITICAL = "critical" # 严重 — 系统级影响


class RecoveryAction(Enum):
    """恢复动作类型"""
    RETRY = "retry"           # 重试
    FALLBACK = "fallback"     # 回退
    CIRCUIT_BREAK = "circuit_break"  # 熔断
    DEGRADE = "degrade"       # 降级
    RESTART = "restart"       # 重启
    ESCALATE = "escalate"     # 升级（通知人工）
    IGNORE = "ignore"         # 忽略


@dataclass
class ErrorContext:
    """错误上下文"""
    error_id: str
    error_type: str
    error_message: str
    traceback_str: str
    component: str
    operation: str
    timestamp: float
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_action: Optional[RecoveryAction] = None
    recovery_success: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_id": self.error_id,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback_str": self.traceback_str,
            "component": self.component,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "context": self.context,
            "recovery_action": self.recovery_action.value if self.recovery_action else None,
            "recovery_success": self.recovery_success,
        }


# ============================================================
# 重试策略
# ============================================================

class RetryPolicy:
    """
    重试策略

    支持指数退避、最大重试次数、可重试异常类型。
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 0.1,
        max_delay: float = 10.0,
        exponential: bool = True,
        retryable_exceptions: Optional[Set[Type[Exception]]] = None,
    ):
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._exponential = exponential
        self._retryable_exceptions = retryable_exceptions or {Exception}
        self._lock = threading.RLock()
        self._stats = {"attempts": 0, "successes": 0, "failures": 0}

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """判断是否应该重试"""
        with self._lock:
            self._stats["attempts"] += 1

        if attempt >= self._max_retries:
            return False

        for exc_type in self._retryable_exceptions:
            if isinstance(exception, exc_type):
                return True
        return False

    def get_delay(self, attempt: int) -> float:
        """获取重试延迟"""
        if self._exponential:
            delay = self._base_delay * (2 ** attempt)
        else:
            delay = self._base_delay
        return min(delay, self._max_delay)

    def execute(self, func: Callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """
        执行函数，带重试

        Returns:
            (result, error) — 成功时error为None
        """
        last_error = None
        for attempt in range(self._max_retries + 1):
            try:
                result = func(*args, **kwargs)
                with self._lock:
                    self._stats["successes"] += 1
                return result, None
            except Exception as e:
                last_error = e
                if not self.should_retry(e, attempt):
                    break
                delay = self.get_delay(attempt)
                if delay > 0:
                    time.sleep(delay)

        with self._lock:
            self._stats["failures"] += 1
        return None, last_error

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


# ============================================================
# 回退处理器
# ============================================================

class FallbackHandler:
    """
    回退处理器

    当主操作失败时，执行回退逻辑。
    """

    def __init__(self):
        self._fallbacks: Dict[str, Callable] = {}  # operation -> fallback_func
        self._lock = threading.RLock()
        self._stats = {"invoked": 0, "successes": 0, "failures": 0}

    def register(self, operation: str, fallback: Callable) -> None:
        """注册回退函数"""
        with self._lock:
            self._fallbacks[operation] = fallback

    def unregister(self, operation: str) -> None:
        """注销回退函数"""
        with self._lock:
            self._fallbacks.pop(operation, None)

    def has_fallback(self, operation: str) -> bool:
        """检查是否有回退"""
        with self._lock:
            return operation in self._fallbacks

    def execute(self, operation: str, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """执行回退"""
        with self._lock:
            fallback = self._fallbacks.get(operation)
            self._stats["invoked"] += 1

        if fallback is None:
            with self._lock:
                self._stats["failures"] += 1
            return None, RuntimeError(f"No fallback for '{operation}'")

        try:
            result = fallback(*args, **kwargs)
            with self._lock:
                self._stats["successes"] += 1
            return result, None
        except Exception as e:
            with self._lock:
                self._stats["failures"] += 1
            return None, e

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._stats)


# ============================================================
# 熔断策略（阶段190: 委托给统一的circuit_breaker.CircuitBreaker）
# ============================================================

try:
    from .circuit_breaker import (
        CircuitBreaker as _UnifiedCircuitBreaker,
        CircuitState as _UnifiedCircuitState,
    )
except ImportError:
    from circuit_breaker import (
        CircuitBreaker as _UnifiedCircuitBreaker,
        CircuitState as _UnifiedCircuitState,
    )


# 阶段190: CircuitState直接别名到统一实现（Python枚举有成员时不能被继承）
CircuitState = _UnifiedCircuitState


class CircuitBreakerPolicy:
    """
    熔断策略（阶段190: 委托给统一的circuit_breaker.CircuitBreaker）

    第一性原理：
    - 熔断逻辑应只有一份实现（DRY原则）
    - 本类保留原有接口，内部委托给统一熔断器
    - 当错误率超过阈值时熔断，经过冷却时间后半开试探
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        # 委托给统一熔断器（recovery_timeout是秒，需转为毫秒）
        self._breaker = _UnifiedCircuitBreaker(
            name="recovery_policy",
            failure_threshold=failure_threshold,
            recovery_timeout_ms=recovery_timeout * 1000.0,
            half_open_max_calls=half_open_max_calls,
            success_threshold=half_open_max_calls,
        )
        # 保留原有属性名以兼容外部访问
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max_calls = half_open_max_calls
        self._lock = threading.RLock()

    @property
    def state(self) -> CircuitState:
        """当前状态（检查超时转换）"""
        self._check_state_transition()
        return CircuitState(self._breaker.state)

    @property
    def _failure_threshold(self) -> int:
        """失败阈值（向后兼容属性，可设置）"""
        return self._breaker.failure_threshold

    @_failure_threshold.setter
    def _failure_threshold(self, value: int) -> None:
        self._breaker.failure_threshold = value
        self.__dict__["_failure_threshold_value"] = value

    def _check_state_transition(self) -> None:
        """检查状态转换（向后兼容方法）"""
        # 委托给统一熔断器的内部逻辑
        # 通过can_execute触发检查，如果返回True说明已转为HALF_OPEN
        if self._breaker.state == _UnifiedCircuitState.OPEN:
            # 检查是否超时，如果超时则手动转换
            import time as _time
            if self._breaker._last_failure_time is not None:
                elapsed_ms = (_time.time() - self._breaker._last_failure_time) * 1000
                if elapsed_ms >= self._breaker.recovery_timeout_ms:
                    with self._breaker._lock:
                        if self._breaker._state == _UnifiedCircuitState.OPEN:
                            self._breaker._state = _UnifiedCircuitState.HALF_OPEN
                            self._breaker._half_open_calls = 0
                            self._breaker._success_count = 0
                            self._breaker._state_changes += 1

    def can_execute(self) -> bool:
        """是否可以执行（委托给统一熔断器）"""
        return self._breaker.can_execute()

    def record_success(self) -> None:
        """记录成功（委托给统一熔断器）"""
        self._breaker.record_success()

    def record_failure(self) -> None:
        """记录失败（委托给统一熔断器）"""
        self._breaker.record_failure()

    def reset(self) -> None:
        """重置熔断器（委托给统一熔断器）"""
        self._breaker.reset()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计（委托给统一熔断器，补充原有字段）"""
        stats = self._breaker.get_stats()
        return {
            "state": stats.get("state", "closed"),
            "failure_count": stats.get("failure_count", 0),
            "success_count": stats.get("success_count", 0),
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
        }


# ============================================================
# 健康监控
# ============================================================

@dataclass
class HealthCheckResult:
    """健康检查结果"""
    component: str
    healthy: bool
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component": self.component,
            "healthy": self.healthy,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
        }


class HealthMonitor:
    """
    健康监控器

    定期检查组件健康状态，触发恢复策略。
    """

    def __init__(self, check_interval: float = 60.0):
        self._check_interval = check_interval
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._on_unhealthy: Optional[Callable[[HealthCheckResult], None]] = None

    def register_check(self, component: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """注册健康检查"""
        with self._lock:
            self._checks[component] = check_func

    def unregister_check(self, component: str) -> None:
        """注销健康检查"""
        with self._lock:
            self._checks.pop(component, None)
            self._results.pop(component, None)

    def set_unhealthy_callback(self, callback: Callable[[HealthCheckResult], None]) -> None:
        """设置不健康回调"""
        self._on_unhealthy = callback

    def check_all(self) -> Dict[str, HealthCheckResult]:
        """执行所有健康检查"""
        with self._lock:
            checks = dict(self._checks)

        for component, check_func in checks.items():
            try:
                result = check_func()
                with self._lock:
                    self._results[component] = result
                if not result.healthy and self._on_unhealthy:
                    try:
                        self._on_unhealthy(result)
                    except Exception:
                        pass
            except Exception as e:
                result = HealthCheckResult(
                    component=component,
                    healthy=False,
                    message=f"Health check error: {e}",
                    timestamp=time.time(),
                )
                with self._lock:
                    self._results[component] = result

        with self._lock:
            return dict(self._results)

    def check_one(self, component: str) -> Optional[HealthCheckResult]:
        """执行单个健康检查"""
        with self._lock:
            check_func = self._checks.get(component)

        if check_func is None:
            return None

        try:
            result = check_func()
        except Exception as e:
            result = HealthCheckResult(
                component=component,
                healthy=False,
                message=f"Health check error: {e}",
                timestamp=time.time(),
            )

        with self._lock:
            self._results[component] = result
        return result

    def get_health(self, component: Optional[str] = None) -> Dict[str, Any]:
        """获取健康状态"""
        with self._lock:
            if component:
                result = self._results.get(component)
                return result.to_dict() if result else {"component": component, "healthy": None}
            return {k: v.to_dict() for k, v in self._results.items()}

    def is_healthy(self, component: Optional[str] = None) -> bool:
        """是否健康"""
        with self._lock:
            if component:
                result = self._results.get(component)
                return result.healthy if result else False
            return all(r.healthy for r in self._results.values()) if self._results else True

    def start(self) -> None:
        """启动定期检查"""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="health-monitor")
        self._thread.start()

    def stop(self) -> None:
        """停止定期检查"""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        """监控循环"""
        while self._running and not self._stop_event.is_set():
            try:
                self.check_all()
            except Exception:
                pass
            self._stop_event.wait(self._check_interval)

    def get_registered_components(self) -> List[str]:
        """获取已注册组件列表"""
        with self._lock:
            return list(self._checks.keys())


# ============================================================
# 恢复策略
# ============================================================

class RecoveryStrategy:
    """
    恢复策略

    组合重试、回退、熔断等策略，为特定操作定义恢复行为。
    """

    def __init__(
        self,
        name: str,
        retry_policy: Optional[RetryPolicy] = None,
        fallback_handler: Optional[FallbackHandler] = None,
        circuit_breaker: Optional[CircuitBreakerPolicy] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        self._name = name
        self._retry = retry_policy
        self._fallback = fallback_handler
        self._circuit = circuit_breaker
        self._severity = severity
        self._lock = threading.RLock()
        self._error_count = 0
        self._recovery_count = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def retry_policy(self) -> Optional[RetryPolicy]:
        return self._retry

    @property
    def fallback_handler(self) -> Optional[FallbackHandler]:
        return self._fallback

    @property
    def circuit_breaker(self) -> Optional[CircuitBreakerPolicy]:
        return self._circuit

    def execute(self, func: Callable, *args, **kwargs) -> Tuple[Any, Optional[ErrorContext]]:
        """
        执行函数，带恢复策略

        Returns:
            (result, error_context) — 成功时error_context为None
        """
        operation = self._name

        # 检查熔断器
        if self._circuit and not self._circuit.can_execute():
            error_ctx = ErrorContext(
                error_id=f"err_{int(time.time())}",
                error_type="CircuitBreakerOpen",
                error_message=f"Circuit breaker open for '{operation}'",
                traceback_str="",
                component=operation,
                operation=operation,
                timestamp=time.time(),
                severity=self._severity,
                recovery_action=RecoveryAction.CIRCUIT_BREAK,
                recovery_success=False,
            )
            with self._lock:
                self._error_count += 1
            return None, error_ctx

        # 带重试执行
        if self._retry:
            result, error = self._retry.execute(func, *args, **kwargs)
        else:
            try:
                result = func(*args, **kwargs)
                error = None
            except Exception as e:
                result = None
                error = e

        if error is None:
            # 成功
            if self._circuit:
                self._circuit.record_success()
            return result, None

        # 失败 — 记录熔断器
        if self._circuit:
            self._circuit.record_failure()

        with self._lock:
            self._error_count += 1

        # 尝试回退
        if self._fallback and self._fallback.has_fallback(operation):
            fallback_result, fallback_error = self._fallback.execute(operation, *args, **kwargs)
            if fallback_error is None:
                with self._lock:
                    self._recovery_count += 1
                error_ctx = ErrorContext(
                    error_id=f"err_{int(time.time())}",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback_str=traceback.format_exc(),
                    component=operation,
                    operation=operation,
                    timestamp=time.time(),
                    severity=self._severity,
                    recovery_action=RecoveryAction.FALLBACK,
                    recovery_success=True,
                )
                return fallback_result, error_ctx

        error_ctx = ErrorContext(
            error_id=f"err_{int(time.time())}",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            component=operation,
            operation=operation,
            timestamp=time.time(),
            severity=self._severity,
            recovery_action=RecoveryAction.RETRY if self._retry else RecoveryAction.ESCALATE,
            recovery_success=False,
        )
        return None, error_ctx

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            stats = {
                "name": self._name,
                "error_count": self._error_count,
                "recovery_count": self._recovery_count,
                "severity": self._severity.value,
            }
        if self._retry:
            stats["retry"] = self._retry.get_stats()
        if self._fallback:
            stats["fallback"] = self._fallback.get_stats()
        if self._circuit:
            stats["circuit"] = self._circuit.get_stats()
        return stats


# ============================================================
# 恢复管理器
# ============================================================

class RecoveryManager:
    """
    恢复管理器

    协调所有恢复策略，提供统一的错误处理接口。
    """

    def __init__(self):
        self._strategies: Dict[str, RecoveryStrategy] = {}
        self._error_history: deque = deque(maxlen=1000)
        self._health_monitor = HealthMonitor()
        self._lock = threading.RLock()
        self._counter = 0
        self._global_retry: Optional[RetryPolicy] = None

    @property
    def health_monitor(self) -> HealthMonitor:
        return self._health_monitor

    def register_strategy(self, strategy: RecoveryStrategy) -> None:
        """注册恢复策略"""
        with self._lock:
            self._strategies[strategy.name] = strategy

    def get_strategy(self, name: str) -> Optional[RecoveryStrategy]:
        """获取恢复策略"""
        with self._lock:
            return self._strategies.get(name)

    def remove_strategy(self, name: str) -> None:
        """移除恢复策略"""
        with self._lock:
            self._strategies.pop(name, None)

    def set_global_retry(self, policy: RetryPolicy) -> None:
        """设置全局重试策略"""
        self._global_retry = policy

    def execute(
        self,
        operation: str,
        func: Callable,
        *args,
        **kwargs,
    ) -> Tuple[Any, Optional[ErrorContext]]:
        """
        执行操作，带恢复策略

        Args:
            operation: 操作名称（用于查找策略）
            func: 要执行的函数
            *args, **kwargs: 函数参数

        Returns:
            (result, error_context)
        """
        with self._lock:
            strategy = self._strategies.get(operation)

        if strategy:
            result, error_ctx = strategy.execute(func, *args, **kwargs)
        elif self._global_retry:
            result, error = self._global_retry.execute(func, *args, **kwargs)
            if error is None:
                return result, None
            error_ctx = self._create_error_context(operation, error)
            result = None
        else:
            try:
                result = func(*args, **kwargs)
                return result, None
            except Exception as e:
                error_ctx = self._create_error_context(operation, e)
                result = None

        # 记录错误历史
        if error_ctx:
            with self._lock:
                self._error_history.append(error_ctx)

        return result, error_ctx

    def _create_error_context(self, operation: str, error: Exception) -> ErrorContext:
        """创建错误上下文"""
        with self._lock:
            self._counter += 1
            error_id = f"err_{int(time.time())}_{self._counter}"

        return ErrorContext(
            error_id=error_id,
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            component=operation,
            operation=operation,
            timestamp=time.time(),
        )

    def get_error_history(
        self,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取错误历史"""
        with self._lock:
            errors = list(self._error_history)
        if component:
            errors = [e for e in errors if e.component == component]
        return [e.to_dict() for e in errors[-limit:]]

    def get_error_count(self, component: Optional[str] = None) -> int:
        """获取错误数"""
        with self._lock:
            errors = list(self._error_history)
        if component:
            errors = [e for e in errors if e.component == component]
        return len(errors)

    def clear_errors(self) -> None:
        """清空错误历史"""
        with self._lock:
            self._error_history.clear()

    def register_health_check(
        self,
        component: str,
        check_func: Callable[[], HealthCheckResult],
    ) -> None:
        """注册健康检查"""
        self._health_monitor.register_check(component, check_func)

    def check_health(self, component: Optional[str] = None) -> Dict[str, Any]:
        """检查健康状态"""
        if component:
            result = self._health_monitor.check_one(component)
            return result.to_dict() if result else {}
        return self._health_monitor.check_all()

    def get_system_health(self) -> Dict[str, Any]:
        """获取系统整体健康状态"""
        health = self._health_monitor.get_health()
        all_healthy = self._health_monitor.is_healthy()
        with self._lock:
            error_count = len(self._error_history)
            strategy_count = len(self._strategies)

        return {
            "healthy": all_healthy,
            "error_count": error_count,
            "strategy_count": strategy_count,
            "components": health,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            strategy_stats = {name: s.get_stats() for name, s in self._strategies.items()}
            return {
                "total_errors": len(self._error_history),
                "strategy_count": len(self._strategies),
                "strategies": strategy_stats,
                "health_components": self._health_monitor.get_registered_components(),
                "system_healthy": self._health_monitor.is_healthy(),
            }

    def start_health_monitoring(self) -> None:
        """启动健康监控"""
        self._health_monitor.start()

    def stop_health_monitoring(self) -> None:
        """停止健康监控"""
        self._health_monitor.stop()


# ============================================================
# 便捷函数
# ============================================================

def create_recovery_manager() -> RecoveryManager:
    """创建恢复管理器"""
    return RecoveryManager()


def create_retry_policy(
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> RetryPolicy:
    """创建重试策略"""
    return RetryPolicy(max_retries=max_retries, base_delay=base_delay)


def create_circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
) -> CircuitBreakerPolicy:
    """创建熔断器"""
    return CircuitBreakerPolicy(
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )


def create_recovery_strategy(
    name: str,
    max_retries: int = 3,
    use_circuit_breaker: bool = True,
    use_fallback: bool = False,
) -> RecoveryStrategy:
    """创建恢复策略"""
    retry = create_retry_policy(max_retries=max_retries)
    circuit = create_circuit_breaker() if use_circuit_breaker else None
    fallback = FallbackHandler() if use_fallback else None
    return RecoveryStrategy(name=name, retry_policy=retry, fallback_handler=fallback, circuit_breaker=circuit)
