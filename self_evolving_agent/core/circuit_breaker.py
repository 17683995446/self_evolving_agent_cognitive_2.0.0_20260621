"""
阶段38: 熔断接口标准化
阶段190: 统一熔断器 — 支持call()和can_execute()/record_*()双模式

第一性原理:
1. 故障会传播。当一个组件持续失败时，继续向它发送请求会浪费资源并可能拖垮整个系统
2. 熔断器在故障持续时快速失败，给故障组件恢复的时间
3. 熔断器必须自动恢复：半开状态下试探，成功后关闭

设计原则:
- 三种状态：关闭（正常）、打开（熔断）、半开（试探）
- 可配置：失败阈值、恢复超时、半开请求数
- 可观测：状态、失败率、切换次数
- 与重试配合：熔断打开时直接失败，不重试
- 阶段190：统一双模式接口，消除recovery.py和production.py的重复实现
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Callable, Optional
from enum import Enum
from datetime import datetime
import threading
import time


class CircuitState(Enum):
    """熔断器状态"""
    CLOSED = "closed"       # 关闭：正常通过
    OPEN = "open"           # 打开：快速失败
    HALF_OPEN = "half_open" # 半开：允许试探请求


class CircuitBreakerAction(Enum):
    """熔断触发动作"""
    FAIL_FAST = "fail_fast"     # 快速失败
    FALLBACK = "fallback"       # 返回降级结果


@dataclass
class CircuitBreakerResult:
    """熔断器执行结果"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    state: CircuitState = CircuitState.CLOSED
    from_fallback: bool = False


# ============================================================
# 熔断器
# ============================================================

class CircuitBreaker:
    """
    熔断器

    第一性原理：熔断是对故障传播的阻断，必须有明确的状态和自动恢复机制
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout_ms: float = 30000.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
        action: CircuitBreakerAction = CircuitBreakerAction.FAIL_FAST,
        fallback_fn: Optional[Callable] = None,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_ms = recovery_timeout_ms
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        self.action = action
        self.fallback_fn = fallback_fn

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._state_changes = 0
        self._total_calls = 0
        self._total_failures = 0
        self._total_successes = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """当前状态"""
        with self._lock:
            return self._state

    def call(self, operation: Callable, *args, **kwargs) -> CircuitBreakerResult:
        """
        执行操作，受熔断器保护

        Args:
            operation: 要执行的操作
            *args, **kwargs: 传递给操作的参数

        Returns:
            执行结果
        """
        with self._lock:
            self._total_calls += 1

            # 检查是否需要从OPEN转到HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._last_failure_time is not None:
                    elapsed_ms = (time.time() - self._last_failure_time) * 1000
                    if elapsed_ms >= self.recovery_timeout_ms:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        self._success_count = 0
                        self._state_changes += 1
                    else:
                        # 仍处于OPEN状态
                        self._total_failures += 1
                        if self.action == CircuitBreakerAction.FALLBACK and self.fallback_fn:
                            return CircuitBreakerResult(
                                success=True,
                                result=self.fallback_fn(*args, **kwargs),
                                state=CircuitState.OPEN,
                                from_fallback=True,
                            )
                        return CircuitBreakerResult(
                            success=False,
                            error=Exception(f"Circuit breaker '{self.name}' is OPEN"),
                            state=CircuitState.OPEN,
                        )

            # HALF_OPEN状态限制请求数
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    self._total_failures += 1
                    return CircuitBreakerResult(
                        success=False,
                        error=Exception(f"Circuit breaker '{self.name}' HALF_OPEN limit reached"),
                        state=CircuitState.HALF_OPEN,
                    )
                self._half_open_calls += 1

        # 执行操作（在锁外执行，避免阻塞）
        try:
            result = operation(*args, **kwargs)
            self._on_success()
            return CircuitBreakerResult(
                success=True,
                result=result,
                state=self.state,
            )
        except Exception as e:
            self._on_failure()
            return CircuitBreakerResult(
                success=False,
                error=e,
                state=self.state,
            )

    def _on_success(self) -> None:
        """处理成功"""
        with self._lock:
            self._total_successes += 1

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    # 恢复成功，关闭熔断器
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    self._half_open_calls = 0
                    self._state_changes += 1
            elif self._state == CircuitState.CLOSED:
                # 连续成功，重置失败计数
                self._failure_count = 0

    def _on_failure(self) -> None:
        """处理失败"""
        with self._lock:
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 半开状态失败，重新打开
                self._state = CircuitState.OPEN
                self._state_changes += 1
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    # 达到阈值，打开熔断器
                    self._state = CircuitState.OPEN
                    self._state_changes += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            failure_rate = (self._total_failures / self._total_calls
                           if self._total_calls > 0 else 0.0)
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "failure_rate": failure_rate,
                "state_changes": self._state_changes,
            }

    def reset(self) -> None:
        """手动重置熔断器"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = None

    # ============================================================
    # 阶段190: 手动模式接口（统一can_execute/record_*模式）
    # ============================================================

    def can_execute(self) -> bool:
        """
        检查是否允许执行（手动模式）

        第一性原理：与call()模式共享同一状态机，但允许调用方
        自行管理操作执行和结果记录。

        Returns:
            True if execution is allowed, False if circuit is open
        """
        with self._lock:
            self._total_calls += 1

            # 检查是否需要从OPEN转到HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._last_failure_time is not None:
                    elapsed_ms = (time.time() - self._last_failure_time) * 1000
                    if elapsed_ms >= self.recovery_timeout_ms:
                        self._state = CircuitState.HALF_OPEN
                        self._half_open_calls = 0
                        self._success_count = 0
                        self._state_changes += 1
                    else:
                        return False
                else:
                    return False

            # HALF_OPEN状态限制请求数
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    return False
                self._half_open_calls += 1

            return True

    def record_success(self) -> None:
        """记录成功（手动模式）"""
        self._on_success()

    def record_failure(self) -> None:
        """记录失败（手动模式）"""
        self._on_failure()


# ============================================================
# 熔断器装饰器
# ============================================================

def circuit_breaker_protected(
    breaker: Optional[CircuitBreaker] = None,
    **breaker_kwargs
):
    """
    熔断器保护装饰器

    自动为函数添加熔断保护
    """
    if breaker is None:
        breaker = CircuitBreaker(**breaker_kwargs)

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            result = breaker.call(func, *args, **kwargs)
            if result.success:
                return result.result
            raise result.error

        wrapper._circuit_breaker = breaker
        wrapper.get_circuit_stats = breaker.get_stats
        return wrapper
    return decorator


# ============================================================
# 便捷函数
# ============================================================

def create_circuit_breaker(**kwargs) -> CircuitBreaker:
    """创建熔断器"""
    return CircuitBreaker(**kwargs)
