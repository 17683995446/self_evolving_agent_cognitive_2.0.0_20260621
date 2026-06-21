"""
阶段132: 可观测性增强层 — 日志、指标、追踪、健康检查、告警
阶段190: 统一日志 — StructuredLogger委托给logging.py统一实现

第一性原理:
1. 可观测性 = 日志 + 指标 + 追踪（Observability三支柱）
2. 日志：离散事件记录 → 结构化日志（可搜索、可关联）
3. 指标：聚合数值监控 → 计数器/仪表/直方图（时间序列）
4. 追踪：请求链路追踪 → 分布式追踪（跨服务调用链）
5. 健康检查：系统状态探针 → 存活/就绪探针（Liveness/Readiness）
6. 告警：异常自动通知 → 阈值告警（主动发现问题）

设计原则:
- 零依赖：仅使用Python标准库
- 结构化：所有可观测数据均为结构化格式
- 可组合：各组件可独立使用，也可集成
- 低开销：可观测性不应成为性能瓶颈
- 阶段190：日志统一委托给logging.py，消除重复实现

提供:
- LogLevel/LogRecord/StructuredLogger: 结构化日志（委托给logging.py）
- Metric/MetricsCollector: 指标收集
- TraceSpan/Tracer: 分布式追踪
- HealthStatus/HealthChecker: 健康检查
- Alert/AlertManager: 告警管理
- ObservabilityManager: 可观测性管理器（集成）
"""

import time
import threading
import json
import traceback
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from collections import defaultdict, deque


# ============================================================
# 结构化日志（阶段190: 委托给logging.py统一实现）
# ============================================================

try:
    from .logging import (
        LogLevel as _UnifiedLogLevel,
        StructuredLogger as _UnifiedLogger,
        LogManager as _UnifiedLogManager,
    )
except ImportError:
    from logging import (
        LogLevel as _UnifiedLogLevel,
        StructuredLogger as _UnifiedLogger,
        LogManager as _UnifiedLogManager,
    )


# 阶段190: LogLevel直接别名到统一实现（Python枚举有成员时不能被继承）
LogLevel = _UnifiedLogLevel


@dataclass
class LogRecord:
    """日志记录（保留原有接口，用于get_records查询）"""
    timestamp: float
    level: LogLevel
    logger_name: str
    message: str
    fields: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "timestamp": self.timestamp,
            "level": self.level.name,
            "logger": self.logger_name,
            "message": self.message,
            "fields": self.fields,
            "exception": self.exception,
        }

    def to_json(self) -> str:
        """转为JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class StructuredLogger:
    """
    结构化日志器（阶段190: 委托给logging.py统一实现）

    第一性原理：
    - 日志逻辑应只有一份实现（DRY原则）
    - 本类保留原有接口（handlers, get_records, get_stats, clear）
    - 实际日志记录委托给logging.py的StructuredLogger
    - 本地保留内存缓冲以支持get_records查询

    支持结构化字段、日志级别过滤、多处理器。
    线程安全。
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        handlers: Optional[List[Callable[[LogRecord], None]]] = None,
    ):
        self._name = name
        self._level = level
        self._handlers: List[Callable[[LogRecord], None]] = handlers or []
        self._lock = threading.RLock()
        self._records: deque = deque(maxlen=10000)  # 内存日志缓冲
        self._counters: Dict[str, int] = defaultdict(int)

        # 阶段190: 委托给统一日志器（LogManager是单例，需实例化）
        self._unified_logger = _UnifiedLogManager().get_logger(name)

    @property
    def name(self) -> str:
        return self._name

    @property
    def level(self) -> LogLevel:
        return self._level

    def set_level(self, level: LogLevel) -> None:
        """设置日志级别"""
        with self._lock:
            self._level = level
            self._unified_logger.set_min_level(level)

    def add_handler(self, handler: Callable[[LogRecord], None]) -> None:
        """添加日志处理器"""
        with self._lock:
            self._handlers.append(handler)

    def debug(self, message: str, **fields) -> None:
        """DEBUG日志"""
        self._log(LogLevel.DEBUG, message, fields)

    def info(self, message: str, **fields) -> None:
        """INFO日志"""
        self._log(LogLevel.INFO, message, fields)

    def warning(self, message: str, **fields) -> None:
        """WARNING日志"""
        self._log(LogLevel.WARNING, message, fields)

    def error(self, message: str, exception: Optional[Exception] = None, **fields) -> None:
        """ERROR日志"""
        exc_str = None
        if exception:
            exc_str = "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        self._log(LogLevel.ERROR, message, fields, exc_str)

    def critical(self, message: str, exception: Optional[Exception] = None, **fields) -> None:
        """CRITICAL日志"""
        exc_str = None
        if exception:
            exc_str = "".join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        self._log(LogLevel.CRITICAL, message, fields, exc_str)

    def _log(
        self,
        level: LogLevel,
        message: str,
        fields: Dict[str, Any],
        exception: Optional[str] = None,
    ) -> None:
        """记录日志（阶段190: 同时委托给统一日志器和本地缓冲）"""
        if level.value < self._level.value:
            return

        record = LogRecord(
            timestamp=time.time(),
            level=level,
            logger_name=self._name,
            message=message,
            fields=fields,
            exception=exception,
        )

        # 本地缓冲（支持get_records查询）
        with self._lock:
            self._records.append(record)
            self._counters[level.name] += 1
            handlers = list(self._handlers)

        # 调用本地handlers
        for handler in handlers:
            try:
                handler(record)
            except Exception:
                pass  # 日志处理器异常不应影响主流程

        # 阶段190: 委托给统一日志器
        try:
            if level == LogLevel.DEBUG:
                self._unified_logger.debug(message, **fields)
            elif level == LogLevel.INFO:
                self._unified_logger.info(message, **fields)
            elif level == LogLevel.WARNING:
                self._unified_logger.warning(message, **fields)
            elif level == LogLevel.ERROR:
                self._unified_logger.error(message, exception=None, **fields)
            elif level == LogLevel.CRITICAL:
                self._unified_logger.critical(message, **fields)
        except Exception:
            pass  # 统一日志器异常不应影响本地日志

    def get_records(
        self,
        level: Optional[LogLevel] = None,
        limit: int = 100,
    ) -> List[LogRecord]:
        """获取日志记录"""
        with self._lock:
            records = list(self._records)
        if level:
            records = [r for r in records if r.level.value >= level.value]
        return records[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """获取日志统计"""
        with self._lock:
            return {
                "name": self._name,
                "level": self._level.name,
                "total_records": len(self._records),
                "counters": dict(self._counters),
            }

    def clear(self) -> None:
        """清空日志"""
        with self._lock:
            self._records.clear()
            self._counters.clear()


class LoggerRegistry:
    """日志器注册表"""

    _loggers: Dict[str, StructuredLogger] = {}
    _lock = threading.RLock()
    _default_level = LogLevel.INFO

    @classmethod
    def get_logger(cls, name: str, level: Optional[LogLevel] = None) -> StructuredLogger:
        """获取或创建日志器"""
        with cls._lock:
            if name not in cls._loggers:
                cls._loggers[name] = StructuredLogger(
                    name=name,
                    level=level or cls._default_level,
                )
            return cls._loggers[name]

    @classmethod
    def set_default_level(cls, level: LogLevel) -> None:
        """设置默认日志级别"""
        with cls._lock:
            cls._default_level = level

    @classmethod
    def get_all_loggers(cls) -> Dict[str, StructuredLogger]:
        """获取所有日志器"""
        with cls._lock:
            return dict(cls._loggers)


def get_logger(name: str) -> StructuredLogger:
    """获取日志器（便捷函数）"""
    return LoggerRegistry.get_logger(name)


# ============================================================
# 指标收集
# ============================================================

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"      # 单调递增
    GAUGE = "gauge"          # 可增可减
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"      # 摘要


@dataclass
class Metric:
    """指标"""
    name: str
    type: MetricType
    value: Union[int, float] = 0
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    # 直方图专用
    buckets: List[float] = field(default_factory=list)
    count: int = 0
    sum_value: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
            "count": self.count,
            "sum": self.sum_value,
        }


class MetricsCollector:
    """
    指标收集器

    支持计数器、仪表、直方图。
    线程安全。
    """

    def __init__(self, namespace: str = ""):
        self._namespace = namespace
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.RLock()
        self._histogram_data: Dict[str, List[float]] = defaultdict(list)

    def _full_name(self, name: str) -> str:
        """完整指标名"""
        if self._namespace:
            return f"{self._namespace}.{name}"
        return name

    def counter(self, name: str, value: int = 1, **labels) -> None:
        """计数器（单调递增）"""
        full_name = self._full_name(name)
        key = f"{full_name}:{tuple(sorted(labels.items()))}"
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Metric(
                    name=full_name, type=MetricType.COUNTER,
                    value=0, labels=labels,
                )
            self._metrics[key].value += value
            self._metrics[key].timestamp = time.time()

    def gauge(self, name: str, value: float, **labels) -> None:
        """仪表（可增可减）"""
        full_name = self._full_name(name)
        key = f"{full_name}:{tuple(sorted(labels.items()))}"
        with self._lock:
            self._metrics[key] = Metric(
                name=full_name, type=MetricType.GAUGE,
                value=value, labels=labels,
            )

    def histogram(self, name: str, value: float, **labels) -> None:
        """直方图"""
        full_name = self._full_name(name)
        key = f"{full_name}:{tuple(sorted(labels.items()))}"
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = Metric(
                    name=full_name, type=MetricType.HISTOGRAM,
                    value=0, labels=labels,
                )
            metric = self._metrics[key]
            metric.count += 1
            metric.sum_value += value
            metric.timestamp = time.time()
            self._histogram_data[key].append(value)
            # 限制直方图数据量
            if len(self._histogram_data[key]) > 10000:
                self._histogram_data[key] = self._histogram_data[key][-5000:]

    def get_metric(self, name: str, **labels) -> Optional[Metric]:
        """获取指标"""
        full_name = self._full_name(name)
        key = f"{full_name}:{tuple(sorted(labels.items()))}"
        with self._lock:
            return self._metrics.get(key)

    def get_all_metrics(self) -> List[Metric]:
        """获取所有指标"""
        with self._lock:
            return list(self._metrics.values())

    def get_histogram_stats(self, name: str, **labels) -> Dict[str, Any]:
        """获取直方图统计"""
        full_name = self._full_name(name)
        key = f"{full_name}:{tuple(sorted(labels.items()))}"
        with self._lock:
            data = list(self._histogram_data.get(key, []))
            if not data:
                return {}
            sorted_data = sorted(data)
            n = len(sorted_data)
            return {
                "count": n,
                "sum": sum(data),
                "avg": sum(data) / n,
                "min": sorted_data[0],
                "max": sorted_data[-1],
                "p50": sorted_data[n // 2],
                "p90": sorted_data[int(n * 0.9)] if n > 1 else sorted_data[0],
                "p99": sorted_data[int(n * 0.99)] if n > 1 else sorted_data[0],
            }

    def export(self) -> Dict[str, Any]:
        """导出所有指标"""
        with self._lock:
            return {
                "namespace": self._namespace,
                "metrics": [m.to_dict() for m in self._metrics.values()],
                "timestamp": time.time(),
            }

    def reset(self) -> None:
        """重置所有指标"""
        with self._lock:
            self._metrics.clear()
            self._histogram_data.clear()


# ============================================================
# 分布式追踪
# ============================================================

@dataclass
class TraceSpan:
    """追踪跨度"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    name: str
    start_time: float
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"  # active, finished, error

    @property
    def duration(self) -> Optional[float]:
        """持续时间"""
        if self.end_time:
            return self.end_time - self.start_time
        return None

    def finish(self) -> None:
        """结束跨度"""
        self.end_time = time.time()
        self.status = "finished"

    def set_error(self) -> None:
        """标记错误"""
        self.status = "error"
        self.end_time = time.time()

    def set_tag(self, key: str, value: Any) -> None:
        """设置标签"""
        self.tags[key] = value

    def log(self, message: str, **fields) -> None:
        """记录日志"""
        self.logs.append({
            "timestamp": time.time(),
            "message": message,
            "fields": fields,
        })

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status,
        }


class Tracer:
    """
    分布式追踪器

    记录请求链路追踪。
    线程安全。
    """

    def __init__(self, service_name: str = "default"):
        self._service_name = service_name
        self._spans: Dict[str, TraceSpan] = {}
        self._traces: Dict[str, List[str]] = defaultdict(list)  # trace_id -> [span_ids]
        self._lock = threading.RLock()
        self._span_counter = 0
        self._trace_counter = 0

    def start_span(
        self,
        name: str,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
    ) -> TraceSpan:
        """开始跨度"""
        with self._lock:
            self._span_counter += 1
            span_id = f"span_{self._span_counter}"
            if trace_id is None:
                self._trace_counter += 1
                trace_id = f"trace_{self._trace_counter}"
            span = TraceSpan(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                name=name,
                start_time=time.time(),
            )
            self._spans[span_id] = span
            self._traces[trace_id].append(span_id)
            return span

    def finish_span(self, span: TraceSpan) -> None:
        """结束跨度"""
        with self._lock:
            span.finish()

    def get_span(self, span_id: str) -> Optional[TraceSpan]:
        """获取跨度"""
        with self._lock:
            return self._spans.get(span_id)

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """获取追踪的所有跨度"""
        with self._lock:
            span_ids = self._traces.get(trace_id, [])
            return [self._spans[sid] for sid in span_ids if sid in self._spans]

    def get_all_traces(self) -> Dict[str, List[Dict[str, Any]]]:
        """获取所有追踪"""
        with self._lock:
            return {
                tid: [self._spans[sid].to_dict() for sid in sids if sid in self._spans]
                for tid, sids in self._traces.items()
            }

    def trace(self, name: str):
        """追踪装饰器"""
        def decorator(func):
            @wraps_decorator(func)
            def wrapper(*args, **kwargs):
                span = self.start_span(name)
                try:
                    result = func(*args, **kwargs)
                    span.finish()
                    return result
                except Exception as e:
                    span.set_error()
                    span.set_tag("error", str(e))
                    raise
            return wrapper
        return decorator

    def stats(self) -> Dict[str, Any]:
        """获取追踪统计"""
        with self._lock:
            return {
                "service_name": self._service_name,
                "total_spans": len(self._spans),
                "total_traces": len(self._traces),
            }

    def clear(self) -> None:
        """清空追踪数据"""
        with self._lock:
            self._spans.clear()
            self._traces.clear()


def wraps_decorator(func):
    """简化的functools.wraps"""
    def wrapper(fn):
        fn.__name__ = func.__name__
        fn.__doc__ = func.__doc__
        return fn
    return wrapper


# ============================================================
# 健康检查
# ============================================================

class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0


class HealthChecker:
    """
    健康检查器

    注册健康检查项，执行检查，汇总结果。
    """

    def __init__(self):
        self._checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self._lock = threading.RLock()
        self._last_results: Dict[str, HealthCheckResult] = {}

    def register_check(self, name: str, check_fn: Callable[[], HealthCheckResult]) -> None:
        """注册健康检查"""
        with self._lock:
            self._checks[name] = check_fn

    def unregister_check(self, name: str) -> None:
        """注销健康检查"""
        with self._lock:
            self._checks.pop(name, None)
            self._last_results.pop(name, None)

    def run_check(self, name: str) -> HealthCheckResult:
        """运行单个检查"""
        with self._lock:
            check_fn = self._checks.get(name)
        if check_fn is None:
            return HealthCheckResult(
                name=name, status=HealthStatus.UNKNOWN,
                message="Check not found",
            )
        start = time.perf_counter()
        try:
            result = check_fn()
            result.duration_ms = (time.perf_counter() - start) * 1000
            with self._lock:
                self._last_results[name] = result
            return result
        except Exception as e:
            result = HealthCheckResult(
                name=name, status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )
            with self._lock:
                self._last_results[name] = result
            return result

    def run_all(self) -> Dict[str, HealthCheckResult]:
        """运行所有检查"""
        with self._lock:
            names = list(self._checks.keys())
        return {name: self.run_check(name) for name in names}

    def get_overall_status(self) -> HealthStatus:
        """获取总体健康状态"""
        results = self.run_all()
        if not results:
            return HealthStatus.UNKNOWN
        statuses = [r.status for r in results.values()]
        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def get_last_results(self) -> Dict[str, HealthCheckResult]:
        """获取上次检查结果"""
        with self._lock:
            return dict(self._last_results)

    def list_checks(self) -> List[str]:
        """列出所有检查项"""
        with self._lock:
            return list(self._checks.keys())


# ============================================================
# 告警管理
# ============================================================

class AlertSeverity(Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """告警"""
    alert_id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "details": self.details,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at,
        }


class AlertRule:
    """告警规则"""

    def __init__(
        self,
        name: str,
        condition: Callable[[], bool],
        severity: AlertSeverity = AlertSeverity.WARNING,
        message: str = "",
        cooldown: float = 60.0,
    ):
        self.name = name
        self._condition = condition
        self.severity = severity
        self.message = message
        self.cooldown = cooldown
        self._last_triggered: Optional[float] = None

    def should_trigger(self) -> bool:
        """是否应触发"""
        now = time.time()
        if self._last_triggered and now - self._last_triggered < self.cooldown:
            return False  # 冷却中
        try:
            return self._condition()
        except Exception:
            return False

    def mark_triggered(self) -> None:
        """标记已触发"""
        self._last_triggered = time.time()


class AlertManager:
    """
    告警管理器

    管理告警规则，触发告警，记录历史。
    线程安全。
    """

    def __init__(self):
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: deque = deque(maxlen=10000)
        self._active_alerts: Dict[str, Alert] = {}
        self._lock = threading.RLock()
        self._alert_counter = 0
        self._handlers: List[Callable[[Alert], None]] = []

    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        with self._lock:
            self._rules[rule.name] = rule

    def remove_rule(self, name: str) -> None:
        """移除告警规则"""
        with self._lock:
            self._rules.pop(name, None)

    def add_handler(self, handler: Callable[[Alert], None]) -> None:
        """添加告警处理器"""
        with self._lock:
            self._handlers.append(handler)

    def check_rules(self) -> List[Alert]:
        """检查所有规则，返回新触发的告警"""
        new_alerts = []
        with self._lock:
            rules = list(self._rules.values())

        for rule in rules:
            if rule.should_trigger():
                alert = self._create_alert(rule)
                new_alerts.append(alert)
                rule.mark_triggered()

        return new_alerts

    def _create_alert(self, rule: AlertRule) -> Alert:
        """创建告警"""
        with self._lock:
            self._alert_counter += 1
            alert = Alert(
                alert_id=f"alert_{self._alert_counter}",
                name=rule.name,
                severity=rule.severity,
                message=rule.message or f"Alert: {rule.name}",
            )
            self._alerts.append(alert)
            self._active_alerts[alert.alert_id] = alert
            handlers = list(self._handlers)

        for handler in handlers:
            try:
                handler(alert)
            except Exception:
                pass

        return alert

    def trigger_alert(
        self,
        name: str,
        severity: AlertSeverity,
        message: str,
        **details,
    ) -> Alert:
        """手动触发告警"""
        with self._lock:
            self._alert_counter += 1
            alert = Alert(
                alert_id=f"alert_{self._alert_counter}",
                name=name,
                severity=severity,
                message=message,
                details=details,
            )
            self._alerts.append(alert)
            self._active_alerts[alert.alert_id] = alert
            handlers = list(self._handlers)

        for handler in handlers:
            try:
                handler(alert)
            except Exception:
                pass

        return alert

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self._lock:
            if alert_id not in self._active_alerts:
                return False
            alert = self._active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolved_at = time.time()
            return True

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        with self._lock:
            return list(self._alerts)[-limit:]

    def stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            return {
                "total_alerts": len(self._alerts),
                "active_alerts": len(self._active_alerts),
                "rules": len(self._rules),
            }

    def clear(self) -> None:
        """清空告警"""
        with self._lock:
            self._alerts.clear()
            self._active_alerts.clear()


# ============================================================
# 可观测性管理器
# ============================================================

class ObservabilityManager:
    """
    可观测性管理器

    集成日志、指标、追踪、健康检查、告警。
    提供统一的可观测性接口。
    """

    def __init__(self, service_name: str = "self_evolving_agent"):
        self._service_name = service_name
        self._logger = StructuredLogger(service_name)
        self._metrics = MetricsCollector(namespace=service_name)
        self._tracer = Tracer(service_name)
        self._health_checker = HealthChecker()
        self._alert_manager = AlertManager()
        self._lock = threading.RLock()

        # 注册默认健康检查
        self._register_default_checks()

    def _register_default_checks(self) -> None:
        """注册默认健康检查"""
        def liveness_check() -> HealthCheckResult:
            return HealthCheckResult(
                name="liveness",
                status=HealthStatus.HEALTHY,
                message="Service is alive",
            )

        def readiness_check() -> HealthCheckResult:
            return HealthCheckResult(
                name="readiness",
                status=HealthStatus.HEALTHY,
                message="Service is ready",
            )

        self._health_checker.register_check("liveness", liveness_check)
        self._health_checker.register_check("readiness", readiness_check)

    @property
    def logger(self) -> StructuredLogger:
        return self._logger

    @property
    def metrics(self) -> MetricsCollector:
        return self._metrics

    @property
    def tracer(self) -> Tracer:
        return self._tracer

    @property
    def health_checker(self) -> HealthChecker:
        return self._health_checker

    @property
    def alert_manager(self) -> AlertManager:
        return self._alert_manager

    def start_span(self, name: str, **tags) -> TraceSpan:
        """开始追踪跨度"""
        span = self._tracer.start_span(name)
        for k, v in tags.items():
            span.set_tag(k, v)
        return span

    def log_event(self, level: LogLevel, message: str, **fields) -> None:
        """记录日志事件"""
        self._logger._log(level, message, fields)

    def record_metric(self, name: str, value: float, **labels) -> None:
        """记录指标"""
        self._metrics.gauge(name, value, **labels)

    def increment_counter(self, name: str, value: int = 1, **labels) -> None:
        """递增计数器"""
        self._metrics.counter(name, value, **labels)

    def observe_histogram(self, name: str, value: float, **labels) -> None:
        """观测直方图"""
        self._metrics.histogram(name, value, **labels)

    def check_health(self) -> Dict[str, Any]:
        """检查健康状态"""
        results = self._health_checker.run_all()
        overall = self._health_checker.get_overall_status()
        return {
            "overall_status": overall.value,
            "checks": {name: r.__dict__ for name, r in results.items()},
        }

    def check_alerts(self) -> List[Alert]:
        """检查告警"""
        return self._alert_manager.check_rules()

    def dashboard(self) -> Dict[str, Any]:
        """获取可观测性仪表板"""
        return {
            "service_name": self._service_name,
            "timestamp": time.time(),
            "logger_stats": self._logger.get_stats(),
            "metrics_export": self._metrics.export(),
            "tracer_stats": self._tracer.stats(),
            "health_status": self._health_checker.get_overall_status().value,
            "health_checks": self._health_checker.list_checks(),
            "alert_stats": self._alert_manager.stats(),
            "active_alerts": [a.to_dict() for a in self._alert_manager.get_active_alerts()],
        }


# ============================================================
# 便捷函数
# ============================================================

def create_observability_manager(service_name: str = "self_evolving_agent") -> ObservabilityManager:
    """创建可观测性管理器"""
    return ObservabilityManager(service_name=service_name)
