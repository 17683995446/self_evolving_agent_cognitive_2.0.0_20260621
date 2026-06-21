"""
阶段133: 生产部署支持层 — 配置、生命周期、健康端点、资源限制

第一性原理:
1. 配置管理：环境差异通过配置隔离（12-Factor App：配置存储于环境）
2. 生命周期管理：优雅启动/关闭（信号处理、资源清理、连接排空）
3. 健康端点：生产探针（Liveness/Readiness分离，Kubernetes标准）
4. 资源限制：防止资源耗尽（限流、配额、熔断）
5. 部署打包：可部署单元（配置模板、启动脚本）
6. 运维接口：运行时管理（管理API、热配置更新）

设计原则:
- 零依赖：仅使用Python标准库
- 12-Factor：配置来自环境变量
- 优雅降级：资源耗尽时降级而非崩溃
- 可观测：所有运维操作可观测

提供:
- Config/ConfigLoader: 多环境配置管理
- LifecycleManager: 生命周期管理（启动/关闭）
- HealthEndpoint: HTTP健康检查端点
- RateLimiter: 限流器（令牌桶）
- CircuitBreaker: 熔断器
- ResourceManager: 资源管理（配额）
- DeploymentPackager: 部署打包器
- ProductionManager: 生产管理器（集成）
"""

import os
import time
import json
import threading
import signal
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
from collections import deque, defaultdict


# ============================================================
# 配置管理
# ============================================================

class Environment(Enum):
    """环境类型"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class Config:
    """
    配置容器

    支持多层级配置：默认值 < 配置文件 < 环境变量 < 运行时覆盖
    """
    values: Dict[str, Any] = field(default_factory=dict)
    env: Environment = Environment.DEVELOPMENT

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点号分隔的路径）"""
        keys = key.split(".")
        value = self.values
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set(self, key: str, value: Any) -> None:
        """设置配置值"""
        keys = key.split(".")
        d = self.values
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    def update(self, values: Dict[str, Any]) -> None:
        """批量更新配置"""
        def merge(target: dict, source: dict) -> None:
            for k, v in source.items():
                if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                    merge(target[k], v)
                else:
                    target[k] = v
        merge(self.values, values)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return {"env": self.env.value, "values": self.values}

    def to_json(self) -> str:
        """转为JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class ConfigLoader:
    """
    配置加载器

    从多个来源加载配置：默认值 → JSON文件 → 环境变量 → 运行时
    """

    @staticmethod
    def load_from_env(prefix: str = "APP_") -> Dict[str, Any]:
        """从环境变量加载配置"""
        config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # 尝试解析JSON
                try:
                    config[config_key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    config[config_key] = value
        return config

    @staticmethod
    def load_from_file(file_path: str) -> Dict[str, Any]:
        """从JSON文件加载配置"""
        if not os.path.exists(file_path):
            return {}
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load(
        env: Environment = Environment.DEVELOPMENT,
        config_file: Optional[str] = None,
        env_prefix: str = "APP_",
        defaults: Optional[Dict[str, Any]] = None,
    ) -> Config:
        """加载完整配置"""
        values = {}
        # 1. 默认值
        if defaults:
            values.update(defaults)
        # 2. 配置文件
        if config_file:
            file_config = ConfigLoader.load_from_file(config_file)
            values.update(file_config)
        # 3. 环境变量
        env_config = ConfigLoader.load_from_env(env_prefix)
        values.update(env_config)
        # 4. 环境特定覆盖
        values["env"] = env.value
        return Config(values=values, env=env)


# ============================================================
# 生命周期管理
# ============================================================

class LifecycleState(Enum):
    """生命周期状态"""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


@dataclass
class LifecycleHook:
    """生命周期钩子"""
    name: str
    handler: Callable[[], None]
    priority: int = 0  # 数字越小优先级越高


class LifecycleManager:
    """
    生命周期管理器

    管理服务的启动和关闭流程。
    支持信号处理、钩子注册、优雅关闭。
    """

    def __init__(self):
        self._state = LifecycleState.CREATED
        self._start_hooks: List[LifecycleHook] = []
        self._stop_hooks: List[LifecycleHook] = []
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._signal_handlers_installed = False

    @property
    def state(self) -> LifecycleState:
        return self._state

    def on_start(self, name: str, handler: Callable[[], None], priority: int = 0) -> None:
        """注册启动钩子"""
        with self._lock:
            self._start_hooks.append(LifecycleHook(name, handler, priority))
            self._start_hooks.sort(key=lambda h: h.priority)

    def on_stop(self, name: str, handler: Callable[[], None], priority: int = 0) -> None:
        """注册停止钩子"""
        with self._lock:
            self._stop_hooks.append(LifecycleHook(name, handler, priority))
            self._stop_hooks.sort(key=lambda h: h.priority)

    def start(self) -> bool:
        """启动服务"""
        with self._lock:
            if self._state != LifecycleState.CREATED:
                return False
            self._state = LifecycleState.STARTING

        # 执行启动钩子
        with self._lock:
            hooks = list(self._start_hooks)

        for hook in hooks:
            try:
                hook.handler()
            except Exception:
                with self._lock:
                    self._state = LifecycleState.FAILED
                raise

        with self._lock:
            self._state = LifecycleState.RUNNING
        return True

    def stop(self) -> bool:
        """停止服务"""
        with self._lock:
            if self._state != LifecycleState.RUNNING:
                return False
            self._state = LifecycleState.STOPPING

        # 执行停止钩子（逆序）
        with self._lock:
            hooks = list(reversed(self._stop_hooks))

        for hook in hooks:
            try:
                hook.handler()
            except Exception:
                pass  # 停止钩子异常不应阻止其他钩子

        with self._lock:
            self._state = LifecycleState.STOPPED
        self._shutdown_event.set()
        return True

    def install_signal_handlers(self) -> None:
        """安装信号处理器（优雅关闭）"""
        if self._signal_handlers_installed:
            return

        def handle_signal(signum, frame):
            self.stop()

        try:
            signal.signal(signal.SIGTERM, handle_signal)
            signal.signal(signal.SIGINT, handle_signal)
            self._signal_handlers_installed = True
        except (ValueError, AttributeError):
            # 非主线程或信号不可用
            pass

    def wait_for_shutdown(self, timeout: Optional[float] = None) -> bool:
        """等待关闭信号"""
        return self._shutdown_event.wait(timeout=timeout)

    def is_running(self) -> bool:
        """是否运行中"""
        return self._state == LifecycleState.RUNNING

    def reset(self) -> None:
        """重置生命周期（用于测试）"""
        with self._lock:
            self._state = LifecycleState.CREATED
            self._shutdown_event.clear()


# ============================================================
# 健康检查端点
# ============================================================

@dataclass
class HealthResponse:
    """健康检查响应"""
    status: str  # "healthy", "degraded", "unhealthy"
    code: int    # HTTP状态码
    checks: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class HealthEndpoint:
    """
    健康检查端点

    提供Kubernetes标准的Liveness和Readiness探针。
    """

    def __init__(self):
        self._liveness_checks: Dict[str, Callable[[], bool]] = {}
        self._readiness_checks: Dict[str, Callable[[], bool]] = {}
        self._lock = threading.RLock()

    def add_liveness_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """添加存活检查（服务是否运行）"""
        with self._lock:
            self._liveness_checks[name] = check_fn

    def add_readiness_check(self, name: str, check_fn: Callable[[], bool]) -> None:
        """添加就绪检查（服务是否可接受流量）"""
        with self._lock:
            self._readiness_checks[name] = check_fn

    def liveness(self) -> HealthResponse:
        """存活探针"""
        with self._lock:
            checks = dict(self._liveness_checks)

        results = {}
        all_healthy = True
        for name, check_fn in checks.items():
            try:
                healthy = check_fn()
                results[name] = "healthy" if healthy else "unhealthy"
                if not healthy:
                    all_healthy = False
            except Exception:
                results[name] = "error"
                all_healthy = False

        status = "healthy" if all_healthy else "unhealthy"
        code = 200 if all_healthy else 503
        return HealthResponse(status=status, code=code, checks=results)

    def readiness(self) -> HealthResponse:
        """就绪探针"""
        with self._lock:
            checks = dict(self._readiness_checks)

        results = {}
        all_ready = True
        for name, check_fn in checks.items():
            try:
                ready = check_fn()
                results[name] = "ready" if ready else "not_ready"
                if not ready:
                    all_ready = False
            except Exception:
                results[name] = "error"
                all_ready = False

        status = "healthy" if all_ready else "unhealthy"
        code = 200 if all_ready else 503
        return HealthResponse(status=status, code=code, checks=results)

    def full_health(self) -> HealthResponse:
        """完整健康检查"""
        liveness = self.liveness()
        readiness = self.readiness()

        all_healthy = liveness.code == 200 and readiness.code == 200
        status = "healthy" if all_healthy else "unhealthy"
        code = 200 if all_healthy else 503

        return HealthResponse(
            status=status,
            code=code,
            checks={
                "liveness": liveness.checks,
                "readiness": readiness.checks,
            },
        )


# ============================================================
# 限流器
# ============================================================

class RateLimiter:
    """
    令牌桶限流器

    控制请求速率，防止过载。
    线程安全。
    """

    def __init__(self, rate: float, capacity: Optional[float] = None):
        """
        rate: 每秒令牌数
        capacity: 桶容量（默认等于rate）
        """
        self._rate = rate
        self._capacity = capacity if capacity is not None else rate
        self._tokens = self._capacity
        self._last_refill = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> bool:
        """尝试获取令牌"""
        with self._lock:
            now = time.time()
            # 补充令牌
            elapsed = now - self._last_refill
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last_refill = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def wait_acquire(self, tokens: float = 1.0, timeout: Optional[float] = None) -> bool:
        """等待获取令牌"""
        start = time.time()
        while True:
            if self.acquire(tokens):
                return True
            if timeout and time.time() - start > timeout:
                return False
            # 计算需要等待的时间
            with self._lock:
                needed = tokens - self._tokens
                wait_time = needed / self._rate if self._rate > 0 else 1.0
            time.sleep(min(wait_time, 0.1))

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def capacity(self) -> float:
        return self._capacity

    def available_tokens(self) -> float:
        """可用令牌数"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_refill
            return min(self._capacity, self._tokens + elapsed * self._rate)


# ============================================================
# 熔断器（阶段190: 委托给统一的circuit_breaker.CircuitBreaker）
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


class CircuitBreaker:
    """
    熔断器（阶段190: 委托给统一的circuit_breaker.CircuitBreaker）

    第一性原理：
    - 熔断逻辑应只有一份实现（DRY原则）
    - 本类保留原有接口，内部委托给统一熔断器
    - 当失败率达到阈值时熔断，保护下游服务
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
    ):
        # 委托给统一熔断器（recovery_timeout是秒，需转为毫秒）
        self._breaker = _UnifiedCircuitBreaker(
            name="production",
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

    def _check_state_transition(self) -> None:
        """检查状态转换（向后兼容方法）"""
        if self._breaker.state == _UnifiedCircuitState.OPEN:
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

    @property
    def _failure_count(self) -> int:
        """失败计数（向后兼容属性）"""
        return self._breaker._failure_count

    @property
    def _success_count(self) -> int:
        """成功计数（向后兼容属性）"""
        return self._breaker._success_count

    @property
    def _half_open_calls(self) -> int:
        """半开调用计数（向后兼容属性）"""
        return self._breaker._half_open_calls

    @property
    def _last_failure_time(self) -> Optional[float]:
        """最后失败时间（向后兼容属性）"""
        return self._breaker._last_failure_time

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

    def stats(self) -> Dict[str, Any]:
        """获取统计（委托给统一熔断器，保持原有字段格式）"""
        unified_stats = self._breaker.get_stats()
        return {
            "state": unified_stats.get("state", "closed"),
            "failure_count": unified_stats.get("failure_count", 0),
            "success_count": unified_stats.get("success_count", 0),
            "failure_threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取统计（别名，与统一熔断器接口一致）"""
        return self.stats()


# ============================================================
# 资源管理器
# ============================================================

@dataclass
class ResourceQuota:
    """资源配额"""
    name: str
    limit: int
    current: int = 0


class ResourceManager:
    """
    资源管理器

    管理资源配额，防止资源耗尽。
    线程安全。
    """

    def __init__(self):
        self._quotas: Dict[str, ResourceQuota] = {}
        self._lock = threading.RLock()

    def set_quota(self, name: str, limit: int) -> None:
        """设置资源配额"""
        with self._lock:
            self._quotas[name] = ResourceQuota(name=name, limit=limit)

    def acquire(self, name: str, amount: int = 1) -> bool:
        """获取资源"""
        with self._lock:
            quota = self._quotas.get(name)
            if quota is None:
                return True  # 无配额限制
            if quota.current + amount > quota.limit:
                return False
            quota.current += amount
            return True

    def release(self, name: str, amount: int = 1) -> None:
        """释放资源"""
        with self._lock:
            quota = self._quotas.get(name)
            if quota:
                quota.current = max(0, quota.current - amount)

    def get_usage(self, name: str) -> Dict[str, int]:
        """获取资源使用情况"""
        with self._lock:
            quota = self._quotas.get(name)
            if quota is None:
                return {"limit": -1, "current": 0}
            return {"limit": quota.limit, "current": quota.current}

    def get_all_usage(self) -> Dict[str, Dict[str, int]]:
        """获取所有资源使用情况"""
        with self._lock:
            return {name: self.get_usage(name) for name in self._quotas}

    def reset(self) -> None:
        """重置所有资源计数"""
        with self._lock:
            for quota in self._quotas.values():
                quota.current = 0


# ============================================================
# 部署打包器
# ============================================================

class DeploymentPackager:
    """
    部署打包器

    生成部署配置文件。
    """

    @staticmethod
    def generate_dockerfile(
        app_name: str = "self_evolving_agent",
        python_version: str = "3.11",
        port: int = 8000,
    ) -> str:
        """生成Dockerfile"""
        return f"""# {app_name} Dockerfile
FROM python:{python_version}-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE {port}

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')" || exit 1

# 启动命令
CMD ["python", "-m", "self_evolving_agent", "--port", "{port}"]
"""

    @staticmethod
    def generate_docker_compose(
        app_name: str = "self_evolving_agent",
        port: int = 8000,
        replicas: int = 3,
    ) -> str:
        """生成docker-compose.yml"""
        return f"""version: '3.8'

services:
  {app_name}:
    build: .
    ports:
      - "{port}:{port}"
    environment:
      - APP_ENV=production
      - APP_PORT={port}
    deploy:
      replicas: {replicas}
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'
      restart_policy:
        condition: on-failure
        max_attempts: 3
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:{port}/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
"""

    @staticmethod
    def generate_kubernetes_manifest(
        app_name: str = "self-evolving-agent",
        port: int = 8000,
        replicas: int = 3,
    ) -> str:
        """生成Kubernetes部署清单"""
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  labels:
    app: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
      - name: {app_name}
        image: {app_name}:latest
        ports:
        - containerPort: {port}
        env:
        - name: APP_ENV
          value: "production"
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: {port}
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: {port}
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}-service
spec:
  selector:
    app: {app_name}
  ports:
  - port: 80
    targetPort: {port}
  type: LoadBalancer
"""

    @staticmethod
    def generate_requirements(extra_deps: Optional[List[str]] = None) -> str:
        """生成requirements.txt"""
        deps = ["# Core dependencies (zero external dependencies by default)"]
        if extra_deps:
            deps.extend(extra_deps)
        deps.append("# Add production dependencies as needed:")
        deps.append("# flask>=2.0.0  # for HTTP API")
        deps.append("# gunicorn>=20.0.0  # for WSGI server")
        return "\n".join(deps) + "\n"

    @staticmethod
    def generate_config_template(env: Environment = Environment.PRODUCTION) -> str:
        """生成配置模板"""
        return json.dumps({
            "env": env.value,
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 4,
            },
            "logging": {
                "level": "INFO",
                "format": "json",
            },
            "limits": {
                "max_requests_per_second": 100,
                "max_connections": 1000,
                "request_timeout": 30,
            },
            "health": {
                "check_interval": 30,
                "timeout": 10,
            },
        }, indent=2, ensure_ascii=False)


# ============================================================
# 生产管理器
# ============================================================

class ProductionManager:
    """
    生产管理器

    集成配置、生命周期、健康端点、限流、熔断、资源管理。
    提供统一的生产部署接口。
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or ConfigLoader.load()
        self._lifecycle = LifecycleManager()
        self._health_endpoint = HealthEndpoint()
        self._rate_limiter: Optional[RateLimiter] = None
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._resource_manager = ResourceManager()
        self._lock = threading.RLock()

        # 注册默认健康检查
        self._register_default_health_checks()

        # 注册生命周期钩子
        self._register_lifecycle_hooks()

    def _register_default_health_checks(self) -> None:
        """注册默认健康检查"""
        self._health_endpoint.add_liveness_check("process", lambda: self._lifecycle.is_running())
        self._health_endpoint.add_readiness_check("process", lambda: self._lifecycle.is_running())

    def _register_lifecycle_hooks(self) -> None:
        """注册生命周期钩子"""
        # 启动钩子
        self._lifecycle.on_start("init_rate_limiter", self._init_rate_limiter, priority=10)

    def _init_rate_limiter(self) -> None:
        """初始化限流器"""
        rate = self._config.get("limits.max_requests_per_second", 100)
        self._rate_limiter = RateLimiter(rate=rate)

    @property
    def config(self) -> Config:
        return self._config

    @property
    def lifecycle(self) -> LifecycleManager:
        return self._lifecycle

    @property
    def health_endpoint(self) -> HealthEndpoint:
        return self._health_endpoint

    @property
    def rate_limiter(self) -> Optional[RateLimiter]:
        return self._rate_limiter

    @property
    def resource_manager(self) -> ResourceManager:
        return self._resource_manager

    def get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """获取或创建熔断器"""
        with self._lock:
            if name not in self._circuit_breakers:
                self._circuit_breakers[name] = CircuitBreaker()
            return self._circuit_breakers[name]

    def start(self) -> bool:
        """启动生产服务"""
        self._lifecycle.install_signal_handlers()
        return self._lifecycle.start()

    def stop(self) -> bool:
        """停止生产服务"""
        return self._lifecycle.stop()

    def check_request_allowed(self) -> bool:
        """检查请求是否允许（限流）"""
        if self._rate_limiter:
            return self._rate_limiter.acquire()
        return True

    def health_check(self) -> HealthResponse:
        """健康检查"""
        return self._health_endpoint.full_health()

    def liveness(self) -> HealthResponse:
        """存活探针"""
        return self._health_endpoint.liveness()

    def readiness(self) -> HealthResponse:
        """就绪探针"""
        return self._health_endpoint.readiness()

    def status(self) -> Dict[str, Any]:
        """获取生产状态"""
        return {
            "lifecycle_state": self._lifecycle.state.value,
            "config_env": self._config.env.value,
            "rate_limiter": {
                "rate": self._rate_limiter.rate if self._rate_limiter else None,
                "available": self._rate_limiter.available_tokens() if self._rate_limiter else None,
            },
            "circuit_breakers": {
                name: cb.stats() for name, cb in self._circuit_breakers.items()
            },
            "resources": self._resource_manager.get_all_usage(),
        }


# ============================================================
# 便捷函数
# ============================================================

def create_production_manager(
    env: Environment = Environment.PRODUCTION,
    config_file: Optional[str] = None,
) -> ProductionManager:
    """创建生产管理器"""
    config = ConfigLoader.load(env=env, config_file=config_file)
    return ProductionManager(config=config)


def create_deployment_package(
    output_dir: str,
    app_name: str = "self_evolving_agent",
    port: int = 8000,
    replicas: int = 3,
) -> Dict[str, str]:
    """创建部署包"""
    os.makedirs(output_dir, exist_ok=True)

    files = {
        "Dockerfile": DeploymentPackager.generate_dockerfile(app_name, port=port),
        "docker-compose.yml": DeploymentPackager.generate_docker_compose(app_name, port, replicas),
        "k8s-deployment.yaml": DeploymentPackager.generate_kubernetes_manifest(app_name, port, replicas),
        "requirements.txt": DeploymentPackager.generate_requirements(),
        "config.json": DeploymentPackager.generate_config_template(),
    }

    for filename, content in files.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

    return files
