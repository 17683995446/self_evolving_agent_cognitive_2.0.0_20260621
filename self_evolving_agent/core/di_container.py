"""
阶段191: 依赖注入容器

第一性原理:
1. 控制反转（IoC）：组件不应创建自己的依赖，而应从外部容器接收
2. 依赖倒置（DIP）：高层模块不应依赖低层模块，二者都应依赖抽象
3. 单一职责：容器只负责对象创建和生命周期管理
4. 开闭原则：通过注册新服务扩展功能，无需修改现有代码

设计原则:
- 三种生命周期：单例（singleton）、瞬态（transient）、作用域（scoped）
- 支持工厂函数和直接类型注册
- 支持实例注册（预创建对象）
- 自动依赖解析（通过类型注解）
- 线程安全

使用方式:
    container = DIContainer()
    container.register(IPredictiveCoding, PredictiveCodingEngine, ServiceLifetime.SINGLETON)
    container.register_factory(IDrive, lambda c: create_drive_engine("drive"))

    engine = container.resolve(IPredictiveCoding)
"""

import threading
import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, get_type_hints

T = TypeVar("T")


# ============================================================
# 服务生命周期
# ============================================================

class ServiceLifetime(Enum):
    """服务生命周期"""
    SINGLETON = "singleton"  # 单例：整个容器共享一个实例
    TRANSIENT = "transient"  # 瞬态：每次解析创建新实例
    SCOPED = "scoped"        # 作用域：同一作用域内共享


# ============================================================
# 服务描述符
# ============================================================

@dataclass
class ServiceDescriptor:
    """服务描述符"""
    service_type: type
    implementation: Optional[type] = None  # 实现类型
    factory: Optional[Callable[["DIContainer"], Any]] = None  # 工厂函数
    instance: Optional[Any] = None  # 预创建实例
    lifetime: ServiceLifetime = ServiceLifetime.SINGLETON
    dependencies: List[type] = field(default_factory=list)


# ============================================================
# 依赖注入容器
# ============================================================

class DIContainer:
    """
    依赖注入容器

    第一性原理：
    - 控制反转：对象的创建和依赖解析由容器负责
    - 生命周期管理：单例、瞬态、作用域
    - 自动依赖解析：通过构造函数类型注解

    线程安全。
    """

    def __init__(self):
        self._descriptors: Dict[type, ServiceDescriptor] = {}
        self._singletons: Dict[type, Any] = {}
        self._lock = threading.RLock()
        self._parent: Optional["DIContainer"] = None

    # ============================================================
    # 注册方法
    # ============================================================

    def register(
        self,
        service_type: type,
        implementation: Optional[type] = None,
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "DIContainer":
        """
        注册服务

        Args:
            service_type: 服务类型（通常是接口或抽象类）
            implementation: 实现类型（None则使用service_type本身）
            lifetime: 生命周期

        Returns:
            self（支持链式调用）
        """
        with self._lock:
            impl = implementation or service_type
            descriptor = ServiceDescriptor(
                service_type=service_type,
                implementation=impl,
                lifetime=lifetime,
            )
            # 自动解析构造函数依赖
            descriptor.dependencies = self._extract_dependencies(impl)
            self._descriptors[service_type] = descriptor
        return self

    def register_factory(
        self,
        service_type: type,
        factory: Callable[["DIContainer"], Any],
        lifetime: ServiceLifetime = ServiceLifetime.SINGLETON,
    ) -> "DIContainer":
        """
        注册工厂函数

        Args:
            service_type: 服务类型
            factory: 工厂函数，接收容器参数，返回服务实例
            lifetime: 生命周期

        Returns:
            self
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                factory=factory,
                lifetime=lifetime,
            )
            self._descriptors[service_type] = descriptor
        return self

    def register_instance(
        self,
        service_type: type,
        instance: Any,
    ) -> "DIContainer":
        """
        注册预创建实例（始终为单例）

        Args:
            service_type: 服务类型
            instance: 预创建的实例

        Returns:
            self
        """
        with self._lock:
            descriptor = ServiceDescriptor(
                service_type=service_type,
                instance=instance,
                lifetime=ServiceLifetime.SINGLETON,
            )
            self._descriptors[service_type] = descriptor
            self._singletons[service_type] = instance
        return self

    # ============================================================
    # 解析方法
    # ============================================================

    def resolve(self, service_type: type) -> Any:
        """
        解析服务

        Args:
            service_type: 要解析的服务类型

        Returns:
            服务实例

        Raises:
            KeyError: 服务未注册
        """
        with self._lock:
            if service_type not in self._descriptors:
                # 尝试从父容器解析
                if self._parent is not None:
                    return self._parent.resolve(service_type)
                raise KeyError(f"服务未注册: {service_type.__name__}")

            descriptor = self._descriptors[service_type]

            # 单例：检查缓存
            if descriptor.lifetime == ServiceLifetime.SINGLETON:
                if service_type in self._singletons:
                    return self._singletons[service_type]
                instance = self._create_instance(descriptor)
                self._singletons[service_type] = instance
                return instance

            # 瞬态：每次创建新实例
            elif descriptor.lifetime == ServiceLifetime.TRANSIENT:
                return self._create_instance(descriptor)

            # 作用域：当前容器内单例（不缓存到全局）
            else:
                return self._create_instance(descriptor)

    def resolve_optional(self, service_type: type, default: Any = None) -> Any:
        """解析服务，未注册或实例化失败时返回默认值"""
        try:
            return self.resolve(service_type)
        except (KeyError, RuntimeError):
            return default

    def is_registered(self, service_type: type) -> bool:
        """检查服务是否已注册"""
        with self._lock:
            if service_type in self._descriptors:
                return True
            if self._parent is not None:
                return self._parent.is_registered(service_type)
            return False

    # ============================================================
    # 作用域管理
    # ============================================================

    def create_scope(self) -> "DIContainer":
        """
        创建子作用域

        子作用域可以解析父作用域中注册的服务，
        但子作用域中的单例不会影响父作用域。
        """
        scope = DIContainer()
        scope._parent = self
        # 共享描述符但独立缓存单例
        with self._lock:
            scope._descriptors = dict(self._descriptors)
        return scope

    # ============================================================
    # 内部方法
    # ============================================================

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        # 预创建实例
        if descriptor.instance is not None:
            return descriptor.instance

        # 工厂函数
        if descriptor.factory is not None:
            return descriptor.factory(self)

        # 类型实例化
        if descriptor.implementation is not None:
            return self._instantiate_type(descriptor)

        raise RuntimeError(f"无法创建服务实例: {descriptor.service_type.__name__}")

    def _instantiate_type(self, descriptor: ServiceDescriptor) -> Any:
        """通过类型实例化，自动解析构造函数依赖"""
        impl = descriptor.implementation
        if impl is None:
            raise RuntimeError(f"无实现类型: {descriptor.service_type.__name__}")

        # 解析构造函数参数
        kwargs = {}
        for dep_type in descriptor.dependencies:
            if self.is_registered(dep_type):
                kwargs[dep_type.__name__.lower()] = self.resolve(dep_type)

        try:
            return impl(**kwargs) if kwargs else impl()
        except TypeError:
            # 如果自动依赖注入失败，尝试无参构造
            try:
                return impl()
            except Exception as e:
                raise RuntimeError(
                    f"无法实例化 {impl.__name__}: {e}"
                ) from e

    def _extract_dependencies(self, impl: type) -> List[type]:
        """从构造函数提取依赖类型"""
        try:
            sig = inspect.signature(impl.__init__)
            deps = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.annotation != inspect.Parameter.empty:
                    annotation = param.annotation
                    # 只处理类型注解（跳过基本类型）
                    if isinstance(annotation, type) and annotation.__module__ != "builtins":
                        deps.append(annotation)
            return deps
        except (ValueError, TypeError):
            return []

    # ============================================================
    # 管理方法
    # ============================================================

    def get_registered_services(self) -> Dict[type, ServiceDescriptor]:
        """获取所有已注册服务"""
        with self._lock:
            return dict(self._descriptors)

    def get_stats(self) -> Dict[str, Any]:
        """获取容器统计"""
        with self._lock:
            return {
                "registered_services": len(self._descriptors),
                "cached_singletons": len(self._singletons),
                "has_parent": self._parent is not None,
                "service_types": [
                    t.__name__ for t in self._descriptors.keys()
                ],
            }

    def clear(self) -> None:
        """清空容器"""
        with self._lock:
            self._descriptors.clear()
            self._singletons.clear()

    def clear_singletons(self) -> None:
        """清空单例缓存（保留注册）"""
        with self._lock:
            self._singletons.clear()


# ============================================================
# 核心引擎注册器
# ============================================================

def create_core_container() -> DIContainer:
    """
    创建预注册核心引擎的DI容器

    第一性原理：
    - 核心引擎通过DI容器管理，降低耦合
    - 单例生命周期：整个系统共享一个实例
    - 工厂函数：延迟创建，按需初始化
    """
    container = DIContainer()

    # 延迟导入以避免循环依赖
    try:
        from .predictive_coding_engine import PredictiveCodingEngine, create_predictive_coding_engine
        from .drive_first_principles import DriveFirstPrinciplesEngine, create_drive_engine
        from .learning_first_principles import LearningFirstPrinciplesEngine, create_learning_engine
        from .causal_first_principles import CausalFirstPrinciplesEngine, create_causal_engine
        from .social_first_principles import SocialFirstPrinciplesEngine, create_social_engine
        from .creativity_first_principles import CreativityFirstPrinciplesEngine, create_creativity_engine
        from .unified_first_principles_engine import UnifiedFirstPrinciplesEngine, create_unified_engine
        from .orchestration_pipeline import OrchestrationPipeline, create_orchestration_pipeline
    except ImportError:
        from predictive_coding_engine import PredictiveCodingEngine, create_predictive_coding_engine
        from drive_first_principles import DriveFirstPrinciplesEngine, create_drive_engine
        from learning_first_principles import LearningFirstPrinciplesEngine, create_learning_engine
        from causal_first_principles import CausalFirstPrinciplesEngine, create_causal_engine
        from social_first_principles import SocialFirstPrinciplesEngine, create_social_engine
        from creativity_first_principles import CreativityFirstPrinciplesEngine, create_creativity_engine
        from unified_first_principles_engine import UnifiedFirstPrinciplesEngine, create_unified_engine
        from orchestration_pipeline import OrchestrationPipeline, create_orchestration_pipeline

    # 注册核心引擎（单例 + 工厂函数）
    container.register_factory(
        PredictiveCodingEngine,
        lambda c: create_predictive_coding_engine("predictive_coding"),
    )
    container.register_factory(
        DriveFirstPrinciplesEngine,
        lambda c: create_drive_engine("drive"),
    )
    container.register_factory(
        LearningFirstPrinciplesEngine,
        lambda c: create_learning_engine("learning"),
    )
    container.register_factory(
        CausalFirstPrinciplesEngine,
        lambda c: create_causal_engine("causal"),
    )
    container.register_factory(
        SocialFirstPrinciplesEngine,
        lambda c: create_social_engine("social"),
    )
    container.register_factory(
        CreativityFirstPrinciplesEngine,
        lambda c: create_creativity_engine("creativity"),
    )
    container.register_factory(
        UnifiedFirstPrinciplesEngine,
        lambda c: create_unified_engine("unified"),
    )
    container.register_factory(
        OrchestrationPipeline,
        lambda c: create_orchestration_pipeline(c.resolve(UnifiedFirstPrinciplesEngine)),
    )

    return container
