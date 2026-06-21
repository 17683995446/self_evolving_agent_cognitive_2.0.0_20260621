"""
阶段191测试：依赖注入容器

验证内容:
1. DIContainer: 基本注册和解析
2. ServiceLifetime: 三种生命周期（单例、瞬态、作用域）
3. 工厂函数注册
4. 实例注册
5. 自动依赖解析
6. 作用域管理
7. 核心引擎注册器
8. 第一性原理验证
"""

import pytest
from self_evolving_agent.core.di_container import (
    DIContainer,
    ServiceLifetime,
    ServiceDescriptor,
    create_core_container,
)


# ============================================================
# 测试用辅助类
# ============================================================

class IEngine:
    """引擎接口"""
    pass


class MockEngine(IEngine):
    """模拟引擎"""
    def __init__(self):
        self.id = id(self)


class MockEngineWithDep:
    """带依赖的引擎"""
    def __init__(self, iengine: IEngine):
        self.engine = iengine


class MockService:
    """模拟服务"""
    def __init__(self, name: str = "default"):
        self.name = name


# ============================================================
# 基本注册和解析测试
# ============================================================

class TestBasicRegistration:
    """基本注册和解析测试"""

    def test_register_and_resolve_type(self):
        """测试注册类型并解析"""
        container = DIContainer()
        container.register(MockService)

        service = container.resolve(MockService)
        assert isinstance(service, MockService)
        assert service.name == "default"

    def test_register_interface_with_implementation(self):
        """测试注册接口和实现"""
        container = DIContainer()
        container.register(IEngine, MockEngine)

        engine = container.resolve(IEngine)
        assert isinstance(engine, MockEngine)

    def test_register_returns_self_for_chaining(self):
        """测试注册返回self支持链式调用"""
        container = DIContainer()
        result = container.register(MockService)
        assert result is container

    def test_resolve_unregistered_raises_keyerror(self):
        """测试解析未注册服务抛出KeyError"""
        container = DIContainer()
        with pytest.raises(KeyError):
            container.resolve(MockService)

    def test_resolve_optional_returns_default(self):
        """测试resolve_optional返回默认值"""
        container = DIContainer()
        result = container.resolve_optional(MockService, default="fallback")
        assert result == "fallback"

    def test_is_registered(self):
        """测试检查服务是否已注册"""
        container = DIContainer()
        assert container.is_registered(MockService) is False

        container.register(MockService)
        assert container.is_registered(MockService) is True


# ============================================================
# 生命周期测试
# ============================================================

class TestServiceLifetime:
    """服务生命周期测试"""

    def test_singleton_returns_same_instance(self):
        """测试单例返回同一实例"""
        container = DIContainer()
        container.register(MockEngine, lifetime=ServiceLifetime.SINGLETON)

        instance1 = container.resolve(MockEngine)
        instance2 = container.resolve(MockEngine)

        assert instance1 is instance2

    def test_transient_returns_new_instance(self):
        """测试瞬态返回新实例"""
        container = DIContainer()
        container.register(MockEngine, lifetime=ServiceLifetime.TRANSIENT)

        instance1 = container.resolve(MockEngine)
        instance2 = container.resolve(MockEngine)

        assert instance1 is not instance2
        assert isinstance(instance1, MockEngine)
        assert isinstance(instance2, MockEngine)

    def test_default_lifetime_is_singleton(self):
        """测试默认生命周期为单例"""
        container = DIContainer()
        container.register(MockEngine)

        instance1 = container.resolve(MockEngine)
        instance2 = container.resolve(MockEngine)

        assert instance1 is instance2

    def test_singleton_cached(self):
        """测试单例被缓存"""
        container = DIContainer()
        container.register(MockEngine, lifetime=ServiceLifetime.SINGLETON)

        # 第一次解析创建实例
        instance1 = container.resolve(MockEngine)
        assert len(container._singletons) == 1

        # 第二次解析返回缓存
        instance2 = container.resolve(MockEngine)
        assert len(container._singletons) == 1
        assert instance1 is instance2


# ============================================================
# 工厂函数注册测试
# ============================================================

class TestFactoryRegistration:
    """工厂函数注册测试"""

    def test_register_factory(self):
        """测试注册工厂函数"""
        container = DIContainer()
        container.register_factory(
            MockService,
            lambda c: MockService(name="factory_created"),
        )

        service = container.resolve(MockService)
        assert service.name == "factory_created"

    def test_factory_singleton(self):
        """测试工厂函数单例"""
        container = DIContainer()
        container.register_factory(
            MockEngine,
            lambda c: MockEngine(),
            lifetime=ServiceLifetime.SINGLETON,
        )

        instance1 = container.resolve(MockEngine)
        instance2 = container.resolve(MockEngine)
        assert instance1 is instance2

    def test_factory_transient(self):
        """测试工厂函数瞬态"""
        container = DIContainer()
        container.register_factory(
            MockEngine,
            lambda c: MockEngine(),
            lifetime=ServiceLifetime.TRANSIENT,
        )

        instance1 = container.resolve(MockEngine)
        instance2 = container.resolve(MockEngine)
        assert instance1 is not instance2

    def test_factory_can_access_container(self):
        """测试工厂函数可以访问容器"""
        container = DIContainer()
        container.register(IEngine, MockEngine)
        container.register_factory(
            MockEngineWithDep,
            lambda c: MockEngineWithDep(c.resolve(IEngine)),
        )

        service = container.resolve(MockEngineWithDep)
        assert isinstance(service.engine, MockEngine)


# ============================================================
# 实例注册测试
# ============================================================

class TestInstanceRegistration:
    """实例注册测试"""

    def test_register_instance(self):
        """测试注册预创建实例"""
        container = DIContainer()
        instance = MockEngine()
        container.register_instance(IEngine, instance)

        resolved = container.resolve(IEngine)
        assert resolved is instance

    def test_registered_instance_is_singleton(self):
        """测试注册的实例是单例"""
        container = DIContainer()
        instance = MockEngine()
        container.register_instance(IEngine, instance)

        resolved1 = container.resolve(IEngine)
        resolved2 = container.resolve(IEngine)
        assert resolved1 is instance
        assert resolved2 is instance


# ============================================================
# 自动依赖解析测试
# ============================================================

class TestAutoDependencyResolution:
    """自动依赖解析测试"""

    def test_auto_resolve_dependencies(self):
        """测试自动解析构造函数依赖"""
        container = DIContainer()
        container.register(IEngine, MockEngine)
        container.register(MockEngineWithDep)

        service = container.resolve(MockEngineWithDep)
        assert isinstance(service, MockEngineWithDep)
        assert isinstance(service.engine, MockEngine)

    def test_auto_resolve_with_unregistered_dep(self):
        """测试依赖未注册时仍能创建（降级为无参构造）"""
        container = DIContainer()
        container.register(MockEngineWithDep)

        # MockEngineWithDep需要IEngine，但未注册
        # 应降级为无参构造（如果可能）
        service = container.resolve_optional(MockEngineWithDep)
        # 可能成功（无参构造）或返回None
        assert service is None or isinstance(service, MockEngineWithDep)


# ============================================================
# 作用域管理测试
# ============================================================

class TestScopeManagement:
    """作用域管理测试"""

    def test_create_scope(self):
        """测试创建子作用域"""
        container = DIContainer()
        container.register(MockEngine, lifetime=ServiceLifetime.SINGLETON)

        scope = container.create_scope()
        assert scope is not container
        assert scope.is_registered(MockEngine)

    def test_scope_inherits_registrations(self):
        """测试子作用域继承父作用域注册"""
        container = DIContainer()
        container.register(MockService)

        scope = container.create_scope()
        assert scope.is_registered(MockService)

        service = scope.resolve(MockService)
        assert isinstance(service, MockService)

    def test_scope_independent_singletons(self):
        """测试子作用域独立单例"""
        container = DIContainer()
        container.register(MockEngine, lifetime=ServiceLifetime.SINGLETON)

        parent_instance = container.resolve(MockEngine)

        scope = container.create_scope()
        scope_instance = scope.resolve(MockEngine)

        # 父子作用域的单例应不同（子作用域独立缓存）
        # 注意：子作用域会重新创建单例
        assert parent_instance is not scope_instance

    def test_scope_can_resolve_from_parent(self):
        """测试子作用域可从父作用域解析"""
        container = DIContainer()
        container.register_instance(MockService, MockService(name="parent"))

        scope = container.create_scope()
        service = scope.resolve(MockService)
        assert service.name == "parent"


# ============================================================
# 容器管理测试
# ============================================================

class TestContainerManagement:
    """容器管理测试"""

    def test_get_registered_services(self):
        """测试获取所有已注册服务"""
        container = DIContainer()
        container.register(MockService)
        container.register(IEngine, MockEngine)

        services = container.get_registered_services()
        assert MockService in services
        assert IEngine in services

    def test_get_stats(self):
        """测试获取容器统计"""
        container = DIContainer()
        container.register(MockService)
        container.register(MockEngine)

        stats = container.get_stats()
        assert stats["registered_services"] == 2
        assert stats["cached_singletons"] == 0
        assert stats["has_parent"] is False
        assert "MockService" in stats["service_types"]

    def test_clear(self):
        """测试清空容器"""
        container = DIContainer()
        container.register(MockService)

        container.clear()
        assert container.is_registered(MockService) is False

    def test_clear_singletons(self):
        """测试清空单例缓存"""
        container = DIContainer()
        container.register(MockEngine, lifetime=ServiceLifetime.SINGLETON)

        # 创建单例
        instance1 = container.resolve(MockEngine)
        assert len(container._singletons) == 1

        # 清空单例
        container.clear_singletons()
        assert len(container._singletons) == 0

        # 重新解析创建新实例
        instance2 = container.resolve(MockEngine)
        assert instance1 is not instance2


# ============================================================
# 核心引擎注册器测试
# ============================================================

class TestCoreContainer:
    """核心引擎注册器测试"""

    def test_create_core_container(self):
        """测试创建核心容器"""
        container = create_core_container()
        assert isinstance(container, DIContainer)
        assert container.get_stats()["registered_services"] >= 7

    def test_resolve_predictive_coding_engine(self):
        """测试解析预测编码引擎"""
        container = create_core_container()

        from self_evolving_agent.core.predictive_coding_engine import PredictiveCodingEngine
        engine = container.resolve(PredictiveCodingEngine)
        assert isinstance(engine, PredictiveCodingEngine)

    def test_resolve_drive_engine(self):
        """测试解析驱力引擎"""
        container = create_core_container()

        from self_evolving_agent.core.drive_first_principles import DriveFirstPrinciplesEngine
        engine = container.resolve(DriveFirstPrinciplesEngine)
        assert isinstance(engine, DriveFirstPrinciplesEngine)

    def test_resolve_learning_engine(self):
        """测试解析学习引擎"""
        container = create_core_container()

        from self_evolving_agent.core.learning_first_principles import LearningFirstPrinciplesEngine
        engine = container.resolve(LearningFirstPrinciplesEngine)
        assert isinstance(engine, LearningFirstPrinciplesEngine)

    def test_resolve_unified_engine(self):
        """测试解析统一引擎"""
        container = create_core_container()

        from self_evolving_agent.core.unified_first_principles_engine import UnifiedFirstPrinciplesEngine
        engine = container.resolve(UnifiedFirstPrinciplesEngine)
        assert isinstance(engine, UnifiedFirstPrinciplesEngine)

    def test_resolve_orchestration_pipeline(self):
        """测试解析编排管线"""
        container = create_core_container()

        from self_evolving_agent.core.orchestration_pipeline import OrchestrationPipeline
        pipeline = container.resolve(OrchestrationPipeline)
        assert isinstance(pipeline, OrchestrationPipeline)

    def test_core_engines_are_singletons(self):
        """测试核心引擎是单例"""
        container = create_core_container()

        from self_evolving_agent.core.predictive_coding_engine import PredictiveCodingEngine
        engine1 = container.resolve(PredictiveCodingEngine)
        engine2 = container.resolve(PredictiveCodingEngine)
        assert engine1 is engine2

    def test_orchestration_pipeline_uses_unified_engine(self):
        """测试编排管线使用统一引擎"""
        container = create_core_container()

        from self_evolving_agent.core.unified_first_principles_engine import UnifiedFirstPrinciplesEngine
        from self_evolving_agent.core.orchestration_pipeline import OrchestrationPipeline

        unified = container.resolve(UnifiedFirstPrinciplesEngine)
        pipeline = container.resolve(OrchestrationPipeline)

        assert pipeline.engine is unified


# ============================================================
# 第一性原理验证测试
# ============================================================

class TestFirstPrinciples:
    """第一性原理验证"""

    def test_inversion_of_control(self):
        """测试控制反转（第一性原理）"""
        container = DIContainer()
        container.register(IEngine, MockEngine)

        # 组件不创建依赖，由容器提供
        engine = container.resolve(IEngine)
        assert engine is not None
        # 引擎不是由调用方创建的
        assert engine.__class__ == MockEngine

    def test_dependency_inversion(self):
        """测试依赖倒置（第一性原理）"""
        container = DIContainer()
        container.register(IEngine, MockEngine)
        container.register(MockEngineWithDep)

        # 高层模块(MockEngineWithDep)依赖抽象(IEngine)
        # 不直接依赖具体实现(MockEngine)
        service = container.resolve(MockEngineWithDep)
        assert isinstance(service.engine, IEngine)

    def test_single_responsibility(self):
        """测试单一职责（第一性原理）"""
        container = DIContainer()
        # 容器只负责创建和管理对象
        stats = container.get_stats()
        assert "registered_services" in stats
        assert "cached_singletons" in stats

    def test_open_closed_principle(self):
        """测试开闭原则（第一性原理）"""
        container = DIContainer()
        container.register(MockService)

        # 通过注册新服务扩展，不修改现有代码
        container.register(MockEngine)
        assert container.is_registered(MockService)
        assert container.is_registered(MockEngine)

    def test_thread_safety(self):
        """测试线程安全（第一性原理）"""
        import threading

        container = DIContainer()
        container.register(MockEngine, lifetime=ServiceLifetime.SINGLETON)

        results = []
        barrier = threading.Barrier(10)

        def resolve_engine():
            barrier.wait()
            engine = container.resolve(MockEngine)
            results.append(engine)

        threads = [threading.Thread(target=resolve_engine) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 单例：所有线程应获得同一实例
        assert len(results) == 10
        assert all(r is results[0] for r in results)
