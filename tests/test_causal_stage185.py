"""
阶段185测试：因果推理增强 — 条件独立性检验与PC算法

验证内容:
1. ConditionalIndependenceTest: 偏相关系数计算与条件独立性检验
2. PCAlgorithm: 骨架发现与V结构方向推断
3. CausalFirstPrinciplesEngine: 新增接口的集成与向后兼容
4. 端到端：从观测数据发现因果结构
"""

import pytest
import random
from self_evolving_agent.core.causal_first_principles import (
    CausalFirstPrinciplesEngine,
    ConditionalIndependenceTest,
    PCAlgorithm,
    CausalStrength,
    create_causal_engine,
)


# ============================================================
# ConditionalIndependenceTest 测试
# ============================================================

class TestConditionalIndependenceTest:
    """条件独立性检验器测试"""

    def test_pearson_correlation_basic(self):
        """测试皮尔逊相关系数基础计算"""
        ci = ConditionalIndependenceTest()
        # 完全正相关
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        corr = ci._pearson_correlation(x, y)
        assert corr == pytest.approx(1.0, abs=0.01)

        # 完全负相关
        y_neg = [10.0, 8.0, 6.0, 4.0, 2.0]
        corr_neg = ci._pearson_correlation(x, y_neg)
        assert corr_neg == pytest.approx(-1.0, abs=0.01)

    def test_pearson_correlation_zero(self):
        """测试零相关"""
        ci = ConditionalIndependenceTest()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 1.0, 4.0, 2.0, 3.0]  # 无线性关系
        corr = ci._pearson_correlation(x, y)
        assert abs(corr) < 0.5

    def test_partial_correlation_no_condition(self):
        """测试无条件集时偏相关退化为普通相关"""
        ci = ConditionalIndependenceTest()
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        pcorr = ci.partial_correlation(x, y, [])
        assert pcorr == pytest.approx(1.0, abs=0.01)

    def test_partial_correlation_removes_confounder(self):
        """
        测试偏相关消除混淆变量

        第一性原理：如果X和Y都由Z引起，则X和Y的偏相关(给定Z)应接近0
        """
        ci = ConditionalIndependenceTest()
        random.seed(42)

        # Z是公共原因：X = 2*Z + noise, Y = 3*Z + noise
        n = 100
        z = [random.gauss(0, 1) for _ in range(n)]
        x = [2.0 * z[i] + random.gauss(0, 0.1) for i in range(n)]
        y = [3.0 * z[i] + random.gauss(0, 0.1) for i in range(n)]

        # 普通相关应很高
        corr = ci._pearson_correlation(x, y)
        assert corr > 0.8

        # 偏相关（给定Z）应接近0
        pcorr = ci.partial_correlation(x, y, [z])
        assert abs(pcorr) < 0.3

    def test_conditional_independence_true(self):
        """测试条件独立判定（条件独立时返回True）"""
        ci = ConditionalIndependenceTest(significance_level=0.3)
        random.seed(42)

        n = 100
        z = [random.gauss(0, 1) for _ in range(n)]
        x = [2.0 * z[i] + random.gauss(0, 0.1) for i in range(n)]
        y = [3.0 * z[i] + random.gauss(0, 0.1) for i in range(n)]

        is_indep, pcorr = ci.test_conditional_independence(x, y, [z])
        assert is_indep is True
        assert abs(pcorr) < 0.3

    def test_conditional_independence_false(self):
        """测试条件独立判定（不独立时返回False）"""
        ci = ConditionalIndependenceTest(significance_level=0.1)
        random.seed(42)

        # X直接导致Y，无条件集能消除关系
        n = 100
        x = [random.gauss(0, 1) for _ in range(n)]
        y = [2.0 * x[i] + random.gauss(0, 0.1) for i in range(n)]
        z = [random.gauss(0, 1) for _ in range(n)]  # 无关变量

        is_indep, pcorr = ci.test_conditional_independence(x, y, [z])
        assert is_indep is False
        assert abs(pcorr) > 0.5

    def test_insufficient_data(self):
        """测试数据不足时返回0"""
        ci = ConditionalIndependenceTest()
        pcorr = ci.partial_correlation([1.0], [2.0], [])
        assert pcorr == 0.0


# ============================================================
# PCAlgorithm 测试
# ============================================================

class TestPCAlgorithm:
    """PC算法测试"""

    def test_discover_skeleton_chain(self):
        """
        测试链结构骨架发现：X → Z → Y

        第一性原理：X和Y在给定Z时应条件独立
        - 骨架应保留X-Z和Z-Y边
        - 骨架应移除X-Y边（条件独立）
        """
        random.seed(42)
        n = 200

        # 生成链结构数据：X → Z → Y
        x = [random.gauss(0, 1) for _ in range(n)]
        z = [2.0 * x[i] + random.gauss(0, 0.2) for i in range(n)]
        y = [1.5 * z[i] + random.gauss(0, 0.2) for i in range(n)]

        data = {"X": x, "Z": z, "Y": y}

        pc = PCAlgorithm(ConditionalIndependenceTest(significance_level=0.15))
        skeleton, details = pc.discover_skeleton(data, max_condition_set_size=1)

        # X-Y应被移除（条件独立于Z）
        xy_edge = ("X", "Y") if ("X", "Y") in skeleton else ("Y", "X")
        assert xy_edge not in skeleton
        # X-Z和Z-Y应保留
        assert ("X", "Z") in skeleton or ("Z", "X") in skeleton
        assert ("Z", "Y") in skeleton or ("Y", "Z") in skeleton

    def test_discover_skeleton_fork(self):
        """
        测试叉结构骨架发现：X ← Z → Y

        第一性原理：X和Y在给定Z时应条件独立
        """
        random.seed(42)
        n = 200

        z = [random.gauss(0, 1) for _ in range(n)]
        x = [2.0 * z[i] + random.gauss(0, 0.2) for i in range(n)]
        y = [1.5 * z[i] + random.gauss(0, 0.2) for i in range(n)]

        data = {"X": x, "Z": z, "Y": y}

        pc = PCAlgorithm(ConditionalIndependenceTest(significance_level=0.15))
        skeleton, details = pc.discover_skeleton(data, max_condition_set_size=1)

        # X-Y应被移除
        assert ("X", "Y") not in skeleton
        assert ("Y", "X") not in skeleton
        # X-Z和Z-Y应保留
        assert ("X", "Z") in skeleton or ("Z", "X") in skeleton
        assert ("Z", "Y") in skeleton or ("Y", "Z") in skeleton

    def test_discover_oriented_graph_v_structure(self):
        """
        测试V结构方向推断：X → Z ← Y

        第一性原理：碰撞点Z不在X-Y的分离集中
        - X和Y无条件独立（边际相关）
        - X和Y给定Z时不独立
        - 因此Z是碰撞点
        """
        random.seed(42)
        n = 300

        # V结构：X → Z ← Y（X和Y独立）
        x = [random.gauss(0, 1) for _ in range(n)]
        y = [random.gauss(0, 1) for _ in range(n)]
        z = [x[i] + y[i] + random.gauss(0, 0.1) for i in range(n)]

        data = {"X": x, "Y": y, "Z": z}

        pc = PCAlgorithm(ConditionalIndependenceTest(significance_level=0.15))
        directed, undirected, details = pc.discover_oriented_graph(data, max_condition_set_size=1)

        # 应检测到V结构：X → Z 和 Y → Z
        assert ("X", "Z") in directed
        assert ("Y", "Z") in directed

    def test_single_variable(self):
        """测试单变量情况"""
        pc = PCAlgorithm()
        skeleton, details = pc.discover_skeleton({"X": [1.0, 2.0, 3.0]})
        assert len(skeleton) == 0

    def test_empty_data(self):
        """测试空数据"""
        pc = PCAlgorithm()
        skeleton, details = pc.discover_skeleton({})
        assert len(skeleton) == 0


# ============================================================
# CausalFirstPrinciplesEngine 阶段185接口测试
# ============================================================

class TestCausalEngineStage185:
    """因果引擎阶段185接口测试"""

    def test_new_interfaces_exist(self):
        """测试新增接口存在"""
        engine = create_causal_engine()
        assert hasattr(engine, "test_conditional_independence")
        assert hasattr(engine, "discover_causal_structure")
        assert hasattr(engine, "get_conditional_independence_test")
        assert hasattr(engine, "get_pc_algorithm")

    def test_record_and_test_ci(self):
        """测试记录观测后进行条件独立性检验"""
        engine = create_causal_engine()
        random.seed(42)

        # 记录链结构数据：X → Z → Y
        for _ in range(100):
            x = random.gauss(0, 1)
            z = 2.0 * x + random.gauss(0, 0.1)
            y = 1.5 * z + random.gauss(0, 0.1)
            engine.record_observation({"X": x, "Z": z, "Y": y})

        # X ⊥ Y | Z 应为条件独立
        is_indep, pcorr = engine.test_conditional_independence("X", "Y", ["Z"])
        assert is_indep is True
        assert abs(pcorr) < 0.3

    def test_discover_causal_structure(self):
        """测试从观测数据发现因果结构"""
        engine = create_causal_engine()
        random.seed(42)

        # 记录V结构数据：X → Z ← Y
        for _ in range(200):
            x = random.gauss(0, 1)
            y = random.gauss(0, 1)
            z = x + y + random.gauss(0, 0.1)
            engine.record_observation({"X": x, "Y": y, "Z": z})

        result = engine.discover_causal_structure(max_condition_set_size=1)

        assert "skeleton" in result
        assert "directed_edges" in result
        assert "undirected_edges" in result
        assert "stats" in result
        # V结构应有有向边
        assert len(result["directed_edges"]) > 0

    def test_backward_compatibility(self):
        """测试向后兼容：原有接口仍正常工作"""
        engine = create_causal_engine()

        # 原有 add_causal_link
        link = engine.add_causal_link("A", "B", CausalStrength.STRONG, 0.9)
        assert link.cause == "A"
        assert link.effect == "B"

        # 原有 record_observation
        engine.record_observation({"A": 1.0, "B": 2.0})

        # 原有 discover_causal_links
        links = engine.discover_causal_links(min_correlation=0.3)
        assert isinstance(links, list)

        # 原有 predict_effect
        effects = engine.predict_effect("A", 1.0)
        assert isinstance(effects, dict)

    def test_no_observations_ci(self):
        """测试无观测数据时的条件独立性检验"""
        engine = create_causal_engine()
        is_indep, pcorr = engine.test_conditional_independence("X", "Y", ["Z"])
        # 无数据时偏相关为0，应判定为条件独立
        assert is_indep is True
        assert pcorr == 0.0

    def test_no_observations_discover(self):
        """测试无观测数据时的因果结构发现"""
        engine = create_causal_engine()
        result = engine.discover_causal_structure()
        assert result["stats"]["skeleton_size"] == 0
