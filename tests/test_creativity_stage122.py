"""
阶段122: 创造力强化 — 专项测试

验证新增能力:
1. CreativityMetrics: 创造力度量（新颖性、价值性、惊讶度）
2. CombinatorialCreativity: 组合创新
3. AnalogicalReasoning: 类比推理（结构映射）
4. DivergentThinking: 发散思维（变异、交叉授粉）
5. CreativityFirstPrinciplesEngine: 新增接口
6. 向后兼容性

第一性原理验证:
- 创新是旧元素的新组合
- 类比是结构映射
- 发散思维产生多样性
- 创造力可量化
- 涌现：整体大于部分之和
"""

import math
import threading
import time
import random
import pytest

from self_evolving_agent.core.creativity_first_principles import (
    ConceptType,
    CreativityMetricType,
    Concept,
    Combination,
    CreativityAssessment,
    AnalogyMapping,
    CreativityMetrics,
    CombinatorialCreativity,
    AnalogicalReasoning,
    DivergentThinking,
    CreativityFirstPrinciplesEngine,
    create_creativity_engine,
    create_concept,
    create_creativity_metrics,
)


# ============================================================
# 1. CreativityMetrics 测试
# ============================================================

class TestCreativityMetrics:
    """创造力度量测试"""

    def test_creation(self):
        """创建"""
        m = CreativityMetrics()
        assert m is not None

    def test_assess_basic(self):
        """基本评估"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1, "b": 2},
        )
        result = m.assess(concept, [])
        assert isinstance(result, CreativityAssessment)
        assert result.concept_id == "c1"

    def test_assess_no_existing(self):
        """无已有概念时新颖性=1"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1},
        )
        result = m.assess(concept, [])
        assert result.novelty == 1.0

    def test_compute_novelty_identical(self):
        """与已有概念相同时新颖性=0"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1, "b": 2},
        )
        existing = [Concept(
            concept_id="c2", name="existing", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1, "b": 2},
        )]
        novelty = m.compute_novelty(concept, existing)
        assert novelty == 0.0

    def test_compute_novelty_completely_different(self):
        """完全不同时新颖性=1"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"x": 1, "y": 2},
        )
        existing = [Concept(
            concept_id="c2", name="existing", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1, "b": 2},
        )]
        novelty = m.compute_novelty(concept, existing)
        assert novelty == 1.0

    def test_compute_novelty_no_props(self):
        """无属性时新颖性=0"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={},
        )
        novelty = m.compute_novelty(concept, [concept])
        assert novelty == 0.0

    def test_compute_value_empty_props(self):
        """无属性时价值=0"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={},
        )
        assert m.compute_value(concept) == 0.0

    def test_compute_value_with_cross_domain(self):
        """跨域属性提升价值"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1, "cross_domain": True, "emergent": True},
        )
        value = m.compute_value(concept)
        assert value > 0.5

    def test_compute_surprise_no_existing(self):
        """无已有概念时惊讶度=1"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1},
        )
        assert m.compute_surprise(concept, []) == 1.0

    def test_compute_surprise_rare_props(self):
        """罕见属性提升惊讶度"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"rare": 1, "common": 2},
        )
        # 10个已有概念都有common，没有rare
        existing = [
            Concept(
                concept_id=f"e{i}", name=f"e{i}", concept_type=ConceptType.PRIMITIVE,
                domain="d1", properties={"common": 1},
            ) for i in range(10)
        ]
        surprise = m.compute_surprise(concept, existing)
        # rare是罕见的（0次），common是常见的（10次）
        assert surprise > 0.0

    def test_compute_surprise_no_props(self):
        """无属性时惊讶度=0"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={},
        )
        assert m.compute_surprise(concept, [concept]) == 0.0

    def test_overall_score_weighted(self):
        """整体得分是加权平均"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1, "cross_domain": True, "emergent": True},
        )
        result = m.assess(concept, [])
        expected = 0.4 * result.novelty + 0.4 * result.value + 0.2 * result.surprise
        assert abs(result.overall_score - expected) < 1e-6

    def test_scores_in_range(self):
        """所有得分在[0,1]范围"""
        m = CreativityMetrics()
        concept = Concept(
            concept_id="c1", name="test", concept_type=ConceptType.PRIMITIVE,
            domain="d1", properties={"a": 1, "b": 2, "c": 3},
        )
        existing = [
            Concept(
                concept_id=f"e{i}", name=f"e{i}", concept_type=ConceptType.PRIMITIVE,
                domain="d1", properties={"a": 1},
            ) for i in range(5)
        ]
        result = m.assess(concept, existing)
        assert 0.0 <= result.novelty <= 1.0
        assert 0.0 <= result.value <= 1.0
        assert 0.0 <= result.surprise <= 1.0
        assert 0.0 <= result.overall_score <= 1.0


# ============================================================
# 2. CombinatorialCreativity 测试
# ============================================================

class TestCombinatorialCreativity:
    """组合创新测试"""

    def test_creation(self):
        """创建"""
        cc = CombinatorialCreativity()
        assert cc is not None

    def test_combine_basic(self):
        """基本组合"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d1", {"b": 2})
        result = cc.combine([c1, c2])
        assert result is not None
        assert result.concept_type == ConceptType.COMPOUND

    def test_combine_cross_domain_emergent(self):
        """跨域组合产生涌现"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d2", {"b": 2})
        result = cc.combine([c1, c2])
        assert result is not None
        assert result.concept_type == ConceptType.EMERGENT
        assert result.properties.get("cross_domain") is True
        assert result.properties.get("emergent") is True

    def test_combine_single_concept(self):
        """单概念组合返回None"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1})
        assert cc.combine([c1]) is None

    def test_combine_empty(self):
        """空组合返回None"""
        cc = CombinatorialCreativity()
        assert cc.combine([]) is None

    def test_combine_merges_properties(self):
        """组合合并属性"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d1", {"c": 3, "d": 4})
        result = cc.combine([c1, c2])
        assert result is not None
        assert result.properties.get("a") == 1
        assert result.properties.get("b") == 2
        assert result.properties.get("c") == 3
        assert result.properties.get("d") == 4

    def test_combine_sets_source_concepts(self):
        """组合设置来源概念"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d1", {"b": 2})
        result = cc.combine([c1, c2])
        assert result is not None
        assert "c1" in result.source_concepts
        assert "c2" in result.source_concepts

    def test_combine_assesses_creativity(self):
        """组合评估创造力"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d1", {"b": 2})
        result = cc.combine([c1, c2])
        assert result is not None
        assert result.novelty_score >= 0.0
        assert result.value_score >= 0.0
        assert result.surprise_score >= 0.0

    def test_generate_combinations(self):
        """生成多个组合"""
        cc = CombinatorialCreativity()
        concepts = [
            Concept(f"c{i}", f"C{i}", ConceptType.PRIMITIVE, "d1", {f"p{i}": i})
            for i in range(4)
        ]
        results = cc.generate_combinations(concepts, max_count=10)
        # C(4,2) = 6 个组合
        assert len(results) == 6

    def test_generate_combinations_max_count(self):
        """限制最大组合数"""
        cc = CombinatorialCreativity()
        concepts = [
            Concept(f"c{i}", f"C{i}", ConceptType.PRIMITIVE, "d1", {f"p{i}": i})
            for i in range(5)
        ]
        results = cc.generate_combinations(concepts, max_count=3)
        assert len(results) == 3

    def test_generate_combinations_insufficient(self):
        """概念不足时返回空"""
        cc = CombinatorialCreativity()
        concepts = [Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1})]
        results = cc.generate_combinations(concepts)
        assert results == []

    def test_find_emergent_properties_cross_domain(self):
        """跨域涌现属性"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d2", {"b": 2})
        emergent = cc.find_emergent_properties([c1, c2])
        assert emergent.get("cross_domain") is True
        assert emergent.get("emergent") is True
        assert emergent.get("domain_count") == 2

    def test_find_emergent_properties_same_domain(self):
        """同域无跨域涌现"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d1", {"b": 2})
        emergent = cc.find_emergent_properties([c1, c2])
        assert "cross_domain" not in emergent

    def test_find_emergent_properties_common(self):
        """共同属性"""
        cc = CombinatorialCreativity()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1, "shared": True})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d1", {"b": 2, "shared": True})
        emergent = cc.find_emergent_properties([c1, c2])
        assert "shared" in emergent.get("common_properties", [])


# ============================================================
# 3. AnalogicalReasoning 测试
# ============================================================

class TestAnalogicalReasoning:
    """类比推理测试"""

    def test_creation(self):
        """创建"""
        ar = AnalogicalReasoning()
        assert ar is not None

    def test_find_analogy_basic(self):
        """基本类比"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"a": 1, "b": 2, "c": 3})
        mapping = ar.find_analogy(source, target)
        assert mapping is not None
        assert mapping.source_concept_id == "s1"
        assert mapping.target_concept_id == "t1"

    def test_find_analogy_no_common(self):
        """无共同属性返回None"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"b": 2})
        mapping = ar.find_analogy(source, target)
        assert mapping is None

    def test_find_analogy_similarity(self):
        """相似度计算（阶段187：结构映射增强后相似度计算方式变化）"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"a": 1, "b": 2})
        mapping = ar.find_analogy(source, target)
        assert mapping is not None
        # 阶段187：结构映射综合相似度（属性+关系+系统性）
        assert mapping.similarity > 0.5

    def test_find_analogy_partial_similarity(self):
        """部分相似度（阶段187：结构映射增强后相似度计算方式变化）"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"a": 1, "c": 3})
        mapping = ar.find_analogy(source, target)
        assert mapping is not None
        # 阶段187：部分匹配的相似度应大于0
        assert mapping.similarity > 0.0

    def test_find_analogy_mapped_properties(self):
        """映射属性"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"a": 1, "b": 2})
        mapping = ar.find_analogy(source, target)
        assert mapping is not None
        assert "a" in mapping.mapped_properties
        assert "b" in mapping.mapped_properties

    def test_find_analogy_transferable_relations(self):
        """可迁移关系（阶段187：transferable_relations可能包含关系类型）"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"a": 1, "b": 2})
        mapping = ar.find_analogy(source, target)
        assert mapping is not None
        # 阶段187：可迁移关系可能包含关系类型（如"less_than"）或属性名
        assert len(mapping.transferable_relations) > 0

    def test_transfer_structure_basic(self):
        """基本结构迁移"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2, "extra": True})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"a": 1, "b": 2})
        mapping = ar.find_analogy(source, target)
        assert mapping is not None
        result = ar.transfer_structure(source, target, mapping)
        assert result is not None
        assert result.concept_type == ConceptType.ANALOGY
        # extra属性应该被迁移
        assert result.properties.get("extra") is True

    def test_transfer_structure_keeps_target_props(self):
        """迁移保留目标属性"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "extra": True})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"a": 1, "target_prop": "yes"})
        mapping = ar.find_analogy(source, target)
        assert mapping is not None
        result = ar.transfer_structure(source, target, mapping)
        assert result is not None
        assert result.properties.get("target_prop") == "yes"

    def test_transfer_structure_source_concepts(self):
        """迁移设置来源概念"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "extra": True})
        target = Concept("t1", "目标", ConceptType.PRIMITIVE, "d2", {"a": 1})
        mapping = ar.find_analogy(source, target)
        result = ar.transfer_structure(source, target, mapping)
        assert result is not None
        assert "s1" in result.source_concepts
        assert "t1" in result.source_concepts

    def test_find_analogies_in_domain(self):
        """在候选中寻找类比"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        candidates = [
            Concept("c1", "C1", ConceptType.PRIMITIVE, "d2", {"a": 1, "b": 2}),
            Concept("c2", "C2", ConceptType.PRIMITIVE, "d3", {"x": 1}),
            Concept("c3", "C3", ConceptType.PRIMITIVE, "d4", {"a": 1, "b": 2, "c": 3}),
        ]
        analogies = ar.find_analogies_in_domain(source, candidates, min_similarity=0.2)
        # c1和c3有共同属性，c2没有
        assert len(analogies) == 2
        # 按相似度排序
        assert analogies[0].similarity >= analogies[1].similarity

    def test_find_analogies_in_domain_min_similarity(self):
        """最小相似度过滤"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2, "c": 3})
        candidates = [
            Concept("c1", "C1", ConceptType.PRIMITIVE, "d2", {"a": 1}),  # 相似度低
        ]
        analogies = ar.find_analogies_in_domain(source, candidates, min_similarity=0.9)
        assert len(analogies) == 0

    def test_find_analogies_excludes_self(self):
        """寻找类比排除自己"""
        ar = AnalogicalReasoning()
        source = Concept("s1", "源", ConceptType.PRIMITIVE, "d1", {"a": 1})
        candidates = [source]
        analogies = ar.find_analogies_in_domain(source, candidates)
        assert len(analogies) == 0


# ============================================================
# 4. DivergentThinking 测试
# ============================================================

class TestDivergentThinking:
    """发散思维测试"""

    def test_creation(self):
        """创建"""
        dt = DivergentThinking()
        assert dt is not None

    def test_generate_ideas_basic(self):
        """基本想法生成"""
        random.seed(42)
        dt = DivergentThinking()
        seed = Concept("s1", "种子", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        ideas = dt.generate_ideas(seed, count=3)
        assert len(ideas) == 3

    def test_generate_ideas_empty_seed(self):
        """空种子无想法"""
        dt = DivergentThinking()
        seed = Concept("s1", "种子", ConceptType.PRIMITIVE, "d1", {})
        ideas = dt.generate_ideas(seed, count=3)
        assert ideas == []

    def test_mutate_concept_basic(self):
        """基本变异"""
        random.seed(42)
        dt = DivergentThinking()
        concept = Concept("c1", "C", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2, "c": 3})
        mutated = dt.mutate_concept(concept, [])
        assert mutated is not None
        assert mutated.source_concepts == ["c1"]

    def test_mutate_concept_empty_props(self):
        """无属性变异返回None"""
        dt = DivergentThinking()
        concept = Concept("c1", "C", ConceptType.PRIMITIVE, "d1", {})
        assert dt.mutate_concept(concept, []) is None

    def test_mutate_concept_assesses(self):
        """变异评估创造力"""
        random.seed(42)
        dt = DivergentThinking()
        concept = Concept("c1", "C", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        mutated = dt.mutate_concept(concept, [])
        assert mutated is not None
        assert mutated.novelty_score >= 0.0
        assert mutated.value_score >= 0.0
        assert mutated.surprise_score >= 0.0

    def test_cross_pollinate_basic(self):
        """基本交叉授粉"""
        dt = DivergentThinking()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2, "c": 3, "d": 4})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d2", {"w": 1, "x": 2, "y": 3, "z": 4})
        result = dt.cross_pollinate(c1, c2)
        assert result is not None
        assert result.properties.get("cross_pollinated") is True

    def test_cross_pollinate_cross_domain(self):
        """跨域交叉授粉"""
        dt = DivergentThinking()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2, "c": 3, "d": 4})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d2", {"w": 1, "x": 2, "y": 3, "z": 4})
        result = dt.cross_pollinate(c1, c2)
        assert result is not None
        assert result.concept_type == ConceptType.EMERGENT
        assert result.properties.get("cross_domain") is True

    def test_cross_pollinate_same_domain(self):
        """同域交叉授粉"""
        dt = DivergentThinking()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2, "c": 3, "d": 4})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d1", {"w": 1, "x": 2, "y": 3, "z": 4})
        result = dt.cross_pollinate(c1, c2)
        assert result is not None
        assert result.concept_type == ConceptType.COMPOUND

    def test_cross_pollinate_source_concepts(self):
        """交叉授粉来源"""
        dt = DivergentThinking()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d2", {"c": 3, "d": 4})
        result = dt.cross_pollinate(c1, c2)
        assert result is not None
        assert "c1" in result.source_concepts
        assert "c2" in result.source_concepts

    def test_cross_pollinate_assesses(self):
        """交叉授粉评估"""
        dt = DivergentThinking()
        c1 = Concept("c1", "A", ConceptType.PRIMITIVE, "d1", {"a": 1, "b": 2, "c": 3, "d": 4})
        c2 = Concept("c2", "B", ConceptType.PRIMITIVE, "d2", {"w": 1, "x": 2, "y": 3, "z": 4})
        result = dt.cross_pollinate(c1, c2)
        assert result is not None
        assert result.novelty_score >= 0.0
        assert result.value_score >= 0.0


# ============================================================
# 5. Enhanced Engine 测试
# ============================================================

class TestEnhancedEngine:
    """增强引擎测试"""

    def test_engine_creation(self):
        """引擎创建"""
        eng = CreativityFirstPrinciplesEngine("test")
        assert eng.engine_id == "test"

    def test_get_creativity_metrics(self):
        """获取度量引擎"""
        eng = CreativityFirstPrinciplesEngine()
        m = eng.get_creativity_metrics()
        assert isinstance(m, CreativityMetrics)

    def test_get_combinatorial_engine(self):
        """获取组合引擎"""
        eng = CreativityFirstPrinciplesEngine()
        cc = eng.get_combinatorial_engine()
        assert isinstance(cc, CombinatorialCreativity)

    def test_get_analogical_engine(self):
        """获取类比引擎"""
        eng = CreativityFirstPrinciplesEngine()
        ar = eng.get_analogical_engine()
        assert isinstance(ar, AnalogicalReasoning)

    def test_get_divergent_engine(self):
        """获取发散引擎"""
        eng = CreativityFirstPrinciplesEngine()
        dt = eng.get_divergent_engine()
        assert isinstance(dt, DivergentThinking)

    def test_assess_creativity(self):
        """评估创造力"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1, "b": 2})
        assessment = eng.assess_creativity("c1")
        assert assessment is not None
        assert isinstance(assessment, CreativityAssessment)

    def test_assess_creativity_nonexistent(self):
        """评估不存在的概念"""
        eng = CreativityFirstPrinciplesEngine()
        assert eng.assess_creativity("nonexistent") is None

    def test_combine_with_assessment(self):
        """带评估的组合"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d2", properties={"b": 2})
        result = eng.combine_with_assessment(["c1", "c2"])
        assert result is not None
        assert result.novelty_score >= 0.0
        assert result.value_score >= 0.0

    def test_combine_with_assessment_nonexistent(self):
        """组合不存在的概念"""
        eng = CreativityFirstPrinciplesEngine()
        result = eng.combine_with_assessment(["nonexistent"])
        assert result is None

    def test_generate_combinations(self):
        """生成多个组合"""
        eng = CreativityFirstPrinciplesEngine()
        for i in range(4):
            eng.add_concept(f"c{i}", f"C{i}", "d1", properties={f"p{i}": i})
        results = eng.generate_combinations(["c0", "c1", "c2", "c3"], max_count=10)
        assert len(results) == 6  # C(4,2)

    def test_analogical_reason(self):
        """类比推理接口"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("s1", "源", "d1", properties={"a": 1, "b": 2})
        eng.add_concept("t1", "目标", "d2", properties={"a": 1, "b": 2})
        mapping = eng.analogical_reason("s1", "t1")
        assert mapping is not None
        assert isinstance(mapping, AnalogyMapping)

    def test_analogical_reason_nonexistent(self):
        """类比推理不存在概念"""
        eng = CreativityFirstPrinciplesEngine()
        assert eng.analogical_reason("nonexistent", "also") is None

    def test_transfer_analogy(self):
        """迁移类比"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("s1", "源", "d1", properties={"a": 1, "b": 2, "extra": True})
        eng.add_concept("t1", "目标", "d2", properties={"a": 1, "b": 2})
        result = eng.transfer_analogy("s1", "t1")
        assert result is not None
        assert result.concept_type == ConceptType.ANALOGY

    def test_transfer_analogy_no_common(self):
        """无共同属性迁移返回None"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("s1", "源", "d1", properties={"a": 1})
        eng.add_concept("t1", "目标", "d2", properties={"b": 2})
        assert eng.transfer_analogy("s1", "t1") is None

    def test_find_analogies_for(self):
        """寻找所有类比"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("s1", "源", "d1", properties={"a": 1, "b": 2})
        eng.add_concept("c1", "C1", "d2", properties={"a": 1, "b": 2})
        eng.add_concept("c2", "C2", "d3", properties={"x": 1})
        analogies = eng.find_analogies_for("s1", min_similarity=0.2)
        assert len(analogies) == 1
        assert analogies[0].target_concept_id == "c1"

    def test_find_analogies_for_nonexistent(self):
        """不存在源概念"""
        eng = CreativityFirstPrinciplesEngine()
        assert eng.find_analogies_for("nonexistent") == []

    def test_diverge(self):
        """发散思维接口"""
        random.seed(42)
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("s1", "种子", "d1", properties={"a": 1, "b": 2, "c": 3})
        ideas = eng.diverge("s1", count=3)
        assert len(ideas) == 3

    def test_diverge_nonexistent(self):
        """发散不存在种子"""
        eng = CreativityFirstPrinciplesEngine()
        assert eng.diverge("nonexistent") == []

    def test_cross_pollinate(self):
        """交叉授粉接口"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1, "b": 2, "c": 3, "d": 4})
        eng.add_concept("c2", "B", "d2", properties={"w": 1, "x": 2, "y": 3, "z": 4})
        result = eng.cross_pollinate("c1", "c2")
        assert result is not None
        assert result.properties.get("cross_pollinated") is True

    def test_cross_pollinate_nonexistent(self):
        """交叉授粉不存在概念"""
        eng = CreativityFirstPrinciplesEngine()
        assert eng.cross_pollinate("nonexistent", "also") is None


# ============================================================
# 6. 向后兼容性测试
# ============================================================

class TestBackwardCompat:
    """向后兼容性测试"""

    def test_concept_type_enum(self):
        """概念类型枚举"""
        assert ConceptType.PRIMITIVE.value == "primitive"
        assert ConceptType.COMPOUND.value == "compound"
        assert ConceptType.ANALOGY.value == "analogy"
        assert ConceptType.EMERGENT.value == "emergent"

    def test_creativity_metric_type_enum(self):
        """创造力度量类型枚举"""
        assert CreativityMetricType.NOVELTY.value == "novelty"
        assert CreativityMetricType.VALUE.value == "value"
        assert CreativityMetricType.SURPRISE.value == "surprise"
        assert CreativityMetricType.USEFULNESS.value == "usefulness"

    def test_concept_dataclass(self):
        """概念数据类"""
        c = Concept("c1", "test", ConceptType.PRIMITIVE, "d1")
        assert c.concept_id == "c1"
        assert c.name == "test"
        assert c.concept_type == ConceptType.PRIMITIVE
        assert c.domain == "d1"
        assert c.properties == {}
        assert c.source_concepts == []
        assert c.novelty_score == 0.0
        assert c.value_score == 0.0
        assert c.surprise_score == 0.0

    def test_combination_dataclass(self):
        """组合数据类"""
        comb = Combination(combination_id="comb1", concept_ids=["c1", "c2"])
        assert comb.combination_id == "comb1"
        assert comb.concept_ids == ["c1", "c2"]
        assert comb.result_concept is None
        assert comb.timestamp > 0

    def test_add_concept(self):
        """添加概念（向后兼容）"""
        eng = CreativityFirstPrinciplesEngine()
        c = eng.add_concept("c1", "test", "d1", properties={"a": 1})
        assert c.concept_id == "c1"
        assert c.name == "test"
        assert c.domain == "d1"
        assert c.properties == {"a": 1}

    def test_add_concept_no_properties(self):
        """添加概念无属性"""
        eng = CreativityFirstPrinciplesEngine()
        c = eng.add_concept("c1", "test", "d1")
        assert c.properties == {}

    def test_combine_backward_compat(self):
        """组合（向后兼容）"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d1", properties={"b": 2})
        result = eng.combine(["c1", "c2"])
        assert result is not None
        assert result.concept_type == ConceptType.COMPOUND

    def test_combine_nonexistent(self):
        """组合不存在的概念"""
        eng = CreativityFirstPrinciplesEngine()
        assert eng.combine(["nonexistent"]) is None

    def test_combine_cross_domain_emergent(self):
        """跨域组合涌现"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d2", properties={"b": 2})
        result = eng.combine(["c1", "c2"])
        assert result is not None
        assert result.concept_type == ConceptType.EMERGENT

    def test_find_analogy_backward_compat(self):
        """寻找类比（向后兼容）"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1, "b": 2})
        eng.add_concept("c2", "B", "d2", properties={"a": 1, "b": 2})
        analogies = eng.find_analogy("d1", "d2")
        assert len(analogies) > 0
        assert analogies[0]["source"] == "c1"
        assert analogies[0]["target"] == "c2"

    def test_find_analogy_no_match(self):
        """寻找类比无匹配"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d2", properties={"b": 2})
        analogies = eng.find_analogy("d1", "d2")
        assert analogies == []

    def test_generate_novel_combination(self):
        """生成新颖组合（向后兼容）"""
        random.seed(42)
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d1", properties={"b": 2})
        result = eng.generate_novel_combination()
        assert result is not None

    def test_generate_novel_combination_insufficient(self):
        """概念不足"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1")
        assert eng.generate_novel_combination() is None

    def test_generate_novel_combination_domain_filter(self):
        """域过滤"""
        random.seed(42)
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d1", properties={"b": 2})
        eng.add_concept("c3", "C", "d2", properties={"c": 3})
        result = eng.generate_novel_combination(domain_filter="d1")
        assert result is not None

    def test_get_stats(self):
        """获取统计"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1")
        eng.add_concept("c2", "B", "d1")
        eng.combine(["c1", "c2"])
        stats = eng.get_stats()
        assert stats["concepts_count"] == 3  # 2原始 + 1组合
        assert stats["combinations_count"] == 1
        assert stats["cycle_count"] == 1

    def test_clear(self):
        """清除"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1")
        eng.combine(["c1"])
        eng.clear()
        stats = eng.get_stats()
        assert stats["concepts_count"] == 0
        assert stats["combinations_count"] == 0

    def test_create_creativity_engine(self):
        """创建引擎便捷函数"""
        eng = create_creativity_engine("test")
        assert isinstance(eng, CreativityFirstPrinciplesEngine)
        assert eng.engine_id == "test"

    def test_create_concept(self):
        """创建概念便捷函数"""
        c = create_concept("c1", "test", "d1", {"a": 1})
        assert isinstance(c, Concept)
        assert c.concept_id == "c1"
        assert c.concept_type == ConceptType.PRIMITIVE

    def test_create_creativity_metrics(self):
        """创建度量引擎便捷函数"""
        m = create_creativity_metrics()
        assert isinstance(m, CreativityMetrics)


# ============================================================
# 7. 并发安全测试
# ============================================================

class TestConcurrency:
    """并发安全测试"""

    def test_concurrent_add_concepts(self):
        """并发添加概念"""
        eng = CreativityFirstPrinciplesEngine()
        errors = []

        def worker(idx):
            try:
                for i in range(20):
                    eng.add_concept(f"c_{idx}_{i}", f"C{idx}_{i}", "d1", {"p": i})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = eng.get_stats()
        assert stats["concepts_count"] == 100

    def test_concurrent_combine(self):
        """并发组合"""
        random.seed(42)
        eng = CreativityFirstPrinciplesEngine()
        for i in range(10):
            eng.add_concept(f"c{i}", f"C{i}", "d1", {f"p{i}": i})

        results = []
        results_lock = threading.Lock()
        errors = []

        def worker(i):
            try:
                random.seed(i)
                result = eng.combine([f"c{i}", f"c{(i+1) % 10}"])
                with results_lock:
                    results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_assess(self):
        """并发评估"""
        eng = CreativityFirstPrinciplesEngine()
        for i in range(10):
            eng.add_concept(f"c{i}", f"C{i}", "d1", {f"p{i}": i})

        results = []
        results_lock = threading.Lock()
        errors = []

        def worker(i):
            try:
                r = eng.assess_creativity(f"c{i}")
                with results_lock:
                    results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


# ============================================================
# 8. 性能测试
# ============================================================

class TestPerformance:
    """性能测试"""

    def test_large_concept_set(self):
        """大概念集性能"""
        eng = CreativityFirstPrinciplesEngine()
        start = time.time()
        for i in range(1000):
            eng.add_concept(f"c{i}", f"C{i}", "d1", {f"p{i}": i, "common": True})
        elapsed = time.time() - start
        assert elapsed < 5.0

    def test_combine_performance(self):
        """组合性能"""
        eng = CreativityFirstPrinciplesEngine()
        for i in range(100):
            eng.add_concept(f"c{i}", f"C{i}", "d1", {f"p{i}": i})
        start = time.time()
        for i in range(50):
            eng.combine([f"c{i}", f"c{(i+1) % 100}"])
        elapsed = time.time() - start
        assert elapsed < 5.0

    def test_assess_performance(self):
        """评估性能"""
        eng = CreativityFirstPrinciplesEngine()
        for i in range(100):
            eng.add_concept(f"c{i}", f"C{i}", "d1", {f"p{i}": i})
        start = time.time()
        for i in range(100):
            eng.assess_creativity(f"c{i}")
        elapsed = time.time() - start
        assert elapsed < 5.0

    def test_analogy_performance(self):
        """类比推理性能"""
        eng = CreativityFirstPrinciplesEngine()
        for i in range(50):
            eng.add_concept(f"c{i}", f"C{i}", "d1", {"a": 1, "b": 2, f"p{i}": i})
        start = time.time()
        for i in range(50):
            eng.analogical_reason(f"c0", f"c{i}")
        elapsed = time.time() - start
        assert elapsed < 5.0


# ============================================================
# 9. 异常容错测试
# ============================================================

class TestErrorTolerance:
    """异常容错测试"""

    def test_empty_engine_operations(self):
        """空引擎操作"""
        eng = CreativityFirstPrinciplesEngine()
        assert eng.assess_creativity("nonexistent") is None
        assert eng.combine(["nonexistent"]) is None
        assert eng.find_analogies_for("nonexistent") == []
        assert eng.diverge("nonexistent") == []
        assert eng.cross_pollinate("nonexistent", "also") is None

    def test_combine_empty_list(self):
        """组合空列表"""
        eng = CreativityFirstPrinciplesEngine()
        assert eng.combine([]) is None

    def test_combine_single_concept(self):
        """组合单概念"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1")
        # 单概念组合在原接口中可能返回None或结果
        result = eng.combine(["c1"])
        # 原接口允许单概念组合
        assert result is not None

    def test_assess_concept_no_properties(self):
        """评估无属性概念"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1")
        assessment = eng.assess_creativity("c1")
        assert assessment is not None
        assert assessment.novelty == 1.0  # 无已有概念
        assert assessment.value == 0.0  # 无属性

    def test_analogy_no_common_properties(self):
        """类比无共同属性"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("s1", "源", "d1", properties={"a": 1})
        eng.add_concept("t1", "目标", "d2", properties={"b": 2})
        mapping = eng.analogical_reason("s1", "t1")
        assert mapping is None

    def test_diverge_empty_seed(self):
        """发散空种子"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1")  # 无属性
        ideas = eng.diverge("c1", count=3)
        assert ideas == []

    def test_clear_resets_engines(self):
        """清除重置所有引擎"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", {"a": 1})
        eng.combine(["c1"])
        eng.clear()
        # 引擎应该重置
        assert eng.get_combinatorial_engine() is not None
        assert eng.get_analogical_engine() is not None
        assert eng.get_divergent_engine() is not None

    def test_generate_combinations_nonexistent(self):
        """生成组合不存在概念"""
        eng = CreativityFirstPrinciplesEngine()
        results = eng.generate_combinations(["nonexistent"], max_count=5)
        assert results == []

    def test_generate_combinations_insufficient(self):
        """生成组合概念不足"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1")
        results = eng.generate_combinations(["c1"], max_count=5)
        assert results == []

    def test_find_analogies_for_no_candidates(self):
        """寻找类比无候选"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", {"a": 1})
        analogies = eng.find_analogies_for("c1")
        assert analogies == []


# ============================================================
# 10. 第一性原理验证测试
# ============================================================

class TestFirstPrinciples:
    """第一性原理验证测试"""

    def test_creativity_is_combination(self):
        """第一性原理：创造是旧元素的新组合"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d2", properties={"b": 2})
        result = eng.combine(["c1", "c2"])
        # 组合应该包含两个来源概念
        assert "c1" in result.source_concepts
        assert "c2" in result.source_concepts

    def test_cross_domain_emergence(self):
        """第一性原理：跨域组合产生涌现"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d2", properties={"b": 2})
        result = eng.combine(["c1", "c2"])
        # 跨域组合应该是涌现的
        assert result.concept_type == ConceptType.EMERGENT
        assert result.properties.get("emergent") is True

    def test_analogy_is_structure_mapping(self):
        """第一性原理：类比是结构映射"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("s1", "鸟", "动物", properties={"flies": True, "wings": 2})
        eng.add_concept("t1", "飞机", "机器", properties={"flies": True, "wings": 2, "engine": True})
        mapping = eng.analogical_reason("s1", "t1")
        # 应该映射共同属性
        assert mapping is not None
        assert "flies" in mapping.mapped_properties
        assert "wings" in mapping.mapped_properties

    def test_divergence_produces_variety(self):
        """第一性原理：发散产生多样性"""
        random.seed(42)
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("s1", "种子", "d1", properties={"a": 1, "b": 2, "c": 3, "d": 4})
        ideas = eng.diverge("s1", count=5)
        # 应该产生多个想法
        assert len(ideas) == 5
        # 想法应该有不同ID
        ids = [idea.concept_id for idea in ideas]
        assert len(set(ids)) == 5

    def test_creativity_is_quantifiable(self):
        """第一性原理：创造力可量化"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1, "b": 2})
        assessment = eng.assess_creativity("c1")
        # 应该有量化的得分
        assert 0.0 <= assessment.novelty <= 1.0
        assert 0.0 <= assessment.value <= 1.0
        assert 0.0 <= assessment.surprise <= 1.0
        assert 0.0 <= assessment.overall_score <= 1.0

    def test_emergence_whole_greater_than_parts(self):
        """第一性原理：涌现，整体大于部分"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1})
        eng.add_concept("c2", "B", "d2", properties={"b": 2})
        result = eng.combine(["c1", "c2"])
        # 组合应该有比单个概念更多的属性
        assert len(result.properties) > len(eng._concepts["c1"].properties)
        assert len(result.properties) > len(eng._concepts["c2"].properties)

    def test_cross_pollination_combines_traits(self):
        """第一性原理：交叉授粉结合特征"""
        eng = CreativityFirstPrinciplesEngine()
        eng.add_concept("c1", "A", "d1", properties={"a": 1, "b": 2, "c": 3, "d": 4})
        eng.add_concept("c2", "B", "d2", properties={"w": 1, "x": 2, "y": 3, "z": 4})
        result = eng.cross_pollinate("c1", "c2")
        # 结果应该有来自两个概念的属性
        assert result is not None
        assert result.properties.get("cross_pollinated") is True
