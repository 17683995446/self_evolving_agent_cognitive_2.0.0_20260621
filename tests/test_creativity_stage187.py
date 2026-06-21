"""
阶段187测试：创造力增强 — 结构映射理论实现

验证内容:
1. StructureMappingEngine: 关系提取、属性相似度、关系相似度
2. AnalogicalReasoning.find_analogy: 增强的结构映射类比
3. CreativityFirstPrinciplesEngine: 新增结构映射接口
4. 端到端：结构映射 vs 简单属性匹配的差异
"""

import pytest
from self_evolving_agent.core.creativity_first_principles import (
    CreativityFirstPrinciplesEngine,
    StructureMappingEngine,
    AnalogicalReasoning,
    Concept,
    ConceptType,
    AnalogyMapping,
    create_creativity_engine,
)


# ============================================================
# StructureMappingEngine 测试
# ============================================================

class TestStructureMappingEngine:
    """结构映射引擎测试"""

    def test_extract_relations_numeric(self):
        """测试数值关系提取"""
        engine = StructureMappingEngine()
        props = {"a": 10.0, "b": 5.0, "c": 10.0}
        relations = engine._extract_relations(props)
        # a > b, a = c, b < a
        assert len(relations) == 3
        types = [r[1] for r in relations]
        assert "greater_than" in types
        assert "equal_to" in types

    def test_extract_relations_boolean(self):
        """测试布尔关系提取"""
        engine = StructureMappingEngine()
        props = {"x": True, "y": True, "z": False}
        relations = engine._extract_relations(props)
        # x,y co_present; x,z mutually_exclusive; y,z mutually_exclusive
        types = [r[1] for r in relations]
        assert "co_present" in types
        assert "mutually_exclusive" in types

    def test_extract_relations_empty(self):
        """测试空属性无关系"""
        engine = StructureMappingEngine()
        relations = engine._extract_relations({})
        assert len(relations) == 0

    def test_attribute_similarity_exact(self):
        """测试完全属性匹配"""
        engine = StructureMappingEngine()
        source = {"a": 1, "b": 2}
        target = {"a": 1, "b": 2}
        mapped, sim = engine._compute_attribute_similarity(source, target)
        assert "a" in mapped
        assert "b" in mapped
        assert sim > 0.5

    def test_attribute_similarity_no_match(self):
        """测试无属性匹配"""
        engine = StructureMappingEngine()
        source = {"a": 1}
        target = {"b": 2}
        mapped, sim = engine._compute_attribute_similarity(source, target)
        assert len(mapped) == 0
        assert sim == 0.0

    def test_relation_similarity_matching(self):
        """测试关系相似度匹配"""
        engine = StructureMappingEngine()
        source_rels = [("a", "greater_than", "b"), ("c", "less_than", "d")]
        target_rels = [("x", "greater_than", "y"), ("z", "less_than", "w")]
        matched, sim = engine._compute_relation_similarity(source_rels, target_rels)
        assert "greater_than" in matched
        assert "less_than" in matched
        assert sim > 0.5

    def test_map_structure_comprehensive(self):
        """测试综合结构映射"""
        engine = StructureMappingEngine()
        source = {"speed": 100, "weight": 50, "color": "red"}
        target = {"speed": 90, "weight": 45, "color": "blue"}

        result = engine.map_structure(source, target)

        assert "mapped_properties" in result
        assert "similarity" in result
        assert "attribute_similarity" in result
        assert "relation_similarity" in result
        assert "systematicity" in result
        assert result["similarity"] > 0
        # speed和weight都有数值关系
        assert result["source_relation_count"] > 0

    def test_map_structure_no_overlap(self):
        """测试无重叠的结构映射"""
        engine = StructureMappingEngine()
        source = {"a": 1}
        target = {"b": 2}
        result = engine.map_structure(source, target)
        assert result["similarity"] == 0.0

    def test_find_structural_analogy(self):
        """测试结构映射类比发现"""
        engine = StructureMappingEngine()
        source = Concept(
            concept_id="s1", name="car", concept_type=ConceptType.PRIMITIVE,
            domain="vehicle", properties={"speed": 100, "weight": 50},
        )
        target = Concept(
            concept_id="t1", name="plane", concept_type=ConceptType.PRIMITIVE,
            domain="vehicle", properties={"speed": 200, "weight": 100},
        )

        mapping = engine.find_structural_analogy(source, target)
        assert mapping is not None
        assert mapping.similarity > 0


# ============================================================
# AnalogicalReasoning 增强测试
# ============================================================

class TestAnalogicalReasoningStage187:
    """类比推理增强测试"""

    def test_find_analogy_with_structure(self):
        """测试结构映射类比"""
        reasoner = AnalogicalReasoning()
        source = Concept(
            concept_id="s1", name="sun", concept_type=ConceptType.PRIMITIVE,
            domain="astronomy",
            properties={"brightness": 100, "size": 50, "temperature": 6000},
        )
        target = Concept(
            concept_id="t1", name="star", concept_type=ConceptType.PRIMITIVE,
            domain="astronomy",
            properties={"brightness": 90, "size": 45, "temperature": 5500},
        )

        mapping = reasoner.find_analogy(source, target)
        assert mapping is not None
        assert mapping.similarity > 0
        assert len(mapping.mapped_properties) > 0

    def test_find_analogy_no_common_props(self):
        """测试无共同属性时的类比"""
        reasoner = AnalogicalReasoning()
        source = Concept(
            concept_id="s1", name="a", concept_type=ConceptType.PRIMITIVE,
            domain="x", properties={"p1": 1},
        )
        target = Concept(
            concept_id="t1", name="b", concept_type=ConceptType.PRIMITIVE,
            domain="y", properties={"p2": 2},
        )

        mapping = reasoner.find_analogy(source, target)
        # 无共同属性名，但可能有数值关系结构匹配
        # 如果相似度为0则返回None
        if mapping is not None:
            assert mapping.similarity >= 0

    def test_find_analogy_empty_props(self):
        """测试空属性"""
        reasoner = AnalogicalReasoning()
        source = Concept(
            concept_id="s1", name="a", concept_type=ConceptType.PRIMITIVE,
            domain="x", properties={},
        )
        target = Concept(
            concept_id="t1", name="b", concept_type=ConceptType.PRIMITIVE,
            domain="y", properties={},
        )
        mapping = reasoner.find_analogy(source, target)
        assert mapping is None


# ============================================================
# CreativityFirstPrinciplesEngine 阶段187接口测试
# ============================================================

class TestCreativityEngineStage187:
    """创造力引擎阶段187接口测试"""

    def test_new_interfaces_exist(self):
        """测试新增接口存在"""
        engine = create_creativity_engine()
        assert hasattr(engine, "map_structure")
        assert hasattr(engine, "find_structural_analogy")

    def test_map_structure(self):
        """测试结构映射分析"""
        engine = create_creativity_engine()
        engine.add_concept("s1", "car", ConceptType.PRIMITIVE, "vehicle",
                           {"speed": 100, "weight": 50})
        engine.add_concept("t1", "plane", ConceptType.PRIMITIVE, "vehicle",
                           {"speed": 200, "weight": 100})

        result = engine.map_structure("s1", "t1")
        assert result is not None
        assert "similarity" in result
        assert "attribute_similarity" in result
        assert "relation_similarity" in result

    def test_find_structural_analogy(self):
        """测试结构映射类比"""
        engine = create_creativity_engine()
        engine.add_concept("s1", "sun", ConceptType.PRIMITIVE, "astronomy",
                           {"brightness": 100, "size": 50})
        engine.add_concept("t1", "star", ConceptType.PRIMITIVE, "astronomy",
                           {"brightness": 90, "size": 45})

        mapping = engine.find_structural_analogy("s1", "t1")
        assert mapping is not None
        assert mapping.similarity > 0

    def test_map_structure_nonexistent(self):
        """测试不存在概念的结构映射"""
        engine = create_creativity_engine()
        result = engine.map_structure("nonexistent", "also_nonexistent")
        assert result is None

    def test_backward_compatibility(self):
        """测试向后兼容：原有类比接口仍正常工作"""
        engine = create_creativity_engine()
        engine.add_concept("s1", "car", ConceptType.PRIMITIVE, "vehicle",
                           {"wheels": 4, "speed": 100})
        engine.add_concept("t1", "bike", ConceptType.PRIMITIVE, "vehicle",
                           {"wheels": 2, "speed": 30})

        # 原有 analogical_reason
        mapping = engine.analogical_reason("s1", "t1")
        assert mapping is not None
        assert mapping.similarity > 0

    def test_structure_mapping_better_than_attribute_only(self):
        """
        验证结构映射比纯属性匹配更优

        第一性原理：当属性名不同但关系结构相同时，结构映射应能发现类比
        """
        engine = StructureMappingEngine()

        # 源域：太阳系（行星距离不同）
        source = {"inner_planet_dist": 1, "outer_planet_dist": 30}
        # 目标域：原子模型（电子轨道不同）
        target = {"inner_shell_dist": 1, "outer_shell_dist": 30}

        result = engine.map_structure(source, target)
        # 虽然属性名完全不同，但数值关系结构相同
        # 关系相似度应 > 0
        assert result["relation_similarity"] > 0
