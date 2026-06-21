"""
阶段122: 创造力强化 — 组合创新 + 类比推理
阶段187: 结构映射理论增强 — 关系结构映射 + 系统性

第一性原理:
1. 创造力 = 组合已有概念产生新想法（"组合产生新颖"）
2. 类比推理 = 结构映射（Gentner的结构映射理论）
3. 发散思维 vs 收敛思维
4. 创新距离 = 新想法与已有想法的差异度
5. 创造力可量化：新颖性 + 价值性 + 惊讶度
6. 涌现：整体大于部分之和
7. 结构映射（阶段187）：类比基于关系结构而非表面属性
8. 系统性原则（阶段187）：连通的关系结构优先映射

强化内容:
- CreativityMetrics: 创造力度量（新颖性、价值性、惊讶度）
- CombinatorialCreativity: 组合创新引擎
- AnalogicalReasoning: 类比推理（结构映射）
- DivergentThinking: 发散思维（变异、交叉授粉）
- StructureMappingEngine（阶段187）: Gentner结构映射引擎
- CreativityFirstPrinciplesEngine: 增强

向后兼容: 保留原有 Concept / Combination / ConceptType / 接口
"""

import threading
import time
import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from itertools import combinations


# ============================================================
# 基础枚举与数据结构（向后兼容 + 增强）
# ============================================================

class ConceptType(Enum):
    """概念类型"""
    PRIMITIVE = "primitive"        # 原始概念
    COMPOUND = "compound"          # 复合概念
    ANALOGY = "analogy"            # 类比概念
    EMERGENT = "emergent"          # 涌现概念


class CreativityMetricType(Enum):
    """创造力度量类型"""
    NOVELTY = "novelty"          # 新颖性
    VALUE = "value"              # 价值性
    SURPRISE = "surprise"        # 惊讶度
    USEFULNESS = "usefulness"    # 有用性


@dataclass
class Concept:
    """概念（增强）"""
    concept_id: str
    name: str
    concept_type: ConceptType
    domain: str
    properties: Dict[str, Any] = field(default_factory=dict)
    source_concepts: List[str] = field(default_factory=list)
    novelty_score: float = 0.0      # 新颖度 [0,1]
    value_score: float = 0.0        # 价值度 [0,1]（阶段122新增）
    surprise_score: float = 0.0     # 惊讶度 [0,1]（阶段122新增）


@dataclass
class Combination:
    """组合"""
    combination_id: str
    concept_ids: List[str]
    result_concept: Optional[Concept] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class CreativityAssessment:
    """创造力评估结果（阶段122新增）"""
    concept_id: str
    novelty: float
    value: float
    surprise: float
    overall_score: float


@dataclass
class AnalogyMapping:
    """类比映射（阶段122新增）"""
    source_concept_id: str
    target_concept_id: str
    mapped_properties: Dict[str, str] = field(default_factory=dict)  # source_prop -> target_prop
    similarity: float = 0.0
    transferable_relations: List[str] = field(default_factory=list)


# ============================================================
# 阶段122新增：创造力度量
# ============================================================

class CreativityMetrics:
    """
    创造力度量引擎

    第一性原理：
    - 新颖性 = 与已有概念的距离（属性差异）
    - 价值性 = 概念属性的丰富度与连贯性
    - 惊讶度 = 与预期偏离的程度
    - 整体创造力 = 加权组合
    """

    def __init__(self):
        self._lock = threading.RLock()

    def assess(
        self,
        concept: Concept,
        existing_concepts: List[Concept],
    ) -> CreativityAssessment:
        """评估概念的创造力"""
        novelty = self.compute_novelty(concept, existing_concepts)
        value = self.compute_value(concept)
        surprise = self.compute_surprise(concept, existing_concepts)
        # 整体得分：加权平均
        overall = 0.4 * novelty + 0.4 * value + 0.2 * surprise
        return CreativityAssessment(
            concept_id=concept.concept_id,
            novelty=novelty,
            value=value,
            surprise=surprise,
            overall_score=overall,
        )

    def compute_novelty(
        self,
        concept: Concept,
        existing_concepts: List[Concept],
    ) -> float:
        """
        计算新颖性

        第一性原理：新颖性 = 与所有已有概念的最小距离的补
        距离 = 1 - 属性重叠度
        """
        if not existing_concepts:
            return 1.0

        concept_props = set(concept.properties.keys())
        if not concept_props:
            return 0.0

        max_overlap = 0.0
        for existing in existing_concepts:
            existing_props = set(existing.properties.keys())
            if not existing_props:
                continue
            overlap = len(concept_props & existing_props) / len(concept_props | existing_props)
            if overlap > max_overlap:
                max_overlap = overlap

        novelty = 1.0 - max_overlap
        return max(0.0, min(1.0, novelty))

    def compute_value(self, concept: Concept) -> float:
        """
        计算价值性

        第一性原理：价值 = 属性丰富度 × 跨域性
        """
        props = concept.properties
        if not props:
            return 0.0

        # 属性丰富度：属性数量（归一化）
        richness = min(1.0, len(props) / 10.0)

        # 跨域性：是否跨域组合
        cross_domain = 1.0 if props.get("cross_domain", False) else 0.5

        # 涌现性：是否有涌现属性
        emergent = 1.0 if props.get("emergent", False) else 0.5

        value = richness * 0.4 + cross_domain * 0.3 + emergent * 0.3
        return max(0.0, min(1.0, value))

    def compute_surprise(
        self,
        concept: Concept,
        existing_concepts: List[Concept],
    ) -> float:
        """
        计算惊讶度

        第一性原理：惊讶度 = 与已有概念属性分布的偏离
        """
        if not existing_concepts:
            return 1.0

        # 统计已有概念中每个属性的出现频率
        prop_freq: Dict[str, float] = {}
        for existing in existing_concepts:
            for prop in existing.properties.keys():
                prop_freq[prop] = prop_freq.get(prop, 0.0) + 1.0

        total = len(existing_concepts)
        concept_props = set(concept.properties.keys())

        # 惊讶度 = 罕见属性的比例
        if not concept_props:
            return 0.0

        rare_count = 0
        for prop in concept_props:
            freq = prop_freq.get(prop, 0.0) / total
            # 频率低于0.3视为罕见
            if freq < 0.3:
                rare_count += 1

        surprise = rare_count / len(concept_props)
        return max(0.0, min(1.0, surprise))


# ============================================================
# 阶段122新增：组合创新引擎
# ============================================================

class CombinatorialCreativity:
    """
    组合创新引擎

    第一性原理：
    - 创新 = 旧元素的新组合
    - 跨域组合产生涌现属性
    - 组合空间搜索
    """

    def __init__(self, metrics: Optional[CreativityMetrics] = None):
        self._metrics = metrics or CreativityMetrics()
        self._lock = threading.RLock()
        self._combination_history: Set[Tuple[str, ...]] = set()

    def combine(
        self,
        concepts: List[Concept],
        existing_concepts: Optional[List[Concept]] = None,
    ) -> Optional[Concept]:
        """组合概念"""
        if len(concepts) < 2:
            return None

        with self._lock:
            sorted_ids = tuple(sorted(c.concept_id for c in concepts))
            is_new = sorted_ids not in self._combination_history
            self._combination_history.add(sorted_ids)

        # 合并属性
        merged_props: Dict[str, Any] = {}
        for c in concepts:
            merged_props.update(c.properties)

        # 跨域检测
        domains = set(c.domain for c in concepts)
        if len(domains) > 1:
            merged_props["cross_domain"] = True
            merged_props["emergent"] = True

        # 创建结果概念
        result = Concept(
            concept_id=f"comb_{int(time.time() * 1000)}_{random.randint(0, 9999)}",
            name=" + ".join(c.name for c in concepts),
            concept_type=ConceptType.EMERGENT if len(domains) > 1 else ConceptType.COMPOUND,
            domain="/".join(sorted(domains)),
            properties=merged_props,
            source_concepts=[c.concept_id for c in concepts],
        )

        # 评估创造力
        all_existing = list(existing_concepts or []) + [c for c in concepts]
        assessment = self._metrics.assess(result, all_existing)
        result.novelty_score = assessment.novelty
        result.value_score = assessment.value
        result.surprise_score = assessment.surprise

        return result

    def generate_combinations(
        self,
        concepts: List[Concept],
        max_count: int = 10,
        existing_concepts: Optional[List[Concept]] = None,
    ) -> List[Concept]:
        """生成所有可能的组合（C(n,2)）"""
        results: List[Concept] = []
        if len(concepts) < 2:
            return results

        # 生成所有2元组合
        for pair in combinations(concepts, 2):
            if len(results) >= max_count:
                break
            combined = self.combine(list(pair), existing_concepts)
            if combined is not None:
                results.append(combined)

        return results

    def find_emergent_properties(self, concepts: List[Concept]) -> Dict[str, Any]:
        """发现涌现属性"""
        emergent: Dict[str, Any] = {}
        domains = set(c.domain for c in concepts)
        if len(domains) > 1:
            emergent["cross_domain"] = True
            emergent["emergent"] = True
            emergent["domain_count"] = len(domains)

        # 属性交集（共同属性）
        if concepts:
            common = set(concepts[0].properties.keys())
            for c in concepts[1:]:
                common &= set(c.properties.keys())
            if common:
                emergent["common_properties"] = list(common)

        return emergent


# ============================================================
# 阶段187: 结构映射引擎（Gentner结构映射理论）
# ============================================================

class StructureMappingEngine:
    """
    结构映射引擎（阶段187）

    第一性原理（Gentner结构映射理论）：
    1. 属性匹配：表面特征（属性名/值）的相似度
    2. 关系匹配：关系结构（属性间的关联）的相似度
    3. 系统性原则：连通的关系结构比孤立属性优先映射
    4. 一对一约束：每个源域元素最多映射到一个目标域元素

    与阶段122的区别：
    - 阶段122仅基于属性名重叠（表面匹配）
    - 阶段187增加关系结构匹配和系统性评分
    """

    def __init__(self):
        self._lock = threading.RLock()

    @staticmethod
    def _extract_relations(properties: Dict[str, Any]) -> List[Tuple[str, str, Any]]:
        """
        从属性中提取关系

        第一性原理：关系 = 属性间的关联
        - 数值属性间的关系：大小、比例、顺序
        - 布尔属性间的关系：共存、互斥
        - 字符串属性间的关系：包含、相似
        """
        relations: List[Tuple[str, str, Any]] = []
        items = list(properties.items())

        for i, (key1, val1) in enumerate(items):
            for key2, val2 in items[i + 1:]:
                # 布尔关系（必须先检查，因为bool是int的子类）
                if isinstance(val1, bool) and isinstance(val2, bool):
                    if val1 == val2:
                        relations.append((key1, "co_present", key2))
                    else:
                        relations.append((key1, "mutually_exclusive", key2))
                # 数值关系
                elif isinstance(val1, (int, float)) and not isinstance(val1, bool) and \
                     isinstance(val2, (int, float)) and not isinstance(val2, bool):
                    if val1 > val2:
                        relations.append((key1, "greater_than", key2))
                    elif val1 < val2:
                        relations.append((key1, "less_than", key2))
                    else:
                        relations.append((key1, "equal_to", key2))
                # 字符串关系
                elif isinstance(val1, str) and isinstance(val2, str):
                    if val1 in val2:
                        relations.append((key1, "contained_in", key2))
                    elif val2 in val1:
                        relations.append((key2, "contained_in", key1))

        return relations

    @staticmethod
    def _compute_attribute_similarity(
        source_props: Dict[str, Any],
        target_props: Dict[str, Any],
    ) -> Tuple[Dict[str, str], float]:
        """
        计算属性相似度

        Returns:
            (mapped_attributes, attribute_similarity)
        """
        source_keys = set(source_props.keys())
        target_keys = set(target_props.keys())

        # 完全匹配的属性名
        exact_match = source_keys & target_keys

        # 值也匹配的属性
        value_match = 0
        mapped: Dict[str, str] = {}
        for key in exact_match:
            mapped[key] = key
            sv = source_props[key]
            tv = target_props[key]
            if type(sv) == type(tv):
                if sv == tv:
                    value_match += 1
                elif isinstance(sv, (int, float)) and isinstance(tv, (int, float)):
                    # 数值接近匹配
                    if abs(sv - tv) < 0.1 * max(abs(sv), abs(tv), 1):
                        value_match += 0.5

        # 相似度 = 值匹配数 / 并集大小
        union_size = len(source_keys | target_keys)
        if union_size == 0:
            return mapped, 0.0

        similarity = (len(exact_match) * 0.5 + value_match * 0.5) / union_size
        return mapped, min(1.0, similarity)

    @staticmethod
    def _compute_relation_similarity(
        source_relations: List[Tuple[str, str, Any]],
        target_relations: List[Tuple[str, str, Any]],
    ) -> Tuple[List[str], float]:
        """
        计算关系结构相似度

        第一性原理：
        - 关系类型匹配（如"greater_than"匹配"greater_than"）
        - 系统性原则：连通的关系结构得分更高

        Returns:
            (matched_relation_types, relation_similarity)
        """
        if not source_relations or not target_relations:
            return [], 0.0

        # 提取关系类型
        source_types = [r[1] for r in source_relations]
        target_types = [r[1] for r in target_relations]

        # 共同关系类型
        common_types = set(source_types) & set(target_types)
        if not common_types:
            return [], 0.0

        # 关系类型匹配数
        source_type_counts: Dict[str, int] = {}
        for t in source_types:
            source_type_counts[t] = source_type_counts.get(t, 0) + 1

        target_type_counts: Dict[str, int] = {}
        for t in target_types:
            target_type_counts[t] = target_type_counts.get(t, 0) + 1

        matched_count = 0
        for t in common_types:
            matched_count += min(source_type_counts[t], target_type_counts[t])

        # 系统性加分：多种关系类型匹配 → 更高系统性
        systematicity_bonus = min(0.2, len(common_types) * 0.05)

        # 关系相似度 = 匹配数 / 较大关系集大小 + 系统性加分
        max_relations = max(len(source_relations), len(target_relations))
        similarity = matched_count / max_relations + systematicity_bonus

        return list(common_types), min(1.0, similarity)

    def map_structure(
        self,
        source_props: Dict[str, Any],
        target_props: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        结构映射（阶段187核心方法）

        第一性原理：
        1. 提取源域和目标域的关系结构
        2. 计算属性相似度和关系相似度
        3. 系统性原则：关系相似度权重 > 属性相似度
        4. 综合相似度 = w1 × 属性相似度 + w2 × 关系相似度

        Returns:
            {
                "mapped_properties": {source_prop: target_prop},
                "similarity": 综合相似度,
                "attribute_similarity": 属性相似度,
                "relation_similarity": 关系相似度,
                "transferable_relations": 可迁移关系类型,
                "systematicity": 系统性评分,
            }
        """
        with self._lock:
            # 步骤1：属性匹配
            mapped_attrs, attr_sim = self._compute_attribute_similarity(
                source_props, target_props
            )

            # 步骤2：关系提取
            source_relations = self._extract_relations(source_props)
            target_relations = self._extract_relations(target_props)

            # 步骤3：关系匹配
            matched_types, rel_sim = self._compute_relation_similarity(
                source_relations, target_relations
            )

            # 步骤4：系统性评分
            # 系统性 = 连通的关系结构比例
            systematicity = 0.0
            if source_relations:
                # 关系类型多样性 → 系统性
                unique_types = len(set(r[1] for r in source_relations))
                systematicity = min(1.0, unique_types * 0.2)

            # 步骤5：综合相似度
            # 系统性原则：关系相似度权重更高（Gentner理论核心）
            # w_relation > w_attribute
            w_attr = 0.3
            w_rel = 0.5
            w_sys = 0.2

            overall_sim = (
                w_attr * attr_sim
                + w_rel * rel_sim
                + w_sys * systematicity
            )

            # 可迁移关系 = 匹配的关系类型
            transferable = matched_types if matched_types else list(mapped_attrs.keys())

            return {
                "mapped_properties": mapped_attrs,
                "similarity": overall_sim,
                "attribute_similarity": attr_sim,
                "relation_similarity": rel_sim,
                "transferable_relations": transferable,
                "systematicity": systematicity,
                "source_relation_count": len(source_relations),
                "target_relation_count": len(target_relations),
            }

    def find_structural_analogy(
        self,
        source: "Concept",
        target: "Concept",
    ) -> Optional[AnalogyMapping]:
        """
        基于结构映射的类比发现

        第一性原理：优先匹配关系结构而非表面属性
        """
        result = self.map_structure(source.properties, target.properties)

        if result["similarity"] <= 0:
            return None

        return AnalogyMapping(
            source_concept_id=source.concept_id,
            target_concept_id=target.concept_id,
            mapped_properties=result["mapped_properties"],
            similarity=result["similarity"],
            transferable_relations=result["transferable_relations"],
        )


# ============================================================
# 阶段122新增：类比推理（结构映射）
# ============================================================

class AnalogicalReasoning:
    """
    类比推理引擎（基于Gentner的结构映射理论）

    第一性原理：
    - 类比 = 将源域的结构关系映射到目标域
    - 结构映射保持关系，忽略表面特征
    - 相似度 = 映射的属性比例
    - 阶段187增强：关系结构映射 + 系统性原则
    """

    def __init__(self):
        self._lock = threading.RLock()
        # 阶段187：结构映射引擎
        self._structure_mapper = StructureMappingEngine()

    def find_analogy(
        self,
        source: Concept,
        target: Concept,
    ) -> Optional[AnalogyMapping]:
        """
        寻找两个概念间的类比映射（阶段187增强：结构映射）

        第一性原理：
        - 阶段122：基于属性名重叠
        - 阶段187：基于关系结构映射 + 系统性原则
        """
        source_props = set(source.properties.keys())
        target_props = set(target.properties.keys())

        # 共同属性
        common = source_props & target_props
        if not common and not source_props and not target_props:
            return None

        # 阶段187：使用结构映射引擎进行深度匹配
        structure_mapping = self._structure_mapper.map_structure(
            source.properties, target.properties
        )

        # 相似度 = 结构映射相似度（优于简单属性重叠）
        if structure_mapping["similarity"] > 0:
            similarity = structure_mapping["similarity"]
            mapped = structure_mapping["mapped_properties"]
            transferable = structure_mapping["transferable_relations"]
        else:
            # 退化为属性重叠计算
            union = source_props | target_props
            similarity = len(common) / len(union) if union else 0.0
            mapped = {p: p for p in common}
            transferable = list(common)

        if similarity <= 0.0:
            return None

        return AnalogyMapping(
            source_concept_id=source.concept_id,
            target_concept_id=target.concept_id,
            mapped_properties=mapped,
            similarity=similarity,
            transferable_relations=transferable,
        )

    def transfer_structure(
        self,
        source: Concept,
        target: Concept,
        mapping: AnalogyMapping,
    ) -> Concept:
        """
        迁移结构：将源概念的关系属性迁移到目标概念

        第一性原理：类比推理的核心是关系迁移
        """
        new_props = dict(target.properties)
        # 迁移源概念的属性（除已有共同属性外）
        for prop, value in source.properties.items():
            if prop not in mapping.mapped_properties and prop not in new_props:
                new_props[prop] = value

        return Concept(
            concept_id=f"analogy_{int(time.time() * 1000)}_{random.randint(0, 9999)}",
            name=f"{target.name}←{source.name}",
            concept_type=ConceptType.ANALOGY,
            domain=target.domain,
            properties=new_props,
            source_concepts=[source.concept_id, target.concept_id],
        )

    def find_analogies_in_domain(
        self,
        source: Concept,
        candidates: List[Concept],
        min_similarity: float = 0.2,
    ) -> List[AnalogyMapping]:
        """在候选概念中寻找类比"""
        analogies: List[AnalogyMapping] = []
        for candidate in candidates:
            if candidate.concept_id == source.concept_id:
                continue
            mapping = self.find_analogy(source, candidate)
            if mapping and mapping.similarity >= min_similarity:
                analogies.append(mapping)

        # 按相似度排序
        analogies.sort(key=lambda m: m.similarity, reverse=True)
        return analogies


# ============================================================
# 阶段122新增：发散思维
# ============================================================

class DivergentThinking:
    """
    发散思维引擎

    第一性原理：
    - 发散思维 = 从一个种子产生多个不同想法
    - 变异 = 修改属性产生新概念
    - 交叉授粉 = 融合不同概念的特征
    """

    def __init__(self, metrics: Optional[CreativityMetrics] = None):
        self._metrics = metrics or CreativityMetrics()
        self._lock = threading.RLock()

    def generate_ideas(
        self,
        seed: Concept,
        count: int = 5,
        existing_concepts: Optional[List[Concept]] = None,
    ) -> List[Concept]:
        """从种子概念生成多个想法"""
        ideas: List[Concept] = []
        for i in range(count):
            mutated = self.mutate_concept(seed, existing_concepts or [])
            if mutated is not None:
                ideas.append(mutated)
        return ideas

    def mutate_concept(
        self,
        concept: Concept,
        existing_concepts: List[Concept],
    ) -> Optional[Concept]:
        """变异概念：随机修改属性"""
        if not concept.properties:
            return None

        new_props = dict(concept.properties)
        # 随机添加一个新属性或修改现有属性
        mutation_type = random.choice(["add", "modify", "remove"])

        if mutation_type == "add" and existing_concepts:
            # 从其他概念借用属性
            donor = random.choice(existing_concepts)
            if donor.properties:
                prop_key = random.choice(list(donor.properties.keys()))
                if prop_key not in new_props:
                    new_props[prop_key] = donor.properties[prop_key]
        elif mutation_type == "modify" and new_props:
            # 修改一个属性值
            prop_key = random.choice(list(new_props.keys()))
            val = new_props[prop_key]
            if isinstance(val, (int, float)):
                new_props[prop_key] = val * random.uniform(0.5, 1.5)
            elif isinstance(val, bool):
                new_props[prop_key] = not val
            elif isinstance(val, str):
                new_props[prop_key] = val + "_mutated"
        elif mutation_type == "remove" and len(new_props) > 1:
            # 移除一个属性
            prop_key = random.choice(list(new_props.keys()))
            del new_props[prop_key]

        result = Concept(
            concept_id=f"mut_{int(time.time() * 1000)}_{random.randint(0, 9999)}",
            name=f"{concept.name}_variant",
            concept_type=concept.concept_type,
            domain=concept.domain,
            properties=new_props,
            source_concepts=[concept.concept_id],
        )

        # 评估
        assessment = self._metrics.assess(result, existing_concepts)
        result.novelty_score = assessment.novelty
        result.value_score = assessment.value
        result.surprise_score = assessment.surprise

        return result

    def cross_pollinate(
        self,
        concept1: Concept,
        concept2: Concept,
        existing_concepts: Optional[List[Concept]] = None,
    ) -> Optional[Concept]:
        """交叉授粉：融合两个概念的特征"""
        new_props: Dict[str, Any] = {}

        # 各取一半属性
        props1 = list(concept1.properties.items())
        props2 = list(concept2.properties.items())

        half1 = len(props1) // 2
        half2 = len(props2) // 2

        for k, v in props1[:half1]:
            new_props[k] = v
        for k, v in props2[half2:]:
            new_props[k] = v

        # 标记交叉授粉
        new_props["cross_pollinated"] = True
        if concept1.domain != concept2.domain:
            new_props["cross_domain"] = True
            new_props["emergent"] = True

        result = Concept(
            concept_id=f"cross_{int(time.time() * 1000)}_{random.randint(0, 9999)}",
            name=f"{concept1.name}×{concept2.name}",
            concept_type=ConceptType.EMERGENT if concept1.domain != concept2.domain else ConceptType.COMPOUND,
            domain=f"{concept1.domain}/{concept2.domain}",
            properties=new_props,
            source_concepts=[concept1.concept_id, concept2.concept_id],
        )

        # 评估
        all_existing = list(existing_concepts or []) + [concept1, concept2]
        assessment = self._metrics.assess(result, all_existing)
        result.novelty_score = assessment.novelty
        result.value_score = assessment.value
        result.surprise_score = assessment.surprise

        return result


# ============================================================
# 创造第一性原理引擎（增强版）
# ============================================================

class CreativityFirstPrinciplesEngine:
    """
    创造第一性原理引擎（阶段122增强版）

    新增能力:
    - CreativityMetrics: 创造力度量
    - CombinatorialCreativity: 组合创新
    - AnalogicalReasoning: 类比推理
    - DivergentThinking: 发散思维

    向后兼容: 保留原有 add_concept / combine / find_analogy / generate_novel_combination 接口
    """

    def __init__(self, engine_id: str = "creativity_engine"):
        self.engine_id = engine_id
        self._concepts: Dict[str, Concept] = {}
        self._combinations: List[Combination] = []
        self._combination_history: Set[Tuple[str, ...]] = set()
        self._lock = threading.RLock()
        self._cycle_count = 0

        # 阶段122新增组件
        self._metrics = CreativityMetrics()
        self._combinatorial = CombinatorialCreativity(self._metrics)
        self._analogical = AnalogicalReasoning()
        self._divergent = DivergentThinking(self._metrics)

    # ============================================================
    # 原有接口（向后兼容）
    # ============================================================

    def add_concept(
        self,
        concept_id: str,
        name: str,
        domain: str,
        concept_type: ConceptType = ConceptType.PRIMITIVE,
        properties: Dict[str, Any] = None,
    ) -> Concept:
        """添加概念（向后兼容）"""
        concept = Concept(
            concept_id=concept_id,
            name=name,
            concept_type=concept_type,
            domain=domain,
            properties=properties or {},
        )
        with self._lock:
            self._concepts[concept_id] = concept
        return concept

    def combine(self, concept_ids: List[str]) -> Optional[Concept]:
        """组合概念（向后兼容）"""
        # 阶段122防御：空列表返回None
        if not concept_ids:
            return None
        with self._lock:
            concepts = [self._concepts.get(cid) for cid in concept_ids]
            if None in concepts:
                return None

            # 检查是否已尝试过此组合
            sorted_ids = tuple(sorted(concept_ids))
            if sorted_ids in self._combination_history:
                novelty = 0.1
            else:
                self._combination_history.add(sorted_ids)
                domains = set(c.domain for c in concepts)
                novelty = 0.5 + 0.5 * (len(domains) - 1) / max(1, len(concepts))

            # 合并属性
            merged_props: Dict[str, Any] = {}
            for c in concepts:
                merged_props.update(c.properties)

            domains = set(c.domain for c in concepts)
            if len(domains) > 1:
                merged_props["cross_domain"] = True
                merged_props["emergent"] = True

            result = Concept(
                concept_id=f"comb_{int(time.time() * 1000)}_{self._cycle_count}",
                name=" + ".join(c.name for c in concepts),
                concept_type=ConceptType.COMPOUND if len(domains) == 1 else ConceptType.EMERGENT,
                domain="/".join(sorted(domains)),
                properties=merged_props,
                source_concepts=list(concept_ids),
                novelty_score=min(1.0, novelty),
            )

            combination = Combination(
                combination_id=f"cmb_{int(time.time() * 1000)}",
                concept_ids=list(concept_ids),
                result_concept=result,
            )

            self._combinations.append(combination)
            self._concepts[result.concept_id] = result
            self._cycle_count += 1

            return result

    def find_analogy(self, source_domain: str, target_domain: str) -> List[Dict[str, Any]]:
        """寻找跨域类比（向后兼容）"""
        with self._lock:
            source_concepts = [c for c in self._concepts.values() if c.domain == source_domain]
            target_concepts = [c for c in self._concepts.values() if c.domain == target_domain]

        analogies = []
        for sc in source_concepts:
            for tc in target_concepts:
                common_props = set(sc.properties.keys()) & set(tc.properties.keys())
                if len(common_props) > 0:
                    similarity = len(common_props) / max(len(sc.properties), len(tc.properties), 1)
                    if similarity > 0.3:
                        analogies.append({
                            "source": sc.concept_id,
                            "target": tc.concept_id,
                            "similarity": similarity,
                            "common_properties": list(common_props),
                        })

        analogies.sort(key=lambda x: x["similarity"], reverse=True)
        return analogies

    def generate_novel_combination(self, domain_filter: Optional[str] = None) -> Optional[Concept]:
        """生成新颖组合（向后兼容）"""
        with self._lock:
            candidates = list(self._concepts.values())
            if domain_filter:
                candidates = [c for c in candidates if c.domain == domain_filter]

        if len(candidates) < 2:
            return None

        random.shuffle(candidates)
        selected = candidates[:2]
        return self.combine([c.concept_id for c in selected])

    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        with self._lock:
            concepts = list(self._concepts.values())
            return {
                "engine_id": self.engine_id,
                "concepts_count": len(concepts),
                "combinations_count": len(self._combinations),
                "cycle_count": self._cycle_count,
                "avg_novelty": sum(c.novelty_score for c in concepts) / len(concepts) if concepts else 0,
                "emergent_concepts": sum(1 for c in concepts if c.concept_type == ConceptType.EMERGENT),
            }

    def clear(self) -> None:
        """清除"""
        with self._lock:
            self._concepts.clear()
            self._combinations.clear()
            self._combination_history.clear()
            self._cycle_count = 0
            self._metrics = CreativityMetrics()
            self._combinatorial = CombinatorialCreativity(self._metrics)
            self._analogical = AnalogicalReasoning()
            self._divergent = DivergentThinking(self._metrics)

    # ============================================================
    # 阶段122新增：创造力度量接口
    # ============================================================

    def assess_creativity(self, concept_id: str) -> Optional[CreativityAssessment]:
        """评估概念创造力"""
        with self._lock:
            concept = self._concepts.get(concept_id)
            if concept is None:
                return None
            existing = [c for c in self._concepts.values() if c.concept_id != concept_id]
        return self._metrics.assess(concept, existing)

    def get_creativity_metrics(self) -> CreativityMetrics:
        """获取创造力度量引擎"""
        return self._metrics

    # ============================================================
    # 阶段122新增：组合创新接口
    # ============================================================

    def combine_with_assessment(self, concept_ids: List[str]) -> Optional[Concept]:
        """带评估的组合"""
        with self._lock:
            concepts = [self._concepts.get(cid) for cid in concept_ids]
            if None in concepts:
                return None
            existing = [c for c in self._concepts.values() if c.concept_id not in concept_ids]

        result = self._combinatorial.combine(concepts, existing)
        if result is not None:
            with self._lock:
                self._concepts[result.concept_id] = result
                self._combinations.append(Combination(
                    combination_id=f"cmb_{int(time.time() * 1000)}",
                    concept_ids=list(concept_ids),
                    result_concept=result,
                ))
                self._cycle_count += 1
        return result

    def generate_combinations(
        self,
        concept_ids: List[str],
        max_count: int = 10,
    ) -> List[Concept]:
        """生成多个组合"""
        with self._lock:
            concepts = [self._concepts.get(cid) for cid in concept_ids]
            if None in concepts:
                return []
            existing = list(self._concepts.values())

        results = self._combinatorial.generate_combinations(concepts, max_count, existing)
        with self._lock:
            for r in results:
                self._concepts[r.concept_id] = r
        return results

    def get_combinatorial_engine(self) -> CombinatorialCreativity:
        """获取组合创新引擎"""
        return self._combinatorial

    # ============================================================
    # 阶段122新增：类比推理接口
    # ============================================================

    def analogical_reason(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[AnalogyMapping]:
        """类比推理"""
        with self._lock:
            source = self._concepts.get(source_id)
            target = self._concepts.get(target_id)
        if source is None or target is None:
            return None
        return self._analogical.find_analogy(source, target)

    def transfer_analogy(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[Concept]:
        """迁移类比结构"""
        with self._lock:
            source = self._concepts.get(source_id)
            target = self._concepts.get(target_id)
        if source is None or target is None:
            return None
        mapping = self._analogical.find_analogy(source, target)
        if mapping is None:
            return None
        result = self._analogical.transfer_structure(source, target, mapping)
        with self._lock:
            self._concepts[result.concept_id] = result
        return result

    def find_analogies_for(
        self,
        source_id: str,
        min_similarity: float = 0.2,
    ) -> List[AnalogyMapping]:
        """为源概念寻找所有类比"""
        with self._lock:
            source = self._concepts.get(source_id)
            candidates = [c for c in self._concepts.values() if c.concept_id != source_id]
        if source is None:
            return []
        return self._analogical.find_analogies_in_domain(source, candidates, min_similarity)

    def get_analogical_engine(self) -> AnalogicalReasoning:
        """获取类比推理引擎"""
        return self._analogical

    # ============================================================
    # 阶段187新增：结构映射接口
    # ============================================================

    def map_structure(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        结构映射分析（阶段187）

        第一性原理：返回详细的结构映射信息，包括属性相似度和关系相似度
        """
        with self._lock:
            source = self._concepts.get(source_id)
            target = self._concepts.get(target_id)
        if source is None or target is None:
            return None
        return self._analogical._structure_mapper.map_structure(
            source.properties, target.properties
        )

    def find_structural_analogy(
        self,
        source_id: str,
        target_id: str,
    ) -> Optional[AnalogyMapping]:
        """
        基于结构映射的类比发现（阶段187）

        第一性原理：优先匹配关系结构而非表面属性
        """
        with self._lock:
            source = self._concepts.get(source_id)
            target = self._concepts.get(target_id)
        if source is None or target is None:
            return None
        return self._analogical._structure_mapper.find_structural_analogy(source, target)

    # ============================================================
    # 阶段122新增：发散思维接口
    # ============================================================

    def diverge(
        self,
        seed_id: str,
        count: int = 5,
    ) -> List[Concept]:
        """发散思维：从种子生成多个想法"""
        with self._lock:
            seed = self._concepts.get(seed_id)
            existing = list(self._concepts.values())
        if seed is None:
            return []
        ideas = self._divergent.generate_ideas(seed, count, existing)
        with self._lock:
            for idea in ideas:
                self._concepts[idea.concept_id] = idea
        return ideas

    def cross_pollinate(
        self,
        concept_id1: str,
        concept_id2: str,
    ) -> Optional[Concept]:
        """交叉授粉"""
        with self._lock:
            c1 = self._concepts.get(concept_id1)
            c2 = self._concepts.get(concept_id2)
            existing = list(self._concepts.values())
        if c1 is None or c2 is None:
            return None
        result = self._divergent.cross_pollinate(c1, c2, existing)
        if result is not None:
            with self._lock:
                self._concepts[result.concept_id] = result
        return result

    def get_divergent_engine(self) -> DivergentThinking:
        """获取发散思维引擎"""
        return self._divergent


# ============================================================
# 便捷函数（向后兼容 + 新增）
# ============================================================

def create_creativity_engine(engine_id: str = "creativity_engine") -> CreativityFirstPrinciplesEngine:
    """创建创造引擎"""
    return CreativityFirstPrinciplesEngine(engine_id=engine_id)


def create_concept(
    concept_id: str,
    name: str,
    domain: str,
    properties: Dict[str, Any] = None,
) -> Concept:
    """创建概念"""
    return Concept(
        concept_id=concept_id,
        name=name,
        concept_type=ConceptType.PRIMITIVE,
        domain=domain,
        properties=properties or {},
    )


def create_creativity_metrics() -> CreativityMetrics:
    """创建创造力度量引擎"""
    return CreativityMetrics()
