"""评估数据一致性校验（不启动 Neo4j / 模型文件 / 外部 LLM）。

三类被校验的数据资产：
1. tests/ner_eval.py 的 NER_TEST_CASES — 标注结构合法、实体类型闭合、样本量达标
2. tests/multiturn_eval.py 的 MULTITURN_SCENARIOS — 场景结构合法、指代类覆盖充足
3. scripts/generate_catalog_orders.py 的生成器 — 订单 product_id 与商品库对齐、
   存在共现结构（否则 Item-CF 恒为 0）

这些是评估脚本的"地基"：评估脚本只在标注结构非法时静默崩掉或产出误导性数字，
因此用纯离线校验守住数据质量。
"""

import os
import re
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

import dedupe_catalog as dedupe  # noqa: E402
import generate_catalog_orders as gen_orders  # noqa: E402
import pytest  # noqa: E402

from tests.multiturn_eval import MULTITURN_SCENARIOS  # noqa: E402
from tests.ner_eval import ENTITY_TYPES, NER_TEST_CASES  # noqa: E402

ALLOWED_EVAL_TYPES = {"relevance", "coreference", "context_inherit",
                      "long_term_recall", "topic_switch"}


# ------------------------------------------------------------------ #
#  NER 测试用例结构
# ------------------------------------------------------------------ #

class TestNerCases:
    def test_case_count_meets_minimum(self):
        """NER 样本量应达到 100+，覆盖品牌/商品/款式/规格/复合/闲聊/短查询/边界。"""
        assert len(NER_TEST_CASES) >= 100

    def test_expected_keys_are_closed(self):
        """每条 expected 必须是 {品牌,商品,款式,规格} 四键，不多不少。"""
        for case in NER_TEST_CASES:
            assert set(case["expected"].keys()) == set(ENTITY_TYPES), case["query"]

    def test_expected_values_are_string_lists(self):
        """每个实体类型下的期望值都是非空字符串列表（无 None / 无空串 / 无杂键）。"""
        for case in NER_TEST_CASES:
            for etype in ENTITY_TYPES:
                values = case["expected"][etype]
                assert isinstance(values, list), (case["query"], etype)
                assert all(isinstance(v, str) and v for v in values), (case["query"], etype)

    def test_queries_are_nonempty(self):
        for case in NER_TEST_CASES:
            assert case["query"].strip(), "存在空 query"

    def test_each_entity_type_appears(self):
        """四类实体各有样本覆盖（避免某种实体长期无人测试）。"""
        covered = {etype: False for etype in ENTITY_TYPES}
        for case in NER_TEST_CASES:
            for etype in ENTITY_TYPES:
                if case["expected"][etype]:
                    covered[etype] = True
        assert all(covered.values()), f"存在无样本的实体类型: {covered}"


# ------------------------------------------------------------------ #
#  多轮场景结构
# ------------------------------------------------------------------ #

class TestMultiturnScenarios:
    def test_scenario_count_meets_minimum(self):
        assert len(MULTITURN_SCENARIOS) >= 20

    def test_scenario_structure(self):
        for scenario in MULTITURN_SCENARIOS:
            assert isinstance(scenario["name"], str) and scenario["name"].strip()
            assert isinstance(scenario["turns"], list) and scenario["turns"]

    def test_turn_structure(self):
        for scenario in MULTITURN_SCENARIOS:
            for turn in scenario["turns"]:
                assert isinstance(turn["user"], str) and turn["user"].strip(), scenario["name"]
                assert isinstance(turn["expected_keywords"], list), scenario["name"]
                assert turn["evaluation"] in ALLOWED_EVAL_TYPES, (
                    scenario["name"], turn["evaluation"]
                )
                if "needs_context" in turn:
                    assert isinstance(turn["needs_context"], bool), scenario["name"]

    def test_coreference_coverage(self):
        """指代消解类（coreference）覆盖应充足——这是多轮最弱项，扩样重点。"""
        coref_turns = [
            t for s in MULTITURN_SCENARIOS for t in s["turns"]
            if t["evaluation"] == "coreference"
        ]
        assert len(coref_turns) >= 15, f"coreference 轮数不足: {len(coref_turns)}"

    def test_context_inherit_and_recall_coverage(self):
        """上下文继承 + 跨轮记忆也要有覆盖，避免只测了单轮相关性。"""
        types = {t["evaluation"] for s in MULTITURN_SCENARIOS for t in s["turns"]}
        assert "context_inherit" in types
        assert "long_term_recall" in types


# ------------------------------------------------------------------ #
#  订单生成器：与商品库对齐
# ------------------------------------------------------------------ #

_CATALOG_PATH = gen_orders.CATALOG


class TestOrderGenerator:
    @pytest.mark.skipif(not os.path.exists(_CATALOG_PATH), reason="商品库未生成")
    def test_generated_product_ids_align_with_catalog(self):
        """订单 product_id 必须全部来自商品库，否则 Item-CF 推荐映射回不存在的商品。"""
        products = gen_orders.load_catalog()
        catalog_ids = {p["id"] for p in products}
        orders = gen_orders.generate_orders(products, n_users=200, seed=1)
        assert orders, "未生成任何订单"
        assert all(o["product_id"] in catalog_ids for o in orders)

    @pytest.mark.skipif(not os.path.exists(_CATALOG_PATH), reason="商品库未生成")
    def test_product_name_contains_valid_category(self):
        """product_name 必须含合法类目名，否则 Item-CF 的 LIKE %类目% 种子查询恒为空。"""
        products = gen_orders.load_catalog()
        orders = gen_orders.generate_orders(products, n_users=200, seed=1)
        valid = gen_orders.VALID_CATEGORIES
        assert all(any(c in o["product_name"] for c in valid) for o in orders)

    @pytest.mark.skipif(not os.path.exists(_CATALOG_PATH), reason="商品库未生成")
    def test_synthetic_name_matches_pretty_name_convention(self):
        """合成名形如 {品牌}{类目}商品{序号}，能被 recommend_agent._pretty_name 清理。"""
        products = gen_orders.load_catalog()
        orders = gen_orders.generate_orders(products, n_users=200, seed=1)
        assert all(re.search(r"商品\d+$", o["product_name"]) for o in orders)

    @pytest.mark.skipif(not os.path.exists(_CATALOG_PATH), reason="商品库未生成")
    def test_cooccurrence_structure_exists(self):
        """至少要有若干商品被多个用户共同购买，否则 Item-CF 的共现信号是空集。"""
        products = gen_orders.load_catalog()
        orders = gen_orders.generate_orders(products, n_users=200, seed=1)
        users_per_product = defaultdict(set)
        for o in orders:
            users_per_product[o["product_id"]].add(o["user_id"])
        co_occurring = sum(1 for u in users_per_product.values() if len(u) >= 2)
        assert co_occurring >= 10, f"共现锚点过少: {co_occurring}"

    @pytest.mark.skipif(not os.path.exists(_CATALOG_PATH), reason="商品库未生成")
    def test_deterministic_given_seed(self):
        """同一 seed 重复运行结果一致（可复现）。"""
        products = gen_orders.load_catalog()
        a = gen_orders.generate_orders(products, n_users=100, seed=7)
        b = gen_orders.generate_orders(products, n_users=100, seed=7)
        assert a == b


# ------------------------------------------------------------------ #
#  商品库去重逻辑
# ------------------------------------------------------------------ #

class TestCatalogDedupe:
    @staticmethod
    def _product(pid, name, brand="华为", category="手机数码", source="simulated"):
        return {"id": pid, "name": name, "category": category, "brand": brand,
                "price": 100.0, "price_source": source}

    def test_normalize_title_strips_decorator_and_marketing(self):
        """装饰段 + 渠道/营销词 + 标点应被剥离，不影响型号判断。"""
        t = dedupe.normalize_title("【包邮】华为 Mate 60 官方旗舰店 全新", merge_specs=False)
        assert "包邮" not in t and "官方旗舰店" not in t and "全新" not in t
        assert "mate60" in t

    def test_dedupe_collapses_exact_duplicates(self):
        """同款仅装饰/标点/空白差异 → 合并为 1 件。"""
        products = [
            self._product("A", "华为Mate60 Pro 手机"),
            self._product("B", "【包邮】华为 Mate60 Pro 手机"),
            self._product("C", "华为Mate60Pro手机"),
        ]
        kept, removed = dedupe.dedupe(products, merge_specs=False)
        assert len(kept) == 1
        assert len(removed) == 1 and len(removed[0]) == 3

    def test_dedupe_keeps_real_price_representative(self):
        """重复组内优先保留真实价格的商品。"""
        products = [
            self._product("A", "华为Mate60", source="simulated"),
            self._product("B", "华为Mate60", source="real"),
        ]
        kept, _ = dedupe.dedupe(products, merge_specs=False)
        assert kept[0]["id"] == "B" and kept[0]["price_source"] == "real"

    def test_dedupe_does_not_merge_spec_variants_by_default(self):
        """保守模式（默认）不合并不同容量：128G 与 256G 是不同 SKU。"""
        products = [
            self._product("A", "iPhone15 128GB"),
            self._product("B", "iPhone15 256GB"),
        ]
        kept, _ = dedupe.dedupe(products, merge_specs=False)
        assert len(kept) == 2

    def test_merge_specs_collapses_capacity_variants(self):
        """--merge-specs 才会连同容量/颜色变体一起合并。"""
        products = [
            self._product("A", "iPhone15 128GB 蓝色"),
            self._product("B", "iPhone15 256GB 黑色"),
        ]
        kept, _ = dedupe.dedupe(products, merge_specs=True)
        assert len(kept) == 1

    def test_dedupe_is_deterministic(self):
        """同输入重复运行，保留结果一致。"""
        products = [self._product(p, "华为Mate60") for p in "ABCDE"]
        a, _ = dedupe.dedupe(products, merge_specs=False)
        b, _ = dedupe.dedupe(products, merge_specs=False)
        assert [p["id"] for p in a] == [p["id"] for p in b]
