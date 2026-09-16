"""
Agent 单元测试

对各个子 Agent 进行独立测试，验证核心功能正常
不依赖外部服务（Neo4j/MySQL）时使用 Mock

运行方法:
    cd /opt/ecommerce-agent
    python -m pytest tests/test_agents.py -v

    或直接运行:
    python tests/test_agents.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unittest.mock import MagicMock

import pytest

from config.settings import settings
from orchestration.router import RouterAgent

# ------------------------------------------------------------------ #
#  Router Agent 测试
# ------------------------------------------------------------------ #


class TestRouterAgent:
    """路由器测试"""

    @pytest.fixture
    def router(self):
        """创建路由器（使用 Mock LLM）"""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="chitchat")
        return RouterAgent(mock_llm)

    def test_keyword_route_order(self, router):
        """测试订单意图识别"""
        result = router.route("查询我的订单 ORD123456")
        assert result["intent"] == "order"
        assert result["agent"] == "business_data_agent"
        assert result["mode"] == "query_order"

    def test_keyword_route_customer_service(self, router):
        """测试客服意图识别"""
        result = router.route("怎么退货？")
        assert result["intent"] == "customer_service"
        assert result["agent"] == "cs_agent"

    def test_keyword_route_search(self, router):
        """测试搜索意图识别"""
        result = router.route("搜索蓝牙耳机")
        assert result["intent"] == "search"
        assert result["agent"] == "catalog_agent"
        assert result["mode"] == "search"

    def test_keyword_route_analytics(self, router):
        """测试数据分析意图识别"""
        result = router.route("最近销量排行")
        assert result["intent"] == "analytics"
        assert result["agent"] == "business_data_agent"
        assert result["mode"] == "analyze"

    def test_keyword_route_classify(self, router):
        """测试分类意图识别（直接分类请求，非问句式）"""
        # "属于什么类别" 是问句，会路由到 kg_qa（优先级更高）
        # classify 仅处理直接分类请求
        result = router.route("分类: 纸尿裤")
        assert result["intent"] == "classify"
        assert result["agent"] == "catalog_agent"
        assert result["mode"] == "classify"

    def test_keyword_route_kg_qa_question(self, router):
        """测试问句式分类查询路由到知识图谱"""
        result = router.route("纸尿裤属于什么类别")
        assert result["intent"] == "kg_qa"
        assert result["agent"] == "kg_qa_agent"
        assert result["mode"] == "run"

    def test_keyword_route_recommend(self, router):
        """测试推荐意图识别"""
        result = router.route("有什么推荐的")
        assert result["intent"] == "recommend"
        assert result["agent"] == "catalog_agent"
        assert result["mode"] == "recommend"

    def test_keyword_route_chitchat(self, router):
        """测试闲聊意图识别"""
        result = router.route("你好")
        assert result["intent"] == "chitchat"
        assert result["agent"] == ""

    def test_normalize_intent(self, router):
        """测试意图标准化"""
        assert router._normalize_intent("cs") == "customer_service"
        assert router._normalize_intent("qa") == "kg_qa"
        assert router._normalize_intent("unknown_intent") == "chitchat"


# ------------------------------------------------------------------ #
#  Classify Agent 测试（规则降级模式）
# ------------------------------------------------------------------ #


class TestClassifyAgent:
    """分类 Agent 测试（使用规则降级模式，不依赖模型文件）"""

    @pytest.fixture
    def agent(self):
        from agents.classify_agent import ClassifyAgent

        # 指定不存在的路径，触发规则降级
        return ClassifyAgent(
            model_path="/nonexistent/model",
            labels_path="/nonexistent/labels.txt",
        )

    def test_rule_based_classify_phone(self, agent):
        """测试规则分类 - 手机"""
        result = agent.classify_product("小米手机 红米 Note 12")
        assert "手机数码" in result

    def test_rule_based_classify_clothes(self, agent):
        """测试规则分类 - 家电（当前模型为 4 类，服装类不再输出）"""
        result = agent.classify_product("美的空调 1.5匹 挂机")
        assert "家用电器" in result

    def test_rule_based_classify_food(self, agent):
        """测试规则分类 - 食品"""
        result = agent.classify_product("进口牛肉 牛排")
        assert "食品生鲜" in result

    def test_preprocess(self, agent):
        """测试文本预处理"""
        # 全角转半角
        result = agent._preprocess("Ａｐｐｌｅ手机　　128GB")
        assert "Apple" in result or "apple" in result.lower()
        # 多余空白被清除
        assert "  " not in result

    def test_get_top_k(self, agent):
        """测试 Top-K 返回"""
        results = agent.get_top_k("手机", k=3)
        assert isinstance(results, list)
        assert len(results) >= 1


# ------------------------------------------------------------------ #
#  Order Agent 测试
# ------------------------------------------------------------------ #


class TestOrderAgent:
    """订单 Agent 测试"""

    @pytest.fixture
    def agent(self):
        from agents.order_agent import OrderAgent

        return OrderAgent(
            db_config={
                "host": "localhost",
                "port": 3306,
                "user": "root",
                "password": "",
                "database": "test",
            }
        )

    def test_extract_order_id_alpha(self, agent):
        """测试订单号提取 - 字母数字混合"""
        order_id = agent._extract_order_id("查询订单 ORD123456")
        assert order_id == "ORD123456"

    def test_extract_order_id_numeric(self, agent):
        """测试订单号提取 - 纯数字"""
        order_id = agent._extract_order_id("我的订单号是 12345678901")
        assert order_id == "12345678901"

    def test_extract_order_id_none(self, agent):
        """测试订单号提取 - 无订单号"""
        order_id = agent._extract_order_id("我的订单到哪了？")
        assert order_id is None

    def test_map_status(self, agent):
        """测试状态码映射"""
        assert agent._map_order_status(1) == "待付款"
        assert agent._map_order_status("3") == "已发货"
        assert agent._map_order_status("unknown_code") == "unknown_code"

    def test_run_without_order_id(self, agent):
        """测试无订单号时的提示"""
        result = agent.run("查询订单")
        assert "订单号" in result


# ------------------------------------------------------------------ #
#  Recommend Agent 测试
# ------------------------------------------------------------------ #


class TestRecommendAgent:
    """推荐 Agent 测试"""

    @pytest.fixture
    def agent(self):
        from agents.recommend_agent import RecommendAgent

        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        return RecommendAgent(neo4j_driver=mock_neo4j, llm=mock_llm)

    def test_parse_budget_range(self, agent):
        """测试预算解析 - 范围"""
        result = agent._parse_budget("100-200")
        assert result == (100.0, 200.0)

    def test_parse_budget_single(self, agent):
        """测试预算解析 - 单值"""
        result = agent._parse_budget("500")
        assert result == (0, 500.0)

    def test_parse_budget_int(self, agent):
        """测试预算解析 - 整数"""
        result = agent._parse_budget(300)
        assert result == (0, 300.0)

    def test_in_range(self, agent):
        """测试价格范围判断"""
        assert agent._in_range(150, 100, 200) is True
        assert agent._in_range(50, 100, 200) is False
        assert agent._in_range(250, 100, 200) is False


# ------------------------------------------------------------------ #
#  Catalog Agent（商品域门面）测试
# ------------------------------------------------------------------ #


class TestCatalogAgent:
    """商品域 Agent 测试（子 Agent 使用 Mock）"""

    @pytest.fixture
    def agent(self):
        from agents.catalog_agent import CatalogAgent

        search_mock = MagicMock()
        search_mock.run.return_value = "搜索结果"
        recommend_mock = MagicMock()
        recommend_mock.run.return_value = "推荐结果"
        classify_mock = MagicMock()
        classify_mock.run.return_value = "分类结果"
        return CatalogAgent(
            search_agent=search_mock,
            recommend_agent=recommend_mock,
            classify_agent=classify_mock,
        )

    def test_run_default_search(self, agent):
        """默认 mode 走搜索"""
        result = agent.run("蓝牙耳机")
        assert result == "搜索结果"
        agent.search_agent.run.assert_called_with(query="蓝牙耳机")

    def test_run_recommend_mode(self, agent):
        """mode=recommend 走推荐"""
        result = agent.run("推荐户外装备", mode="recommend")
        assert result == "推荐结果"
        agent.recommend_agent.run.assert_called_with(query="推荐户外装备", mode="recommend")

    def test_recommend_passes_user_profile(self, agent):
        """推荐方法透传用户画像"""
        agent.recommend("推荐母婴用品", user_profile={"偏好分类": "母婴用品"})
        agent.recommend_agent.run.assert_called_with(
            query="推荐母婴用品", user_profile={"偏好分类": "母婴用品"}
        )

    def test_run_classify_mode(self, agent):
        """mode=classify 走分类"""
        result = agent.run("小米手机", mode="classify")
        assert result == "分类结果"
        agent.classify_agent.run.assert_called_with(query="小米手机", mode="classify")

    def test_hybrid_search_delegation(self, agent):
        """独立搜索接口委托给 search_agent.hybrid_search"""
        agent.hybrid_search("耳机", top_k=5, category="手机数码")
        agent.search_agent.hybrid_search.assert_called_with(
            query="耳机", top_k=5, category="手机数码", min_price=None, max_price=None
        )

    def test_get_top_k_delegation(self, agent):
        """独立分类接口委托给 classify_agent.get_top_k"""
        agent.classify_agent.get_top_k.return_value = [{"category": "手机数码"}]
        result = agent.get_top_k("手机", k=3)
        assert result == [{"category": "手机数码"}]
        agent.classify_agent.get_top_k.assert_called_with("手机", 3)


# ------------------------------------------------------------------ #
#  Business Data Agent（业务数据域门面）测试
# ------------------------------------------------------------------ #


class TestBusinessDataAgent:
    """业务数据域 Agent 测试（子 Agent 使用 Mock）"""

    @pytest.fixture
    def agent(self):
        from agents.business_data_agent import BusinessDataAgent

        order_mock = MagicMock()
        order_mock.run.return_value = "订单信息"
        analytics_mock = MagicMock()
        analytics_mock.run.return_value = "分析结果"
        return BusinessDataAgent(order_agent=order_mock, analytics_agent=analytics_mock)

    def test_run_default_query_order(self, agent):
        """默认 mode 走订单查询"""
        result = agent.run("查询订单 ORD123")
        assert result == "订单信息"
        agent.order_agent.run.assert_called_with(query="查询订单 ORD123")

    def test_run_analyze_mode(self, agent):
        """mode=analyze 走数据分析"""
        result = agent.run("最近销量排行", mode="analyze")
        assert result == "分析结果"
        agent.analytics_agent.run.assert_called_with(query="最近销量排行", mode="analyze")


# ------------------------------------------------------------------ #
#  闲聊兜底（替代 chitchat_agent）测试
# ------------------------------------------------------------------ #


class TestChitchatFallback:
    """编排器闲聊兜底测试"""

    def test_match_greeting_hello(self):
        from orchestration.graph import ECommerceOrchestrator

        result = ECommerceOrchestrator._match_greeting("你好")
        assert "你好" in result
        assert "电商" in result

    def test_match_greeting_thanks(self):
        from orchestration.graph import ECommerceOrchestrator

        result = ECommerceOrchestrator._match_greeting("谢谢")
        assert "不客气" in result

    def test_match_greeting_none(self):
        from orchestration.graph import ECommerceOrchestrator

        result = ECommerceOrchestrator._match_greeting("搜索蓝牙耳机")
        assert result == ""


# ------------------------------------------------------------------ #
#  用户画像构建测试（记忆 → 推荐）
# ------------------------------------------------------------------ #


class TestUserProfile:
    """用户画像构建与消费测试"""

    def test_build_user_profile_merges_explicit_and_memory(self):
        """显式画像 + 长期记忆偏好合并"""
        from orchestration.memory import MemoryItem, MemoryManager

        manager = MemoryManager(llm=None, embedder=None)
        manager._long_term = MagicMock()
        manager._long_term.memories = [
            MemoryItem(content="用户偏好母婴用品", category="preference", session_id="s1"),
            MemoryItem(content="用户喜欢户外运动", category="preference", session_id="s1"),
            MemoryItem(content="其他会话的偏好", category="preference", session_id="s2"),
            MemoryItem(content="用户上个月买了手机", category="decision", session_id="s1"),
            MemoryItem(content="普通事实，不参与画像", category="fact", session_id="s1"),
        ]

        profile = manager.build_user_profile(
            session_id="s1",
            current_query="推荐户外装备",
            explicit={"预算": "100-300"},
        )

        assert profile["预算"] == "100-300"
        notes = profile.get("long_term_notes", [])
        assert "用户偏好母婴用品" in notes
        assert "用户喜欢户外运动" in notes
        assert "用户上个月买了手机" in notes
        # 其他会话与 fact 类记忆不进入画像
        assert "其他会话的偏好" not in notes
        assert "普通事实" not in notes

    def test_build_user_profile_no_memory(self):
        """无长期记忆时只返回显式画像"""
        from orchestration.memory import MemoryManager

        manager = MemoryManager(llm=None, embedder=None)
        manager._long_term = None

        profile = manager.build_user_profile("s1", explicit={"偏好品牌": "华为"})
        assert profile == {"偏好品牌": "华为"}

    def test_extract_category_uses_profile_notes(self):
        """分类提取参考用户画像中的长期偏好"""
        from agents.recommend_agent import RecommendAgent

        mock_neo4j = MagicMock()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = "家用电器"
        agent = RecommendAgent(neo4j_driver=mock_neo4j, llm=mock_llm)

        result = agent._extract_category(
            "有什么推荐的",
            user_profile={"long_term_notes": ["用户最近想买空调和冰箱"]},
        )
        assert result == "家用电器"

        # 无画像时也能正常工作（向后兼容）
        result2 = agent._extract_category("有什么推荐的")
        assert result2 == "家用电器"


# ------------------------------------------------------------------ #
#  多轮实体复用（指代消解）测试
# ------------------------------------------------------------------ #


class _FakeNer:
    """可控的 NER 桩：返回预置实体 dict。"""

    def __init__(self, entities=None):
        self.entities = entities or {}

    def extract(self, query):
        return self.entities


class TestMultiTurnEntityReuse:
    """多轮实体复用：追问时回填上一轮 NER 实体消解指代（不依赖模型/LLM）。"""

    # -- 纯函数：merge_entity_context ---------------------------------- #

    def test_merge_reuses_carried_on_anaphoric_followup(self):
        from agents.catalog_agent import merge_entity_context

        result = merge_entity_context(
            "那有降噪的吗",
            current_entities={"商品": [], "品牌": [], "款式": []},
            carried_entities={"商品": ["蓝牙耳机"]},
        )
        assert "蓝牙耳机" in result

    def test_merge_skips_when_current_has_new_entity(self):
        from agents.catalog_agent import merge_entity_context

        result = merge_entity_context(
            "那跑步鞋呢",
            current_entities={"商品": ["跑步鞋"]},
            carried_entities={"商品": ["蓝牙耳机"]},
        )
        assert result == "那跑步鞋呢"

    def test_merge_skips_when_no_carried(self):
        from agents.catalog_agent import merge_entity_context

        assert merge_entity_context("那有降噪的吗", {}, {}) == "那有降噪的吗"

    def test_merge_ignores_spec_only_carried(self):
        from agents.catalog_agent import merge_entity_context

        # 规格不参与指代消解，只有规格时原样返回
        assert merge_entity_context("那这个呢", {}, {"规格": ["512G"]}) == "那这个呢"

    def test_merge_skips_long_non_followup(self):
        from agents.catalog_agent import merge_entity_context

        # 长句且无回指词 → 视为新话题，不回填
        result = merge_entity_context(
            "帮我找一款适合送人的生日礼物",
            {},
            {"商品": ["蓝牙耳机"]},
        )
        assert result == "帮我找一款适合送人的生日礼物"

    # -- 纯函数：_is_anaphoric_followup -------------------------------- #

    def test_is_anaphoric_followup_short_or_marker(self):
        from agents.catalog_agent import _is_anaphoric_followup

        assert _is_anaphoric_followup("那这个呢")
        assert _is_anaphoric_followup("第二个")
        assert _is_anaphoric_followup("有降噪的吗，帮我看看这个牌子还有哪些")
        assert not _is_anaphoric_followup("帮我推荐一款适合户外露营的帐篷")
        assert not _is_anaphoric_followup("")

    # -- 记忆层：实体上下文存取 ----------------------------------------- #

    def test_memory_entity_context_roundtrip_and_carry_forward(self, monkeypatch):
        from orchestration.memory import MemoryManager

        monkeypatch.setattr(settings, "memory_long_term_enabled", False)
        mm = MemoryManager(llm=None, embedder=None)

        assert mm.get_entity_context("s1") == {}
        mm.set_entity_context("s1", {"商品": ["蓝牙耳机"], "品牌": ["华为"]})
        assert mm.get_entity_context("s1") == {"商品": ["蓝牙耳机"], "品牌": ["华为"]}

        # 空实体 no-op：链式指代不断链（推荐蓝牙耳机 → 那有降噪的吗 → 那第二个呢）
        mm.set_entity_context("s1", {"商品": [], "品牌": [], "款式": [], "规格": []})
        assert mm.get_entity_context("s1") == {"商品": ["蓝牙耳机"], "品牌": ["华为"]}

        # 新实体替换旧实体
        mm.set_entity_context("s1", {"商品": ["跑步鞋"]})
        assert mm.get_entity_context("s1") == {"商品": ["跑步鞋"]}

    def test_memory_clear_session_removes_entity_context(self, monkeypatch):
        from orchestration.memory import MemoryManager

        monkeypatch.setattr(settings, "memory_long_term_enabled", False)
        mm = MemoryManager(llm=None, embedder=None)
        mm.set_entity_context("s1", {"商品": ["蓝牙耳机"]})
        mm.clear_session("s1")
        assert mm.get_entity_context("s1") == {}

    def test_memory_entity_context_is_session_scoped(self, monkeypatch):
        from orchestration.memory import MemoryManager

        monkeypatch.setattr(settings, "memory_long_term_enabled", False)
        mm = MemoryManager(llm=None, embedder=None)
        mm.set_entity_context("s1", {"商品": ["蓝牙耳机"]})
        mm.set_entity_context("s2", {"商品": ["跑步鞋"]})
        assert mm.get_entity_context("s1") == {"商品": ["蓝牙耳机"]}
        assert mm.get_entity_context("s2") == {"商品": ["跑步鞋"]}

    # -- CatalogAgent：实体上下文注入搜索/推荐 --------------------------- #

    def test_catalog_search_injects_entity_context(self):
        from agents.catalog_agent import CatalogAgent

        search_mock = MagicMock()
        search_mock.run.return_value = "搜索结果"
        agent = CatalogAgent(
            search_agent=search_mock,
            recommend_agent=MagicMock(),
            classify_agent=MagicMock(),
            ner_agent=_FakeNer({"商品": [], "品牌": [], "款式": [], "规格": []}),
        )
        agent.search("那有降噪的吗", entity_context={"商品": ["蓝牙耳机"]})
        search_mock.run.assert_called_once()
        query = search_mock.run.call_args.kwargs.get("query")
        assert query and "蓝牙耳机" in query

    def test_catalog_recommend_injects_entity_context(self):
        from agents.catalog_agent import CatalogAgent

        recommend_mock = MagicMock()
        recommend_mock.run.return_value = "推荐结果"
        agent = CatalogAgent(
            search_agent=MagicMock(),
            recommend_agent=recommend_mock,
            classify_agent=MagicMock(),
            ner_agent=_FakeNer({"商品": [], "品牌": [], "款式": [], "规格": []}),
        )
        agent.recommend("那这款呢", entity_context={"商品": ["蓝牙耳机"]})
        recommend_mock.run.assert_called_once()
        query = recommend_mock.run.call_args.kwargs.get("query")
        assert query and "蓝牙耳机" in query

    def test_catalog_search_without_entity_context_unchanged(self):
        from agents.catalog_agent import CatalogAgent

        search_mock = MagicMock()
        search_mock.run.return_value = "搜索结果"
        agent = CatalogAgent(
            search_agent=search_mock,
            recommend_agent=MagicMock(),
            classify_agent=MagicMock(),
            ner_agent=_FakeNer({"商品": ["耳机"]}),
        )
        # 无 entity_context：仅走原有 NER 查询增强，不回填
        agent.search("蓝牙耳机")
        search_mock.run.assert_called_once()
        query = search_mock.run.call_args.kwargs.get("query")
        assert query == "蓝牙耳机 耳机"


# ------------------------------------------------------------------ #
#  直接运行入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    # 不使用 pytest 时直接运行
    print("=" * 60)
    print("  电商智能体单元测试")
    print("=" * 60)

    tests = [
        ("Router - 订单意图", lambda: RouterAgent(MagicMock()).route("查询订单 ORD123")),
        ("Router - 客服意图", lambda: RouterAgent(MagicMock()).route("怎么退货")),
        ("Router - 搜索意图", lambda: RouterAgent(MagicMock()).route("搜索耳机")),
        ("Router - 闲聊意图", lambda: RouterAgent(MagicMock()).route("你好")),
    ]

    passed = 0
    failed = 0
    for name, test_func in tests:
        try:
            result = test_func()
            if result:
                print(f"  ✅ {name}")
                passed += 1
            else:
                print(f"  ❌ {name} - 返回空")
                failed += 1
        except Exception as e:
            print(f"  ❌ {name} - {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  通过: {passed} | 失败: {failed}")
    print(f"{'=' * 60}")
