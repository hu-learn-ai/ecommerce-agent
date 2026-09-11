"""模型链路回归测试。

覆盖审查发现并已修复的问题：
1. 分类模型域外拒识（关键词黑名单 + 逐类马氏距离门控）
2. NER 受限解码 + ModelCache 复用
3. 搜索价格/预算解析
4. 模型加载期 int8 量化

运行:
    python -m pytest tests/model_regression.py -v
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

# ------------------------------------------------------------------ #
#  价格解析（纯函数，无模型依赖）
# ------------------------------------------------------------------ #


class TestPriceParsing:
    """搜索链路价格/预算解析"""

    def _parse(self, query):
        from agents.search_agent import ProductSearchAgent

        return ProductSearchAgent._parse_price_constraints(query)

    def test_max_price(self):
        assert self._parse("500元以下的手机") == (None, 500.0)
        assert self._parse("预算200以内买耳机") == (None, 200.0)
        assert self._parse("不超过1000的冰箱") == (None, 1000.0)

    def test_range(self):
        assert self._parse("100-300元的蓝牙耳机") == (100.0, 300.0)
        assert self._parse("200至500元 电饭煲") == (200.0, 500.0)

    def test_min_price(self):
        assert self._parse("300元以上 智能手表") == (300.0, None)
        assert self._parse("至少800的冰箱") == (800.0, None)

    def test_no_price(self):
        assert self._parse("推荐一些适合户外的装备") is None
        assert self._parse("") is None


# ------------------------------------------------------------------ #
#  分类模型：域内正确 / 域外拒识
# ------------------------------------------------------------------ #


class TestClassifyModel:
    """分类 Agent 域内/域外回归"""

    @pytest.fixture(scope="class")
    def agent(self):
        from agents.classify_agent import ClassifyAgent
        from config.settings import settings

        return ClassifyAgent(
            settings.classify_model_path,
            os.path.join(settings.classify_model_path, "labels.txt"),
        )

    @pytest.mark.parametrize(
        "title,expected",
        [
            ("华为Mate60 Pro 12+512G 5G手机", "手机数码"),
            ("iPhone 15 Pro Max 256GB 钛金属", "手机数码"),
            ("戴森 V12 Detect Slim 无线吸尘器", "家用电器"),
            ("新鲜智利车厘子JJ级 2斤装", "食品生鲜"),
            ("鱼跃血压计 家用上臂式", "医药保健"),
            ("海尔冰箱 一级能效 双开门", "家用电器"),
            ("小米蓝牙耳机 降噪版 黑色", "手机数码"),
        ],
    )
    def test_in_domain(self, agent, title, expected):
        result = agent.classify_product(title)
        assert expected in result, f"{title} -> {result}"
        assert "无法判断" not in result, f"{title} 被误拒: {result}"

    @pytest.mark.parametrize(
        "title",
        [
            "跑步机 家用电动 静音",
            "儿童玩具 积木 益智",
            "保温杯 316不锈钢 500ml",
            "雨伞 晴雨两用 自动",
            "拖鞋 浴室防滑 家用",
            "宜家简约书桌 实木",
            "口红 哑光丝绒",
            "考研数学教材 高数",
            "山地自行车 21速 变速",
            "红色连衣裙 夏季新款",
            "吉他 民谣单板 初学者",
        ],
    )
    def test_out_of_domain(self, agent, title):
        result = agent.classify_product(title)
        assert "无法判断" in result, f"域外文本被硬塞分类: {title} -> {result}"

    def test_ood_stats_exist(self):
        """域外门控统计文件必须存在（缺失则门控静默失效）"""
        from config.settings import settings

        assert os.path.isfile(settings.ood_stats_path), "缺少 ood_stats.npz"


# ------------------------------------------------------------------ #
#  NER 实体抽取
# ------------------------------------------------------------------ #


class TestNerModel:
    """NER 实体抽取回归（受限解码 + ModelCache）"""

    @pytest.fixture(scope="class")
    def agent(self):
        from agents.ner_agent import NerAgent
        from config.settings import settings

        return NerAgent(settings.ner_model_path)

    def test_brand_product(self, agent):
        r = agent.extract("华为Mate60 Pro 5G手机")
        assert "华为" in r["品牌"]
        assert any("手机" in p for p in r["商品"])

    def test_brand_only(self, agent):
        r = agent.extract("海尔冰箱 一级能效 双开门")
        assert "海尔" in r["品牌"]
        assert "冰箱" in r["商品"]

    def test_empty_input(self, agent):
        assert agent.extract("") == {"品牌": [], "商品": [], "款式": [], "规格": []}
        assert agent.extract("😀😀😀") == {"品牌": [], "商品": [], "款式": [], "规格": []}


# ------------------------------------------------------------------ #
#  模型加载期量化
# ------------------------------------------------------------------ #


class TestQuantization:
    """int8 加载期量化：模型可加载、可推理、默认开启（CPU）"""

    def test_settings_default(self):
        from config.settings import settings

        assert settings.model_quantize is True

    def test_classification_model_quantized(self):
        import torch

        from models.persistence import load_classification_model

        if torch.cuda.is_available():
            pytest.skip("GPU 环境不量化")
        _, model, labels, _ = load_classification_model("models/best")
        quantized = sum(
            1
            for m in model.modules()
            if isinstance(m, torch.ao.nn.quantized.modules.linear.Linear)
        )
        assert quantized > 0, "分类模型未应用 int8 量化"
        assert len(labels) == 4

    def test_ner_model_quantized(self):
        import torch

        from models.persistence import load_token_classification_model

        if torch.cuda.is_available():
            pytest.skip("GPU 环境不量化")
        _, model, labels, _ = load_token_classification_model("models/ner/best_model")
        quantized = sum(
            1
            for m in model.modules()
            if isinstance(m, torch.ao.nn.quantized.modules.linear.Linear)
        )
        assert quantized > 0, "NER 模型未应用 int8 量化"
        assert len(labels) == 9


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
