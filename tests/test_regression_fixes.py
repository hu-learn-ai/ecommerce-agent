"""回归测试：4 个已修复缺陷的守护用例。

每个用例都对应一次真实发生过的缺陷，改动相关代码时若退化会立即失败：

1. ``/api/classify`` 的 ``top_k`` 越界
   —— 模型只有 4 个类别，接口却允许 top_k 到 10，``torch.topk`` 直接抛
   ``RuntimeError: selected index k out of range``。

2. ``RepeatDetectionCallback`` 防死循环失效
   —— ``BaseCallbackHandler.raise_error`` 默认 False，LangChain 的
   ``CallbackManager.handle_event`` 会吞掉回调抛出的异常，保护形同虚设。

3. 短期记忆里的 ``role="tool"`` 事件被静默丢弃
   —— ``add_event()`` 写入的工具结果在 ``_build_chat_history()`` 里没有
   任何分支处理，"那第二个呢"之类的指代无法解析到上一轮候选。

4. L3 语义缓存跨意图误命中
   —— 语义缓存只按 query 相似度匹配、不区分 intent，
   "蓝牙耳机"的搜索回答会被当成分类回答返回（实测相似度 1.0）。

全部不启动 Neo4j / 真实模型文件 / 外部 LLM。
"""

import os
import sys

import numpy as np
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.classify_agent import ClassifyAgent  # noqa: E402
from orchestration.memory import MemoryManager  # noqa: E402
from orchestration.model_router import CostOptimizer, MultiLevelCache, SemanticCache  # noqa: E402
from orchestration.react_orchestrator import (  # noqa: E402
    ReactOrchestrator,
    RepeatDetectionCallback,
)


# ------------------------------------------------------------------ #
#  1. top_k 越界
# ------------------------------------------------------------------ #
class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }


class _FakeOutputs:
    # 第一类 logit 远高于其余，softmax 后 top-1 置信度接近 1，
    # 保证走完 topk 分支而不是被 MIN_CONFIDENCE 提前判为"无法判断"
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    hidden_states = (torch.randn(1, 3, 768),)


class _FakeModel:
    def __call__(self, **kwargs):
        return _FakeOutputs()


def _make_agent() -> ClassifyAgent:
    agent = ClassifyAgent(model_path="models/best", labels_path="models/best/labels.txt")
    agent._tokenizer = _FakeTokenizer()
    agent._model = _FakeModel()
    agent._labels = ["医药保健", "家用电器", "手机数码", "食品生鲜"]
    agent._ood_gate = None
    return agent


class TestTopKOutOfRange:
    @pytest.mark.parametrize("k", [1, 3, 4])
    def test_normal_k(self, k):
        """k 不超过类别数时按请求返回。"""
        assert len(_make_agent().get_top_k("小米手机", k)) == k

    @pytest.mark.parametrize("k", [5, 10, 999])
    def test_k_clamped_to_num_classes(self, k):
        """k 超过类别数时自动裁剪为 4，绝不抛 RuntimeError。"""
        result = _make_agent().get_top_k("小米手机", k)
        assert len(result) == 4

    def test_non_positive_k_is_safe(self):
        """k<=0 也不能崩（裁剪到至少 1）。"""
        assert len(_make_agent().get_top_k("小米手机", 0)) == 1


# ------------------------------------------------------------------ #
#  2. 防死循环回调
# ------------------------------------------------------------------ #
class TestRepeatDetectionCallback:
    def test_raise_error_is_enabled(self):
        """raise_error 必须为 True，否则 on_tool_start 的异常会被框架吞掉。"""
        assert RepeatDetectionCallback().raise_error is True

    def test_loop_is_actually_interrupted(self):
        """通过真实的 CallbackManager 走一遍，确认异常能穿透到调用方。"""
        from langchain_core.callbacks.manager import CallbackManager

        cb = RepeatDetectionCallback(max_repeats=2)
        cm = CallbackManager.configure([cb])
        with pytest.raises(ValueError, match="重复调用"):
            for i in range(5):
                cm.on_tool_start({"name": "search_products"}, "蓝牙耳机", run_id=f"r{i}")

    def test_different_args_do_not_trip(self):
        """参数不同属于正常多步推理，不应被误判。"""
        cb = RepeatDetectionCallback(max_repeats=2)
        for i in range(5):
            cb.on_tool_start({"name": "search_products"}, f"关键词{i}")
        assert len(cb.call_history) == 5


# ------------------------------------------------------------------ #
#  3. 工具事件进入上下文
# ------------------------------------------------------------------ #
class TestToolEventInContext:
    def _build(self):
        mm = MemoryManager(llm=None, embedder=None)
        mm.add_message("s-tool", "user", "推荐蓝牙耳机", intent="recommend")
        mm.add_event(
            "s-tool",
            "tool_call",
            {"tool": "recommend_products", "result": "华为蓝牙耳机 ¥299"},
        )
        mm.add_message("s-tool", "assistant", "为您推荐...", intent="recommend")
        return mm

    def test_build_context_keeps_tool_role(self):
        ctx = self._build().build_context("s-tool", "那有降噪的吗")
        assert "tool" in [m["role"] for m in ctx]

    def test_build_chat_history_surfaces_tool_result(self):
        mm = self._build()
        orch = ReactOrchestrator.__new__(ReactOrchestrator)
        orch.memory = mm
        history = orch._build_chat_history("s-tool", "那有降噪的吗")
        assert any("华为蓝牙耳机" in str(m.content) for m in history)

    def test_system_message_never_emitted_without_session(self):
        """无有效会话 ID 时不注入任何历史（M2 修复）。"""
        mm = self._build()
        orch = ReactOrchestrator.__new__(ReactOrchestrator)
        orch.memory = mm
        assert orch._build_chat_history("default", "那有降噪的吗") == []


# ------------------------------------------------------------------ #
#  4. L3 语义缓存 intent 隔离
# ------------------------------------------------------------------ #
def _constant_embed(_text):
    """任意两句都"完全相似"，用于逼出最坏情况下的串意图误命中。"""
    return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class TestSemanticCacheNamespace:
    def test_semantic_cache_requires_same_namespace(self):
        c = SemanticCache(embed_fn=_constant_embed)
        c.set("k", "搜索结果", query="蓝牙耳机多少钱", namespace="search|")
        assert c.get("蓝牙耳机多少钱", namespace="search|") == "搜索结果"
        assert c.get("蓝牙耳机是什么分类", namespace="classify|") is None

    def test_multi_level_cache_hits_in_same_intent(self):
        c = MultiLevelCache(embed_fn=_constant_embed)
        ns = MultiLevelCache.make_namespace("search")
        c.set("search|蓝牙耳机", "搜索结果", query="蓝牙耳机", namespace=ns)
        assert c.get("search|蓝牙耳机", query="蓝牙耳机", namespace=ns) == "搜索结果"

    def test_cost_optimizer_isolates_intent_and_params(self):
        opt = CostOptimizer()
        opt.set_embed_fn(_constant_embed)
        opt.cache_set("search", "蓝牙耳机多少钱", "【搜索】蓝牙耳机 10 条结果")
        # 同 intent + 同参数 → 命中
        assert opt.cache_get("search", "蓝牙耳机多少钱") == "【搜索】蓝牙耳机 10 条结果"
        # 不同 intent → 不命中
        assert opt.cache_get("classify", "蓝牙耳机是什么分类") is None
        # 同 intent 但参数不同（top_k 会改变答案）→ 不命中
        assert opt.cache_get("search", "蓝牙耳机多少钱", top_k=10) is None

    def test_namespace_includes_sorted_params(self):
        assert MultiLevelCache.make_namespace("search", top_k=5) == "search|top_k=5"
        assert (
            MultiLevelCache.make_namespace("search", top_k=5, page=1)
            == MultiLevelCache.make_namespace("search", page=1, top_k=5)
        )
