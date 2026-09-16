"""鲁棒性测试（降级路径 + 并发安全）。

补强 C6/C7 之后仍缺的两类测试：
- 降级路径：LLM / 嵌入模型 / FAISS / 长期记忆 / 编排器缺失时优雅降级、不崩溃
- 并发安全：TraceContext (contextvars) 与短时会话在 asyncio 并发下不串台

全部不启动 Neo4j / 模型文件 / 外部 LLM。
"""

import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient

import api.app as app_module
from config.settings import settings
from orchestration.memory import LongTermMemory, MemoryManager
from orchestration.observability import TraceContext

# ------------------------------------------------------------------ #
#  降级路径
# ------------------------------------------------------------------ #


class TestDegradation:
    def test_build_context_works_without_long_term(self, monkeypatch):
        """长期记忆禁用时，build_context 仍能基于短期记忆构建上下文。"""
        monkeypatch.setattr(settings, "memory_long_term_enabled", False)
        mm = MemoryManager(llm=None, embedder=None)
        assert mm._long_term is None
        mm.add_message("s1", "user", "你好")
        ctx = mm.build_context("s1", "")
        assert len(ctx) == 1
        assert ctx[0]["role"] == "user"
        assert ctx[0]["content"] == "你好"

    def test_extract_and_store_without_llm_is_noop(self, tmp_path):
        """LLM 缺失时长期记忆提取直接返回，不抛异常、不写入。"""
        ltm = LongTermMemory(store_path=str(tmp_path), embedder=None)
        ltm.extract_and_store("我想买蓝牙耳机", "好的", "s1")
        assert ltm.memories == []

    def test_search_keyword_fallback_without_embedder(self, tmp_path):
        """无嵌入模型/FAISS 时，检索降级为关键词匹配，而非静默返回空。"""
        ltm = LongTermMemory(store_path=str(tmp_path), embedder=None)
        ltm._add_memory("用户偏好蓝牙耳机", "preference", "s1")
        results = ltm.search("蓝牙", k=3, session_id="s1")
        assert any("蓝牙" in r["content"] for r in results)

    def test_shared_context_empty_without_embedder(self, tmp_path):
        """无嵌入模型时，共享记忆检索优雅返回空。"""
        ltm = LongTermMemory(store_path=str(tmp_path), embedder=None)
        ltm._add_memory("偏好华为品牌", "preference", "s1")
        mm = MemoryManager(llm=None, embedder=None)
        mm._long_term = ltm
        assert mm._get_shared_context("s1", "华为") == ""

    def test_chat_returns_503_when_orchestrator_not_ready(self, monkeypatch):
        """编排器未就绪（lifespan 初始化失败/未启动）时，业务接口返回 503 而非崩溃。"""
        monkeypatch.setattr(app_module, "_API_ACCESS_KEY", "k")
        monkeypatch.setattr(app_module, "_rate_buckets", app_module.defaultdict(list))
        monkeypatch.setattr(app_module, "orchestrator", None)
        client = TestClient(app_module.app)
        r = client.post("/api/chat", json={"message": "你好"}, headers={"X-API-Key": "k"})
        assert r.status_code == 503
        # 健康检查不受影响
        assert client.get("/api/health").status_code == 200


# ------------------------------------------------------------------ #
#  并发安全
# ------------------------------------------------------------------ #


class TestConcurrency:
    def test_trace_context_async_isolation(self):
        """TraceContext 在 asyncio 多任务并发下隔离，各任务只看得到自己的 span。"""

        async def worker(name):
            TraceContext.start_trace(f"trace-{name}")
            await asyncio.sleep(0)  # 让出控制权，制造任务交错
            await asyncio.sleep(0)
            return TraceContext._current_span.get().name

        async def main():
            TraceContext.clear()
            TraceContext.start_trace("main")
            results = await asyncio.gather(
                worker("A"), worker("B"), worker("C"), worker("D"), worker("E")
            )
            return results

        results = asyncio.run(main())
        assert sorted(results) == [
            "trace-A",
            "trace-B",
            "trace-C",
            "trace-D",
            "trace-E",
        ]
        TraceContext.clear()

    def test_concurrent_sessions_do_not_cross_contaminate(self, monkeypatch):
        """并发写入不同会话的短期记忆不串台。"""
        monkeypatch.setattr(settings, "memory_long_term_enabled", False)
        mm = MemoryManager(llm=None, embedder=None)

        async def writer(sid):
            for i in range(50):
                mm.add_message(sid, "user", f"{sid}-msg-{i}")
            return [m["content"] for m in mm.get_history(sid)]

        async def main():
            return await asyncio.gather(writer("s1"), writer("s2"), writer("s3"))

        h1, h2, h3 = asyncio.run(main())
        assert all("s1-msg" in c for c in h1)
        assert all("s2-msg" in c for c in h2)
        assert all("s3-msg" in c for c in h3)
        # 无跨会话串台
        assert all(("s2-msg" not in c and "s3-msg" not in c) for c in h1)
