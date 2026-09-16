"""
安全加固回归测试

覆盖本轮修复的 4 个安全/正确性项，防止回归：
1. 订单 PII 脱敏（user_id / 地址 / 收货人）+ 越权查单防枚举
2. 长期记忆去重（原始余弦相似度 + 过期记忆跳过）
3. TraceContext 并发隔离（contextvars）
4. KG-QA Cypher 校验 + 只读会话

全部为纯函数 / Mock / 假嵌入模型，不依赖 Neo4j / MySQL / 模型文件。
"""

import contextvars
import os
import sys
import time
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pytest

from agents.kg_qa_agent import KGQAAgent
from agents.order_agent import OrderAgent
from config.settings import settings
from orchestration.memory import (
    LongTermMemory,
    MemoryManager,
    _same_user_namespace,
    _user_namespace,
)
from orchestration.observability import TraceContext

# ------------------------------------------------------------------ #
#  1. 订单 PII 脱敏
# ------------------------------------------------------------------ #


class TestOrderPII:
    ORDER = {
        "order_id": "ORD123456",
        "status": "已发货",
        "total_amount": 199.0,
        "create_time": "2026-09-01 10:00:00",
        "user_id": "u_8888",
        "consignee": "上海市浦东新区xx路1号",
        "consignee_name": "张三",
    }

    def _fmt(self, show_pii):
        return OrderAgent(db_config={})._format_order(self.ORDER, None, show_pii=show_pii)

    def test_mask_user_id_when_anonymous(self):
        out = self._fmt(show_pii=False)
        assert "u_8888" not in out
        assert "用户ID: u_" in out  # 脱敏保留前 2 位

    def test_show_user_id_for_owner(self):
        out = self._fmt(show_pii=True)
        assert "u_8888" in out

    def test_mask_address_and_consignee(self):
        out = self._fmt(show_pii=False)
        assert "浦东新区xx路" not in out
        assert "张三" not in out

    def test_unknown_placeholder_not_masked(self):
        out = OrderAgent(db_config={})._format_order({}, None, show_pii=False)
        assert "未知" in out

    def test_mask_pii(self):
        assert OrderAgent._mask_pii("a") == "*"
        assert OrderAgent._mask_pii("ab") == "**"
        assert OrderAgent._mask_pii("abcd") == "ab**"
        assert OrderAgent._mask_pii("abcdefghij") == "ab******"


class TestOrderCrossUser:
    def test_cross_user_order_returns_not_found(self):
        """越权查单与"未找到"返回相同措辞，防止订单号存在性枚举。"""
        pytest.importorskip("sqlalchemy")
        agent = OrderAgent(db_config={})
        fake_conn = MagicMock()
        fake_conn.execute.return_value.mappings.return_value.first.return_value = {
            "order_id": "ORD1",
            "user_id": "victim",
            "status": "已发货",
            "total_amount": 100,
            "create_time": "2026-01-01",
        }
        agent.engine = MagicMock()
        agent.engine.connect.return_value = fake_conn

        out = agent.query_order("ORD1", user_id="attacker")

        assert "未找到订单" in out
        assert "无权查看" not in out


# ------------------------------------------------------------------ #
#  2. 长期记忆去重
# ------------------------------------------------------------------ #


class _FakeEmbedder:
    """确定性假嵌入：相同文本 → 相同向量；不同文本 → 随机（低相似）向量。"""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self._cache = {}

    def encode(self, texts, normalize_embeddings=True):
        vecs = []
        for t in texts:
            key = str(t)
            if key not in self._cache:
                rng = np.random.RandomState(sum(ord(c) for c in key) % 1009 + len(key))
                v = rng.rand(self.dim).astype(np.float32)
                v /= float(np.linalg.norm(v)) + 1e-9
                self._cache[key] = v
            vecs.append(self._cache[key])
        return np.stack(vecs).astype(np.float32)


class _ControlledEmbedder:
    """按文本显式指定归一化向量，便于构造确定的相似度排序（测试 FAISS 检索路径）。"""

    def __init__(self, vectors: dict):
        self._vectors = {k: np.asarray(v, dtype=np.float32) for k, v in vectors.items()}

    def encode(self, texts, normalize_embeddings=True):
        return np.stack([self._vectors[t] for t in texts]).astype(np.float32)


class TestLongTermMemoryDedup:
    def _ltm(self, tmp_path):
        return LongTermMemory(store_path=str(tmp_path), embedder=_FakeEmbedder())

    def test_dedup_identical_content_and_raise_importance(self, tmp_path):
        ltm = self._ltm(tmp_path)
        ltm._add_memory("用户偏好蓝牙耳机", "preference", "s1", importance=0.5)
        ltm._add_memory("用户偏好蓝牙耳机", "preference", "s1", importance=0.9)
        assert len(ltm.memories) == 1
        assert ltm.memories[0].importance == 0.9

    def test_no_dedup_dissimilar_content(self, tmp_path):
        ltm = self._ltm(tmp_path)
        ltm._add_memory("用户偏好蓝牙耳机", "preference", "s1")
        ltm._add_memory("用户偏好红色外套", "preference", "s1")
        assert len(ltm.memories) == 2

    def test_expired_memory_not_deduped(self, tmp_path):
        """过期记忆不参与去重——否则新事实会被写进检索不到的旧条目（静默丢写）。"""
        ltm = self._ltm(tmp_path)
        ltm._add_memory("用户偏好蓝牙耳机", "preference", "s1")
        # 将已存记忆置为过期
        ltm.memories[0].ttl_days = 30
        ltm.memories[0].timestamp = time.time() - 999 * 86400
        ltm._add_memory("用户偏好蓝牙耳机", "preference", "s1")
        assert len(ltm.memories) == 2

    def test_search_fetches_beyond_active_count_when_expired(self, tmp_path):
        """FAISS 检索 fetch 上限按索引总量而非活跃数——否则过期项占比高时漏召回。

        构造：仅 1 条活跃记忆 A，但 A 在索引里相似度排名垫底（第 4），
        3 条过期记忆 E1/E2/E3 排名更靠前。旧实现 fetch_k = min(k*3, 活跃数=1)=1
        只取到过期项 → 过滤后为空；修复后按 ntotal=4 取，能召回 A。
        """
        pytest.importorskip("faiss")
        vecs = {
            "memory E1": [0.966, 0.259],  # 与 query Q 夹角 15° → cos 0.966
            "memory E2": [0.707, 0.707],  # 45° → cos 0.707
            "memory E3": [0.259, 0.966],  # 75° → cos 0.259
            "memory A": [-0.5, 0.866],  # 120° → cos -0.5（排名最后）
            "query Q": [1.0, 0.0],
        }
        ltm = LongTermMemory(store_path=str(tmp_path), embedder=_ControlledEmbedder(vecs))
        for c in ("memory E1", "memory E2", "memory E3"):
            ltm._add_memory(c, "fact", "s1")
        ltm._add_memory("memory A", "fact", "s1")
        # 将 E1/E2/E3 置为过期，仅 A 活跃
        for m in ltm.memories:
            if m.content.startswith("memory E"):
                m.timestamp = time.time() - 999 * 86400
                m.ttl_days = 1
        results = ltm.search("query Q", k=3, exclude_expired=True, use_forgetting=False)
        assert [r["content"] for r in results] == ["memory A"]


# ------------------------------------------------------------------ #
#  3. 跨会话共享记忆按 user 命名空间隔离
# ------------------------------------------------------------------ #


class TestSharedMemoryIsolation:
    """共享记忆（preference/decision 自动 shared=True）必须限定同一 user_id 命名空间。

    否则开启 memory_shared_enabled 后，A 用户的偏好会被注入 B 用户的上下文（IDOR）。
    """

    def _mm(self, tmp_path):
        ltm = LongTermMemory(store_path=str(tmp_path), embedder=_FakeEmbedder())
        mm = MemoryManager(llm=None, embedder=_FakeEmbedder())
        mm._long_term = ltm
        return mm

    def test_namespace_helpers(self):
        assert _user_namespace("u_A::s1") == "u_A"
        assert _user_namespace("s1") == ""
        assert _same_user_namespace("u_A::s1", "u_A::s2") is True
        assert _same_user_namespace("u_A::s1", "u_B::s2") is False
        # 匿名会话（空命名空间）之间不共享
        assert _same_user_namespace("s1", "s2") is False

    def test_shared_notes_exclude_other_users(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "memory_shared_enabled", True)
        mm = self._mm(tmp_path)
        mm._long_term._add_memory("偏好华为品牌", "preference", "u_A::s1")
        mm._long_term._add_memory("偏好苹果品牌", "preference", "u_B::s2")

        # 用户 B 的画像只能看到 B 自己的共享记忆，看不到 A 的
        notes = mm.build_user_profile("u_B::s3").get("shared_notes", [])
        assert "偏好苹果品牌" in notes
        assert "偏好华为品牌" not in notes

    def test_shared_notes_same_user_across_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "memory_shared_enabled", True)
        mm = self._mm(tmp_path)
        mm._long_term._add_memory("偏好华为品牌", "preference", "u_A::s1")

        # 同一用户另一会话能看到该偏好
        notes = mm.build_user_profile("u_A::s2").get("shared_notes", [])
        assert "偏好华为品牌" in notes

    def test_shared_notes_exclude_anonymous_cross_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "memory_shared_enabled", True)
        mm = self._mm(tmp_path)
        mm._long_term._add_memory("偏好华为品牌", "preference", "s1")
        mm._long_term._add_memory("偏好苹果品牌", "preference", "s2")

        # 匿名会话之间不共享（不同匿名用户是不同的人）
        profile = mm.build_user_profile("s3")
        assert profile.get("shared_notes", []) == []


# ------------------------------------------------------------------ #
#  4. TraceContext 并发隔离
# ------------------------------------------------------------------ #


class TestTraceContextIsolation:
    def test_contexts_do_not_clobber(self):
        TraceContext.clear()
        TraceContext.start_trace("main")
        main_span = TraceContext._current_span.get()

        def worker(name):
            TraceContext.start_trace(f"trace-{name}")
            return TraceContext._current_span.get().name

        r1 = contextvars.copy_context().run(worker, "A")
        r2 = contextvars.copy_context().run(worker, "B")

        assert r1 == "trace-A"
        assert r2 == "trace-B"
        # 主上下文不受子上下文影响
        assert TraceContext._current_span.get() is main_span
        assert TraceContext._current_span.get().name == "main"
        TraceContext.clear()


# ------------------------------------------------------------------ #
#  5. KG-QA Cypher 校验 + 只读会话
# ------------------------------------------------------------------ #


class TestKGQACypher:
    def _agent(self, driver=None):
        return KGQAAgent(neo4j_driver=driver, llm=None)

    def test_accept_valid_match(self):
        out = self._agent()._entity_alignment(
            "MATCH (t:Trademark) WHERE t.name CONTAINS '华为' RETURN t.name LIMIT 10"
        )
        assert out

    def test_reject_write_keywords(self):
        agent = self._agent()
        for bad in ("CREATE (n)", "MATCH (n) DELETE n", "MERGE (n:SPU)", "CALL db.labels()"):
            assert agent._entity_alignment(bad) == "", bad

    def test_reject_non_match_start(self):
        assert self._agent()._entity_alignment("RETURN 1") == ""

    def test_reject_multistatement(self):
        out = self._agent()._entity_alignment("MATCH (n) RETURN n; MATCH (m) RETURN m")
        assert out == ""

    def test_inject_limit(self):
        out = self._agent()._entity_alignment("MATCH (n:SPU) RETURN n.name")
        assert "LIMIT" in out

    def test_reject_cartesian_product(self):
        out = self._agent()._entity_alignment("MATCH (a:SPU), (b:SPU) RETURN a.name, b.name")
        assert out == ""

    def test_reject_unwind(self):
        # UNWIND 可做数据展开（DoS 面），应被黑名单拒绝
        out = self._agent()._entity_alignment(
            "MATCH (n:SPU) UNWIND range(0, 1000000) AS i RETURN n"
        )
        assert out == ""

    def test_reject_union(self):
        # UNION 可拼接第二条子查询绕过白名单，应被黑名单拒绝
        out = self._agent()._entity_alignment(
            "MATCH (n:SPU) RETURN n UNION MATCH (m) RETURN m"
        )
        assert out == ""

    def test_reject_block_comment_obfuscation(self):
        # 用 /* */ 拆开禁止关键词绕过黑名单：剥离注释后应命中 CREATE 被拒绝
        out = self._agent()._entity_alignment("MATCH (n) CR/*x*/EATE (m) RETURN m")
        assert out == ""

    def test_read_session_uses_read_access(self):
        from neo4j import READ_ACCESS

        mock_driver = MagicMock()
        agent = self._agent(driver=mock_driver)
        agent._read_session()
        mock_driver.session.assert_called_once_with(default_access_mode=READ_ACCESS)
