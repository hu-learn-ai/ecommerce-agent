"""API 层安全回归测试（C7）。

覆盖 api/app.py 的安全中间件与身份处理，全部不启动 Neo4j / LLM：
- API Key 认证 fail-closed（未配置 → 503，密钥错 → 401，正确 → 放行）
- IP 限流（超限 → 429）+ TRUST_PROXY 防 X-Forwarded-For 伪造
- 可信头身份解析 + user_profile 身份字段剥离
- reasoning 剥离
- 会话清除按 user_id 命名空间隔离（C6）
"""

import os
import sys
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.app as app_module
from orchestration.memory import memory_namespace_key

# ReactOrchestrator 依赖 langchain → aiohttp；本地 anaconda 存在 attr/attrs 冲突会导致
# 导入失败，而 CI（ubuntu + requirements-dev.txt）可正常导入。降级为 skip，不阻塞其余用例。
try:
    from orchestration.react_orchestrator import ReactOrchestrator

    _HAS_REACT = True
except Exception:  # noqa: BLE001
    ReactOrchestrator = None
    _HAS_REACT = False


def _build_app(*middlewares):
    """构造一个只挂指定中间件的最小 FastAPI 应用（隔离中间件顺序/状态）。"""
    app = FastAPI()
    for mw in middlewares:
        app.middleware("http")(mw)

    @app.get("/api/health")
    def _health():
        return {"status": "ok"}

    @app.get("/api/test")
    def _test():
        return {"ok": True}

    return app


# ------------------------------------------------------------------ #
#  纯函数：身份解析 / 画像剥离 / reasoning 剥离 / 命名空间键
# ------------------------------------------------------------------ #


class TestResolveUserID:
    def test_from_trusted_header(self, monkeypatch):
        monkeypatch.setattr(app_module, "_USER_ID_HEADER", "X-User-Id")
        req = MagicMock()
        req.headers = {"X-User-Id": "u_8888"}
        assert app_module._resolve_user_id(req) == "u_8888"

    def test_none_when_header_unconfigured(self, monkeypatch):
        monkeypatch.setattr(app_module, "_USER_ID_HEADER", "")
        req = MagicMock()
        req.headers = {"X-User-Id": "u_8888"}
        assert app_module._resolve_user_id(req) is None

    def test_none_when_header_missing(self, monkeypatch):
        monkeypatch.setattr(app_module, "_USER_ID_HEADER", "X-User-Id")
        req = MagicMock()
        req.headers = {}
        assert app_module._resolve_user_id(req) is None


class TestSanitizeProfile:
    def test_strips_identity_keys(self):
        out = app_module._sanitize_user_profile(
            {"user_id": "x", "uid": "y", "open_id": "z", "openid": "w", "budget": 100}
        )
        assert out == {"budget": 100}

    def test_none_for_empty(self):
        assert app_module._sanitize_user_profile(None) is None
        assert app_module._sanitize_user_profile({}) is None


class TestStripReasoning:
    def test_extract_final_answer_field(self):
        assert app_module._strip_reasoning('{"final_answer": "你好"}') == "你好"

    def test_strip_prefix(self):
        assert app_module._strip_reasoning("最终答案: 你好") == "你好"


class TestMemoryNamespaceKey:
    def test_namespaced_when_user_id(self):
        assert memory_namespace_key("u_1", "abc") == "u_1::abc"

    def test_unchanged_when_anonymous(self):
        assert memory_namespace_key(None, "abc") == "abc"
        assert memory_namespace_key("", "abc") == "abc"


# ------------------------------------------------------------------ #
#  API Key 中间件（fail-closed）
# ------------------------------------------------------------------ #


class TestApiKeyMiddleware:
    def test_fail_closed_when_no_key_configured(self, monkeypatch):
        monkeypatch.setattr(app_module, "_API_ACCESS_KEY", "")
        client = TestClient(_build_app(app_module.verify_api_key))
        assert client.get("/api/test").status_code == 503
        # 健康检查豁免
        assert client.get("/api/health").status_code == 200

    def test_401_on_missing_or_bad_key(self, monkeypatch):
        monkeypatch.setattr(app_module, "_API_ACCESS_KEY", "secret123")
        client = TestClient(_build_app(app_module.verify_api_key))
        assert client.get("/api/test").status_code == 401  # 无 key
        assert client.get("/api/test", headers={"X-API-Key": "wrong"}).status_code == 401

    def test_200_on_good_key(self, monkeypatch):
        monkeypatch.setattr(app_module, "_API_ACCESS_KEY", "secret123")
        client = TestClient(_build_app(app_module.verify_api_key))
        assert client.get("/api/test", headers={"X-API-Key": "secret123"}).status_code == 200


# ------------------------------------------------------------------ #
#  限流中间件 + TRUST_PROXY 防 XFF 伪造
# ------------------------------------------------------------------ #


class TestRateLimitMiddleware:
    def test_429_after_limit(self, monkeypatch):
        monkeypatch.setattr(app_module, "_RATE_LIMIT", 2)
        monkeypatch.setattr(app_module, "_TRUST_PROXY", False)
        monkeypatch.setattr(app_module, "_rate_buckets", app_module.defaultdict(list))
        client = TestClient(_build_app(app_module.rate_limit))
        assert client.get("/api/test").status_code == 200
        assert client.get("/api/test").status_code == 200
        assert client.get("/api/test").status_code == 429

    def test_xff_ignored_when_not_trusting_proxy(self, monkeypatch):
        # TRUST_PROXY=False：X-Forwarded-For 可被客户端伪造，须忽略，统一按真实 client 计数
        monkeypatch.setattr(app_module, "_RATE_LIMIT", 1)
        monkeypatch.setattr(app_module, "_TRUST_PROXY", False)
        monkeypatch.setattr(app_module, "_rate_buckets", app_module.defaultdict(list))
        client = TestClient(_build_app(app_module.rate_limit))
        assert (
            client.get("/api/test", headers={"X-Forwarded-For": "1.2.3.4"}).status_code
            == 200
        )
        # 换个伪造 IP 仍算同一客户端 → 超限
        assert (
            client.get("/api/test", headers={"X-Forwarded-For": "9.9.9.9"}).status_code
            == 429
        )

    def test_xff_used_when_trusting_proxy(self, monkeypatch):
        monkeypatch.setattr(app_module, "_RATE_LIMIT", 1)
        monkeypatch.setattr(app_module, "_TRUST_PROXY", True)
        monkeypatch.setattr(app_module, "_rate_buckets", app_module.defaultdict(list))
        client = TestClient(_build_app(app_module.rate_limit))
        assert (
            client.get("/api/test", headers={"X-Forwarded-For": "1.2.3.4"}).status_code
            == 200
        )
        # 不同 XFF = 不同客户端，各自首次放行
        assert (
            client.get("/api/test", headers={"X-Forwarded-For": "5.6.7.8"}).status_code
            == 200
        )
        # 同一 XFF 再来 → 超限
        assert (
            client.get("/api/test", headers={"X-Forwarded-For": "1.2.3.4"}).status_code
            == 429
        )


# ------------------------------------------------------------------ #
#  会话清除按 user_id 命名空间隔离（C6）
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _HAS_REACT, reason="langchain/attrs 环境不可导入 ReactOrchestrator")
class TestClearSessionNamespace:
    def test_namespaced_by_user_id(self):
        orch = object.__new__(ReactOrchestrator)
        orch.memory = MagicMock()
        orch.clear_session("abc", user_id="u_1")
        orch.memory.clear_session.assert_called_once_with("u_1::abc")

    def test_anonymous_unchanged(self):
        orch = object.__new__(ReactOrchestrator)
        orch.memory = MagicMock()
        orch.clear_session("abc", user_id=None)
        orch.memory.clear_session.assert_called_once_with("abc")


# ------------------------------------------------------------------ #
#  DELETE 端点集成：可信头 user_id 正确透传给 clear_session
# ------------------------------------------------------------------ #


class TestDeleteSessionEndpoint:
    def test_delete_session_passes_user_id(self, monkeypatch):
        monkeypatch.setattr(app_module, "_API_ACCESS_KEY", "k")
        monkeypatch.setattr(app_module, "_USER_ID_HEADER", "X-User-Id")
        monkeypatch.setattr(app_module, "_rate_buckets", app_module.defaultdict(list))
        fake_orch = MagicMock()
        fake_orch.clear_session = MagicMock()
        monkeypatch.setattr(app_module, "orchestrator", fake_orch)

        client = TestClient(app_module.app)
        r = client.delete(
            "/api/session/abc",
            headers={"X-API-Key": "k", "X-User-Id": "u_7"},
        )
        assert r.status_code == 200
        fake_orch.clear_session.assert_called_once_with("abc", user_id="u_7")
