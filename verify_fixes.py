# -*- coding: utf-8 -*-
"""针对安全修复的轻量逻辑验证（不依赖重型第三方包，重型依赖以桩代替）"""
import importlib.util
import os
import sys
import types

# ================= 桩模块引导 =================
_stub_modules = [
    "dotenv", "dotenv.main",
    "langchain", "langchain.agents", "langchain_classic", "langchain_classic.agents",
    "langchain_core", "langchain_core.callbacks", "langchain_core.messages",
    "langchain_core.prompts", "langchain_core.tools", "langchain_openai",
    "langchain_community", "langchain_community.callbacks",
    "langgraph", "langsmith", "cachetools", "numpy",
]


def _make_stub(name):
    m = types.ModuleType(name)

    def __getattr__(attr):
        # 返回可作基类、可实例化、可属性访问的通用桩类
        class _StubCls:
            def __init__(self, *a, **k):
                pass

            def __call__(self, *a, **k):
                return _StubCls()

            def __getattr__(self, item):
                return _StubCls()

            def __iter__(self):
                return iter(())

            def __len__(self):
                return 0

            def __bool__(self):
                return True

            def __repr__(self):
                return f"<stub:{name}.{attr}>"

        return _StubCls

    m.__getattr__ = __getattr__
    return m


for _n in _stub_modules:
    sys.modules.setdefault(_n, _make_stub(_n))

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


# ============ 1. MemoryManager.clear_session 删除长期记忆 ============
from orchestration.memory import MemoryManager  # noqa: E402


class FakeLongTerm:
    def __init__(self, memories):
        self.memories = memories
        self.saved = 0
        self.rebuilt = 0

    def _ensure_loaded(self):
        return None

    def _save(self):
        self.saved += 1

    def _build_faiss_index(self):
        self.rebuilt += 1


def make_mem(sid, content):
    class M:
        pass

    m = M()
    m.session_id = sid
    m.content = content
    return m


memories = [make_mem("A", "A 的偏好"), make_mem("A", "A 的决策"), make_mem("B", "B 的偏好")]
fake_lt = FakeLongTerm(memories)
mm = MemoryManager.__new__(MemoryManager)
mm._short_term = {
    "A": types.SimpleNamespace(messages=[1, 2], summary="s", clear=lambda: None),
    "B": types.SimpleNamespace(messages=[3], summary="", clear=lambda: None),
}
mm._long_term = fake_lt

mm.clear_session("A")

check(
    "clear_session 删除该会话长期记忆",
    all(m.session_id != "A" for m in fake_lt.memories) and len(fake_lt.memories) == 1,
    f"剩余 {len(fake_lt.memories)} 条（应为 1）",
)
check("clear_session 保留其他会话记忆", fake_lt.memories[0].session_id == "B")
check("clear_session 触发保存与重建", fake_lt.saved == 1 and fake_lt.rebuilt == 1)
check("clear_session 清除短期记忆", "A" not in mm._short_term and "B" in mm._short_term)

mm2 = MemoryManager.__new__(MemoryManager)
mm2._short_term = {}
mm2._long_term = FakeLongTerm([make_mem("A", "x")])
mm2.clear_session("NONEXIST")
check("无该会话长期记忆时不触发保存", mm2._long_term.saved == 0)

# ============ 2. 共享记忆门控 ============
import importlib

cfg = importlib.import_module("config.settings")
cfg.settings.memory_shared_enabled = False

from orchestration.memory import MemoryManager as MM2  # noqa: E402


class FakeStm:
    def __init__(self):
        self.messages = []

    def build_context_messages(self):
        return []


class FakeLT2:
    def __init__(self):
        self.embedder = object()

    def _ensure_loaded(self):
        return None

    def get_context_for_query(self, q):
        return "[当前会话长期记忆]"


mm3 = MM2.__new__(MM2)
mm3._short_term = {}
mm3._long_term = FakeLT2()
mm3.get_short_term = lambda sid: FakeStm()

ctx = mm3.build_context("s1", "查询蓝牙耳机")
shared_present = any("[跨会话共享记忆]" in (m.get("content") or "") for m in ctx)
long_term_present = any("[当前会话长期记忆]" in (m.get("content") or "") for m in ctx)
check("共享记忆默认关闭（不注入）", not shared_present, f"ctx={ctx}")
check("当前会话长期记忆仍注入", long_term_present)

# ============ 3. 会话记忆开关（MEMORYLESS）与 ReAct 模块可导入 ============
from orchestration.react_orchestrator import ReactOrchestrator  # noqa: E402

check("default 会话禁用记忆", not ReactOrchestrator._memory_enabled("default"))
check("空串会话禁用记忆", not ReactOrchestrator._memory_enabled(""))
check("None 会话禁用记忆", not ReactOrchestrator._memory_enabled(None))
check("唯一会话 ID 启用记忆", ReactOrchestrator._memory_enabled("s-abc-123"))

# ============ 4. DiskCache 过期清理（真实落盘验证） ============
import json
import tempfile
import time as _t

from orchestration.model_router import DiskCache  # noqa: E402

tmpdir = tempfile.mkdtemp(prefix="dc_test_")
cache = DiskCache(cache_dir=tmpdir)

# 写入一条新缓存
cache.set("k1", "v1")
p1 = cache._key_to_path("k1")
check("DiskCache 正常写入", os.path.exists(p1))

# 伪造一条过期缓存（timestamp 很早）
import hashlib

fake_key = "expired_key"
fake_path = os.path.join(tmpdir, hashlib.md5(fake_key.encode()).hexdigest() + ".json")
with open(fake_path, "w", encoding="utf-8") as f:
    json.dump({"value": "old", "timestamp": _t.time() - 3600 * 24 * 30}, f)

got = cache.get(fake_key)
check("DiskCache 过期条目返回 None", got is None)
check("DiskCache 过期条目被删除", not os.path.exists(fake_path))

# 新缓存仍可读
check("DiskCache 未过期条目可读", cache.get("k1") == "v1")

# 启动时清理
fake_key2 = "expired_key2"
fake_path2 = os.path.join(tmpdir, hashlib.md5(fake_key2.encode()).hexdigest() + ".json")
with open(fake_path2, "w", encoding="utf-8") as f:
    json.dump({"value": "old", "timestamp": _t.time() - 3600 * 24 * 30}, f)
DiskCache(cache_dir=tmpdir)  # 新实例启动 → 清理
check("DiskCache 启动时清理过期文件", not os.path.exists(fake_path2))

# ============ 5. TokenTracker 上限 ============
from orchestration.observability import TokenTracker  # noqa: E402

tt = TokenTracker()
for i in range(12000):
    tt.record(model="m", prompt_tokens=1, completion_tokens=1)
check("TokenTracker 明细有上限", len(tt._usage) == TokenTracker.MAX_USAGE_RECORDS)
check("TokenTracker 汇总仍正确", tt.get_summary()["total_calls"] == TokenTracker.MAX_USAGE_RECORDS)

# ============ 6. 源码级静态断言 ============
def read(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


kg = read(os.path.join("agents", "kg_qa_agent.py"))
check("kg_qa 白名单不再放行 CYPHER", '("MATCH", "OPTIONAL MATCH")' in kg)

app_src = read(os.path.join("api", "app.py"))
check("app.py 存在 fail-closed 中间件", "未配置 API_ACCESS_KEY" in app_src and "503" in app_src)
check("app.py 剥离身份字段", "_IDENTITY_KEYS" in app_src and "_sanitize_user_profile" in app_src)
check("app.py 限流信任受控", "TRUST_PROXY" in app_src)
check("app.py 生产关闭 reload", 'reload=os.getenv("UVICORN_RELOAD", "false")' in app_src)

r_src = read(os.path.join("orchestration", "react_orchestrator.py"))
check("订单意图禁止读缓存", 'intent in ("recommend", "order")' in r_src)
check("订单意图禁止写缓存", 'intent not in ("chitchat", "recommend", "order")' in r_src)
check("追踪不再记录原始工具输入", 'span.set_attribute("repeat_history"' not in r_src and "repeat_tools" in r_src)
check("SSE 错误脱敏", "请稍后重试" in r_src and 'yield f"抱歉，处理您的请求时出错: {e}"' not in r_src)

mr_src = read(os.path.join("orchestration", "model_router.py"))
check("磁盘缓存启动清理", "_cleanup_expired_files" in mr_src)

ob_src = read(os.path.join("orchestration", "observability.py"))
check("TokenTracker 有上限", "MAX_USAGE_RECORDS" in ob_src and "deque(maxlen=" in ob_src)

compose_src = read(os.path.join("docker-compose.yml"))
check("compose 不再暴露 3306", "3306:3306" not in compose_src)
check("compose 不再暴露 7474/7687", "7474:7474" not in compose_src and "7687:7687" not in compose_src)
check("faiss 只读挂载", "/app/data/faiss_index:ro" in compose_src)

df_src = read("Dockerfile")
check("Dockerfile 非 root 运行", "USER appuser" in df_src and "useradd" in df_src)

fe_src = read(os.path.join("frontend", "streamlit_app.py"))
check("前端携带 X-API-Key", "X-API-Key" in fe_src and "api_headers" in fe_src)

st_src = read(os.path.join("scripts", "start.sh"))
check("start.sh 单 worker", "--workers 1" in st_src)
check("start.sh source 方式加载 .env", "source .env" in st_src and "grep -v '^#' .env | xargs" not in st_src)

lic = read("LICENSE")
check("LICENSE 占位符已补全", "<你的姓名" not in lic and "ecommerce-agent contributors" in lic)

req = read("requirements.txt")
check("依赖补充上限", "fastapi>=0.115.0,<1.0.0" in req and "torch>=2.1.0,<3.0.0" in req)

mem_src = read(os.path.join("orchestration", "memory.py"))
check("共享记忆门控已接入 settings", "settings.memory_shared_enabled" in mem_src)
check("clear_session 删除长期记忆", "m.session_id != session_id" in mem_src)

# 清理临时目录
import shutil

shutil.rmtree(tmpdir, ignore_errors=True)

print()
failed = [r for r in results if not r[1]]
print(f"共 {len(results)} 项，通过 {len(results) - len(failed)} 项，失败 {len(failed)} 项")
if failed:
    for name, _ in failed:
        print(f"  FAIL: {name}")
    sys.exit(1)
print("全部验证通过")
