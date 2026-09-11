"""
FastAPI 统一 API 层 — 电商智能体 HTTP 接口

提供以下接口：
- POST /api/chat          : 统一对话入口（ReAct 工具循环 + 记忆系统）
- POST /api/chat/stream   : 流式对话入口（SSE 逐 token 返回）
- POST /api/classify      : 独立商品分类接口
- POST /api/search        : 独立商品搜索接口
- GET  /api/health        : 健康检查
- GET  /api/agents        : 查看已注册的 Agent 列表
- GET  /api/stats         : 系统统计（Token 用量、缓存命中率、记忆状态）
- GET  /api/trace         : 获取最近请求的追踪树
- GET  /docs              : Swagger API 文档
"""

import json
import os
import secrets
import sys
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import re

from config.settings import settings


def _strip_reasoning(text: str) -> str:
    """剥离 LLM 输出的 chain-of-thought 推理段，只保留最终答案。

    规则：
    1. 优先提取结构化字段 final_answer
    2. 砍掉最后一个独立 "---" 分隔线之前的内容
    3. 去掉 "最终答案:" / "Final Answer:" 前缀
    """
    if not text:
        return text

    # 1. 结构化输出 final_answer 字段
    m = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        return m.group(1).replace("\\n", "\n")

    # 2. 砍掉最后一个独立 --- 分隔线之前的内容（表格中的 |---| 不受影响）
    lines = text.split("\n")
    sep_idx = None
    for i, line in enumerate(lines):
        if re.fullmatch(r"\s*-{3,}\s*", line):
            sep_idx = i
    if sep_idx is not None:
        text = "\n".join(lines[sep_idx + 1 :]).strip()

    # 3. 去掉前缀
    text = re.sub(r"^(最终答案[:：]|Final Answer[:：])\s*", "", text.strip())
    return text.strip()

# ------------------------------------------------------------------ #
#  请求/响应模型
# ------------------------------------------------------------------ #


class ChatRequest(BaseModel):
    """对话请求"""

    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    session_id: Optional[str] = Field("default", description="会话 ID（用于多轮记忆）")
    user_profile: Optional[dict] = Field(None, description="用户画像")


class ChatResponse(BaseModel):
    """对话响应"""

    message: str
    intent: str
    agent_used: str
    mode: str = "react"  # react | fallback | cached
    tools_called: list = []
    steps: int = 0


class ClassifyRequest(BaseModel):
    """分类请求"""

    title: str = Field(..., min_length=1, description="商品标题")
    top_k: int = Field(3, ge=1, le=10, description="返回 Top-K 结果")


class SearchRequest(BaseModel):
    """搜索请求"""

    query: str = Field(..., min_length=1, description="搜索关键词")
    top_k: int = Field(10, ge=1, le=50, description="返回数量")
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None


# ------------------------------------------------------------------ #
#  全局实例
# ------------------------------------------------------------------ #

orchestrator = None  # ReactOrchestrator (主)
fallback_orchestrator = None  # ECommerceOrchestrator (降级)
neo4j_driver = None
shared_embedder = None


# ------------------------------------------------------------------ #
#  生命周期管理
# ------------------------------------------------------------------ #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期：启动时初始化，关闭时清理
    """
    global orchestrator, fallback_orchestrator, neo4j_driver, shared_embedder

    from orchestration.memory import MemoryManager
    from orchestration.observability import obs
    from orchestration.react_orchestrator import ReactOrchestrator
    from orchestration.registry import create_llm, create_neo4j_driver, register_all_agents
    from orchestration.router import RouterAgent

    # 设置 HuggingFace 镜像
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    # 设置 LangSmith tracing
    # obs 已在模块加载时自动初始化

    print("\n" + "=" * 60)
    print("  电商智能体系统启动中...")
    print(f"  ReAct 模式: {'[ON] 已启用' if settings.react_enabled else '[OFF] 未启用'}")
    print(f"  流式输出: {'[ON] 已启用' if settings.streaming_enabled else '[OFF] 未启用'}")
    print(f"  记忆系统: {'[ON] 长期记忆' if settings.memory_long_term_enabled else '[OK] 短期记忆'}")
    print(f"  LangSmith: {'[ON] 已启用' if obs.langsmith_enabled else '[OFF] 未启用'}")
    print("=" * 60)

    # 初始化 Neo4j 连接
    try:
        neo4j_driver = create_neo4j_driver()
    except Exception as e:
        print(f"[Startup] Neo4j 初始化失败: {e}")
        neo4j_driver = None

    # 创建 LLM
    llm = create_llm()

    # 注册所有 Agent
    agents = register_all_agents(
        neo4j_driver=neo4j_driver,
        llm=llm,
    )

    # 预加载客服 Agent
    try:
        cs_agent = agents.get("cs_agent")
        if cs_agent:
            cs_agent.preload()
    except Exception as e:
        print(f"[Startup] 客服 Agent 预加载失败（不影响启动）: {e}")

    # 获取共享 embedder (用于记忆系统)
    try:
        # 优先复用 registry 已加载的共享实例（避免重复加载 400MB 模型）
        # 复用 registry 共享实例并移出 agents 字典（避免污染 /api/agents）
        shared_embedder = agents.pop("__shared_embedder__", None)
        if shared_embedder is None:
            from sentence_transformers import SentenceTransformer

            shared_embedder = SentenceTransformer(settings.embedding_model)
            print(f"[Startup] 共享嵌入模型已加载: {settings.embedding_model}")
        else:
            print("[Startup] 复用 registry 共享嵌入模型（无重复加载）")
            # 激活 L3 语义缓存（此前从未注入 embed_fn，三级缓存实际只有两级）
            try:
                from orchestration.model_router import cost_optimizer

                cost_optimizer.set_embed_fn(
                    lambda q: shared_embedder.encode([q], normalize_embeddings=True)[0]
                )
                print("[Startup] L3 语义缓存已激活")
            except Exception as exc:  # noqa: BLE001
                print(f"[Startup] L3 语义缓存激活失败: {exc}")
    except Exception as e:
        print(f"[Startup] 嵌入模型加载失败（记忆系统降级为短期记忆）: {e}")
        shared_embedder = None

    # 创建路由器
    router = RouterAgent(llm)

    # 创建记忆管理器
    memory_manager = MemoryManager(llm=llm, embedder=shared_embedder)

    # 创建 ReAct 编排器 (主)
    if settings.react_enabled:
        orchestrator = ReactOrchestrator(
            llm=llm,
            agents=agents,
            router=router,
            memory_manager=memory_manager,
        )
        print("[Startup] ReAct 编排器已创建")
    else:
        # 未启用 ReAct, 使用原编排器
        from orchestration.graph import ECommerceOrchestrator

        orchestrator = ECommerceOrchestrator(llm, agents, router)
        print("[Startup] 固定路由编排器已创建 (ReAct 未启用)")

    # 创建降级编排器
    from orchestration.graph import ECommerceOrchestrator

    fallback_orchestrator = ECommerceOrchestrator(llm, agents, router)

    print("=" * 60)
    print("  [OK] 系统启动完成!")
    print(f"  API: http://{settings.api_host}:{settings.api_port}")
    print(f"  文档: http://{settings.api_host}:{settings.api_port}/docs")
    print("=" * 60 + "\n")

    yield

    # 关闭连接
    if neo4j_driver:
        neo4j_driver.close()
        print("[Shutdown] Neo4j 连接已关闭")

    print("[Shutdown] 系统已关闭")


# ------------------------------------------------------------------ #
#  FastAPI 应用
# ------------------------------------------------------------------ #

app = FastAPI(
    title="电商领域智能体 API",
    description="基于 LangGraph + ReAct Agent + 多工具架构的电商智能助手",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
_allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# API Key 认证
_API_ACCESS_KEY = os.getenv("API_ACCESS_KEY", "")


@app.middleware("http")
async def verify_api_key(request: Request, call):
    """API Key 认证中间件"""
    if (
        _API_ACCESS_KEY
        and request.url.path.startswith("/api/")
        and request.url.path != "/api/health"
    ):
        api_key = request.headers.get("X-API-Key")
        if not secrets.compare_digest(api_key or "", _API_ACCESS_KEY):
            return JSONResponse(401, {"detail": "Unauthorized: invalid or missing API key"})
    return await call(request)


# 限流
_RATE_LIMIT = int(os.getenv("API_RATE_LIMIT", "60"))
_RATE_WINDOW = 60
_rate_buckets: dict = defaultdict(list)


@app.middleware("http")
async def rate_limit(request: Request, call):
    """IP 限流中间件"""
    if not request.url.path.startswith("/api/") or request.url.path == "/api/health":
        return await call(request)

    # 反向代理场景：优先取 X-Forwarded-For 第一个 IP（需可信代理场景才可靠）
    forwarded = request.headers.get("X-Forwarded-For", "")
    client_ip = (
        forwarded.split(",")[0].strip()
        if forwarded
        else (request.client.host if request.client else "unknown")
    )
    now = time.time()

    # 定期清理过期桶，防止长期运行内存增长
    if len(_rate_buckets) > 1000:
        for ip in list(_rate_buckets.keys()):
            if not _rate_buckets[ip] or now - _rate_buckets[ip][-1] > _RATE_WINDOW * 2:
                del _rate_buckets[ip]

    bucket = _rate_buckets[client_ip]
    _rate_buckets[client_ip] = [ts for ts in bucket if now - ts < _RATE_WINDOW]

    if len(_rate_buckets[client_ip]) >= _RATE_LIMIT:
        return JSONResponse(429, {"detail": f"请求过于频繁，每分钟限 {_RATE_LIMIT} 次"})

    _rate_buckets[client_ip].append(now)
    return await call(request)


# ------------------------------------------------------------------ #
#  API 路由
# ------------------------------------------------------------------ #


@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "ok",
        "agents_registered": len(orchestrator.agents)
        if hasattr(orchestrator, "agents") and orchestrator.agents
        else 0,
        "react_enabled": settings.react_enabled,
        "streaming_enabled": settings.streaming_enabled,
    }


@app.get("/api/agents")
async def list_agents():
    """查看已注册的 Agent 列表"""
    agents_dict = getattr(orchestrator, "agents", None)
    if not agents_dict:
        raise HTTPException(503, "服务未就绪")
    return {
        "agents": [
            {
                "name": name,
                "class": agent.__class__.__name__,
                "description": agent.description,
            }
            for name, agent in agents_dict.items()
            if not name.startswith("__")  # 跳过内部键（如 __shared_embedder__）
        ]
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    统一对话入口

    使用 ReAct 工具循环 + 记忆系统处理用户消息。
    支持多轮对话 (通过 session_id 隔离)。
    """
    if not orchestrator:
        raise HTTPException(503, "服务未就绪，请稍后重试")

    session_id = request.session_id or "default"

    # 调用编排器
    if settings.react_enabled and hasattr(orchestrator, "ainvoke"):
        response, state_info = await orchestrator.ainvoke(
            request.message,
            session_id=session_id,
            user_profile=request.user_profile,
        )
    else:
        # 降级到原编排器
        response, state_info = await fallback_orchestrator.ainvoke(request.message)
        state_info["mode"] = "fixed_route"

    response = _strip_reasoning(response)

    return ChatResponse(
        message=response,
        intent=state_info.get("intent", "unknown"),
        agent_used=state_info.get("agent", "unknown"),
        mode=state_info.get("mode", "react"),
        tools_called=state_info.get("tools_called", []),
        steps=state_info.get("steps", 0),
    )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式对话入口 (SSE)

    返回 Server-Sent Events, 逐 token 推送回答。
    前端使用 EventSource 或 fetch + ReadableStream 消费。

    事件格式:
    - {"type": "token", "content": "..."} — 逐 token 内容
    - {"type": "meta", "intent": "...", "mode": "..."} — 元信息
    - {"type": "done", "full_response": "..."} — 完成
    - {"type": "error", "message": "..."} — 错误
    """
    if not orchestrator:
        raise HTTPException(503, "服务未就绪，请稍后重试")

    if not settings.streaming_enabled:
        # 流式未启用, 降级为普通接口
        raise HTTPException(504, "流式输出未启用，请使用 /api/chat")

    session_id = request.session_id or "default"

    async def event_generator():
        """SSE 事件生成器"""
        try:
            # 发送开始事件
            yield f"data: {json.dumps({'type': 'start'}, ensure_ascii=False)}\n\n"

            full_response = ""

            # 流式获取回答
            if settings.react_enabled and hasattr(orchestrator, "ainvoke_stream"):
                async for chunk in orchestrator.ainvoke_stream(
                    request.message,
                    session_id=session_id,
                    user_profile=request.user_profile,
                ):
                    full_response += chunk
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
            else:
                # 降级: 一次性获取然后流式返回
                response, state_info = await fallback_orchestrator.ainvoke(request.message)
                full_response = response
                # 模拟流式
                chunk_size = 5
                for i in range(0, len(response), chunk_size):
                    chunk = response[i : i + chunk_size]
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"

            # 发送完成事件
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response}, ensure_ascii=False)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Nginx 不缓冲
        },
    )


@app.post("/api/classify")
def classify_product(request: ClassifyRequest):
    """独立商品分类接口（同步 def，FastAPI 自动放入线程池，避免阻塞事件循环）"""
    agents_dict = getattr(orchestrator, "agents", None)
    if not agents_dict:
        raise HTTPException(503, "服务未就绪")

    agent = agents_dict.get("catalog_agent")
    if not agent:
        raise HTTPException(503, "分类能力不可用")

    top_k_results = agent.get_top_k(request.title, request.top_k)
    return {
        "title": request.title,
        "top_k": top_k_results,
    }


@app.post("/api/search")
def search_products(request: SearchRequest):
    """独立商品搜索接口（同步 def，FastAPI 自动放入线程池）"""
    agents_dict = getattr(orchestrator, "agents", None)
    if not agents_dict:
        raise HTTPException(503, "服务未就绪")

    agent = agents_dict.get("catalog_agent")
    if not agent:
        raise HTTPException(503, "搜索能力不可用")

    result = agent.hybrid_search(
        query=request.query,
        top_k=request.top_k,
        category=request.category,
        min_price=request.min_price,
        max_price=request.max_price,
    )
    return {
        "query": request.query,
        "result": result,
    }


@app.get("/api/stats")
async def get_stats(session_id: str = None):
    """
    系统统计 — Token 用量、缓存命中率、记忆状态、追踪信息

    ?session_id=xxx 时只返回该会话的统计。
    """
    from orchestration.model_router import cost_optimizer
    from orchestration.observability import obs

    stats = {
        "token_usage": obs.get_usage_summary(),
        "cache_stats": cost_optimizer.get_stats(),
    }

    # 记忆系统统计
    if hasattr(orchestrator, "get_memory_stats"):
        stats["memory"] = orchestrator.get_memory_stats(session_id=session_id)

    # 追踪信息
    spans = obs.get_trace_tree()
    if session_id:
        spans = [
            s for s in spans if s.get("attributes", {}).get("session_id") == session_id
        ]
    stats["trace"] = {"spans": len(spans)}

    return stats


@app.get("/api/trace")
async def get_trace(session_id: str = None):
    """获取最近请求的追踪树（?session_id=xxx 时按会话过滤）"""
    from orchestration.observability import TraceContext

    spans = TraceContext.get_trace_tree()
    if session_id:
        spans = [
            s for s in spans if s.get("attributes", {}).get("session_id") == session_id
        ]
    return {"spans": spans}


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """清除会话记忆"""
    if hasattr(orchestrator, "clear_session"):
        orchestrator.clear_session(session_id)
        return {"status": "ok", "session_id": session_id}
    raise HTTPException(404, "会话管理不可用")


# ------------------------------------------------------------------ #
#  启动入口
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level="info",
    )
