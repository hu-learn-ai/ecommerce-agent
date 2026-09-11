"""
可观测性系统 — 全链路追踪、结构化日志、Token 用量监控

三层可观测性：
1. LangSmith tracing: 自动捕获 LLM 调用、工具调用、链路轨迹
2. 结构化日志: 统一 JSON 格式日志，支持日志级别和上下文
3. Token 追踪: 记录每次 LLM 调用的 token 用量和成本

使用方式:
    from orchestration.observability import obs

    # 自动追踪（装饰器）
    @obs.trace("agent_name")
    def my_function(...): ...

    # 手动记录
    obs.log_event("search_completed", {"query": "...", "results": 10})
    obs.record_tokens(model="deepseek-chat", prompt_tokens=100, completion_tokens=50)
"""

import functools
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

from config.settings import settings

# ------------------------------------------------------------------ #
#  Token 用量追踪
# ------------------------------------------------------------------ #


@dataclass
class TokenUsage:
    """单次 LLM 调用的 token 用量"""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: float = field(default_factory=time.time)

    # DeepSeek 定价 (USD per 1M tokens, 可按需调整)
    PRICING = {
        "deepseek-chat": {"input": 0.14, "output": 0.28},
        "deepseek-coder": {"input": 0.14, "output": 0.28},
    }

    def calculate_cost(self):
        """计算成本 (USD)"""
        pricing = self.PRICING.get(self.model, {"input": 0.14, "output": 0.28})
        self.cost_usd = (
            self.prompt_tokens / 1_000_000 * pricing["input"]
            + self.completion_tokens / 1_000_000 * pricing["output"]
        )
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self.cost_usd


class TokenTracker:
    """Token 用量聚合追踪器"""

    # M6 修复：明细记录上限，防止长运行服务内存无限增长
    MAX_USAGE_RECORDS = 10000

    def __init__(self):
        self._usage: deque = deque(maxlen=self.MAX_USAGE_RECORDS)
        self._by_model: dict[str, dict] = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "calls": 0,
            }
        )

    def record(self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        """记录一次 LLM 调用的 token 用量"""
        usage = TokenUsage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        usage.calculate_cost()
        self._usage.append(usage)

        agg = self._by_model[model]
        agg["prompt_tokens"] += prompt_tokens
        agg["completion_tokens"] += completion_tokens
        agg["total_tokens"] += usage.total_tokens
        agg["cost_usd"] += usage.cost_usd
        agg["calls"] += 1

    def get_summary(self) -> dict:
        """获取 token 用量汇总"""
        return {
            "total_calls": len(self._usage),
            "total_tokens": sum(u.total_tokens for u in self._usage),
            "total_cost_usd": round(sum(u.cost_usd for u in self._usage), 6),
            "by_model": dict(self._by_model),
        }

    def reset(self):
        """重置统计"""
        self._usage.clear()
        self._by_model.clear()


# ------------------------------------------------------------------ #
#  结构化日志
# ------------------------------------------------------------------ #


class StructuredLogger:
    """结构化 JSON 日志记录器"""

    def __init__(self, name: str = "ecommerce_agent", level: str = None):
        self.logger = logging.getLogger(name)
        level = level or settings.log_level
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # 避免重复添加 handler
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
            self.logger.addHandler(handler)

    def _log(self, level: str, message: str, **context):
        """结构化日志"""
        entry = {
            "msg": message,
            "ts": time.time(),
        }
        if context:
            entry["ctx"] = context
        getattr(self.logger, level)(json.dumps(entry, ensure_ascii=False, default=str))

    def info(self, message: str, **ctx):
        self._log("info", message, **ctx)

    def warning(self, message: str, **ctx):
        self._log("warning", message, **ctx)

    def error(self, message: str, **ctx):
        self._log("error", message, **ctx)

    def debug(self, message: str, **ctx):
        self._log("debug", message, **ctx)


# ------------------------------------------------------------------ #
#  链路追踪上下文
# ------------------------------------------------------------------ #


@dataclass
class TraceSpan:
    """单个追踪 span"""

    trace_id: str
    span_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    status: str = "ok"
    parent_span_id: Optional[str] = None

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def add_event(self, name: str, **attrs):
        self.events.append(
            {
                "name": name,
                "ts": time.time(),
                "attrs": attrs,
            }
        )

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def finish(self, status: str = "ok"):
        self.end_time = time.time()
        self.status = status


class TraceContext:
    """追踪上下文管理器"""

    _spans: deque = deque(maxlen=1000)  # 限长，防止长运行内存增长
    _current_span: Optional[TraceSpan] = None
    _trace_id: Optional[str] = None

    @classmethod
    def start_trace(cls, name: str) -> "TraceSpan":
        """启动一个新的追踪"""
        cls._trace_id = str(uuid.uuid4())
        span = TraceSpan(
            trace_id=cls._trace_id,
            span_id=str(uuid.uuid4())[:8],
            name=name,
            start_time=time.time(),
        )
        cls._spans.append(span)
        cls._current_span = span
        return span

    @classmethod
    def start_span(cls, name: str, parent: Optional[TraceSpan] = None) -> TraceSpan:
        """启动一个子 span"""
        parent = parent or cls._current_span
        trace_id = parent.trace_id if parent else str(uuid.uuid4())
        span = TraceSpan(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:8],
            name=name,
            start_time=time.time(),
            parent_span_id=parent.span_id if parent else None,
        )
        cls._spans.append(span)
        return span

    @classmethod
    def get_trace_tree(cls) -> list[dict]:
        """获取追踪树结构"""
        return [
            {
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "name": s.name,
                "duration_ms": round(s.duration_ms, 2),
                "status": s.status,
                "attributes": s.attributes,
                "events": s.events,
                "parent_span_id": s.parent_span_id,
            }
            for s in cls._spans
        ]

    @classmethod
    def clear(cls):
        cls._spans.clear()
        cls._current_span = None
        cls._trace_id = None


# ------------------------------------------------------------------ #
#  可观测性统一入口
# ------------------------------------------------------------------ #


class Observability:
    """可观测性统一入口"""

    def __init__(self):
        self.logger = StructuredLogger()
        self.tokens = TokenTracker()
        self._langsmith_enabled = False
        self._setup_langsmith()

    def _setup_langsmith(self):
        """配置 LangSmith tracing"""
        if settings.tracing_enabled and settings.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langsmith_endpoint
            os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
            self._langsmith_enabled = True
            self.logger.info("LangSmith tracing 已启用", project=settings.langsmith_project)
        else:
            self.logger.info(
                "LangSmith tracing 未启用 (设置 TRACING_ENABLED=true 和 LANGSMITH_API_KEY 启用)"
            )

    @property
    def langsmith_enabled(self) -> bool:
        return self._langsmith_enabled

    # --- 追踪装饰器 ---

    def trace(self, name: str):
        """
        装饰器：自动追踪函数执行

        @obs.trace("search_agent")
        def search(query): ...
        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                span = TraceContext.start_span(name)
                span.set_attribute("function", func.__name__)
                try:
                    result = func(*args, **kwargs)
                    span.finish("ok")
                    return result
                except Exception as e:
                    span.finish("error")
                    span.set_attribute("error", str(e))
                    self.logger.error(f"{name} 执行失败", error=str(e), span_id=span.span_id)
                    raise

            return wrapper

        return decorator

    def trace_async(self, name: str):
        """异步版本追踪装饰器"""

        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                span = TraceContext.start_span(name)
                span.set_attribute("function", func.__name__)
                try:
                    result = await func(*args, **kwargs)
                    span.finish("ok")
                    return result
                except Exception as e:
                    span.finish("error")
                    span.set_attribute("error", str(e))
                    self.logger.error(f"{name} 执行失败", error=str(e), span_id=span.span_id)
                    raise

            return wrapper

        return decorator

    # --- 事件记录 ---

    def log_event(self, event_name: str, **attrs):
        """记录自定义事件"""
        self.logger.info(event_name, **attrs)
        if self._current_span_active():
            TraceContext._current_span.add_event(event_name, **attrs)

    def _current_span_active(self) -> bool:
        return TraceContext._current_span is not None

    # --- Token 追踪 ---

    def record_tokens(self, model: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        """记录 LLM token 用量"""
        self.tokens.record(model, prompt_tokens, completion_tokens)

    def get_usage_summary(self) -> dict:
        """获取用量汇总"""
        return self.tokens.get_summary()

    # --- 追踪上下文管理 ---

    @contextmanager
    def trace_context(self, name: str):
        """上下文管理器：创建追踪 span"""
        span = TraceContext.start_trace(name)
        try:
            yield span
        finally:
            span.finish()

    def get_trace_tree(self) -> list[dict]:
        """获取完整追踪树"""
        return TraceContext.get_trace_tree()


# 全局单例
obs = Observability()
