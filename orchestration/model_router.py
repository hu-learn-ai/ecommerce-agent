"""
成本/延迟优化 — 模型分级路由 + Prompt 压缩 + 多级缓存

三层成本优化:
1. 模型分级 (ModelTierRouter):
   - 简单任务 (闲聊、问候、简单 FAQ) → lite_model (更低 max_tokens, 更低 temperature)
   - 复杂任务 (KG QA、推荐重排、数据分析) → full model
   - 根据意图类型和消息长度自动路由

2. Prompt 压缩 (PromptCompressor):
   - 裁剪冗余空白和重复内容
   - 限制 context 长度
   - 对系统 prompt 做缓存 (DeepSeek 支持 prompt caching)

3. 多级缓存 (MultiLevelCache):
   - L1: 内存缓存 (cachetools TTLCache) — 5 分钟, 500 条
   - L2: 磁盘持久化缓存 — 跨重启有效
   - L3: 语义缓存 (Embedding 余弦相似度) — 对语义相同/相近的查询命中缓存
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from cachetools import TTLCache
from langchain_openai import ChatOpenAI

from config.settings import settings
from orchestration.observability import obs

# ------------------------------------------------------------------ #
#  模型分级路由
# ------------------------------------------------------------------ #


@dataclass
class ModelConfig:
    """模型配置"""

    model_name: str
    temperature: float
    max_tokens: int
    streaming: bool = False


class ModelTierRouter:
    """
    模型分级路由器 — 根据任务复杂度选择不同模型 + 不同参数配置

    分级策略 (三层模型差异化):
    - Tier 1 (lite): deepseek-chat (V3) + 512 max_tokens + 0.1 temp + 10s 超时
      → 闲聊、简单 FAQ、意图路由。成本最低，延迟最快。
    - Tier 2 (standard): deepseek-chat (V3) + 2048 max_tokens + 0.3 temp + 30s 超时
      → KG QA、搜索、分类降级、订单查询。通用场景。
    - Tier 3 (heavy): deepseek-reasoner (R1) + 4096 max_tokens + 0.5 temp + 60s 超时
      → 推荐重排、数据分析、复杂推理。R1 推理能力强但成本约 4x。

    成本节省: 约 30-50% (简单任务不再用 R1 推理模型)
    """

    # 意图 → 模型层级映射
    INTENT_TIER_MAP = {
        "chitchat": 1,  # 闲聊 → lite
        "customer_service": 1,  # 简单 FAQ → lite
        "order": 1,  # 订单查询 (DB 查询为主) → lite
        "classify": 1,  # 分类 (BERT 模型, 不用 LLM) → lite (仅降级时用)
        "search": 2,  # 搜索 → standard
        "kg_qa": 2,  # 知识图谱 QA → standard
        "analytics": 3,  # 数据分析 → heavy
        "recommend": 3,  # 推荐 (LLM 重排) → heavy
    }

    # 各层级模型信息（用于日志和成本估算）
    TIER_INFO = {
        1: {"name": "lite", "model": "deepseek-chat", "cost_per_1k": 0.002},
        2: {"name": "standard", "model": "deepseek-chat", "cost_per_1k": 0.002},
        3: {"name": "heavy", "model": "deepseek-reasoner", "cost_per_1k": 0.008},
    }

    def __init__(self):
        self._models: Dict[int, ChatOpenAI] = {}
        self._init_models()

    def _probe_vllm(self) -> bool:
        """探活本地 vLLM（OpenAI 兼容 /v1/models）；失败则由调用方回退 DeepSeek"""
        try:
            from openai import OpenAI

            client = OpenAI(api_key="EMPTY", base_url=settings.cs_model_base_url, timeout=3.0)
            client.models.list()
            return True
        except Exception:
            return False

    def _build_deepseek_lite(self) -> ChatOpenAI:
        """DeepSeek lite 模型（默认 Tier 1；本地 vLLM 探活失败时的兜底）"""
        return ChatOpenAI(
            model=settings.lite_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.lite_temperature,
            max_tokens=settings.lite_max_tokens,
            timeout=settings.lite_timeout,  # 极短超时, 保证低延迟
            max_retries=2,
        )

    def _init_models(self):
        """初始化各层级模型 — 三层使用不同模型/参数实现真正成本差异"""
        # Tier 1: lite — 默认 DeepSeek；SELF_HOSTED_CS=true 时指向本地 vLLM（探活失败自动回退）
        if settings.self_hosted_cs and self._probe_vllm():
            obs.log_event("model_router", local_cs="active", base_url=settings.cs_model_base_url)
            self._models[1] = ChatOpenAI(
                model=settings.cs_model_name,
                api_key="EMPTY",  # vLLM 本地服务不校验 key
                base_url=settings.cs_model_base_url,
                temperature=settings.lite_temperature,
                max_tokens=settings.lite_max_tokens,
                timeout=settings.lite_timeout,
                max_retries=2,
            )
            self.TIER_INFO[1]["model"] = settings.cs_model_name
        else:
            if settings.self_hosted_cs:
                obs.log_event("model_router", local_cs="fallback_deepseek", reason="vllm_probe_failed")
            self._models[1] = self._build_deepseek_lite()

        # Tier 2: standard — deepseek-chat + 标准参数
        self._models[2] = ChatOpenAI(
            model=settings.standard_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.standard_temperature,
            max_tokens=settings.standard_max_tokens,
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
        )

        # Tier 3: heavy — deepseek-reasoner (R1) + 高配参数
        # R1 模型具备更强的链式推理能力, 适合推荐重排和复杂分析
        # 注意: R1 不支持 temperature 调节, API 会忽略该参数
        self._models[3] = ChatOpenAI(
            model=settings.heavy_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.heavy_temperature,
            max_tokens=settings.heavy_max_tokens,
            timeout=settings.heavy_timeout,  # R1 推理需要更长时间
            max_retries=settings.max_retries,
        )

    def get_model(self, intent: str = None, message_length: int = 0) -> ChatOpenAI:
        """
        根据意图和消息长度选择模型

        Args:
            intent: 意图类型 (chitchat, search, kg_qa, ...)
            message_length: 用户消息长度

        Returns:
            对应层级的 ChatOpenAI 实例
        """
        tier = self.get_tier(intent, message_length)

        model = self._models.get(tier, self._models[2])
        tier_info = self.TIER_INFO.get(tier, {})
        obs.log_event(
            "model_tier_selected",
            tier=tier,
            tier_name=tier_info.get("name", "unknown"),
            intent=intent,
            model=model.model_name,
            cost_per_1k=tier_info.get("cost_per_1k", 0),
        )
        return model

    def get_tier(self, intent: str = None, message_length: int = 0) -> int:
        """计算意图对应的模型层级（lite/standard/heavy）"""
        tier = self.INTENT_TIER_MAP.get(intent, 2)
        # 短消息 + 简单意图 → 降级到 lite
        if message_length < 20 and tier <= 2:
            tier = 1
        return tier

    def get_model_by_tier(self, tier: int) -> ChatOpenAI:
        """直接按层级获取模型"""
        return self._models.get(tier, self._models[2])


# ------------------------------------------------------------------ #
#  Prompt 压缩
# ------------------------------------------------------------------ #


class PromptCompressor:
    """
    Prompt 压缩器 — 减少 token 消耗

    策略:
    1. 裁剪多余空白 (连续空行/空格)
    2. 限制 context 长度 (截断超长文档)
    3. 去除重复的 system prompt (prompt caching)
    4. 压缩 JSON context (移除 null/空字段)
    """

    # 各场景的 context 最大长度 (字符数)
    MAX_CONTEXT_LENGTH = {
        "search": 2000,  # 搜索结果
        "kg_qa": 3000,  # KG 查询结果
        "recommend": 2000,  # 推荐候选
        "cs": 1500,  # FAQ context
        "analytics": 3000,  # 分析数据
        "default": 2000,
    }

    @classmethod
    def compress_whitespace(cls, text: str) -> str:
        """压缩多余空白"""
        import re

        # 连续空行 → 单空行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 行首行尾空白
        lines = [line.strip() for line in text.split("\n")]
        # 连续空格 → 单空格
        text = "\n".join(lines)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    @classmethod
    def truncate_context(cls, text: str, max_length: int = None, scene: str = "default") -> str:
        """截断过长的 context"""
        max_length = max_length or cls.MAX_CONTEXT_LENGTH.get(scene, 2000)
        if len(text) <= max_length:
            return text
        return text[:max_length] + "\n...(内容已截断)"

    @classmethod
    def compress_json(cls, data: Any) -> str:
        """压缩 JSON (移除空值)"""
        if isinstance(data, list):
            cleaned = [
                {k: v for k, v in item.items() if v is not None and v != "" and v != []}
                for item in data
                if isinstance(item, dict)
            ]
        elif isinstance(data, dict):
            cleaned = {k: v for k, v in data.items() if v is not None and v != "" and v != []}
        else:
            cleaned = data
        return json.dumps(cleaned, ensure_ascii=False)

    @classmethod
    def compress_prompt(cls, prompt: str, scene: str = "default") -> str:
        """综合压缩 prompt"""
        # 1. 压缩空白
        prompt = cls.compress_whitespace(prompt)
        # 2. 截断
        prompt = cls.truncate_context(prompt, scene=scene)
        return prompt


# ------------------------------------------------------------------ #
#  多级缓存
# ------------------------------------------------------------------ #


class DiskCache:
    """L2 磁盘缓存 — 持久化存储"""

    def __init__(self, cache_dir: str = None):
        self.cache_dir = cache_dir or settings.cache_persist_path
        os.makedirs(self.cache_dir, exist_ok=True)
        # M6 修复：启动时清理过期缓存文件，防止磁盘无限增长
        self._cleanup_expired_files()

    def _key_to_path(self, key: str) -> str:
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{key_hash}.json")

    def _cleanup_expired_files(self):
        """清理已过期缓存文件（TTL × 6 视为过期），防止磁盘无限增长"""
        try:
            now = time.time()
            for fname in os.listdir(self.cache_dir):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(self.cache_dir, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if now - data.get("timestamp", 0) > settings.cache_ttl_seconds * 6:
                        os.remove(path)
                except Exception:
                    continue
        except Exception:
            pass

    def get(self, key: str) -> Optional[str]:
        path = self._key_to_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 检查 TTL
            if (
                time.time() - data.get("timestamp", 0) > settings.cache_ttl_seconds * 6
            ):  # L2 TTL 更长
                # 过期即删除，避免残留
                try:
                    os.remove(path)
                except Exception:
                    pass
                return None
            return data.get("value")
        except Exception:
            return None

    def set(self, key: str, value: str):
        path = self._key_to_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"value": value, "timestamp": time.time()}, f, ensure_ascii=False)
        except Exception:
            pass

    def clear(self):
        """清空磁盘缓存"""
        if os.path.exists(self.cache_dir):
            for f in os.listdir(self.cache_dir):
                if f.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, f))


class SemanticCache:
    """
    L3 语义缓存 — 基于 Embedding 相似度的缓存命中

    当 L1/L2 精确匹配未命中时, 计算查询的 embedding 向量,
    与已缓存查询的 embedding 做余弦相似度比较:
    - 相似度 ≥ threshold → 命中, 返回缓存的 value
    - 相似度 < threshold → 未命中

    适用场景: 用户换了措辞但语义相同的查询
    例: "有什么便宜的耳机" vs "推荐个低价耳机" → 语义命中

    注意: embedding 函数需要外部注入 (延迟加载 BGE 模型)
    """

    def __init__(
        self,
        embed_fn: Callable[[str], np.ndarray] = None,
        similarity_threshold: float = 0.92,
        max_size: int = 200,
    ):
        self._embed_fn = embed_fn
        self._threshold = similarity_threshold
        self._max_size = max_size
        # 存储: key → (embedding_vector, value, timestamp)
        self._store: Dict[str, Tuple[np.ndarray, str, float]] = {}
        # 嵌入缓存: 规范化查询 → 向量（避免重复查询重复计算 BGE）
        self._embed_cache: Dict[str, np.ndarray] = {}
        self._embed_cache_max: int = 500
        self._semantic_hits = 0
        self._semantic_misses = 0

    def set_embed_fn(self, embed_fn: Callable[[str], np.ndarray]):
        """注入 embedding 函数 (延迟加载 BGE 模型后调用)"""
        self._embed_fn = embed_fn

    def _cache_embed(self, norm: str, vec: np.ndarray) -> None:
        """写入嵌入缓存并做上限淘汰（dict 保持插入顺序）"""
        self._embed_cache[norm] = vec
        if len(self._embed_cache) > self._embed_cache_max:
            self._embed_cache.pop(next(iter(self._embed_cache)))

    @staticmethod
    def _norm(text: str) -> str:
        """规范化查询文本（去除多余空白、统一小写）"""
        return " ".join(text.strip().lower().split())

    def get_exact(self, query: str) -> Optional[str]:
        """精确快速路径：规范化 key 直接查字典，不计算 embedding。"""
        key = self._norm(query)
        item = self._store.get(key)
        if item is not None:
            _, value, ts = item
            if time.time() - ts <= settings.cache_ttl_seconds:
                self._semantic_hits += 1
                return value
        return None

    def get(self, query: str) -> Optional[str]:
        """语义查询: 返回最相似缓存项的 value (如果超过阈值)"""
        if not self._embed_fn or not self._store:
            self._semantic_misses += 1
            return None

        try:
            norm = self._norm(query)
            # 精确快速路径（避免重复计算 embedding）
            item = self._store.get(norm)
            if item is not None:
                _, value, ts = item
                if time.time() - ts <= settings.cache_ttl_seconds:
                    self._semantic_hits += 1
                    return value

            # 嵌入缓存：同一规范化查询不重复计算 BGE
            query_vec = self._embed_cache.get(norm)
            if query_vec is None:
                query_vec = self._embed_fn(query)
                if query_vec is None:
                    self._semantic_misses += 1
                    return None
                self._cache_embed(norm, query_vec)
            if query_vec is None:
                self._semantic_misses += 1
                return None

            best_sim = 0.0
            best_value = None

            for key, (cached_vec, value, ts) in self._store.items():
                # 检查 TTL
                if time.time() - ts > settings.cache_ttl_seconds:
                    continue
                # 余弦相似度
                sim = float(
                    np.dot(query_vec, cached_vec)
                    / (np.linalg.norm(query_vec) * np.linalg.norm(cached_vec) + 1e-8)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_value = value

            if best_sim >= self._threshold and best_value is not None:
                self._semantic_hits += 1
                obs.log_event("semantic_cache_hit", query=query[:50], similarity=round(best_sim, 4))
                return best_value
        except Exception as e:
            obs.log_event("semantic_cache_error", error=str(e))

        self._semantic_misses += 1
        return None

    def set(self, key: str, value: str, query: str = None):
        """写入语义缓存 (同时存储 embedding)"""
        if not self._embed_fn:
            return

        # query 参数用于计算 embedding (可能比 key 更自然)
        embed_text = query or key
        norm = self._norm(embed_text)

        try:
            vec = self._embed_cache.get(norm)
            if vec is None:
                vec = self._embed_fn(embed_text)
                if vec is not None:
                    self._cache_embed(norm, vec)
            if vec is not None:
                self._store[norm] = (vec, value, time.time())

                # LRU 淘汰: 超过 max_size 时移除最旧的
                if len(self._store) > self._max_size:
                    oldest_key = min(self._store, key=lambda k: self._store[k][2])
                    del self._store[oldest_key]
        except Exception:
            pass

    def get_stats(self) -> dict:
        total = self._semantic_hits + self._semantic_misses
        return {
            "semantic_hits": self._semantic_hits,
            "semantic_misses": self._semantic_misses,
            "semantic_hit_rate": f"{self._semantic_hits / total * 100:.1f}%"
            if total > 0
            else "N/A",
            "semantic_cache_size": len(self._store),
            "threshold": self._threshold,
        }


class MultiLevelCache:
    """
    多级缓存 — L1 内存 + L2 磁盘 + L3 语义

    查询流程: L1 精确 → L2 精确 → L3 语义 → miss
    写入流程: 同时写入 L1, L2, L3

    L3 语义缓存使用 embedding 余弦相似度匹配, 命中阈值 0.92。
    需要外部注入 embed_fn 后才生效 (延迟加载 BGE 模型)。
    """

    def __init__(self, embed_fn: Callable[[str], np.ndarray] = None):
        # L1: 内存缓存 (TTL=5min, maxsize=500)
        self._l1 = TTLCache(
            maxsize=settings.cache_max_size,
            ttl=settings.cache_ttl_seconds,
        )
        # L2: 磁盘缓存
        self._l2 = DiskCache()
        # L3: 语义缓存 (可选, 需注入 embed_fn)
        self._l3 = SemanticCache(embed_fn=embed_fn)
        self._hits = 0
        self._misses = 0

    def set_embed_fn(self, embed_fn: Callable[[str], np.ndarray]):
        """注入 embedding 函数, 激活 L3 语义缓存"""
        self._l3.set_embed_fn(embed_fn)

    def get(self, key: str, query: str = None) -> Optional[str]:
        """
        查询缓存

        Args:
            key: 精确匹配 key (intent|query|params)
            query: 原始用户查询 (用于 L3 语义匹配, 默认等于 key)
        """
        # L1 精确匹配
        if key in self._l1:
            self._hits += 1
            return self._l1[key]

        # L2 精确匹配
        value = self._l2.get(key)
        if value is not None:
            # 回填 L1
            self._l1[key] = value
            self._hits += 1
            return value

        # L3 语义匹配
        semantic_query = query or key
        # 3a. 精确快速路径（不计算 embedding）
        value = self._l3.get_exact(semantic_query)
        if value is not None:
            self._l1[key] = value
            self._hits += 1
            return value
        # 3b. 语义相似度匹配
        value = self._l3.get(semantic_query)
        if value is not None:
            # 回填 L1
            self._l1[key] = value
            self._hits += 1
            return value

        self._misses += 1
        return None

    def set(self, key: str, value: str, query: str = None):
        """写入缓存 (同时写入 L1, L2, L3)"""
        self._l1[key] = value
        self._l2.set(key, value)
        self._l3.set(key, value, query=query or key)

    def get_stats(self) -> dict:
        """获取缓存统计"""
        total = self._hits + self._misses
        stats = {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self._hits / total * 100:.1f}%" if total > 0 else "N/A",
            "l1_size": len(self._l1),
        }
        # 合并 L3 语义缓存统计
        l3_stats = self._l3.get_stats()
        stats.update(l3_stats)
        return stats

    @staticmethod
    def make_key(intent: str, query: str, **params) -> str:
        """生成缓存 key"""
        key_parts = [intent, query.strip().lower()]
        for k in sorted(params.keys()):
            key_parts.append(f"{k}={params[k]}")
        return "|".join(key_parts)


# ------------------------------------------------------------------ #
#  统一成本优化入口
# ------------------------------------------------------------------ #


class CostOptimizer:
    """
    成本优化统一入口

    整合模型分级、prompt 压缩、多级缓存 (L1 精确 + L2 磁盘 + L3 语义)
    """

    def __init__(self):
        self.model_router = ModelTierRouter()
        self.cache = MultiLevelCache()
        self.compressor = PromptCompressor()

    def set_embed_fn(self, embed_fn: Callable[[str], np.ndarray]):
        """注入 embedding 函数, 激活 L3 语义缓存"""
        self.cache.set_embed_fn(embed_fn)

    def get_llm(self, intent: str = None, message_length: int = 0) -> ChatOpenAI:
        """获取优化后的 LLM 实例"""
        return self.model_router.get_model(intent, message_length)

    def compress_prompt(self, prompt: str, scene: str = "default") -> str:
        """压缩 prompt"""
        return self.compressor.compress_prompt(prompt, scene)

    def cache_get(self, intent: str, query: str, **params) -> Optional[str]:
        """查询缓存 (L1 精确 → L2 磁盘 → L3 语义)"""
        key = MultiLevelCache.make_key(intent, query, **params)
        result = self.cache.get(key, query=query)
        if result:
            obs.log_event("cache_hit", intent=intent, query=query[:50])
        return result

    def cache_set(self, intent: str, query: str, value: str, **params):
        """写入缓存 (同时写入 L1, L2, L3)"""
        key = MultiLevelCache.make_key(intent, query, **params)
        self.cache.set(key, value, query=query)

    def get_stats(self) -> dict:
        """获取成本优化统计"""
        return {
            "cache": self.cache.get_stats(),
            "token_usage": obs.get_usage_summary(),
        }


# 全局单例
cost_optimizer = CostOptimizer()
