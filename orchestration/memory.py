"""
记忆系统 — 短期记忆 + 长期记忆

短期记忆 (ShortTermMemory):
  - 维护最近 N 轮对话 (滑动窗口)
  - 超过阈值时自动摘要旧消息
  - 提供 LLM 上下文构建

长期记忆 (LongTermMemory):
  - 从对话中提取关键事实 (用户偏好、历史决策)
  - 向量化存储 (复用 FAISS + BGE)
  - 检索时按相关性召回

MemoryManager:
  - 统一管理短期 + 长期记忆
  - 按 session_id 隔离不同会话
  - 提供 build_context() 构建 LLM 上下文
"""

import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config.settings import settings

# TTL 策略配置
CATEGORY_TTL = {
    "preference": 365,   # 用户偏好: 长期保留 (1 年)
    "fact": 90,          # 事实信息: 中期 (3 个月)
    "decision": 180,     # 购买决策: 中长期 (半年)
    "context": 30,       # 上下文: 短期 (1 个月)
    "ephemeral": 1,      # 临时信息: 不隔夜
}

# 重要记忆的 TTL 延长倍率
IMPORTANCE_TTL_BONUS = {
    0.8: 2.0,   # 重要性 ≥ 0.8 → TTL × 2
    0.9: 3.0,   # 重要性 ≥ 0.9 → TTL × 3
    1.0: 5.0,   # 重要性 = 1.0 → TTL × 5 (永久保留)
}


def auto_ttl(category: str, importance: float) -> int:
    """根据类型和重要性自动计算 TTL (天)"""
    base_ttl = CATEGORY_TTL.get(category, 30)
    for threshold, multiplier in sorted(IMPORTANCE_TTL_BONUS.items(), reverse=True):
        if importance >= threshold:
            return int(base_ttl * multiplier)
    return base_ttl


@dataclass
class Message:
    """单条消息"""

    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)  # 存储 intent, agent 等元信息


class ShortTermMemory:
    """
    短期记忆 — 滑动窗口 + 递归摘要

    工作原理:
    1. 保留最近 memory_window 条消息 (默认 10 条)
    2. 当消息数超过 memory_summary_threshold (默认 6) 时
       将较早的消息压缩为摘要，只保留最近窗口
    3. 摘要由 LLM 生成，提取关键信息
    4. 递归摘要: 当已有摘要 + 新摘要超过长度上限时，
       将两者合并后再次压缩，确保摘要不会无限增长
    """

    MAX_SUMMARY_LENGTH = 500  # 摘要最大字符数（超过则递归压缩）

    def __init__(self, window_size: int = None, summary_threshold: int = None, llm=None):
        self.window_size = window_size or settings.memory_window
        self.summary_threshold = summary_threshold or settings.memory_summary_threshold
        self.llm = llm
        self.messages: List[Message] = []
        self.summary: str = ""  # 历史摘要
        self._lock = threading.Lock()

    def add(self, role: str, content: str, **metadata):
        """添加一条消息"""
        msg = Message(role=role, content=content, metadata=metadata)
        self.messages.append(msg)
        # 摘要生成涉及 LLM 调用，放到后台线程，避免阻塞响应
        if len(self.messages) > self.summary_threshold:
            threading.Thread(target=self._maybe_summarize, daemon=True).start()

    def _maybe_summarize(self):
        """当消息数超过阈值时触发递归摘要"""
        with self._lock:
            if len(self.messages) <= self.summary_threshold:
                return
            old_messages = self.messages[: -self.window_size]
            if not old_messages:
                return
            try:
                new_summary = self._generate_summary(old_messages)
                if self.summary:
                    combined = f"{self.summary}\n{new_summary}"
                    if len(combined) > self.MAX_SUMMARY_LENGTH:
                        self.summary = self._recursive_compress(combined)
                    else:
                        self.summary = combined
                else:
                    self.summary = new_summary
                self.messages = self.messages[-self.window_size :]
            except Exception as exc:  # noqa: BLE001
                print(f"[ShortTermMemory] 摘要生成失败: {exc}")

    def _generate_summary(self, messages: List[Message]) -> str:
        """使用 LLM 摘要旧消息"""
        if not self.llm:
            # 无 LLM 时使用简单截断
            return self._simple_summary(messages)

        conversation = "\n".join(
            f"{'用户' if m.role == 'user' else '助手'}: {m.content[:200]}" for m in messages
        )

        prompt = f"""请将以下对话历史压缩为简洁摘要。

必须原样保留以下关键信息(不要概括或省略):
- 数字: 预算、价格、尺寸、规格、数量
- 专有名词: 品牌型号、商品名、技术参数
- 用户偏好和决策

对话历史:
{conversation}

摘要 (300字以内):"""

        try:
            result = self.llm.invoke(prompt)
            return result.content.strip()
        except Exception:
            return self._simple_summary(messages)

    def _recursive_compress(self, combined_summary: str) -> str:
        """
        递归压缩: 将过长的合并摘要再次用 LLM 压缩

        当已有摘要 + 新摘要合并后超过 MAX_SUMMARY_LENGTH 时调用。
        确保摘要长度始终受控，不会无限增长。
        """
        if not self.llm:
            # 无 LLM 时简单截断
            return combined_summary[: self.MAX_SUMMARY_LENGTH]

        prompt = f"""以下是一段过长的对话历史摘要，请将其进一步压缩。
压缩时必须原样保留数字(预算/价格/规格)和专有名词(品牌型号), 只删除冗余措辞。

原始摘要:
{combined_summary}

压缩后摘要 ({self.MAX_SUMMARY_LENGTH}字以内):"""

        try:
            result = self.llm.invoke(prompt)
            compressed = result.content.strip()
            # 确保压缩后的摘要确实更短
            if len(compressed) < len(combined_summary):
                return compressed
            return combined_summary[: self.MAX_SUMMARY_LENGTH]
        except Exception:
            return combined_summary[: self.MAX_SUMMARY_LENGTH]

    def _simple_summary(self, messages: List[Message]) -> str:
        """无 LLM 时的简单摘要"""
        user_msgs = [m.content[:50] for m in messages if m.role == "user"]
        return f"之前用户问了: {'; '.join(user_msgs)}"

    def get_recent(self, n: int = None) -> List[Message]:
        """获取最近 N 条消息"""
        n = n or self.window_size
        return self.messages[-n:]

    def build_context_messages(self) -> List[dict]:
        """
        构建 LLM 可用的上下文消息列表

        格式: [{"role": "system", "content": "摘要..."}, {"role": "user", "content": "..."}, ...]
        """
        result = []

        # 如果有摘要，作为 system 消息注入
        if self.summary:
            result.append({"role": "system", "content": f"[对话历史摘要]\n{self.summary}"})

        # 添加最近窗口的消息
        for msg in self.messages:
            result.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        return result

    def get_user_history(self) -> List[str]:
        """获取所有用户消息（用于长期记忆提取）"""
        return [m.content for m in self.messages if m.role == "user"]

    def clear(self):
        """清空记忆"""
        self.messages.clear()
        self.summary = ""


@dataclass
class MemoryItem:
    """
    增强版长期记忆条目

    支持:
    - 重要性评分 (importance): 0-1，越高越重要
    - 过期策略 (ttl_days): 过期天数，None 表示不过期
    - 访问记录 (last_accessed, access_count): 用于遗忘曲线
    - 遗忘评分 (forget_score): 基于 Ebbinghaus 曲线计算
    - 跨会话共享 (shared, linked_sessions): 偏好/决策可跨会话迁移
    """

    content: str  # 记忆内容
    category: str  # "preference" | "fact" | "decision"
    session_id: str
    timestamp: float = field(default_factory=time.time)
    embedding: list = field(default_factory=list)

    # 重要性评分 (0-1, 默认 0.5)
    importance: float = 0.5

    # 过期策略: None = 不过期, 否则指定天数
    ttl_days: Optional[int] = None

    # 访问记录 (用于遗忘曲线)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    # 遗忘评分 (0-1, 越高越容易被遗忘)
    forget_score: float = 0.0

    # 跨会话共享标记
    shared: bool = False

    # 关联的会话列表 (用于跨会话迁移追踪)
    linked_sessions: list = field(default_factory=list)

    def touch(self):
        """更新访问记录 (检索命中时调用)"""
        self.last_accessed = time.time()
        self.access_count += 1

    def is_expired(self, now: float = None) -> bool:
        """检查是否过期"""
        if self.ttl_days is None:
            return False
        now = now or time.time()
        return (now - self.timestamp) > self.ttl_days * 86400

    def compute_forget_score(self, now: float = None) -> float:
        """
        基于 Ebbinghaus 遗忘曲线计算遗忘评分

        公式: f(t) = 1 - e^(-λ * t)
        t: 距最近访问的时间 (天)
        λ: 遗忘速率，与重要性成反比

        返回: 0 (完全记得) ~ 1 (完全遗忘)
        """
        now = now or time.time()
        days_since_access = (now - self.last_accessed) / 86400

        # 重要性越高 → 遗忘速率越慢
        base_lambda = 0.01  # 基础遗忘速率 (约 69 天后遗忘 50%)
        adjusted_lambda = base_lambda * (1 - self.importance * 0.7)

        # Ebbinghaus 曲线: 保留率 R = e^(-λt)
        retention = math.exp(-adjusted_lambda * days_since_access)

        # 访问次数加成: 多次访问提升保留率
        access_bonus = min(0.3, self.access_count * 0.05)
        retention = min(1.0, retention + access_bonus)

        # 重要性加成: 重要记忆额外保留
        importance_bonus = self.importance * 0.2
        retention = min(1.0, retention + importance_bonus)

        # 遗忘评分 = 1 - 保留率
        self.forget_score = 1.0 - retention
        return self.forget_score

    def compute_effective_score(self, semantic_score: float) -> float:
        """
        计算检索时的综合得分

        final_score = semantic_score × importance_weight × (1 - forget_penalty)

        semantic_score: 语义相似度 (余弦相似度 0-1)
        importance_weight: 0.5 + importance × 0.5 (0.5-1.0)
        forget_penalty: forget_score × 0.3 (0-0.3)
        """
        importance_weight = 0.5 + self.importance * 0.5
        forget_penalty = self.forget_score * 0.3
        return semantic_score * importance_weight * (1.0 - forget_penalty)


class LongTermMemory:
    """
    长期记忆 — 向量化存储与检索

    工作原理:
    1. 从对话中提取关键事实 (用户偏好、历史决策)
    2. 使用 BGE 模型向量化
    3. 存储到磁盘 (JSON + FAISS)
    4. 检索时按语义相似度召回 Top-K

    注意: 需要传入 shared_embedder (SentenceTransformer) 实例
    """

    def __init__(self, store_path: str = None, embedder=None, llm=None):
        self.store_path = store_path or settings.memory_store_path
        self.embedder = embedder
        self.llm = llm
        self.memories: List[MemoryItem] = []
        self._faiss_index = None
        self._loaded = False

        # 确保存储目录存在
        os.makedirs(self.store_path, exist_ok=True)

    def _ensure_loaded(self):
        """懒加载持久化记忆"""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def _load(self):
        """从磁盘加载记忆 (兼容旧格式)"""
        index_file = os.path.join(self.store_path, "memories.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.memories = []
                for item in data:
                    mem = MemoryItem(
                        content=item["content"],
                        category=item.get("category", "fact"),
                        session_id=item.get("session_id", ""),
                        timestamp=item.get("timestamp", time.time()),
                        importance=item.get("importance", 0.5),
                        ttl_days=item.get("ttl_days"),
                        last_accessed=item.get("last_accessed", item.get("timestamp", time.time())),
                        access_count=item.get("access_count", 0),
                        forget_score=item.get("forget_score", 0.0),
                        shared=item.get("shared", False),
                        linked_sessions=item.get("linked_sessions", []),
                    )
                    self.memories.append(mem)
            except Exception:
                self.memories = []

        # 加载 FAISS 索引
        faiss_file = os.path.join(self.store_path, "memories.index")
        if os.path.exists(faiss_file) and self.embedder:
            try:
                import faiss

                self._faiss_index = faiss.read_index(faiss_file)
            except Exception:
                pass

    def _save(self):
        """持久化记忆到磁盘 (含增强字段)"""
        index_file = os.path.join(self.store_path, "memories.json")
        try:
            data = [
                {
                    "content": m.content,
                    "category": m.category,
                    "session_id": m.session_id,
                    "timestamp": m.timestamp,
                    "importance": m.importance,
                    "ttl_days": m.ttl_days,
                    "last_accessed": m.last_accessed,
                    "access_count": m.access_count,
                    "forget_score": m.forget_score,
                    "shared": m.shared,
                    "linked_sessions": m.linked_sessions,
                }
                for m in self.memories
            ]
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _build_faiss_index(self):
        """构建/重建 FAISS 索引"""
        if not self.embedder or not self.memories:
            return

        try:
            import faiss
            import numpy as np

            texts = [m.content for m in self.memories]
            embeddings = self.embedder.encode(texts, normalize_embeddings=True).astype(np.float32)

            dim = embeddings.shape[1]
            self._faiss_index = faiss.IndexFlatIP(dim)
            self._faiss_index.add(embeddings)

            faiss_file = os.path.join(self.store_path, "memories.index")
            faiss.write_index(self._faiss_index, faiss_file)
        except Exception:
            pass

    # 重要性评估提示词
    IMPORTANCE_PROMPT = """分析以下对话，提取值得长期记住的关键信息（用户偏好、重要事实、购买决策等）。
同时评估每条信息的重要性 (0-1 分)，评分标准:
- 0.9-1.0: 核心偏好/关键决策 (如 "用户坚持使用 TypeScript")
- 0.7-0.8: 重要事实 (如 "用户项目使用 React 18")
- 0.5-0.6: 一般信息 (如 "用户提到用过 Next.js")
- 0.0-0.4: 临时上下文 (如 "用户问了某个临时问题")

如果没有值得记住的信息，回复 "NONE"。
否则用 JSON 格式回复: {{"items": [{{"content": "...", "category": "preference|fact|decision", "importance": 0.8}}]}}

用户: {user_input}
助手: {assistant_response}

提取结果:"""

    def extract_and_store(self, user_input: str, assistant_response: str, session_id: str):
        """
        从对话中提取关键信息并存储 (含重要性评估)

        使用 LLM 判断是否有值得记住的信息，并评估重要性
        """
        if not self.llm:
            return

        prompt = self.IMPORTANCE_PROMPT.format(
            user_input=user_input,
            assistant_response=assistant_response,
        )

        try:
            result = self.llm.invoke(prompt).content.strip()
            if result.upper().startswith("NONE"):
                return

            # 清理 markdown
            if result.startswith("```"):
                result = result.split("\n", 1)[-1]
                result = result.rsplit("```", 1)[0].strip()

            parsed = json.loads(result)
            for item in parsed.get("items", []):
                importance = float(item.get("importance", 0.5))
                # 类型加权: preference/decision 天然更重要
                category_weight = {
                    "preference": 0.8,
                    "decision": 1.0,
                    "fact": 0.6,
                }.get(item.get("category", "fact"), 0.5)
                adjusted_importance = min(1.0, (importance + category_weight) / 2)

                self._add_memory(
                    content=item["content"],
                    category=item.get("category", "fact"),
                    session_id=session_id,
                    importance=adjusted_importance,
                )
        except Exception:
            pass

    def _add_memory(self, content: str, category: str, session_id: str,
                    importance: float = 0.5):
        """
        添加一条长期记忆 (含 TTL 和重要性)

        Args:
            content: 记忆内容
            category: 记忆类型
            session_id: 来源会话
            importance: 重要性评分 (0-1)
        """
        self._ensure_loaded()

        # 去重: 检查是否已有相似记忆
        existing = self.search(content, k=1)
        if existing and existing[0].get("score", 0) > 0.92:
            # 已存在: 只更新访问记录和重要性，不重复添加
            mem = self._find_by_content(content)
            if mem:
                mem.touch()
                # 如果新的重要性更高，则更新
                if importance > mem.importance:
                    mem.importance = importance
                self._save()
            return

        # 计算 TTL
        ttl_days = auto_ttl(category, importance)

        # 自动标记可共享的记忆类型
        shared = category in ("preference", "decision")

        item = MemoryItem(
            content=content,
            category=category,
            session_id=session_id,
            importance=importance,
            ttl_days=ttl_days,
            shared=shared,
            linked_sessions=[session_id] if shared else [],
        )

        # 生成 embedding
        if self.embedder:
            emb = self.embedder.encode([content], normalize_embeddings=True)
            item.embedding = emb[0].tolist()

        self.memories.append(item)
        self._save()
        self._build_faiss_index()

    def _find_by_content(self, content: str) -> Optional[MemoryItem]:
        """根据内容查找记忆 (用于去重时更新)"""
        for mem in self.memories:
            if mem.content == content:
                return mem
        return None

    def search(self, query: str, k: int = 3,
               exclude_expired: bool = True,
               use_forgetting: bool = True) -> List[dict]:
        """
        检索相关记忆 (支持过期过滤和综合得分排序)

        Args:
            query: 查询文本
            k: 返回数量
            exclude_expired: 是否过滤过期记忆
            use_forgetting: 是否应用遗忘评分调整

        Returns:
            按综合得分排序的记忆列表
        """
        self._ensure_loaded()

        if not self.memories:
            return []

        # 过滤过期记忆
        active_memories = self.memories
        if exclude_expired:
            active_memories = [m for m in self.memories if not m.is_expired()]
            if len(active_memories) < len(self.memories):
                # 有过期记忆，需要重建索引或在检索后过滤
                pass

        if not active_memories:
            return []

        # 更新遗忘评分
        if use_forgetting:
            for mem in active_memories:
                mem.compute_forget_score()

        # 有 FAISS 索引时使用向量检索
        if self._faiss_index is not None and self.embedder:
            try:
                import numpy as np

                query_emb = self.embedder.encode([query], normalize_embeddings=True).astype(
                    np.float32
                )
                # 多取一些用于重排序
                fetch_k = min(k * 3, len(active_memories))
                scores, indices = self._faiss_index.search(query_emb, fetch_k)

                candidates = []
                for score, idx in zip(scores[0], indices[0]):
                    if idx < 0 or idx >= len(self.memories):
                        continue
                    mem = self.memories[idx]
                    # 跳过过期记忆
                    if exclude_expired and mem.is_expired():
                        continue
                    # 更新访问记录
                    mem.touch()
                    # 计算综合得分
                    if use_forgetting:
                        final_score = mem.compute_effective_score(float(score))
                    else:
                        final_score = float(score) * (0.5 + mem.importance * 0.5)
                    candidates.append({
                        "content": mem.content,
                        "category": mem.category,
                        "importance": mem.importance,
                        "forget_score": mem.forget_score,
                        "score": final_score,
                    })

                # 按综合得分排序，取 Top-K
                candidates.sort(key=lambda x: x["score"], reverse=True)
                results = candidates[:k]

                # 保存更新后的访问记录
                if use_forgetting or exclude_expired:
                    self._save()

                return results
            except Exception:
                pass

        # 降级: 关键词匹配 + 重要性加权
        results = []
        query_lower = query.lower()
        for mem in active_memories:
            score = sum(1 for kw in query_lower.split() if kw in mem.content.lower())
            if score > 0:
                base_score = score / max(len(query_lower.split()), 1)
                # 应用重要性权重
                if use_forgetting:
                    mem.compute_forget_score()
                    final_score = mem.compute_effective_score(base_score)
                else:
                    final_score = base_score * (0.5 + mem.importance * 0.5)
                mem.touch()
                results.append({
                    "content": mem.content,
                    "category": mem.category,
                    "importance": mem.importance,
                    "forget_score": mem.forget_score,
                    "score": final_score,
                })
        results.sort(key=lambda x: x["score"], reverse=True)

        # 保存访问记录
        if results:
            self._save()

        return results[:k]

    def cleanup_expired(self) -> int:
        """
        清理过期记忆

        Returns:
            删除的记忆数量
        """
        self._ensure_loaded()

        before_count = len(self.memories)
        self.memories = [m for m in self.memories if not m.is_expired()]
        after_count = len(self.memories)

        deleted = before_count - after_count
        if deleted > 0:
            self._save()
            self._build_faiss_index()
            print(f"[LongTermMemory] 清理 {deleted} 条过期记忆")

        return deleted

    def update_forget_scores(self):
        """更新所有记忆的遗忘评分"""
        self._ensure_loaded()
        for mem in self.memories:
            mem.compute_forget_score()

    def apply_forgetting(self, forget_threshold: float = 0.85,
                         protect_important: bool = True) -> int:
        """
        应用遗忘机制

        Args:
            forget_threshold: 遗忘评分阈值 (>此值的被遗忘)
            protect_important: 是否保护高重要性记忆 (importance > 0.7)

        Returns:
            遗忘的记忆数量
        """
        self._ensure_loaded()
        self.update_forget_scores()

        before_count = len(self.memories)

        def should_forget(mem: MemoryItem) -> bool:
            if mem.forget_score < forget_threshold:
                return False
            if protect_important and mem.importance > 0.7:
                return False  # 保护重要记忆
            if mem.access_count >= 10:
                return False  # 频繁访问的记忆不遗忘
            return True

        self.memories = [m for m in self.memories if not should_forget(m)]
        after_count = len(self.memories)

        forgotten = before_count - after_count
        if forgotten > 0:
            self._save()
            self._build_faiss_index()
            print(f"[LongTermMemory] 遗忘 {forgotten} 条记忆 (阈值={forget_threshold})")

        return forgotten

    def get_stats(self) -> dict:
        """获取增强版统计信息"""
        self._ensure_loaded()
        self.update_forget_scores()

        if not self.memories:
            return {"total": 0}

        # 分类统计
        by_category = {}
        for m in self.memories:
            cat = m.category
            if cat not in by_category:
                by_category[cat] = {"count": 0, "avg_importance": 0}
            by_category[cat]["count"] += 1
            by_category[cat]["avg_importance"] += m.importance

        for cat in by_category:
            data = by_category[cat]
            data["avg_importance"] /= max(data["count"], 1)

        # 遗忘分布
        forget_dist = {"low (0-0.3)": 0, "medium (0.3-0.6)": 0,
                       "high (0.6-0.9)": 0, "critical (>0.9)": 0}
        for m in self.memories:
            fs = m.forget_score
            if fs < 0.3:
                forget_dist["low (0-0.3)"] += 1
            elif fs < 0.6:
                forget_dist["medium (0.3-0.6)"] += 1
            elif fs < 0.9:
                forget_dist["high (0.6-0.9)"] += 1
            else:
                forget_dist["critical (>0.9)"] += 1

        return {
            "total": len(self.memories),
            "by_category": by_category,
            "forget_distribution": forget_dist,
            "shared_count": len([m for m in self.memories if m.shared]),
            "expired_count": len([m for m in self.memories if m.is_expired()]),
            "avg_importance": sum(m.importance for m in self.memories) / len(self.memories),
        }

    def get_context_for_query(self, query: str, k: int = 3) -> str:
        """为查询构建长期记忆上下文"""
        memories = self.search(query, k=k)
        if not memories:
            return ""

        lines = ["[用户历史记忆]"]
        for m in memories:
            lines.append(f"- {m['content']} ({m['category']})")
        return "\n".join(lines)


class MemoryManager:
    """
    记忆管理器 — 统一管理短期 + 长期记忆

    按 session_id 隔离不同会话的记忆。
    """

    def __init__(self, llm=None, embedder=None):
        self.llm = llm
        self.embedder = embedder
        self._short_term: Dict[str, ShortTermMemory] = {}
        self._long_term = (
            LongTermMemory(
                embedder=embedder,
                llm=llm,
            )
            if settings.memory_long_term_enabled
            else None
        )

    def get_short_term(self, session_id: str) -> ShortTermMemory:
        """获取或创建会话的短期记忆"""
        if session_id not in self._short_term:
            self._short_term[session_id] = ShortTermMemory(
                llm=self.llm,
            )
        return self._short_term[session_id]

    def add_message(self, session_id: str, role: str, content: str, **metadata):
        """添加消息到短期记忆"""
        stm = self.get_short_term(session_id)
        stm.add(role, content, **metadata)

        # 如果是完整的 user-assistant 对，提取长期记忆
        if role == "assistant" and self._long_term:
            recent = stm.get_recent(2)
            if len(recent) >= 2 and recent[0].role == "user":
                # 长期记忆提取涉及 LLM 调用，后台执行，避免阻塞响应
                threading.Thread(
                    target=self._long_term.extract_and_store,
                    kwargs={
                        "user_input": recent[0].content,
                        "assistant_response": recent[1].content,
                        "session_id": session_id,
                    },
                    daemon=True,
                ).start()

    def add_event(self, session_id: str, event_type: str, payload: dict):
        """记录一条结构化事件（如工具调用 JSON 结果），显式存入短期记忆。

        与 LLM 摘要不同，这里保留原始工具结果，供后续轮次直接引用。
        """
        try:
            import json as _json

            content = _json.dumps(
                {"type": event_type, "payload": payload},
                ensure_ascii=False,
                default=str,
            )
            stm = self.get_short_term(session_id)
            stm.add("tool", content, event=event_type)
        except Exception as exc:  # noqa: BLE001
            print(f"[MemoryManager] 事件写入失败: {exc}")

    def build_context(self, session_id: str, current_query: str = "") -> List[dict]:
        """
        构建 LLM 上下文消息列表 (含跨会话共享记忆)

        包含: 跨会话共享记忆 + 当前会话长期记忆 + 短期记忆摘要 + 最近对话
        """
        messages = []

        # 1. 跨会话共享记忆 (preference/decision 类型)
        # M2 修复：默认关闭（settings.memory_shared_enabled=false），
        # 避免不同会话/用户的偏好互相串味
        if (
            settings.memory_shared_enabled
            and self._long_term
            and current_query
        ):
            shared_ctx = self._get_shared_context(session_id, current_query)
            if shared_ctx:
                messages.append({
                    "role": "system",
                    "content": shared_ctx,
                })

        # 2. 当前会话的长期记忆
        if self._long_term and current_query:
            long_term_ctx = self._long_term.get_context_for_query(current_query)
            if long_term_ctx:
                messages.append({
                    "role": "system",
                    "content": long_term_ctx,
                })

        # 3. 短期记忆 (摘要 + 最近窗口)
        stm = self.get_short_term(session_id)
        messages.extend(stm.build_context_messages())

        return messages

    def _get_shared_context(self, session_id: str, query: str) -> str:
        """
        获取跨会话共享的相关记忆

        检索 shared=True 的记忆，返回与当前查询相关的条目
        """
        if not self._long_term or not self._long_term.embedder:
            return ""

        self._long_term._ensure_loaded()

        # 只在 shared 记忆中检索
        shared_memories = [m for m in self._long_term.memories
                          if m.shared and not m.is_expired()]
        if not shared_memories:
            return ""

        try:
            import numpy as np

            # 对 shared 记忆构建临时索引
            shared_texts = [m.content for m in shared_memories]
            shared_embs = self._long_term.embedder.encode(
                shared_texts, normalize_embeddings=True
            )

            import faiss
            dim = shared_embs.shape[1]
            temp_index = faiss.IndexFlatIP(dim)
            temp_index.add(shared_embs.astype(np.float32))

            query_emb = self._long_term.embedder.encode(
                [query], normalize_embeddings=True
            )
            scores, indices = temp_index.search(
                query_emb.astype(np.float32), min(5, len(shared_memories))
            )

            relevant = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0.5:
                    mem = shared_memories[idx]
                    relevant.append(f"- {mem.content} (偏好/决策)")
                    # 记录关联会话
                    if session_id not in mem.linked_sessions:
                        mem.linked_sessions.append(session_id)

            if relevant:
                self._long_term._save()
                return "[跨会话共享记忆]\n" + "\n".join(relevant)

        except Exception:
            pass

        return ""

    def cleanup_expired_memories(self) -> int:
        """清理所有过期记忆"""
        if self._long_term:
            return self._long_term.cleanup_expired()
        return 0

    def get_history(self, session_id: str) -> List[dict]:
        """获取会话历史 (用于 API 返回)"""
        stm = self.get_short_term(session_id)
        return [
            {"role": m.role, "content": m.content, "metadata": m.metadata} for m in stm.messages
        ]

    def build_user_profile(
        self,
        session_id: str,
        current_query: str = "",
        explicit: dict = None,
    ) -> dict:
        """
        构建用户画像，供推荐工具使用 (含跨会话共享记忆)

        来源：
        1. 显式画像 explicit（API 请求传入）
        2. 当前 session 的 preference/decision 记忆
        3. 跨会话共享的 preference/decision 记忆

        Returns:
            {"偏好分类": ..., "预算": ..., "long_term_notes": [...], "shared_notes": [...]} 等
        """
        profile = dict(explicit or {})

        session_notes = []
        shared_notes = []

        if self._long_term:
            self._long_term._ensure_loaded()
            self._long_term.update_forget_scores()

            # 当前 session 的偏好/决策类记忆，取最近 5 条
            session_memories = [
                m for m in self._long_term.memories
                if m.session_id == session_id
                and m.category in ("preference", "decision")
                and not m.is_expired()
                and m.forget_score < 0.8
            ]
            session_memories.sort(key=lambda m: m.last_accessed, reverse=True)
            session_notes = [m.content for m in session_memories[:5]]

            # 跨会话共享的偏好/决策记忆 (排除当前 session 的)
            # M2 修复：默认关闭共享注入，避免串味
            if settings.memory_shared_enabled:
                shared_memories = [
                    m for m in self._long_term.memories
                    if m.shared
                    and m.category in ("preference", "decision")
                    and m.session_id != session_id
                    and not m.is_expired()
                    and m.forget_score < 0.8
                ]
                shared_memories.sort(key=lambda m: m.last_accessed, reverse=True)
                shared_notes = [m.content for m in shared_memories[:3]]

        if session_notes:
            profile["long_term_notes"] = session_notes
        if shared_notes:
            profile["shared_notes"] = shared_notes

        return profile

    def clear_session(self, session_id: str):
        """清除会话记忆（短期记忆 + 该会话的长期记忆）。

        M2 修复：原实现只清短期记忆，长期记忆会永久残留且跨会话可见，
        不满足数据删除权要求。现在同时删除该 session 的长期记忆并重建索引。
        """
        if session_id in self._short_term:
            self._short_term[session_id].clear()
            del self._short_term[session_id]

        if self._long_term:
            try:
                self._long_term._ensure_loaded()
                before = len(self._long_term.memories)
                self._long_term.memories = [
                    m
                    for m in self._long_term.memories
                    if m.session_id != session_id
                ]
                if len(self._long_term.memories) != before:
                    self._long_term._save()
                    self._long_term._build_faiss_index()
            except Exception as exc:  # noqa: BLE001
                print(f"[MemoryManager] 清除长期记忆失败: {exc}")

    def shutdown(self):
        """关闭记忆系统，保存持久化数据"""
        if self._long_term:
            self._long_term._ensure_loaded()
            self._long_term._save()

    def get_stats(self) -> dict:
        """获取增强版记忆系统统计"""
        stats = {
            "active_sessions": len(self._short_term),
            "sessions": {
                sid: {"messages": len(stm.messages), "has_summary": bool(stm.summary)}
                for sid, stm in self._short_term.items()
            },
        }

        if self._long_term:
            stats["long_term"] = self._long_term.get_stats()
        else:
            stats["long_term"] = {"total": 0, "enabled": False}

        return stats
