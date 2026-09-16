"""
商品域 Agent — 统一封装搜索、推荐、分类三个能力

精简背景：search/recommend/classify 三个原 Agent 共享 Neo4j 商品图谱数据源，
边界重叠导致 ReAct LLM 在相似工具间误路由。合并为 1 个「商品域」Agent，
对外暴露 3 个职责清晰的工具方法，内部各能力实现保持不变。
"""

from typing import Optional

from agents.tools.base import BaseAgentTool

# 多轮指代消解复用的实体键（品牌/商品/款式参与回填；规格不参与指代判断）
_ENTITY_REUSE_KEYS = ("商品", "品牌", "款式")

# 指代式追问触发词（用户用代词/序数词回指上一轮的商品）
_FOLLOWUP_MARKERS = (
    "那", "这个", "那个", "它", "这款", "那款", "这些", "那些",
    "还有", "别的", "其他", "另外", "第二个", "第一个", "最后一个", "换一个",
)


def _is_anaphoric_followup(query: str) -> bool:
    """是否为简短/指代式追问（用于判断是否回填上一轮实体）。"""
    q = (query or "").strip()
    if not q:
        return False
    if len(q) <= 6:
        return True
    return any(m in q for m in _FOLLOWUP_MARKERS)


def merge_entity_context(query: str, current_entities: Optional[dict] = None,
                         carried_entities: Optional[dict] = None) -> str:
    """多轮实体复用：本轮无新实体且上一轮有实体时，回填上一轮实体消解指代。

    规则（确定性）：
    - 上一轮无 品牌/商品/款式 实体可复用 → 原样返回；
    - 本轮已抽到 品牌/商品/款式 新实体 → 不复用（用户已换目标）；
    - 本轮是简短/指代式追问且无新实体 → 把上一轮的品牌/商品/款式追加到 query 尾部。
    """
    current = current_entities or {}
    carried = carried_entities or {}
    if not any(carried.get(k) for k in _ENTITY_REUSE_KEYS):
        return query
    if any(current.get(k) for k in _ENTITY_REUSE_KEYS):
        return query
    if not _is_anaphoric_followup(query):
        return query
    extra = []
    for k in _ENTITY_REUSE_KEYS:
        extra.extend(carried.get(k, []))
    extra = [x for x in extra if x and x not in query]
    if not extra:
        return query
    return f"{query} {' '.join(extra)}"


class CatalogAgent(BaseAgentTool):
    """
    商品域 Agent

    能力：
    - search:     商品搜索（FAISS 向量 + Neo4j 关键词混合检索）
    - recommend:  个性化推荐（图协同 + Item-CF + LLM 重排）
    - classify:   商品标题分类（BERT 15 分类）
    """

    name: str = "catalog_agent"
    description: str = (
        "商品域 Agent：统一提供商品搜索、个性化推荐、商品分类三个能力"
    )

    def __init__(self, search_agent, recommend_agent, classify_agent, ner_agent=None):
        super().__init__()
        self.search_agent = search_agent
        self.recommend_agent = recommend_agent
        self.classify_agent = classify_agent
        self.ner_agent = ner_agent  # 查询实体抽取（可选，增强搜索）

    # ------------------------------------------------------------------ #
    #  工具方法（ReAct Tool 直接绑定）
    # ------------------------------------------------------------------ #

    def search(self, query: str, entity_context: Optional[dict] = None, **kwargs) -> str:
        """商品搜索（向量语义 + 关键词 + 图结构混合检索）"""
        query = self._resolve_entity_context(query, entity_context)
        return self.search_agent.run(query=self._enrich_query(query), **kwargs)

    def recommend(self, query: str, entity_context: Optional[dict] = None, **kwargs) -> str:
        """个性化商品推荐（图协同 + Item-CF + LLM 重排）"""
        query = self._resolve_entity_context(query, entity_context)
        return self.recommend_agent.run(query=query, **kwargs)

    def classify(self, query: str, **kwargs) -> str:
        """商品标题分类（BERT 模型标签分类）"""
        return self.classify_agent.run(query=query, **kwargs)

    def extract_entities(self, query: str) -> dict:
        """从用户查询中抽取品牌/商品/款式/规格实体（NER 模型）"""
        if self.ner_agent is None:
            return {}
        return self.ner_agent.extract(query)

    def _resolve_entity_context(self, query: str, entity_context: Optional[dict]) -> str:
        """把上一轮实体（多轮指代）合并进当前 query；无实体上下文时原样返回。"""
        if not entity_context:
            return query
        current = self.extract_entities(query)
        return merge_entity_context(query, current_entities=current, carried_entities=entity_context)

    def _enrich_query(self, query: str) -> str:
        """用 NER 抽取的商品/款式实体增强搜索 query（模型不可用时原样返回）。"""
        if self.ner_agent is None:
            return query
        try:
            entities = self.ner_agent.extract(query)
            products = entities.get("商品", [])
            styles = entities.get("款式", [])
            extra = " ".join(products + styles)
            if extra:
                return f"{query} {extra}"
        except Exception as e:  # noqa: BLE001
            print(f"[CatalogAgent] NER 查询增强失败: {e}")
        return query

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        category=None,
        min_price=None,
        max_price=None,
    ) -> str:
        """独立搜索接口（/api/search 使用）"""
        return self.search_agent.hybrid_search(
            query=query,
            top_k=top_k,
            category=category,
            min_price=min_price,
            max_price=max_price,
        )

    def get_top_k(self, title: str, k: int = 3) -> list:
        """独立分类接口（/api/classify 使用），返回 Top-K 分类结果"""
        return self.classify_agent.get_top_k(title, k)

    # ------------------------------------------------------------------ #
    #  统一入口（降级路由按 mode 分派）
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        mode = kwargs.get("mode", "search")
        if mode == "recommend":
            return self.recommend(query, **kwargs)
        if mode == "classify":
            return self.classify(query, **kwargs)
        return self.search(query, **kwargs)

    async def arun(self, **kwargs) -> str:
        """异步执行（当前复用同步逻辑）"""
        return self.run(**kwargs)
