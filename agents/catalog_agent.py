"""
商品域 Agent — 统一封装搜索、推荐、分类三个能力

精简背景：search/recommend/classify 三个原 Agent 共享 Neo4j 商品图谱数据源，
边界重叠导致 ReAct LLM 在相似工具间误路由。合并为 1 个「商品域」Agent，
对外暴露 3 个职责清晰的工具方法，内部各能力实现保持不变。
"""

from agents.tools.base import BaseAgentTool


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

    def search(self, query: str, **kwargs) -> str:
        """商品搜索（向量语义 + 关键词 + 图结构混合检索）"""
        return self.search_agent.run(query=self._enrich_query(query), **kwargs)

    def recommend(self, query: str, **kwargs) -> str:
        """个性化商品推荐（图协同 + Item-CF + LLM 重排）"""
        return self.recommend_agent.run(query=query, **kwargs)

    def classify(self, query: str, **kwargs) -> str:
        """商品标题分类（BERT 模型标签分类）"""
        return self.classify_agent.run(query=query, **kwargs)

    def extract_entities(self, query: str) -> dict:
        """从用户查询中抽取品牌/商品/款式/规格实体（NER 模型）"""
        if self.ner_agent is None:
            return {}
        return self.ner_agent.extract(query)

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
