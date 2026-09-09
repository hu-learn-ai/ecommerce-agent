"""
推荐 Agent — 个性化商品推荐引擎

结合三种推荐策略：
1. 图结构推荐 (Neo4j 分类/品牌协同关系)
2. LLM 推理增强 (基于用户画像和需求的推理)
3. Item-CF 协同过滤 (基于 MySQL 订单数据的物品协同过滤)

学习参考: llm-based-recommender 的 LangGraph 推荐流程
          ec_graph 的 GNN 实体嵌入 (可扩展)
"""

import json
import os
from typing import List, Optional

from agents.tools.base import BaseAgentTool


class RecommendAgent(BaseAgentTool):
    """
    个性化商品推荐引擎

    流程：
        用户需求 + 用户画像
        → 图结构初筛 (Neo4j 协同查询)
        → Item-CF 补充 (MySQL 订单协同过滤)
        → LLM 重排序 (基于用户偏好推理)
        → 格式化推荐列表
    """

    name: str = "recommend_agent"
    description: str = (
        "商品推荐：根据用户偏好、需求描述和画像信息进行个性化推荐，"
        "结合知识图谱协同关系、Item-CF 协同过滤和 LLM 推理"
    )

    @staticmethod
    def _pretty_name(name: str) -> str:
        """
        美化合成商品名: '华为手机数码商品302' → '华为手机数码'

        样本数据集商品名是 品牌+品类+序号 的合成格式，序号对用户无意义，
        去掉后展示更自然（真实数据集的商品名不受影响）。
        """
        if not name:
            return name
        import re

        cleaned = re.sub(r"商品\d+$", "", name).strip()
        return cleaned or name

    def __init__(self, neo4j_driver, llm, db_config: dict = None, search_agent=None, engine=None):
        super().__init__()
        self.neo4j_driver = neo4j_driver
        self.llm = llm
        self.db_config = db_config  # MySQL 连接配置 (用于 Item-CF)
        self.search_agent = search_agent  # FAISS 向量搜索兜底（DB 不可用时）
        self.engine = engine  # SQLAlchemy 连接池（优先复用，避免每次新建连接）

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行推荐"""
        user_profile = kwargs.get("user_profile")
        top_k = kwargs.get("top_k", 5)
        return self.recommend(query, user_profile, top_k)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心推荐逻辑
    # ------------------------------------------------------------------ #

    def recommend(self, query: str, user_profile: Optional[dict] = None, top_k: int = 5) -> str:
        """
        根据用户偏好和当前上下文推荐商品

        Args:
            query: 用户需求描述
            user_profile: 用户画像（可选），如 {"偏好分类": "母婴", "预算": "100-200"}
            top_k: 推荐数量

        Returns:
            格式化的推荐结果字符串
        """
        user_profile = user_profile or {}

        # Step 1: 图结构推荐（基于分类/品牌的协同关系）
        graph_recs = self._graph_based_recommend(query, user_profile, top_k * 2)

        # Step 2: Item-CF 协同过滤补充（基于 MySQL 订单共现）
        cf_recs = self._item_cf_recommend(query, user_profile, top_k * 2)

        # 合并去重: 图推荐 + CF 推荐
        seen_ids = set()
        merged_recs = []
        for item in graph_recs + cf_recs:
            item_id = str(item.get("id", ""))
            if item_id and item_id not in seen_ids:
                seen_ids.add(item_id)
                merged_recs.append(item)

        # 如果合并后无结果：先尝试 FAISS 向量兜底（真实商品库），再降级到宽泛查询/如实告知
        if not merged_recs:
            fallback = self._vector_fallback_recommend(query, user_profile, top_k)
            if fallback:
                return fallback
            return self._llm_direct_recommend(query, user_profile, top_k)

        # 预算硬过滤：工具层直接排除超出预算的商品（避免 LLM 重排时"软"处理）
        budget = self._extract_budget(query)
        if budget:
            merged_recs = [r for r in merged_recs if self._within_budget(r, budget)]
            if not merged_recs:
                return (
                    f"抱歉，根据您的需求「{query}」，在 {budget:.0f} 元预算内"
                    "暂未找到匹配的商品。"
                )

        # Step 3: LLM 重排序（基于用户画像的推理增强）
        reranked = self._llm_rerank(query, merged_recs, user_profile, top_k)

        # Step 4: 格式化输出
        return self._format_recommendations(reranked[:top_k])

    @staticmethod
    def _extract_budget(query: str) -> Optional[float]:
        """从需求描述中解析预算上限（支持 万/千/w/k/块/元）。"""
        import re

        if not query:
            return None
        patterns = [
            r"(?:预算|价格|价位|控制在|不超过|不超|低于)\s*[：:为是]?\s*(\d+(?:\.\d+)?)\s*(万|千|w|k|块|元)?",
            r"(\d+(?:\.\d+)?)\s*(万|千|w|k|块|元)?\s*(?:以内|以下|左右|预算)",
        ]
        for pat in patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                num = float(m.group(1))
                unit = (m.group(2) or "").lower()
                if unit in ("万", "w"):
                    num *= 10000
                elif unit in ("千", "k"):
                    num *= 1000
                return num
        return None

    @staticmethod
    def _within_budget(item: dict, budget: float) -> bool:
        """判断商品是否在预算内；无价格信息时视为通过（无法核验不误杀）。"""
        price = item.get("price", "") or ""
        if not price and item.get("prices"):
            prices = [p for p in item["prices"] if p]
            price = prices[0] if prices else ""
        if not price:
            return True
        try:
            return float(price) <= budget
        except (TypeError, ValueError):
            return True

    def _vector_fallback_recommend(
        self, query: str, user_profile: dict, top_k: int
    ) -> Optional[str]:
        """FAISS 向量兜底：图/CF 均无结果时，从真实商品库检索并走 LLM 重排。"""
        if self.search_agent is None:
            return None
        try:
            records = self.search_agent.search_structured(query, top_k=top_k * 2)
            if not records:
                return None

            candidates = []
            for r in records:
                brand = r.get("brand", "") or ""
                price = r.get("price", "")
                candidates.append(
                    {
                        "product": r.get("name", ""),
                        "id": r.get("id", ""),
                        "category": r.get("category", ""),
                        "brand": brand,
                        "brands": [brand] if brand else [],
                        "price": price,
                        # 相似度(0-1) 转 10 分制，便于统一展示
                        "score": round(float(r.get("score", 0)) * 10, 1),
                    }
                )

            # 预算硬过滤（向量兜底同样适用）
            budget = self._extract_budget(query)
            if budget:
                candidates = [c for c in candidates if self._within_budget(c, budget)]
                if not candidates:
                    return None

            reranked = self._llm_rerank(query, candidates, user_profile, top_k)
            if not reranked:
                return None
            return self._format_recommendations(reranked[:top_k])
        except Exception as e:  # noqa: BLE001
            print(f"[RecommendAgent] 向量兜底推荐失败: {e}")
            return None

    def _graph_based_recommend(self, query: str, user_profile: dict, top_k: int) -> list:
        """
        基于知识图谱的协同推荐

        策略：
        1. 从用户画像提取偏好分类和品牌
        2. 在图中查找该分类/品牌下的商品
        3. 通过品牌-分类协同关系扩展候选
        """
        # 提取用户偏好
        preferred_category = user_profile.get("偏好分类") or user_profile.get("category")
        preferred_brand = user_profile.get("偏好品牌") or user_profile.get("brand")
        budget = user_profile.get("预算") or user_profile.get("budget")

        # 如果用户画像无明确分类，用 LLM 从 query 中提取
        if not preferred_category:
            preferred_category = self._extract_category(query, user_profile)

        if not preferred_category and not preferred_brand:
            # 无明确偏好，返回热门商品
            return self._get_popular_products(top_k)

        cypher = """
        MATCH (p:SPU)-[:Belong]->(c:Category3)
        WHERE ($category IS NULL OR c.name CONTAINS $category)
        OPTIONAL MATCH (p)-[:Have]->(t:Trademark)
        WHERE ($brand IS NULL OR t.name CONTAINS $brand OR t.tm_name CONTAINS $brand)
        OPTIONAL MATCH (p)-[:Have]->(sku:SKU)
        RETURN p.name AS product, p.id AS id,
               c.name AS category,
               collect(DISTINCT t.name)[0..2] AS brands,
               collect(DISTINCT sku.price)[0..3] AS prices
        LIMIT $top_k
        """

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    cypher,
                    {
                        "category": preferred_category,
                        "brand": preferred_brand,
                        "top_k": top_k,
                    },
                )
                records = []
                for r in result:
                    brands = r.get("brands", []) or []
                    prices = r.get("prices", []) or []
                    record = {
                        "product": r["product"],
                        "id": str(r["id"]),
                        "category": r.get("category", ""),
                        "brands": brands,
                        "prices": prices,
                        # 默认取首个品牌/价格，保证展示时有真实信息
                        "brand": brands[0] if brands else "",
                        "price": prices[0] if prices else "",
                    }
                    # 价格过滤
                    if budget:
                        price_range = self._parse_budget(budget)
                        if price_range:
                            min_p, max_p = price_range
                            valid_prices = [
                                float(p)
                                for p in prices
                                if p and self._in_range(float(p), min_p, max_p)
                            ]
                            if not valid_prices:
                                continue
                            record["price"] = valid_prices[0]
                    records.append(record)
                return records
        except Exception as e:
            print(f"[RecommendAgent] 图查询失败: {e}")
            return []

    def _item_cf_recommend(self, query: str, user_profile: dict, top_k: int) -> list:
        """
        Item-CF 协同过滤推荐 — 基于 MySQL 订单数据的物品协同过滤

        原理:
        1. 从用户偏好分类中提取种子商品 (seed items)
        2. 查询 MySQL: 购买过种子商品的用户还购买了什么
        3. 按共现频率排序 (co-occurrence count)
        4. 归一化为相似度分数 (cosine normalization)

        这是经典的 "购买了该商品的用户还购买了" 推荐策略。
        """
        if not self.db_config:
            return []

        import pymysql

        preferred_category = user_profile.get("偏好分类") or user_profile.get("category")
        if not preferred_category:
            preferred_category = self._extract_category(query, user_profile)
        if not preferred_category:
            return []

        try:
            # 统一走 registry 创建的 SQLAlchemy 连接池；无 engine 时降级 pymysql
            if self.engine is not None:
                from sqlalchemy import text

                with self.engine.connect() as conn:
                    # Step 1: 种子商品
                    result = conn.execute(
                        text(
                            "SELECT DISTINCT od.product_id, od.product_name "
                            "FROM order_detail od WHERE od.product_name LIKE :kw LIMIT 10"
                        ),
                        {"kw": f"%{preferred_category}%"},
                    )
                    seed_products = [dict(r._mapping) for r in result]
                    if not seed_products:
                        return []
                    seed_ids = [p["product_id"] for p in seed_products]

                    # Step 2: 购买过种子的用户
                    placeholders = ",".join([":s%d" % i for i in range(len(seed_ids))])
                    params = {f"s{i}": sid for i, sid in enumerate(seed_ids)}
                    result = conn.execute(
                        text(
                            f"SELECT DISTINCT user_id FROM order_detail "
                            f"WHERE product_id IN ({placeholders})"
                        ),
                        params,
                    )
                    user_ids = [dict(r._mapping)["user_id"] for r in result]
                    if not user_ids:
                        return []
                    user_ids = user_ids[:500]

                    # Step 3: 共同购买商品
                    u_placeholders = ",".join([":u%d" % i for i in range(len(user_ids))])
                    s_placeholders = ",".join([":p%d" % i for i in range(len(seed_ids))])
                    result = conn.execute(
                        text(
                            f"SELECT product_id, product_name, COUNT(*) AS co_count "
                            f"FROM order_detail "
                            f"WHERE user_id IN ({u_placeholders}) "
                            f"AND product_id NOT IN ({s_placeholders}) "
                            f"GROUP BY product_id, product_name "
                            f"ORDER BY co_count DESC LIMIT {int(top_k)}"
                        ),
                        {**{f"u{i}": uid for i, uid in enumerate(user_ids)},
                         **{f"p{i}": sid for i, sid in enumerate(seed_ids)}},
                    )
                    co_purchased = [dict(r._mapping) for r in result]
            else:
                conn = pymysql.connect(**self.db_config)
                with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                # Step 1: 查找该分类下的种子商品 (用户购买过的)
                    cursor.execute(
                        """SELECT DISTINCT od.product_id, od.product_name
                           FROM order_detail od
                           WHERE od.product_name LIKE %s
                           LIMIT 10""",
                        (f"%{preferred_category}%",),
                    )
                    seed_products = cursor.fetchall()
                    if not seed_products:
                        return []
                    seed_ids = [p["product_id"] for p in seed_products]
                    placeholders = ",".join(["%s"] * len(seed_ids))
                    cursor.execute(
                        f"SELECT DISTINCT user_id FROM order_detail "
                        f"WHERE product_id IN ({placeholders})",
                        seed_ids,
                    )
                    user_ids = [row["user_id"] for row in cursor.fetchall()]
                    if not user_ids:
                        return []
                    user_ids = user_ids[:500]
                    user_placeholders = ",".join(["%s"] * len(user_ids))
                    cursor.execute(
                        f"SELECT product_id, product_name, COUNT(*) as co_count "
                        f"FROM order_detail "
                        f"WHERE user_id IN ({user_placeholders}) "
                        f"AND product_id NOT IN ({placeholders}) "
                        f"GROUP BY product_id, product_name "
                        f"ORDER BY co_count DESC LIMIT %s",
                        user_ids + seed_ids + [top_k],
                    )
                    co_purchased = cursor.fetchall()

            # Step 4: 计算归一化相似度分数并构建推荐列表（engine/pymysql 两种数据源共用）
            max_count = co_purchased[0]["co_count"] if co_purchased else 1
            results = []
            for item in co_purchased:
                results.append(
                    {
                        "product": item["product_name"],
                        "id": str(item["product_id"]),
                        "category": preferred_category,
                        "score": round(item["co_count"] / max_count * 10, 1),
                        "reason": f"{item['co_count']}位相似用户共同购买",
                    }
                )

            return results
        except Exception as e:
            print(f"[RecommendAgent] Item-CF 查询失败: {e}")
            return []

    def _llm_rerank(self, query: str, candidates: list, user_profile: dict, top_k: int) -> list:
        """
        LLM 重排序：基于用户画像对候选商品进行推理重排

        输入候选商品列表 → LLM 评估每个商品的匹配度 → 按匹配度排序
        """
        if not candidates:
            return []
        from config.settings import settings

        # 开关 + 候选上限：控制 heavy 模型的 token 成本与延迟
        if not settings.recommend_rerank_enabled:
            return candidates[:top_k]

        # 压缩候选为精简字段（去掉 description 等冗余，限制候选数）
        compact = []
        for c in candidates[:12]:
            compact.append(
                {
                    "id": c.get("id"),
                    "product": c.get("product", c.get("name", "")),
                    "category": c.get("category", ""),
                    "price": c.get("price", ""),
                    "brand": c.get("brand", "")
                    or ((c.get("brands") or [""])[0] if isinstance(c.get("brands"), list) else ""),
                }
            )
        candidates_text = json.dumps(compact, ensure_ascii=False)

        prompt = f"""你是一个电商推荐专家。请根据用户需求重新排序候选商品。

用户需求: {query}
用户画像: {json.dumps(user_profile, ensure_ascii=False)}

候选商品(JSON):
{candidates_text}

请根据用户需求和画像，对候选商品按推荐优先级排序。
返回 JSON 数组格式，每个元素包含:
  - "product": 商品名
  - "id": 商品 ID（必须原样返回候选中的 id）
  - "reason": 推荐理由（一句话）
  - "score": 匹配度评分(0-10)

只返回 JSON 数组，不要其他文字。"""

        try:
            if self.llm is None:
                return candidates[:top_k]
            response = self.llm.invoke(prompt)
            content = response.content.strip()

            # 清理 markdown 标记
            if content.startswith("```"):
                content = content.split("\n", 1)[1] if "\n" in content else content
                content = content.rsplit("```", 1)[0] if "```" in content else content
                content = content.strip()

            reranked = json.loads(content)
            # 按 score 降序
            reranked.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

            # 合并原始信息
            for item in reranked:
                for orig in candidates:
                    # 优先按 id 匹配（LLM 改写商品名时仍能找回原始元数据）
                    if (
                        orig.get("id")
                        and item.get("id")
                        and str(orig.get("id")) == str(item.get("id"))
                    ):
                        item.setdefault("id", orig.get("id", ""))
                        item.setdefault("category", orig.get("category", ""))
                        item.setdefault("price", orig.get("price", ""))
                        break
                    if orig.get("product") == item.get("product"):
                        item.setdefault("id", orig.get("id", ""))
                        item.setdefault("category", orig.get("category", ""))
                        item.setdefault("price", orig.get("price", ""))
                        break

            return reranked[:top_k]
        except Exception as e:
            print(f"[RecommendAgent] LLM 重排序失败: {e}")
            return candidates[:top_k]

    def _llm_direct_recommend(self, query: str, user_profile: dict, top_k: int) -> str:
        """
        图查询和 Item-CF 均无结果时的降级处理

        严禁让 LLM 自行编造商品推荐。
        尝试用 Neo4j 宽泛查询作为最后兜底，如果仍无结果则如实告知用户。
        """
        # 尝试 Neo4j 宽泛查询（不做分类过滤，直接按名称模糊匹配）
        if self.neo4j_driver:
            try:
                cypher = """
                MATCH (p:SPU)-[:Belong]->(c:Category3)
                OPTIONAL MATCH (p)-[:Have]->(sku:SKU)
                RETURN p.name AS product, p.id AS id,
                       c.name AS category,
                       collect(DISTINCT sku.price)[0..1] AS prices
                LIMIT $top_k
                """
                with self.neo4j_driver.session() as session:
                    result = session.run(cypher, {"top_k": top_k})
                    items = []
                    for r in result:
                        name = r["product"] or ""
                        category = r.get("category", "")
                        prices = r.get("prices", [])
                        price = prices[0] if prices else ""
                        line = f"**{name}**"
                        if category:
                            line += f" | 分类: {category}"
                        if price:
                            line += f" | 价格: ¥{price}"
                        items.append(line)

                    if items:
                        header = f"💡 为您推荐 {len(items)} 个商品：\n"
                        return header + "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))
            except Exception as e:
                print(f"[RecommendAgent] Neo4j 兜底查询失败: {e}")

        # 所有数据源均无结果，如实告知
        return (
            f"抱歉，根据您的需求「{query}」，暂未在商品库中找到匹配的商品。\n"
            "您可以尝试：\n"
            "1. 换一个关键词搜索（如「搜索 耳机」）\n"
            "2. 浏览热门品类（如母婴、户外、食品）\n"
            "3. 描述更具体的需求（如「预算100以内的零食」）"
        )

    @staticmethod
    def _category_candidates() -> List[str]:
        """候选分类：动态读取分类模型 labels.txt，避免写死 15 类与模型不一致。"""
        try:
            from config.settings import settings

            labels_file = os.path.join(settings.classify_model_path, "labels.txt")
            if os.path.isfile(labels_file):
                with open(labels_file, encoding="utf-8") as f:
                    labels = [line.strip() for line in f if line.strip()]
                if labels:
                    return labels
        except Exception:  # noqa: BLE001
            pass
        return ["医药保健", "家用电器", "手机数码", "食品生鲜"]

    def _extract_category(self, query: str, user_profile: dict = None) -> Optional[str]:
        """
        从用户查询 + 用户画像（含长期记忆偏好）中提取商品分类

        限定在分类模型候选标签内，避免 LLM 自由发挥。
        """
        candidate_list = self._category_candidates()
        candidates = "、".join(candidate_list)
        profile_hint = ""
        notes = (user_profile or {}).get("long_term_notes") or []
        if notes:
            profile_hint = (
                "\n\n用户历史偏好（可参考，不作为唯一依据）:\n"
                + "\n".join(f"- {note}" for note in notes)
            )

        prompt = f"""从以下用户需求中提取最匹配的商品分类，必须从候选分类中选择一个，只回复分类名，不要其他文字。
如果需求与任何候选分类都不相关，回复 "unknown"。

候选分类: {candidates}

用户需求: {query}{profile_hint}

分类:"""
        try:
            result = self.llm.invoke(prompt).content.strip()
            # 校验结果必须是候选分类之一
            for cat in candidate_list:
                if cat in result:
                    return cat
            return None
        except Exception:
            return None

    def _get_popular_products(self, top_k: int) -> list:
        """获取热门商品（无明确偏好时的兜底）"""
        # 按销量/热度降序（替代 ORDER BY rand() / 按 id 取前 N）
        cypher = """
        MATCH (p:SPU)
        OPTIONAL MATCH (p)-[:Have]->(sku:SKU)
        OPTIONAL MATCH (p)-[:Belong]->(c:Category3)
        WITH p, c, collect(DISTINCT sku.price)[0..1] AS prices
        RETURN p.name AS product, p.id AS id, prices[0] AS price, c.name AS category
        ORDER BY coalesce(p.sales_count, 0) DESC, coalesce(p.popularity_score, 0.0) DESC
        LIMIT $top_k
        """
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(cypher, {"top_k": top_k})
                return [
                    {
                        "product": r["product"],
                        "id": str(r["id"]),
                        "price": str(r.get("price", "")),
                        "category": r.get("category") or "",
                    }
                    for r in result
                ]
        except Exception:
            return []

    def _parse_budget(self, budget) -> tuple:
        """解析预算字符串，返回 (min, max) 元组"""
        if isinstance(budget, (int, float)):
            return (0, float(budget))
        if isinstance(budget, str):
            import re

            nums = re.findall(r"\d+\.?\d*", budget)
            if len(nums) >= 2:
                return (float(nums[0]), float(nums[1]))
            elif len(nums) == 1:
                return (0, float(nums[0]))
        return None

    def _in_range(self, price: float, min_p: float, max_p: float) -> bool:
        """检查价格是否在范围内"""
        return min_p <= price <= max_p

    def _format_recommendations(self, items: list) -> str:
        """格式化推荐列表"""
        if not items:
            return "暂无推荐商品，请稍后再试。"

        lines = [f"💡 为您推荐 {len(items)} 个商品：\n"]
        for i, item in enumerate(items, 1):
            name = self._pretty_name(item.get("product", item.get("name", "未知商品")))
            reason = item.get("reason", "")
            category = item.get("category", "")
            # 图查询返回的是 brands 列表，需兼容两种字段
            brand = item.get("brand", "")
            if not brand and item.get("brands"):
                brand = (
                    item["brands"][0]
                    if isinstance(item["brands"], list) and item["brands"]
                    else item["brands"]
                )
            price = item.get("price", "")
            score = item.get("score", "")

            line = f"{i}. **{name}**"
            if brand:
                line += f" | 品牌: {brand}"
            if category:
                line += f" | 分类: {category}"
            if price:
                line += f" | 价格: ¥{price}"
            if reason:
                line += f"\n   推荐理由: {reason}"
            elif score:
                line += f" | 匹配度: {score}/10"
            lines.append(line)

        return "\n".join(lines)
