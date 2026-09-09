"""
商品搜索 Agent — 混合搜索引擎

融合三种检索策略：
1. 向量语义搜索 (FAISS + BGE 嵌入)
2. 关键词匹配 (Neo4j 全文索引)
3. 图结构增强 (分类/品牌关联)

学习参考: llm-based-recommender 的混合检索方案
          E-Commerce Shopping Assistant 的 BGE + FAISS RAG
"""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from agents.tools.base import BaseAgentTool


class ProductSearchAgent(BaseAgentTool):
    """
    混合商品搜索引擎

    检索流程：
        用户查询 → BGE 向量化
                 → FAISS 语义检索 (Top-K*2)
                 → Neo4j 全文索引关键词检索
                 → Reciprocal Rank Fusion 融合排序
                 → 返回 Top-K 结果
    """

    name: str = "search_agent"
    description: str = (
        "商品搜索：根据用户查询进行向量语义+关键词+图结构混合搜索，支持按分类和价格范围过滤"
    )

    # 余弦相似度噪声下限：IndexFlatIP + 归一化向量 ≈ cosine 相似度，
    # 低于该值的命中基本与查询无关（如 0.016 的噪声），直接过滤避免误导 LLM
    MIN_COSINE_SCORE: float = 0.30

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

    def __init__(
        self,
        embedding_model_name: str,
        neo4j_driver,
        faiss_index_path: str,
        shared_embedder=None,
    ):
        super().__init__()
        self.embedding_model_name = embedding_model_name
        self.neo4j_driver = neo4j_driver
        self.faiss_index_path = faiss_index_path

        # 使用共享嵌入模型或懒加载
        self._embedder = shared_embedder
        try:
            from config.settings import settings

            self.min_cosine_score = float(
                getattr(settings, "search_min_cosine_score", self.MIN_COSINE_SCORE)
            )
        except Exception:
            self.min_cosine_score = self.MIN_COSINE_SCORE

        # 加载 FAISS 索引（需要预先构建，见 scripts/build_faiss_index.py）
        self.faiss_index = None
        self.product_ids = None
        # 商品 ID → 元数据映射（FAISS 只存向量和 ID，真实名称/价格/分类靠它补充）
        self._product_meta: Dict[str, Dict[str, Any]] = {}
        self._load_faiss_index()

    # ------------------------------------------------------------------ #
    #  懒加载
    # ------------------------------------------------------------------ #

    @property
    def embedder(self):
        """懒加载 SentenceTransformer 嵌入模型"""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _load_faiss_index(self):
        """加载预构建的 FAISS 索引"""
        index_file = os.path.join(self.faiss_index_path, "products.index")
        ids_file = os.path.join(self.faiss_index_path, "product_ids.npy")

        try:
            import faiss

            if os.path.exists(index_file) and os.path.exists(ids_file):
                self.faiss_index = faiss.read_index(index_file)
                self.product_ids = np.load(ids_file)
                print(f"[SearchAgent] 已加载 FAISS 索引: {len(self.product_ids)} 个商品")
                # 加载商品元数据映射（FAISS 命中后补充真实名称/价格/分类）
                self._load_product_meta()
            else:
                print("[SearchAgent] FAISS 索引未找到，请先运行 scripts/build_faiss_index.py")
        except ImportError:
            print("[SearchAgent] faiss-cpu 未安装，向量检索不可用")
        except Exception as e:
            print(f"[SearchAgent] 加载 FAISS 索引失败: {e}")

    def _load_product_meta(self):
        """
        加载 商品ID → 元数据 映射，用于 FAISS 命中后补充真实名称/价格/分类/品牌

        优先读取构建索引时的源 JSON（data/processed/products_for_faiss.json），
        JSON 缺失时降级为 Neo4j 批量查询。
        """
        # 1. 本地 JSON（与 FAISS 索引同源，2000 商品，含 name/category/brand/price）
        json_path = os.path.join(
            os.path.dirname(self.faiss_index_path),
            "processed",
            "products_for_faiss.json",
        )
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    products = json.load(f)
                self._product_meta = {str(p["id"]): p for p in products if p.get("id")}
                print(f"[SearchAgent] 已加载商品元数据映射: {len(self._product_meta)} 条 (JSON)")
                return
            except Exception as e:
                print(f"[SearchAgent] 加载商品元数据 JSON 失败: {e}")

        # 2. Neo4j 兜底（懒加载，仅当 JSON 不可用时触发）
        if self.neo4j_driver:
            try:
                cypher = """
                MATCH (p:SPU)-[:Belong]->(c:Category3)
                OPTIONAL MATCH (p)-[:Have]->(sku:SKU)
                RETURN p.id AS id, p.name AS name, c.name AS category,
                       collect(DISTINCT sku.price)[0..1] AS prices
                """
                with self.neo4j_driver.session() as session:
                    result = session.run(cypher)
                    for r in result:
                        prices = r.get("prices") or []
                        self._product_meta[str(r["id"])] = {
                            "id": str(r["id"]),
                            "name": r["name"] or "",
                            "category": r.get("category") or "",
                            "price": prices[0] if prices else "",
                        }
                print(f"[SearchAgent] 已加载商品元数据映射: {len(self._product_meta)} 条 (Neo4j)")
            except Exception as e:
                print(f"[SearchAgent] Neo4j 商品元数据加载失败: {e}")

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行混合搜索"""
        top_k = kwargs.get("top_k", 10)
        category = kwargs.get("category")
        min_price = kwargs.get("min_price")
        max_price = kwargs.get("max_price")

        return self.hybrid_search(
            query=query,
            top_k=top_k,
            category=category,
            min_price=min_price,
            max_price=max_price,
        )

    async def arun(self, **kwargs) -> str:
        """异步执行（当前复用同步逻辑）"""
        return self.run(**kwargs)

    def search_structured(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """结构化向量检索（供推荐链路兜底等场景复用）。

        返回记录列表: {id, name, category, brand, price, sales_count,
                       popularity_score, score, source, rank}
        """
        results = self._vector_search(query, top_k * 2)
        if category:
            results = [r for r in results if category in (r.get("category") or "")]
        return results[:top_k]

    # ------------------------------------------------------------------ #
    #  核心搜索逻辑
    # ------------------------------------------------------------------ #

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> str:
        """
        混合搜索商品

        Args:
            query: 搜索关键词或描述
            top_k: 返回数量
            category: 按分类过滤（可选）
            min_price: 最低价格过滤（可选）
            max_price: 最高价格过滤（可选）

        Returns:
            格式化后的搜索结果字符串
        """
        # 从用户查询中解析预算/价格约束（如「500元以下」「100-200元」）
        price_constraints = self._parse_price_constraints(query)
        if min_price is None and max_price is None and price_constraints:
            min_price, max_price = price_constraints

        # Step 1: 向量语义检索
        vector_results = self._vector_search(query, top_k * 2)
        # 向量结果按价格过滤（元数据含价格时才生效；JSON 数据源无价格则交由关键词检索过滤）
        if min_price is not None or max_price is not None:
            vector_results = self._filter_by_price(vector_results, min_price, max_price)

        # Step 2: Neo4j 关键词全文检索（复用 ec_graph 的索引）
        cypher_results = self._neo4j_keyword_search(query, top_k, category, min_price, max_price)

        # Step 3: Reciprocal Rank Fusion 融合排序
        merged = self._reciprocal_rank_fusion(vector_results, cypher_results)

        # Step 4: 格式化输出
        return self._format_results(merged[:top_k])

    @staticmethod
    def _parse_price_constraints(query: str):
        """从自然语言查询中解析价格区间，返回 (min_price, max_price) 或 None。

        支持: 「500元以下」「预算200以内」「100-300元」「至少800」「300元以上」
        """
        import re

        if not query:
            return None
        text = query.strip()
        num = r"(\d{1,7}(?:\.\d{1,2})?)"

        # 1. 区间: 100-300元 / 100~300 / 100至300
        m = re.search(num + r"\s*[-~至到]\s*" + num + r"\s*(?:元|块)?", text)
        if m:
            lo, hi = float(m.group(1)), float(m.group(2))
            return (min(lo, hi), max(lo, hi))

        # 2. 上限: 预算200 / 500元以下 / 300以内 / 不超过1000
        m = re.search(
            r"(?:预算|不超过|低于|以内|最多|封顶)\s*" + num + r"\s*(?:元|块)?", text
        ) or re.search(num + r"\s*(?:元|块)\s*(?:以|之)?内", text) or re.search(
            num + r"\s*(?:元|块)\s*(?:以|之)?下", text
        )
        if m:
            return (None, float(m.group(1)))

        # 3. 下限: 至少800 / 300元以上 / 超过1000
        m = re.search(
            r"(?:至少|不低于|超过|高于|起步)\s*" + num + r"\s*(?:元|块)?", text
        ) or re.search(num + r"\s*(?:元|块)\s*(?:以|之)?上", text)
        if m:
            return (float(m.group(1)), None)

        return None

    @staticmethod
    def _filter_by_price(results: List[Dict[str, Any]], min_price, max_price) -> List:
        """按价格过滤向量检索结果；无价格信息的结果保留（交由关键词检索兜底）。"""
        if not results:
            return results
        filtered = []
        for r in results:
            p = r.get("price")
            if p in (None, ""):
                filtered.append(r)
                continue
            try:
                pv = float(p)
            except (TypeError, ValueError):
                filtered.append(r)
                continue
            if min_price is not None and pv < min_price:
                continue
            if max_price is not None and pv > max_price:
                continue
            filtered.append(r)
        return filtered

    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        FAISS 向量语义检索（自动补充真实商品元数据）

        注意：FAISS 索引只存向量和商品 ID，必须通过 _product_meta 映射
        补充真实名称/分类/品牌/价格，否则 LLM 拿到的只有编号（如 商品#P001057）。
        """
        if not self.faiss_index or self.product_ids is None:
            return []

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype(
            np.float32
        )

        distances, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < 0:
                continue
            # 余弦相似度噪声过滤（低于阈值的基本与查询无关）
            if float(dist) < self.min_cosine_score:
                continue
            product_id = str(self.product_ids[idx])
            meta = self._product_meta.get(product_id, {})
            results.append(
                {
                    "id": product_id,
                    "name": meta.get("name") or f"商品#{product_id}",
                    "category": meta.get("category", ""),
                    "brand": meta.get("brand", ""),
                    "price": meta.get("price", ""),
                    "sales_count": meta.get("sales_count", ""),
                    "popularity_score": meta.get("popularity_score", ""),
                    "score": float(dist),
                    "source": "vector",
                    "rank": rank,
                }
            )
        return results

    def _neo4j_keyword_search(
        self,
        query: str,
        top_k: int,
        category: Optional[str],
        min_price: Optional[float],
        max_price: Optional[float],
    ) -> List[Dict[str, Any]]:
        """
        Neo4j 全文索引搜索（复用 ec_graph 的索引能力）

        使用 fulltext index 对 SPU 节点进行关键词匹配
        同时支持按分类和价格范围过滤
        """
        if not self.neo4j_driver:
            return []

        cypher = """
        CALL db.index.fulltext.queryNodes('spu_fulltext', $query)
        YIELD node, score
        MATCH (node)-[:Belong]->(c:Category3)
        WHERE ($category IS NULL OR c.name CONTAINS $category)
        OPTIONAL MATCH (node)-[:Have]->(sku:SKU)
        WHERE ($min_price IS NULL OR coalesce(toFloat(sku.price), 0) >= $min_price)
          AND ($max_price IS NULL OR coalesce(toFloat(sku.price), 0) <= $max_price)
        WITH node, score, c, collect(DISTINCT sku)[0..3] AS skus
        OPTIONAL MATCH (node)-[:Have]->(t:Trademark)
        RETURN node.name AS name, node.id AS id, score, c.name AS category,
               [s in skus | coalesce(s.price, '')][0] AS price,
               collect(DISTINCT t.name)[0] AS brand
        ORDER BY score DESC LIMIT $top_k
        """

        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    cypher,
                    {
                        "query": query,
                        "top_k": top_k,
                        "category": category,
                        "min_price": min_price,
                        "max_price": max_price,
                    },
                )
                return [
                    {
                        "id": str(r["id"]),
                        "name": r["name"],
                        "score": r["score"],
                        "category": r.get("category", ""),
                        "brand": r.get("brand") or "",
                        "price": r.get("price") or "",
                        "source": "keyword",
                        "rank": i,
                    }
                    for i, r in enumerate(result)
                ]
        except Exception as e:
            # fulltext index 可能不存在，降级为 CONTAINS 匹配
            print(f"[SearchAgent] Neo4j fulltext 搜索失败，降级为 CONTAINS: {e}")
            return self._neo4j_contains_search(query, top_k, category)

    def _neo4j_contains_search(
        self, query: str, top_k: int, category: Optional[str]
    ) -> List[Dict[str, Any]]:
        """降级方案：使用 CONTAINS 模糊匹配"""
        cypher = """
        MATCH (p:SPU)-[:Belong]->(c:Category3)
        WHERE p.name CONTAINS $query
          AND ($category IS NULL OR c.name CONTAINS $category)
        OPTIONAL MATCH (p)-[:Have]->(sku:SKU)
        OPTIONAL MATCH (p)-[:Have]->(t:Trademark)
        RETURN p.name AS name, p.id AS id, c.name AS category,
               [s in collect(DISTINCT sku) | coalesce(s.price, '')][0] AS price,
               collect(DISTINCT t.name)[0] AS brand
        LIMIT $top_k
        """
        try:
            with self.neo4j_driver.session() as session:
                result = session.run(
                    cypher,
                    {
                        "query": query,
                        "top_k": top_k,
                        "category": category,
                    },
                )
                return [
                    {
                        "id": str(r["id"]),
                        "name": r["name"],
                        "category": r.get("category", ""),
                        "brand": r.get("brand") or "",
                        "price": r.get("price") or "",
                        "source": "keyword",
                        "rank": i,
                    }
                    for i, r in enumerate(result)
                ]
        except Exception as e:
            print(f"[SearchAgent] Neo4j CONTAINS 降级搜索也失败: {e}")
            import traceback

            traceback.print_exc()
            return []

    def _reciprocal_rank_fusion(
        self,
        vec_results: List[Dict[str, Any]],
        cypher_results: List[Dict[str, Any]],
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion 融合排序算法

        RRF score = Σ 1/(k + rank_i)
        将向量检索和关键词检索的结果按排名融合
        """
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vec_results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank + 1)
            # 合并时保留原始相似度分数（用于展示，RRF 融合分仅用于排序）
            if item_id in items:
                items[item_id]["score"] = max(items[item_id].get("score", 0), item.get("score", 0))
            else:
                items[item_id] = item

        for rank, item in enumerate(cypher_results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0) + 1 / (k + rank + 1)
            # 合并信息：如果已有向量结果，用关键词结果补充 category/name；
            # Lucene score 与余弦相似度量纲不同，单独存 kw_score，不再混入 score
            if item_id in items:
                if "category" in item and "category" not in items[item_id]:
                    items[item_id]["category"] = item["category"]
                if "name" in item and (
                    not items[item_id].get("name") or items[item_id]["name"].startswith("商品#")
                ):
                    items[item_id]["name"] = item["name"]
                items[item_id]["kw_score"] = max(
                    items[item_id].get("kw_score", 0), item.get("score", 0)
                )
            else:
                merged = dict(item)
                merged["kw_score"] = item.get("score", 0)
                merged.pop("score", None)
                items[item_id] = merged

        # 按融合分数降序排列
        sorted_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [{**items[item_id], "fusion_score": score} for item_id, score in sorted_ids]

    def _format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化搜索结果为可读字符串

        展示真实商品名/品牌/分类/价格，相关度用原始相似度分数
        （RRF 融合分仅用于排序，数值极小且不直观，不能展示给 LLM）。
        """
        if not results:
            return "未找到匹配的商品，请尝试更换关键词。"

        lines = [f"🔍 共找到 {len(results)} 个相关商品：\n"]
        for i, item in enumerate(results, 1):
            name = self._pretty_name(item.get("name", "未知商品"))
            brand = item.get("brand", "")
            category = item.get("category", "")
            price = item.get("price", "")
            sales_count = item.get("sales_count", "")
            # 优先展示原始相似度分数，避免把 RRF 融合分(≈0.016)误当相关度
            score = item.get("score")
            kw_score = item.get("kw_score")

            line = f"{i}. **{name}**"
            if brand:
                line += f" | 品牌: {brand}"
            if category:
                line += f" | 分类: {category}"
            if price:
                line += f" | 价格: ¥{price}"
            if sales_count:
                line += f" | 销量: {sales_count}"
            if score is not None:
                line += f" | 相似度: {float(score):.4f}"
            elif kw_score:
                line += f" | 关键词相关度: {float(kw_score):.2f}"
            lines.append(line)

        return "\n".join(lines)
