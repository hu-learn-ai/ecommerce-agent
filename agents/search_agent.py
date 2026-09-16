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
import re
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
        # BM25 关键词检索（懒构建；jieba + rank_bm25 缺失时回落 Neo4j 全文索引）
        self._bm25 = None
        self._bm25_docs: List[Dict[str, Any]] = []
        self._bm25_failed = False
        try:
            from config.settings import settings

            self.keyword_backend = getattr(settings, "search_keyword_backend", "bm25")
        except Exception:
            self.keyword_backend = "bm25"
        try:
            from config.settings import settings

            self.adaptive_rrf_weights = bool(
                getattr(settings, "rrf_adaptive_weights", False)
            )
        except Exception:
            self.adaptive_rrf_weights = False
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
        return self._format_results(
            self.search_results(
                query,
                top_k=top_k,
                category=category,
                min_price=min_price,
                max_price=max_price,
            )
        )

    def search_results(
        self,
        query: str,
        top_k: int = 10,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """生产检索链路的结构化返参（与 `hybrid_search` 同一条路径，仅不做格式化）。

        评估脚本应调用本方法，而不是自行拼 `_vector_search` + 关键词检索：
        这样评估才覆盖与线上一致的逻辑（价格解析、关键词后端选择、RRF 融合）。
        """
        # 从用户查询中解析预算/价格约束（如「500元以下」「100-200元」）
        price_constraints = self._parse_price_constraints(query)
        if min_price is None and max_price is None and price_constraints:
            min_price, max_price = price_constraints

        # Step 1: 向量语义检索（带价格过滤）
        # 有价格约束时先加深召回再过滤：否则"Top-20 恰好都超预算"会导致
        # 向量分支整体为空，只剩关键词分支的数字噪声（如"1000元以下手机"）
        has_price_constraint = min_price is not None or max_price is not None
        depth = max(top_k * 2, 200) if has_price_constraint else top_k * 2
        vector_results = self._vector_search(query, depth)
        if has_price_constraint:
            vector_results = self._filter_by_price(vector_results, min_price, max_price)
            vector_results = vector_results[: top_k * 2]

        # Step 2: 关键词检索（BM25 或 Neo4j 全文，见 keyword_backend）
        # 深度取 2×top_k：实测关键词候选给到 20 条时融合指标更好
        # （全量 NDCG@5 0.9086 → 0.9140、留出集 0.8396 → 0.8433、3/4 折占优，见 tests/tune_fusion.py）
        keyword_results = self.keyword_search(query, top_k * 2, category, min_price, max_price)

        # Step 3: Reciprocal Rank Fusion 融合排序
        w_vec, w_kw = self.rrf_weights_for(query)
        merged = self._reciprocal_rank_fusion(
            vector_results, keyword_results, w_vec=w_vec, w_kw=w_kw
        )

        # Step 4: 软预算排序（只对模拟价格生效：超预算的后置，但不删除）
        if min_price is not None or max_price is not None:
            merged = self._apply_soft_budget(merged, min_price, max_price)

        return merged[:top_k]

    @staticmethod
    def _apply_soft_budget(
        results: List[Dict[str, Any]], min_price, max_price
    ) -> List[Dict[str, Any]]:
        """软预算：**模拟价格**的商品若超出预算，排到最后但不删除。

        真实价格（手机数码 3000 件）在检索阶段已被硬过滤；模拟价格由类目分布生成
        （见 `scripts/fill_catalog_prices.py`），硬过滤会把真实相关的商品误滤掉
        ——例如"预算200以内的蓝牙耳机"的候选全是模拟价格，硬过滤会直接返回空。
        因此这里改成排序偏好：符合预算的优先，超预算的靠后。
        """
        preferred, over_budget = [], []
        for item in results:
            price = item.get("price")
            if item.get("price_source") != "simulated" or price in (None, ""):
                preferred.append(item)
                continue
            try:
                value = float(price)
            except (TypeError, ValueError):
                preferred.append(item)
                continue
            if (min_price is not None and value < min_price) or (
                max_price is not None and value > max_price
            ):
                over_budget.append(item)
            else:
                preferred.append(item)
        return preferred + over_budget

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
        """按价格过滤向量检索结果。

        保留两类商品：① 无价格信息（交由关键词检索兜底）；② **模拟价格**——
        商品库只有 3000 件（手机数码）是真实价格，其余 4000 件由
        `scripts/fill_catalog_prices.py` 按类目分布生成，仅用于展示与排序参考。
        用模拟价格做硬过滤会把真实存在的相关商品误滤掉，因此这里只对真实价格生效。
        """
        if not results:
            return results
        filtered = []
        for r in results:
            if r.get("price_source") == "simulated":
                filtered.append(r)
                continue
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
                    # 价格来源：real（真实价格）/ simulated（由脚本按类目分布生成）
                    # 供价格过滤与软预算判断使用（见 scripts/fill_catalog_prices.py）
                    "price_source": meta.get("price_source", "unknown"),
                    "sales_count": meta.get("sales_count", ""),
                    "popularity_score": meta.get("popularity_score", ""),
                    "score": float(dist),
                    "source": "vector",
                    "rank": rank,
                }
            )
        return results

    @classmethod
    def _bm25_tokenize(cls, text: str) -> List[str]:
        """jieba 分词 + 过滤中文虚词（语料与查询必须走同一套分词口径）。

        整句都是虚词时回落到未过滤结果，避免查出空集。
        """
        import jieba

        tokens = [t for t in jieba.cut(text or "") if t.strip()]
        filtered = [t for t in tokens if t not in cls._BM25_STOPWORDS]
        return filtered or tokens

    def _ensure_bm25(self) -> bool:
        """懒构建 BM25 索引（jieba 分词 + BM25Okapi，语料=商品名+类目+品牌）。

        依赖缺失或构建失败时返回 False，由调用方回落 Neo4j 全文索引。
        """
        if self._bm25 is not None:
            return True
        if self._bm25_failed or not self._product_meta:
            return False
        try:
            import importlib.util

            if importlib.util.find_spec("jieba") is None:
                raise ImportError("jieba 依赖缺失")
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # 未安装 rank-bm25/jieba
            print(f"[SearchAgent] BM25 依赖缺失（{exc}），关键词检索回落 Neo4j 全文索引")
            self._bm25_failed = True
            return False
        docs = list(self._product_meta.values())
        corpus = [
            f"{d.get('name', '')} {d.get('category', '')} {d.get('brand', '')}" for d in docs
        ]
        self._bm25_docs = docs
        self._bm25 = BM25Okapi([self._bm25_tokenize(doc) for doc in corpus])
        print(f"[SearchAgent] BM25 关键词索引已构建: {len(docs)} 条商品")
        return True

    def _bm25_keyword_search(
        self,
        query: str,
        top_k: int,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """BM25Okapi 关键词检索（jieba 分词）。

        替代 Neo4j 全文索引作为默认关键词分支：16 用例人工标注 GT 下
        BM25 NDCG@5 0.8995 / P@5 0.8875，明显优于 Neo4j 全文（0.7672 / 0.7750）。
        价格约束按「未知价格视为不满足」处理，避免把无价格商品当作符合预算。
        """
        if not self._ensure_bm25():
            return self._neo4j_keyword_search(query, top_k, category, min_price, max_price)

        scores = self._bm25.get_scores(self._bm25_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: List[Dict[str, Any]] = []
        for idx in order:
            if scores[idx] <= 0:
                break
            meta = self._bm25_docs[idx]
            if category and category not in (meta.get("category") or ""):
                continue
            price = meta.get("price")
            if min_price is not None or max_price is not None:
                # 模拟价格不参与硬过滤（见 _filter_by_price 说明）
                if meta.get("price_source") != "simulated":
                    if price in (None, ""):
                        continue
                    try:
                        price_value = float(price)
                    except (TypeError, ValueError):
                        continue
                    if min_price is not None and price_value < min_price:
                        continue
                    if max_price is not None and price_value > max_price:
                        continue
            results.append(
                {
                    "id": str(meta.get("id", "")),
                    "name": meta.get("name", ""),
                    "category": meta.get("category", ""),
                    "brand": meta.get("brand", ""),
                    "price": price if price not in (None, "") else "",
                    "price_source": meta.get("price_source", "unknown"),
                    "score": float(scores[idx]),
                    "source": "bm25",
                }
            )
            if len(results) >= top_k:
                break
        return results

    def keyword_search(
        self,
        query: str,
        top_k: int,
        category: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """关键词检索统一入口：按配置选择 BM25（默认）或 Neo4j 全文索引。"""
        if (self.keyword_backend or "bm25").lower() == "neo4j":
            return self._neo4j_keyword_search(query, top_k, category, min_price, max_price)
        return self._bm25_keyword_search(query, top_k, category, min_price, max_price)

    def warmup(self) -> bool:
        """启动期预热：提前构建 BM25 索引（约 10s），避免首个查询承担冷启动。

        返回是否预热成功；失败不抛错（关键词检索会回落 Neo4j 全文索引）。
        """
        if (self.keyword_backend or "bm25").lower() == "neo4j":
            return False
        return self._ensure_bm25()

    # 查询类型 → RRF 权重（向量权重, 关键词权重）
    # 依据（16 用例人工标注 GT 实测）：属性/规格约束型查询里"关键词字面匹配"更强
    # （1.5匹空调 BM25 1.000 vs 向量 0.699、滚筒洗衣机 1.000 vs 0.723）；
    # 意图/语义型查询里向量更强（"打游戏用的手机" 向量 0.854 vs BM25 0.000）。
    _STRUCTURED_UNIT_RE = re.compile(
        r"\d+(?:\.\d+)?\s*(?:匹|元|块|kg|公斤|克|升|寸|英寸|GB|TB|W|瓦|mAh|毫安|小时|天|核)"
    )
    # 价格类约束：预算条件已由结构化解析 + 过滤处理，关键词只会把 "500"/"元"
    # 当词面匹配引入噪声（实测 BM25 在 "500元以下手机" 上 NDCG@5 = 0.000）
    _PRICE_QUERY_RE = re.compile(r"预算|\d+(?:\.\d+)?\s*(?:元|块)")
    _STRUCTURED_WORDS = (
        "滚筒", "对开门", "对开", "充电", "进口", "儿童", "二手", "便携",
        "折叠", "变频", "静音", "大容量", "迷你", "一级能效",
    )
    _INTENT_WORDS = (
        "打游戏", "游戏用", "送给", "适合", "用的", "戴的", "办公用", "学习用",
        "性价比", "好用", "便宜", "值得", "求推荐", "怎么选", "哪个好",
    )
    # BM25 停用词：**只过滤纯虚词**（封闭类功能词）。
    # 这类词在商品名里几乎不承载区分信息，却会因为出现在大量文档里而给无关商品加分
    # （"预算200以内的蓝牙耳机"里命中的"的"）。注意不要再往里加"推荐/买/用/好"这类
    # 词：它们在商品标题里是有信息的，实测把它们一起过滤会让推荐侧查询召回变差
    # （8 用例人工标注：query_aware NDCG@5 0.7656→0.7172）。
    # 依据：35 条人工标注查询上关键词单路 NDCG@5 0.8294→0.8384（留出集持平、
    # 4 折 0 负），hybrid 全部持平（0.9110）。
    _BM25_STOPWORDS = frozenset(
        "的 了 是 在 和 与 我 你 他 她 它 们 这 那 就 都 而 及 或 也 还 又 很 太 "
        "吗 呢 吧 啊 哦 嗯 之 其 且".split()
    )

    @classmethod
    def classify_query_type(cls, query: str) -> str:
        """按查询措辞粗分为四类：price（价格约束）/ structured（规格约束）/ intent / plain。"""
        text = (query or "").lower()
        if cls._PRICE_QUERY_RE.search(text):
            return "price"
        if cls._STRUCTURED_UNIT_RE.search(text) or any(
            word in text for word in cls._STRUCTURED_WORDS
        ):
            return "structured"
        if any(word in text for word in cls._INTENT_WORDS):
            return "intent"
        return "plain"

    def rrf_weights_for(self, query: str) -> tuple:
        """返回该查询的 RRF 两路权重（仅在自适应加权开启时生效）。

        自适应加权默认关闭（`RRF_ADAPTIVE_WEIGHTS=false`）：需先在 ≥30 个标注查询上用
        `tests/tune_rrf.py --adaptive` 做留出/交叉验证，确认稳定提升后再启用。
        """
        if not getattr(self, "adaptive_rrf_weights", False):
            return (1.0, 1.0)
        kind = self.classify_query_type(query)
        if kind == "price":
            # 价格约束偏向量：预算已由结构化过滤处理，关键词侧只有数字噪声
            return (1.3, 0.7)
        if kind == "structured":
            return (0.7, 1.3)
        if kind == "intent":
            return (1.3, 0.7)
        return (1.0, 1.0)

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
        keyword_results: List[Dict[str, Any]],
        k: int = 60,
        w_vec: float = 1.0,
        w_kw: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion 融合排序算法

        RRF score = w_vec/(k + rank_vec) + w_kw/(k + rank_kw)
        将向量检索和关键词检索的结果按排名融合

        w_vec / w_kw 用于按查询类型调节两路权重（见 `rrf_weights_for`），默认等权。
        """
        scores: Dict[str, float] = {}
        items: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(vec_results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0) + w_vec / (k + rank + 1)
            # 合并时保留原始相似度分数（用于展示，RRF 融合分仅用于排序）
            if item_id in items:
                items[item_id]["score"] = max(items[item_id].get("score", 0), item.get("score", 0))
            else:
                items[item_id] = item

        for rank, item in enumerate(keyword_results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0) + w_kw / (k + rank + 1)
            # 合并信息：如果已有向量结果，用关键词结果补充 category/name；
            # 关键词分数（Lucene/BM25）与余弦相似度量纲不同，单独存 kw_score，不再混入 score
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
