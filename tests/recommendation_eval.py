"""
推荐 Agent 离线评估框架

评估推荐系统的质量, 对比不同策略 (图结构推荐 vs 图+LLM重排)

评估维度:
1. 排序质量指标:
   - NDCG@K: 归一化折损累计增益 (考虑排序位置)
   - MAP@K: 平均精度均值
   - MRR: 平均倒数排名
   - Hit Rate@K: 命中率

2. 多样性指标:
   - Coverage: 推荐商品覆盖率 (推荐了多少不同商品)
   - Diversity: 推荐列表内部分类多样性
   - Novelty: 推荐新颖性 (推荐长尾商品的比例)

3. LLM 重排效果评估:
   - 重排前后 NDCG 变化 (Δ NDCG)
   - 重排延迟开销
   - 重排一致性 (多次调用结果稳定性)

4. 模拟点击/转化评估:
   - CTR@K (模拟点击率): 基于相关性分数模拟用户点击
   - CVR@K (模拟转化率): 基于用户画像匹配度模拟转化

使用方法:
    python tests/recommendation_eval.py
"""

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import List

# 必须在 import sentence-transformers/huggingface 相关模块前设置
# 否则 SentenceTransformer 加载时仍请求 huggingface.co 超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root

ensure_project_root()

from langchain_openai import ChatOpenAI

from config.settings import settings

# 固定随机种子 — 让模拟 CTR/CVR 可复现,避免每次评估结果漂移
random.seed(42)

# ------------------------------------------------------------------ #
#  评估数据结构
# ------------------------------------------------------------------ #


@dataclass
class RecommendationTestCase:
    """推荐评估测试用例"""

    user_id: str
    query: str
    user_profile: dict
    # Ground truth: 用户实际购买/浏览过的商品 ID
    purchased_ids: List[str] = field(default_factory=list)
    viewed_ids: List[str] = field(default_factory=list)
    preferred_categories: List[str] = field(default_factory=list)
    # 动态 ground truth 构建的关键词（当 purchased_ids/viewed_ids 为空时使用）
    relevant_keywords: List[str] = field(default_factory=list)
    # 标记是否使用动态 ground truth
    use_dynamic_gt: bool = True


@dataclass
class RecommendationResult:
    """推荐结果"""

    items: List[dict]  # [{product, id, score, reason, category, price}]
    strategy: str  # "graph_only" | "graph+llm_rerank" | "llm_direct"
    latency_ms: float = 0.0


# ------------------------------------------------------------------ #
#  排序质量指标
# ------------------------------------------------------------------ #


class RankingMetrics:
    """推荐排序质量指标"""

    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """DCG@K"""
        return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))

    @staticmethod
    def ndcg_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        NDCG@K: 归一化折损累计增益

        考虑推荐排序位置: 排在前面且相关 → 高分
        """
        if not relevant_ids:
            return 0.0

        # 实际相关性 (二值: 相关=1, 不相关=0)
        actual_rels = [1.0 if item_id in relevant_ids else 0.0 for item_id in recommended_ids[:k]]

        # 理想排序
        ideal_rels = [1.0] * min(len(relevant_ids), k)

        dcg = RankingMetrics.dcg_at_k(actual_rels, k)
        idcg = RankingMetrics.dcg_at_k(ideal_rels, k)

        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def average_precision(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        AP@K: 平均精度

        = (1/min(k, |relevant|)) * Σ Precision@i (for each relevant item in top-k)
        """
        if not relevant_ids:
            return 0.0

        hits = 0
        precision_sum = 0.0

        for i, item_id in enumerate(recommended_ids[:k]):
            if item_id in relevant_ids:
                hits += 1
                precision_sum += hits / (i + 1)

        return precision_sum / min(k, len(relevant_ids))

    @staticmethod
    def mrr(recommended_ids: List[str], relevant_ids: List[str]) -> float:
        """MRR: 平均倒数排名"""
        for i, item_id in enumerate(recommended_ids):
            if item_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def hit_rate_at_k(recommended_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """Hit Rate@K"""
        top_k = recommended_ids[:k]
        return 1.0 if set(top_k) & set(relevant_ids) else 0.0

    @classmethod
    def compute_all(cls, recommended_ids: List[str], relevant_ids: List[str]) -> dict:
        """计算所有排序指标"""
        return {
            "ndcg@5": cls.ndcg_at_k(recommended_ids, relevant_ids, 5),
            "ndcg@10": cls.ndcg_at_k(recommended_ids, relevant_ids, 10),
            "map@5": cls.average_precision(recommended_ids, relevant_ids, 5),
            "map@10": cls.average_precision(recommended_ids, relevant_ids, 10),
            "mrr": cls.mrr(recommended_ids, relevant_ids),
            "hit_rate@5": cls.hit_rate_at_k(recommended_ids, relevant_ids, 5),
            "hit_rate@10": cls.hit_rate_at_k(recommended_ids, relevant_ids, 10),
        }


# ------------------------------------------------------------------ #
#  多样性指标
# ------------------------------------------------------------------ #


class DiversityMetrics:
    """推荐多样性指标"""

    @staticmethod
    def coverage(all_recommended: List[List[str]], total_items: int) -> float:
        """
        Coverage: 推荐商品覆盖率

        = |∪ recommended_items| / |total_items|
        """
        unique = set()
        for recs in all_recommended:
            unique.update(recs)
        return len(unique) / total_items if total_items > 0 else 0.0

    @staticmethod
    def intra_list_diversity(items: List[dict]) -> float:
        """
        ILD: 列表内部多样性

        基于分类的多样性: 推荐列表中有多少不同分类
        """
        if not items:
            return 0.0
        categories = set()
        for item in items:
            cat = item.get("category", "")
            if cat:
                categories.add(cat)
        return len(categories) / len(items)

    @staticmethod
    def novelty(items: List[dict], popularity_map: dict) -> float:
        """
        Novelty: 推荐新颖性

        基于商品流行度: 推荐冷门商品的比例
        popularity_map: {item_id: popularity_score} (0-1, 1=最热门)
        """
        if not items:
            return 0.0
        novelty_scores = []
        for item in items:
            item_id = item.get("id", "")
            pop = popularity_map.get(item_id, 0.5)
            novelty_scores.append(1.0 - pop)  # 越冷门越新颖
        return sum(novelty_scores) / len(novelty_scores)


# ------------------------------------------------------------------ #
#  模拟点击/转化评估
# ------------------------------------------------------------------ #


class SimulationMetrics:
    """
    模拟用户行为评估推荐效果

    基于相关性分数和用户画像匹配度模拟用户点击和转化
    """

    @staticmethod
    def simulate_ctr(
        recommended: List[dict],
        relevant_ids: List[str],
        k: int = 5,
    ) -> float:
        """
        模拟 CTR@K

        规则:
        - 相关商品 → 70-90% 点击概率
        - 不相关商品 → 5-15% 点击概率
        """
        if not recommended:
            return 0.0

        clicks = 0
        for item in recommended[:k]:
            item_id = item.get("id", "")
            if item_id in relevant_ids:
                clicks += random.uniform(0.7, 0.9)
            else:
                clicks += random.uniform(0.05, 0.15)

        return clicks / min(k, len(recommended))

    @staticmethod
    def simulate_cvr(
        recommended: List[dict],
        user_profile: dict,
        k: int = 5,
    ) -> float:
        """
        模拟 CVR@K

        规则: 基于价格匹配度和分类匹配度
        """
        if not recommended:
            return 0.0

        conversions = 0
        budget = user_profile.get("预算") or user_profile.get("budget")
        preferred_cat = user_profile.get("偏好分类") or user_profile.get("category")

        for item in recommended[:k]:
            cvr_prob = 0.5  # 基础转化率

            # 分类匹配
            item_cat = item.get("category", "")
            if preferred_cat and item_cat and preferred_cat in item_cat:
                cvr_prob += 0.2

            # 价格匹配
            if budget:
                price = item.get("price", "")
                if price:
                    try:
                        price_float = float(str(price).replace("¥", "").strip())
                        if isinstance(budget, (int, float)) and price_float <= float(budget):
                            cvr_prob += 0.15
                    except (ValueError, TypeError):
                        pass

            # 推荐理由影响
            reason = item.get("reason", "")
            if reason and len(reason) > 10:
                cvr_prob += 0.05

            conversions += min(cvr_prob, 0.95)

        return conversions / min(k, len(recommended))


# ------------------------------------------------------------------ #
#  推荐评估器
# ------------------------------------------------------------------ #


class RecommendationEvaluator:
    """
    推荐 Agent 离线评估器

    对比策略:
    1. graph_only: 仅图结构推荐 (无 LLM 重排)
    2. graph+llm_rerank: 图结构推荐 + LLM 重排
    3. llm_direct: LLM 直接推荐 (无图查询)

    评估指标: NDCG, MAP, MRR, Hit Rate, Coverage, Diversity, 模拟 CTR/CVR
    """

    def __init__(self, recommend_agent=None, llm=None, neo4j_driver=None):
        self.agent = recommend_agent
        self.llm = llm
        self.neo4j_driver = neo4j_driver or (
            recommend_agent.neo4j_driver if recommend_agent else None
        )
        # Catalog 总数 — 用于 Coverage 指标分母(从商品数据集加载,失败降级为 0)
        self._catalog_size = self._load_catalog_size()
        # 商品全量(含 name/category), 用于构建与策略无关的中立 Ground Truth
        self._catalog = None
        # popularity_map — 用于 Novelty 指标; 评估时根据所有推荐结果出现频率动态构建
        # (被多个策略推荐的商品更热门, 推荐冷门商品比例越高 novelty 越高)
        self._popularity_map: dict = {}

    def _load_catalog_size(self) -> int:
        """加载商品总数(用于 Coverage 分母)"""
        try:
            path = os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    products = json.load(f)
                return len(products)
        except Exception:
            pass
        return 0

    def _build_popularity_map(self, all_recommended: List[List[str]]) -> dict:
        """
        根据所有推荐结果构建 popularity_map

        策略: 商品在所有策略所有用例的推荐列表中出现的频率(归一化到 0-1)
        - 被频繁推荐 → 高 popularity(热门)
        - 仅少数推荐 → 低 popularity(冷门)
        此代理不依赖外部数据,语义合理: novelty 衡量推荐冷门商品的比例
        """
        freq = {}
        total = 0
        for recs in all_recommended:
            for item_id in recs:
                freq[item_id] = freq.get(item_id, 0) + 1
                total += 1
        if not total:
            return {}
        return {k: v / total for k, v in freq.items()}

    def _load_catalog(self) -> list:
        """加载商品全量(懒加载, 供中立 GT 使用)"""
        if self._catalog is None:
            try:
                path = os.path.join(
                    PROJECT_ROOT, "data", "processed", "products_for_faiss.json"
                )
                with open(path, "r", encoding="utf-8") as f:
                    self._catalog = json.load(f)
            except Exception:
                self._catalog = []
        return self._catalog

    def _build_ground_truth(self, test_case: RecommendationTestCase) -> List[str]:
        """
        动态构建推荐评估的 Ground Truth — 中立口径, 与被评估策略无关

        规则:
        - 商品名或分类包含任一 relevant_keywords → 相关
        - 或分类命中 preferred_categories → 相关

        不再把"推荐结果中分类匹配的商品"标为相关(此前会把被评估策略
        自己推荐的商品写进 GT, 造成 NDCG/MAP 虚高到 1.0 的循环引用)。
        """
        # 优先使用显式标注的 ground truth
        explicit_gt = test_case.purchased_ids + test_case.viewed_ids
        if explicit_gt:
            return explicit_gt

        prefs = [p.lower() for p in (test_case.preferred_categories or [])]
        kws = [k.lower() for k in (test_case.relevant_keywords or [])]

        relevant_set = set()
        for p in self._load_catalog():
            text = f"{p.get('name', '')} {p.get('category', '')}".lower()
            if any(pref in text for pref in prefs) or any(kw in text for kw in kws):
                relevant_set.add(str(p.get("id", "")))

        relevant_set.discard("")
        return sorted(relevant_set)

    def evaluate_strategy(
        self,
        test_case: RecommendationTestCase,
        strategy: str = "graph+llm_rerank",
        top_k: int = 5,
    ) -> dict:
        """
        评估单条测试用例的指定策略

        Args:
            test_case: 测试用例
            strategy: "graph_only" | "graph+llm_rerank" | "llm_direct"
            top_k: 推荐数量
        """
        if not self.agent:
            return {"error": "recommend_agent 不可用"}

        start_time = time.time()

        recommended_items = []

        try:
            if strategy == "graph_only":
                # 仅图结构推荐
                candidates = self.agent._graph_based_recommend(
                    test_case.query, test_case.user_profile, top_k * 2
                )
                recommended_items = candidates[:top_k]

            elif strategy == "graph+llm_rerank":
                # 图结构 + LLM 重排 (默认策略)
                candidates = self.agent._graph_based_recommend(
                    test_case.query, test_case.user_profile, top_k * 2
                )
                if candidates:
                    recommended_items = self.agent._llm_rerank(
                        test_case.query, candidates, test_case.user_profile, top_k
                    )
                else:
                    # 图查询无结果, 降级
                    recommended_items = self.agent._get_popular_products(top_k)

            elif strategy == "llm_direct":
                # LLM 直接推荐
                result_str = self.agent._llm_direct_recommend(
                    test_case.query, test_case.user_profile, top_k
                )
                # LLM 直接推荐返回字符串, 无法提取结构化 ID
                recommended_items = [{"product": result_str, "id": "llm_direct"}]

            elif strategy == "popularity":
                # Baseline: 热门商品（无个性化、无模型调用）
                candidates = self.agent._get_popular_products(top_k * 2)
                recommended_items = candidates[:top_k]

            elif strategy == "item_cf_only":
                # Baseline: 仅 Item-CF 协同过滤（无图、无 LLM 重排）
                candidates = self.agent._item_cf_recommend(
                    test_case.query, test_case.user_profile, top_k * 2
                )
                recommended_items = candidates[:top_k]

            elif strategy == "graph+item_cf":
                # Baseline: 图协同 + Item-CF 融合（无 LLM 重排）
                graph_recs = self.agent._graph_based_recommend(
                    test_case.query, test_case.user_profile, top_k * 2
                )
                cf_recs = self.agent._item_cf_recommend(
                    test_case.query, test_case.user_profile, top_k * 2
                )
                seen_ids = set()
                merged = []
                for item in graph_recs + cf_recs:
                    item_id = str(item.get("id", ""))
                    if item_id and item_id not in seen_ids:
                        seen_ids.add(item_id)
                        merged.append(item)
                recommended_items = merged[:top_k]

            elif strategy == "graph+item_cf+llm_rerank":
                # 完整管线: 图协同 + Item-CF 融合 → LLM 重排
                graph_recs = self.agent._graph_based_recommend(
                    test_case.query, test_case.user_profile, top_k * 2
                )
                cf_recs = self.agent._item_cf_recommend(
                    test_case.query, test_case.user_profile, top_k * 2
                )
                seen_ids = set()
                merged = []
                for item in graph_recs + cf_recs:
                    item_id = str(item.get("id", ""))
                    if item_id and item_id not in seen_ids:
                        seen_ids.add(item_id)
                        merged.append(item)
                if merged:
                    recommended_items = self.agent._llm_rerank(
                        test_case.query, merged, test_case.user_profile, top_k
                    )
                else:
                    recommended_items = self.agent._get_popular_products(top_k)

        except Exception as e:
            return {"error": f"推荐失败: {e}"}

        latency_ms = (time.time() - start_time) * 1000

        # 提取推荐 ID
        recommended_ids = [str(item.get("id", "")) for item in recommended_items]

        # Ground truth: 优先使用显式标注，否则中立动态构建
        relevant_ids = test_case.purchased_ids + test_case.viewed_ids
        if relevant_ids:
            gt_source = "explicit"
        elif test_case.use_dynamic_gt:
            relevant_ids = self._build_ground_truth(test_case)
            gt_source = "dynamic"
        else:
            relevant_ids = []
            gt_source = "none"

        # 排序指标
        ranking_metrics = RankingMetrics.compute_all(recommended_ids, relevant_ids)

        # 多样性指标
        diversity = DiversityMetrics.intra_list_diversity(recommended_items)

        # 模拟 CTR/CVR
        ctr = SimulationMetrics.simulate_ctr(recommended_items, relevant_ids, top_k)
        cvr = SimulationMetrics.simulate_cvr(recommended_items, test_case.user_profile, top_k)

        # Novelty — 需 popularity_map, 单条用例先用空 map 兜底, 汇总时再算
        # (popularity 需聚合所有策略结果, 单条评估阶段还无法计算准确值)
        novelty_score = DiversityMetrics.novelty(recommended_items, self._popularity_map)

        return {
            "strategy": strategy,
            "query": test_case.query,
            "recommended_ids": recommended_ids,
            "relevant_ids": relevant_ids,
            "gt_source": gt_source,
            "gt_count": len(relevant_ids),
            "ranking_metrics": ranking_metrics,
            "diversity": round(diversity, 4),
            "novelty": round(novelty_score, 4),
            "simulated_ctr": round(ctr, 4),
            "simulated_cvr": round(cvr, 4),
            "latency_ms": round(latency_ms, 2),
            "num_candidates": len(recommended_items),
        }

    def compare_strategies(
        self,
        test_cases: List[RecommendationTestCase],
        strategies: List[str] = None,
    ) -> dict:
        """
        对比不同推荐策略

        Args:
            test_cases: 测试用例列表
            strategies: 要对比的策略列表

        Returns:
            对比报告
        """
        strategies = strategies or [
            "popularity",  # Baseline: 热门商品
            "graph_only",  # Baseline: 仅图协同
            "item_cf_only",  # Baseline: 仅 Item-CF
            "graph+item_cf",  # Baseline: 图 + CF 融合，无重排
            "graph+item_cf+llm_rerank",  # 完整管线（图 + CF + LLM 重排）
        ]
        all_results = {}

        for strategy in strategies:
            print(f"\n{'=' * 40}")
            print(f"  评估策略: {strategy}")
            print(f"{'=' * 40}")

            results = []
            for case in test_cases:
                result = self.evaluate_strategy(case, strategy)
                result["test_case"] = case.query
                results.append(result)

                if "error" not in result:
                    ndcg = result["ranking_metrics"].get("ndcg@5", 0)
                    print(f"  [{ndcg:.4f}] {case.query[:30]}... → {result['latency_ms']:.0f}ms")
                else:
                    print(f"  [ERROR] {case.query[:30]}... → {result['error']}")

            all_results[strategy] = results

        # 用所有策略的推荐结果聚合构建 popularity_map, 回填每条结果的 novelty
        # (单条评估阶段 popularity_map 为空, novelty 兜底为 0.5; 此处用真实频率重算)
        all_recs_for_pop = [
            r["recommended_ids"]
            for results in all_results.values()
            for r in results
            if "error" not in r and r.get("recommended_ids")
        ]
        self._popularity_map = self._build_popularity_map(all_recs_for_pop)
        if self._popularity_map:
            for results in all_results.values():
                for r in results:
                    if "error" in r:
                        continue
                    # 需 recommended_items(含 category) 才能算 novelty; 但 novelty 只用 id+popularity
                    # 这里重建最小 items 结构(只含 id) 调用 DiversityMetrics.novelty
                    items_for_novelty = [{"id": iid} for iid in r.get("recommended_ids", [])]
                    r["novelty"] = round(
                        DiversityMetrics.novelty(items_for_novelty, self._popularity_map), 4
                    )

        # 汇总对比
        comparison = self._compute_comparison(all_results)
        return {
            "strategies": all_results,
            "comparison": comparison,
        }

    def _compute_comparison(self, all_results: dict) -> dict:
        """计算策略对比汇总"""
        comparison = {}

        for strategy, results in all_results.items():
            valid = [r for r in results if "error" not in r]
            if not valid:
                comparison[strategy] = {"error": "所有用例失败"}
                continue

            # 排序指标平均
            ret_metrics = ["ndcg@5", "ndcg@10", "map@5", "mrr", "hit_rate@5"]
            avg_metrics = {}
            for metric in ret_metrics:
                values = [r["ranking_metrics"].get(metric, 0) for r in valid]
                avg_metrics[metric] = round(sum(values) / len(values), 4) if values else 0

            # 多样性平均
            diversities = [r.get("diversity", 0) for r in valid]
            avg_diversity = round(sum(diversities) / len(diversities), 4) if diversities else 0

            # Novelty 平均(单条已用聚合 popularity_map 重算)
            novelties = [r.get("novelty", 0) for r in valid]
            avg_novelty = round(sum(novelties) / len(novelties), 4) if novelties else 0

            # Coverage — 该策略所有用例推荐出去的 unique 商品数 / catalog 总数
            all_recommended_for_strategy = [
                r.get("recommended_ids", []) for r in valid
            ]
            coverage = DiversityMetrics.coverage(
                all_recommended_for_strategy, self._catalog_size
            )

            # CTR/CVR 平均
            ctrs = [r.get("simulated_ctr", 0) for r in valid]
            cvrs = [r.get("simulated_cvr", 0) for r in valid]
            avg_ctr = round(sum(ctrs) / len(ctrs), 4) if ctrs else 0
            avg_cvr = round(sum(cvrs) / len(cvrs), 4) if cvrs else 0

            # 延迟平均
            latencies = [r.get("latency_ms", 0) for r in valid]
            avg_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0

            # 延迟百分位 — 比 avg 更能反映真实用户体验
            def _percentile(vals, p):
                if not vals:
                    return 0.0
                s = sorted(vals)
                k = (len(s) - 1) * p / 100
                f = int(k)
                c = f + 1
                if c >= len(s):
                    return s[-1]
                return s[f] + (k - f) * (s[c] - s[f])

            p50_latency = round(_percentile(latencies, 50), 2)
            p90_latency = round(_percentile(latencies, 90), 2)
            p95_latency = round(_percentile(latencies, 95), 2)

            # Ground truth 统计
            gt_counts = [r.get("gt_count", 0) for r in valid]
            avg_gt = round(sum(gt_counts) / len(gt_counts), 1) if gt_counts else 0
            dynamic_gt_count = sum(1 for r in valid if r.get("gt_source") == "dynamic")

            comparison[strategy] = {
                "ranking": avg_metrics,
                "diversity": avg_diversity,
                "novelty": avg_novelty,
                "coverage": round(coverage, 4),
                "catalog_size": self._catalog_size,
                "simulated_ctr": avg_ctr,
                "simulated_cvr": avg_cvr,
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": p50_latency,
                "p90_latency_ms": p90_latency,
                "p95_latency_ms": p95_latency,
                "valid_cases": len(valid),
                "avg_gt_count": avg_gt,
                "dynamic_gt_cases": dynamic_gt_count,
            }

        return comparison

    def evaluate_rerank_impact(self, test_cases: List[RecommendationTestCase]) -> dict:
        """
        专门评估 LLM 重排的影响

        对比: graph_only vs graph+llm_rerank
        输出: Δ NDCG, 重排延迟开销, 重排稳定性
        """
        print("\n" + "=" * 60)
        print("  LLM 重排效果评估")
        print("=" * 60)

        results = self.compare_strategies(test_cases, ["graph_only", "graph+llm_rerank"])

        # 计算 Δ 指标
        comparison = results["comparison"]
        graph_only = comparison.get("graph_only", {})
        graph_rerank = comparison.get("graph+llm_rerank", {})

        if "ranking" in graph_only and "ranking" in graph_rerank:
            delta = {}
            for metric in graph_only["ranking"]:
                delta[f"Δ_{metric}"] = round(
                    graph_rerank["ranking"][metric] - graph_only["ranking"][metric], 4
                )

            delta["Δ_latency_ms"] = round(
                graph_rerank.get("avg_latency_ms", 0) - graph_only.get("avg_latency_ms", 0), 2
            )
            delta["Δ_diversity"] = round(
                graph_rerank.get("diversity", 0) - graph_only.get("diversity", 0), 4
            )

            results["rerank_impact"] = delta

            print("\n  重排影响:")
            for k, v in delta.items():
                sign = "+" if v > 0 else ""
                print(f"    {k:20s}: {sign}{v}")

        return results


# ------------------------------------------------------------------ #
#  测试用例
# ------------------------------------------------------------------ #


def generate_test_cases() -> List[RecommendationTestCase]:
    """生成推荐评估测试用例（含动态 ground truth 关键词）"""
    return [
        RecommendationTestCase(
            user_id="user_001",
            query="推荐一些适合户外的装备",
            user_profile={"偏好分类": "运动户外", "预算": "200-500"},
            purchased_ids=[],
            viewed_ids=[],
            preferred_categories=["运动户外"],
            relevant_keywords=["户外", "运动", "登山", "露营", "跑步"],
        ),
        RecommendationTestCase(
            user_id="user_002",
            query="预算200以内的母婴用品",
            user_profile={"偏好分类": "母婴用品", "预算": "200"},
            purchased_ids=[],
            viewed_ids=[],
            preferred_categories=["母婴用品", "母婴"],
            relevant_keywords=["婴儿", "奶粉", "纸尿裤", "母婴", "宝宝"],
        ),
        RecommendationTestCase(
            user_id="user_003",
            query="有什么好的手机推荐",
            user_profile={"偏好分类": "手机数码", "预算": "2000-5000"},
            purchased_ids=[],
            viewed_ids=[],
            preferred_categories=["手机数码", "手机"],
            relevant_keywords=["手机", "智能", "数码", "蓝牙", "充电"],
        ),
        RecommendationTestCase(
            user_id="user_004",
            query="推荐美妆护肤品",
            user_profile={"偏好分类": "美妆护肤", "预算": "100-300"},
            purchased_ids=[],
            viewed_ids=[],
            preferred_categories=["美妆护肤", "美妆"],
            relevant_keywords=["护肤", "面膜", "精华", "口红", "化妆品"],
        ),
        RecommendationTestCase(
            user_id="user_005",
            query="家用小电器推荐",
            user_profile={"偏好分类": "家用电器", "预算": "500-1000"},
            purchased_ids=[],
            viewed_ids=[],
            preferred_categories=["家用电器", "家电"],
            relevant_keywords=["电饭", "扫地", "豆浆", "电器", "家用"],
        ),
        RecommendationTestCase(
            user_id="user_006",
            query="帮我选个礼物",
            user_profile={"偏好分类": "礼品鲜花", "预算": "100-300"},
            purchased_ids=[],
            viewed_ids=[],
            preferred_categories=["礼品鲜花", "礼品"],
            relevant_keywords=["礼盒", "鲜花", "礼物", "巧克力", "礼"],
        ),
        RecommendationTestCase(
            user_id="user_007",
            query="食品零食推荐",
            user_profile={"偏好分类": "食品生鲜", "预算": "50-100"},
            purchased_ids=[],
            viewed_ids=[],
            preferred_categories=["食品生鲜", "食品"],
            relevant_keywords=["零食", "坚果", "饼干", "食品", "特产"],
        ),
        RecommendationTestCase(
            user_id="user_008",
            query="宠物用品推荐",
            user_profile={"偏好分类": "宠物用品", "预算": "100-200"},
            purchased_ids=[],
            viewed_ids=[],
            preferred_categories=["宠物用品", "宠物"],
            relevant_keywords=["猫粮", "狗粮", "宠物", "猫砂", "牵引"],
        ),
    ]


# ------------------------------------------------------------------ #
#  主入口
# ------------------------------------------------------------------ #


def main():
    """运行推荐评估"""
    print("=" * 60)
    print("  推荐 Agent 离线评估系统")
    print("=" * 60)

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.1,
        max_tokens=1024,
    )

    # 初始化推荐 Agent
    from orchestration.registry import create_neo4j_driver, register_all_agents

    neo4j_driver = None
    try:
        neo4j_driver = create_neo4j_driver()
    except Exception:
        pass

    agents = register_all_agents(neo4j_driver=neo4j_driver, llm=llm)
    recommend_agent = agents.get("catalog_agent").recommend_agent

    if not recommend_agent:
        print("❌ 推荐 Agent 不可用")
        return

    evaluator = RecommendationEvaluator(
        recommend_agent=recommend_agent, llm=llm, neo4j_driver=neo4j_driver
    )

    test_cases = generate_test_cases()
    print(f"\n共 {len(test_cases)} 个测试用例\n")

    # 评估 LLM 重排效果（graph_only vs graph+llm_rerank 的增量）
    result = evaluator.evaluate_rerank_impact(test_cases)

    # 全策略 baseline 对比
    baseline = evaluator.compare_strategies(test_cases)
    result["baseline_comparison"] = baseline.get("comparison", {})

    # 打印对比汇总
    print(f"\n{'=' * 60}")
    print("  策略对比汇总")
    print(f"{'=' * 60}")

    comparison = baseline.get("comparison", {})
    for strategy, metrics in comparison.items():
        print(f"\n  [{strategy}]")
        if "ranking" in metrics:
            for metric, value in metrics["ranking"].items():
                print(f"    {metric:20s}: {value:.4f}")
        print(f"    {'diversity':20s}: {metrics.get('diversity', 0):.4f}")
        print(f"    {'novelty':20s}: {metrics.get('novelty', 0):.4f}")
        print(f"    {'coverage':20s}: {metrics.get('coverage', 0):.4f}  (catalog={metrics.get('catalog_size', 0)})")
        print(f"    {'simulated_ctr':20s}: {metrics.get('simulated_ctr', 0):.4f}")
        print(f"    {'simulated_cvr':20s}: {metrics.get('simulated_cvr', 0):.4f}")
        print(f"    {'avg_latency_ms':20s}: {metrics.get('avg_latency_ms', 0):.2f}")
        print(f"    {'p50_latency_ms':20s}: {metrics.get('p50_latency_ms', 0):.2f}")
        print(f"    {'p90_latency_ms':20s}: {metrics.get('p90_latency_ms', 0):.2f}")
        print(f"    {'p95_latency_ms':20s}: {metrics.get('p95_latency_ms', 0):.2f}")

    print(f"\n{'=' * 60}\n")

    # 保存报告
    output_path = os.path.join(PROJECT_ROOT, "tests", "recommendation_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
