"""
RAG 评估体系 — 检索质量 + 生成质量全维度评估

评估维度:
1. 检索指标 (Retrieval Metrics):
   - Recall@K: 召回率 (Top-K 中包含正确文档的比例)
   - Precision@K: 精确率 (Top-K 中正确文档的比例)
   - MRR (Mean Reciprocal Rank): 平均倒数排名
   - NDCG@K (Normalized Discounted Cumulative Gain): 归一化折损累计增益
   - Hit Rate@K: 命中率 (至少有一个正确文档的比例)

2. 生成指标 (Generation Metrics):
   - Faithfulness: 答案忠实度 (答案是否基于检索到的 context)
   - Answer Relevance: 答案相关性 (答案是否回答了问题)
   - Context Precision: 上下文精确率 (检索到的 context 是否与问题相关)
   - Context Recall: 上下文召回率 (是否检索到了所有需要的 context)

3. 端到端指标:
   - Latency: 端到端延迟
   - Token Cost: Token 消耗

使用方法:
    python tests/rag_evaluation.py

    # 或在代码中调用
    from tests.rag_evaluation import RAGEvaluator
    evaluator = RAGEvaluator(llm, search_agent, cs_agent)
    results = evaluator.evaluate(test_cases)
"""

import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List

# 必须在 import sentence-transformers/huggingface 相关模块前设置
# 否则 SentenceTransformer 加载时仍请求 huggingface.co 超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import TokenRecordingLLM, ensure_project_root, require_assets

ensure_project_root()

from langchain_openai import ChatOpenAI

from config.settings import settings
from orchestration.observability import obs


def _extract_token_usage(result, model_name: str) -> dict:
    """
    从 LangChain AIMessage 提取 token 用量

    兼容两种格式:
    - usage_metadata (LangChain 0.2+): {input_tokens, output_tokens, total_tokens}
    - response_metadata.token_usage (OpenAI 原生): {prompt_tokens, completion_tokens, total_tokens}
    """
    prompt_tokens = completion_tokens = 0

    # 1. usage_metadata (LangChain 0.2+ 标准)
    um = getattr(result, "usage_metadata", None)
    if isinstance(um, dict):
        prompt_tokens = int(um.get("input_tokens", 0))
        completion_tokens = int(um.get("output_tokens", 0))

    # 2. response_metadata.token_usage (OpenAI 原生)
    if prompt_tokens == 0:
        rm = getattr(result, "response_metadata", None) or {}
        tu = rm.get("token_usage") or rm.get("tokenUsage") or {}
        if isinstance(tu, dict):
            prompt_tokens = int(tu.get("prompt_tokens", tu.get("input_tokens", 0)))
            completion_tokens = int(tu.get("completion_tokens", tu.get("output_tokens", 0)))

    return {"model": model_name, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

# ------------------------------------------------------------------ #
#  评估数据结构
# ------------------------------------------------------------------ #


@dataclass
class RAGTestCase:
    """RAG 评估测试用例"""

    query: str  # 用户查询
    relevant_doc_ids: List[str]  # 相关文档 ID (ground truth)
    expected_answer_keywords: List[str]  # 期望答案中包含的关键词
    category: str = "general"  # 分类 (search/cs/kg_qa)
    description: str = ""  # 用例描述


@dataclass
class RetrievalResult:
    """单条检索结果"""

    doc_id: str
    score: float
    content: str = ""


@dataclass
class EvaluationMetrics:
    """评估指标"""

    # 检索指标
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    hit_rate_at_5: float = 0.0

    # 生成指标
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0

    # 性能指标
    latency_ms: float = 0.0
    token_cost: int = 0


# ------------------------------------------------------------------ #
#  检索指标计算
# ------------------------------------------------------------------ #


class RetrievalMetrics:
    """检索质量指标计算器"""

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Recall@K: Top-K 检索结果中包含多少比例的相关文档

        = |relevant ∩ retrieved_top_k| / |relevant|
        """
        if not relevant_ids:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & set(relevant_ids))
        return hits / len(relevant_ids)

    @staticmethod
    def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Precision@K: Top-K 检索结果中有多少比例是相关的

        = |relevant ∩ retrieved_top_k| / k
        """
        if k == 0:
            return 0.0
        top_k = retrieved_ids[:k]
        hits = len(set(top_k) & set(relevant_ids))
        return hits / k

    @staticmethod
    def reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
        """
        RR: 第一个相关文档的倒数排名

        如果第一个相关文档排第 3 位, RR = 1/3
        如果没有相关文档, RR = 0
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        NDCG@K: 归一化折损累计增益

        DCG = Σ (2^rel_i - 1) / log2(i + 2)  for i in top_k
        IDCG = DCG of ideal ranking
        NDCG = DCG / IDCG
        """

        def dcg(rels: List[int]) -> float:
            return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))

        # 实际相关性
        actual_rels = [1 if doc_id in relevant_ids else 0 for doc_id in retrieved_ids[:k]]

        # 理想相关性 (所有相关文档排在前面)
        ideal_rels = [1] * min(len(relevant_ids), k)
        ideal_rels += [0] * (k - len(ideal_rels))

        dcg_value = dcg(actual_rels)
        idcg_value = dcg(ideal_rels)

        if idcg_value == 0:
            return 0.0
        return dcg_value / idcg_value

    @staticmethod
    def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
        """
        Hit Rate@K: Top-K 中是否至少包含一个相关文档

        = 1 if |relevant ∩ retrieved_top_k| > 0 else 0
        """
        top_k = retrieved_ids[:k]
        return 1.0 if set(top_k) & set(relevant_ids) else 0.0

    @classmethod
    def compute_all(cls, retrieved_ids: List[str], relevant_ids: List[str]) -> dict:
        """计算所有检索指标"""
        return {
            "recall@1": cls.recall_at_k(retrieved_ids, relevant_ids, 1),
            "recall@3": cls.recall_at_k(retrieved_ids, relevant_ids, 3),
            "recall@5": cls.recall_at_k(retrieved_ids, relevant_ids, 5),
            "precision@5": cls.precision_at_k(retrieved_ids, relevant_ids, 5),
            "mrr": cls.reciprocal_rank(retrieved_ids, relevant_ids),
            "ndcg@5": cls.ndcg_at_k(retrieved_ids, relevant_ids, 5),
            "hit_rate@5": cls.hit_rate_at_k(retrieved_ids, relevant_ids, 5),
        }


# ------------------------------------------------------------------ #
#  生成质量评估 (LLM as Judge)
# ------------------------------------------------------------------ #


class GenerationMetrics:
    """
    生成质量评估器 — 使用 LLM 评估答案质量

    评估维度:
    - Faithfulness: 答案是否完全基于检索到的 context (无幻觉)
    - Answer Relevance: 答案是否直接回答了用户问题
    - Context Precision: 检索到的 context 是否与问题相关
    - Context Recall: 是否检索到了回答问题所需的全部信息
    """

    EVAL_PROMPT = """你是一个 RAG 系统评估专家。请评估以下 RAG 系统的回答质量。

用户问题: {query}
检索到的 Context:
{context}

系统回答: {answer}

请从以下 4 个维度评分 (0-1 分, 保留 2 位小数):

1. **faithfulness** (忠实度): 回答是否完全基于 context, 没有编造信息?
   - 1.0: 完全基于 context, 无任何编造
   - 0.5: 部分基于 context, 有少量编造
   - 0.0: 大量编造, 与 context 无关

2. **answer_relevance** (答案相关性): 回答是否直接回答了用户问题?
   - 1.0: 完全回答了问题
   - 0.5: 部分回答
   - 0.0: 完全没有回答

3. **context_precision** (上下文精确率): 检索到的 context 是否与问题相关?
   - 1.0: 所有 context 都高度相关
   - 0.5: 部分相关
   - 0.0: 完全不相关

4. **context_recall** (上下文召回率): context 是否包含了回答问题所需的全部信息?
   - 1.0: 完全包含
   - 0.5: 部分包含
   - 0.0: 缺失关键信息

请只回复 JSON 格式:
{{"faithfulness": 0.00, "answer_relevance": 0.00, "context_precision": 0.00, "context_recall": 0.00}}"""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    @staticmethod
    def _parse_json_response(result: str) -> dict:
        """
        健壮 JSON 解析 — 兼容 LLM 常见的非标准输出:
        1. ```json ... ``` 代码块包裹
        2. JSON 前后夹杂解释性文字
        3. 尾随逗号 / 单引号等宽松格式
        """
        # 1. 去除 markdown 代码块包裹
        if result.startswith("```"):
            result = result.split("\n", 1)[-1]
            result = result.rsplit("```", 1)[0].strip()

        # 2. 尝试直接解析
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # 3. 提取最外层 {...} 块再解析
        start, end = result.find("{"), result.rfind("}")
        if start != -1 and end > start:
            candidate = result[start : end + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

        # 4. 宽松模式: 修复尾随逗号 / 单引号
        try:
            import re

            fixed = re.sub(r",\s*([}\]])", r"\1", candidate if "candidate" in dir() else result)
            fixed = fixed.replace("'", '"')
            parsed = json.loads(fixed)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        raise ValueError(f"无法解析 LLM 返回为 JSON: {result[:200]!r}")

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
    ) -> dict:
        """评估单条回答"""
        prompt = self.EVAL_PROMPT.format(
            query=query,
            context=context[:2000],  # 限制 context 长度
            answer=answer[:1000],
        )

        try:
            msg = self.llm.invoke(prompt)
            # 记录 LLM Judge 的 token 用量到全局 obs
            try:
                model_name = getattr(self.llm, "model_name", "") or getattr(self.llm, "model", "") or "unknown"
                usage = _extract_token_usage(msg, model_name)
                if usage["prompt_tokens"] or usage["completion_tokens"]:
                    obs.record_tokens(
                        model=usage["model"],
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                    )
            except Exception:
                # token 提取失败不影响评估主流程
                pass
            result = msg.content.strip()
            parsed = self._parse_json_response(result)
            return parsed
        except Exception as e:
            # 失败必须可见 — 不再静默返回全 0, 显式标记 error 便于排查
            print(f"[GenerationMetrics] 评估失败: {type(e).__name__}: {e}")
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0,
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }


# ------------------------------------------------------------------ #
#  RAG 评估器
# ------------------------------------------------------------------ #


class RAGEvaluator:
    """
    RAG 系统全维度评估器

    整合检索指标 + 生成指标, 对 RAG 系统进行端到端评估
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        search_agent=None,
        cs_agent=None,
        kg_qa_agent=None,
    ):
        self.llm = llm
        self.search_agent = search_agent
        self.cs_agent = cs_agent
        self.kg_qa_agent = kg_qa_agent
        self.gen_metrics = GenerationMetrics(llm)
        # 评估开始前重置 token 统计,确保本次评估的用量独立计数
        obs.tokens.reset()

    def _snapshot_token_usage(self) -> dict:
        """获取当前 LLM Judge 累计 token 用量快照"""
        try:
            return obs.get_usage_summary()
        except Exception:
            return {}

    @staticmethod
    def _token_delta(before: dict, after: dict) -> dict:
        """计算两次快照间的 token 增量(单条用例的消耗)"""
        try:
            b_total = before.get("total_tokens", 0)
            a_total = after.get("total_tokens", 0)
            b_cost = before.get("total_cost_usd", 0.0)
            a_cost = after.get("total_cost_usd", 0.0)
            return {
                "prompt_tokens": a_total - b_total,
                "completion_tokens": 0,  # 汇总级只有 total, 不拆分
                "total_tokens": a_total - b_total,
                "cost_usd": round(a_cost - b_cost, 6),
            }
        except Exception:
            return {"total_tokens": 0, "cost_usd": 0.0}

    def evaluate_search(
        self,
        query: str,
        relevant_ids: List[str],
        top_k: int = 10,
        relevant_keywords: List[str] = None,
    ) -> dict:
        """
        评估商品搜索 RAG

        Args:
            query: 搜索查询
            relevant_ids: 相关文档 ID 列表。如果为 ["auto"], 则动态构建 ground truth
            top_k: 返回数量
            relevant_keywords: 用于动态构建 ground truth 的关键词列表
        """
        if not self.search_agent:
            return {"error": "search_agent 不可用"}

        start_time = time.time()

        # 执行搜索并获取原始结果 (含 doc_id)
        retrieved_ids = []
        context_str = ""

        try:
            # 调用内部搜索方法获取结构化结果
            vec_results = self.search_agent._vector_search(query, top_k)
            cypher_results = self.search_agent._neo4j_keyword_search(query, top_k, None, None, None)
            merged = self.search_agent._reciprocal_rank_fusion(vec_results, cypher_results)

            retrieved_ids = [r["id"] for r in merged[:top_k]]
            context_str = "\n".join(
                f"[{r.get('name', '未知')}] id={r['id']} score={r.get('fusion_score', 0):.4f}"
                for r in merged[:top_k]
            )
        except Exception as e:
            return {"error": f"搜索失败: {e}"}

        retrieval_latency_ms = (time.time() - start_time) * 1000

        # 动态构建 ground truth (当 relevant_ids 为 ["auto"] 时)
        if relevant_ids == ["auto"]:
            relevant_ids = self._build_ground_truth(merged, vec_results, relevant_keywords or [])

        # 计算检索指标
        retrieval_metrics = RetrievalMetrics.compute_all(retrieved_ids, relevant_ids)

        # 格式化搜索结果作为回答
        gen_answer_start = time.time()
        answer = self.search_agent._format_results(merged[:top_k]) if merged else "未找到结果"
        gen_latency_ms = (time.time() - gen_answer_start) * 1000

        # 生成指标 (LLM Judge 单独计时, 不污染 gen_latency_ms)
        judge_start = time.time()
        token_before = self._snapshot_token_usage()
        gen_metrics = self.gen_metrics.evaluate(query, answer, context_str)
        judge_latency_ms = (time.time() - judge_start) * 1000
        token_after = self._snapshot_token_usage()
        token_cost = self._token_delta(token_before, token_after)

        return {
            "query": query,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "retrieval_metrics": retrieval_metrics,
            "generation_metrics": gen_metrics,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "gen_latency_ms": round(gen_latency_ms, 2),
            "judge_latency_ms": round(judge_latency_ms, 2),
            "latency_ms": round(retrieval_latency_ms + gen_latency_ms, 2),
            "token_cost": token_cost,
            "answer": answer[:200],
        }

    def _build_ground_truth(
        self,
        merged_results: list,
        vec_results: list,
        relevant_keywords: list,
    ) -> List[str]:
        """
        动态构建 ground truth

        策略 (pseudo relevance labeling):
        1. 向量检索 Top-3 作为高置信相关文档 (向量相似度高 = 语义相关)
        2. 在合并结果中, 商品名包含任一 relevant_keyword 的也标记为相关
        3. 去重后返回

        这种方法虽不如人工标注精确, 但能为评估提供有意义的 baseline。
        生产环境应替换为人工标注或用户点击数据。
        """
        relevant_set = set()

        # 向量检索 Top-3 作为高置信相关
        for item in vec_results[:3]:
            relevant_set.add(str(item["id"]))

        # 关键词匹配补充
        if relevant_keywords:
            for item in merged_results:
                name = item.get("name", "").lower()
                if any(kw.lower() in name for kw in relevant_keywords):
                    relevant_set.add(str(item["id"]))

        return list(relevant_set)

    def evaluate_cs(
        self,
        query: str,
        relevant_keywords: List[str],
    ) -> dict:
        """评估客服 FAQ RAG"""
        if not self.cs_agent:
            return {"error": "cs_agent 不可用"}

        start_time = time.time()

        # 执行检索
        self.cs_agent._ensure_loaded()
        docs = self.cs_agent._retrieve(query, k=3)
        context_str = "\n".join(d.page_content for d in docs) if docs else ""
        retrieval_latency_ms = (time.time() - start_time) * 1000

        # 生成回答 (真实 LLM 生成, 单独计时)
        gen_start = time.time()
        answer = self.cs_agent.policy_qa(query)
        gen_latency_ms = (time.time() - gen_start) * 1000

        # 生成指标 (LLM Judge 单独计时, 不污染 gen_latency_ms)
        judge_start = time.time()
        token_before = self._snapshot_token_usage()
        gen_metrics = self.gen_metrics.evaluate(query, answer, context_str)
        judge_latency_ms = (time.time() - judge_start) * 1000
        token_after = self._snapshot_token_usage()
        token_cost = self._token_delta(token_before, token_after)

        # 关键词命中率 (简化版 recall)
        answer_lower = answer.lower()
        keyword_hits = sum(1 for kw in relevant_keywords if kw.lower() in answer_lower)
        keyword_hit_rate = keyword_hits / len(relevant_keywords) if relevant_keywords else 0

        return {
            "query": query,
            "retrieved_docs": len(docs),
            "keyword_hit_rate": round(keyword_hit_rate, 2),
            "generation_metrics": gen_metrics,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "gen_latency_ms": round(gen_latency_ms, 2),
            "judge_latency_ms": round(judge_latency_ms, 2),
            "latency_ms": round(retrieval_latency_ms + gen_latency_ms, 2),
            "token_cost": token_cost,
            "answer": answer[:200],
        }

    def run_full_evaluation(self, test_cases: List[dict]) -> dict:
        """
        运行完整评估

        Args:
            test_cases: [{"type": "search"|"cs", "query": "...", "relevant_ids": [...], ...}]

        Returns:
            汇总评估报告
        """
        results = []

        for case in test_cases:
            eval_type = case.get("type", "search")

            if eval_type == "search":
                result = self.evaluate_search(
                    query=case["query"],
                    relevant_ids=case.get("relevant_ids", []),
                    relevant_keywords=case.get("relevant_keywords", []),
                )
            elif eval_type == "cs":
                result = self.evaluate_cs(
                    query=case["query"],
                    relevant_keywords=case.get("relevant_keywords", []),
                )
            else:
                continue

            result["type"] = eval_type
            result["description"] = case.get("description", "")
            results.append(result)

            print(
                f"  [{eval_type}] {case['query'][:30]}... → "
                f"latency={result.get('latency_ms', 0):.0f}ms"
            )

        # 汇总
        summary = self._compute_summary(results)
        return {
            "total_cases": len(results),
            "results": results,
            "summary": summary,
        }

    def evaluate_retrieval_strategies(self, test_cases: List[dict]) -> dict:
        """
        检索策略 baseline 对比（同一 pseudo-label GT 下评估）

        策略:
        - keyword_only: 仅 Neo4j 关键词检索
        - vector_only: 仅 FAISS 向量检索
        - hybrid: 向量 + 关键词 RRF 融合（完整管线）
        """
        strategies = ["keyword_only", "vector_only", "hybrid"]
        per_strategy = {s: [] for s in strategies}
        case_count = 0

        for case in test_cases:
            if case.get("type") != "search":
                continue
            case_count += 1
            query = case["query"]
            try:
                vec = self.search_agent._vector_search(query, 10)
                cypher = self.search_agent._neo4j_keyword_search(query, 10, None, None, None)
                merged = self.search_agent._reciprocal_rank_fusion(vec, cypher)
                gt = self._build_ground_truth(merged, vec, case.get("relevant_keywords", []))
            except Exception as e:
                print(f"  [Baseline] 检索失败 {query[:20]}: {e}")
                continue

            for strategy in strategies:
                if strategy == "vector_only":
                    ids = [r["id"] for r in vec[:10]]
                elif strategy == "keyword_only":
                    ids = [r["id"] for r in cypher[:10]]
                else:
                    ids = [r["id"] for r in merged[:10]]
                per_strategy[strategy].append(RetrievalMetrics.compute_all(ids, gt))

        summary = {}
        metric_names = [
            "recall@1",
            "recall@3",
            "recall@5",
            "precision@5",
            "mrr",
            "ndcg@5",
            "hit_rate@5",
        ]
        for strategy, results in per_strategy.items():
            summary[strategy] = {
                m: round(sum(r.get(m, 0) for r in results) / len(results), 4) if results else 0
                for m in metric_names
            }

        return {
            "cases": case_count,
            "summary": summary,
        }

    def _compute_summary(self, results: list) -> dict:
        """计算汇总指标"""
        if not results:
            return {}

        # 检索指标汇总
        search_results = [
            r for r in results if r.get("type") == "search" and "retrieval_metrics" in r
        ]

        summary = {}

        if search_results:
            ret_metrics = [
                "recall@1",
                "recall@3",
                "recall@5",
                "precision@5",
                "mrr",
                "ndcg@5",
                "hit_rate@5",
            ]
            summary["retrieval"] = {}
            for metric in ret_metrics:
                values = [r["retrieval_metrics"].get(metric, 0) for r in search_results]
                summary["retrieval"][metric] = round(sum(values) / len(values), 4) if values else 0

        # 生成指标汇总
        gen_metrics = ["faithfulness", "answer_relevance", "context_precision", "context_recall"]
        summary["generation"] = {}
        gen_error_count = sum(1 for r in results if r.get("generation_metrics", {}).get("error"))
        for metric in gen_metrics:
            values = [r.get("generation_metrics", {}).get(metric, 0) for r in results]
            summary["generation"][metric] = round(sum(values) / len(values), 4) if values else 0
        # 显式标注生成指标有效性 — 有 error 时不可信
        summary["generation"]["valid"] = gen_error_count == 0
        summary["generation"]["error_count"] = gen_error_count
        if gen_error_count:
            summary["generation"]["first_error"] = next(
                (
                    r.get("generation_metrics", {}).get("error", "")
                    for r in results
                    if r.get("generation_metrics", {}).get("error")
                ),
                "",
            )

        # 性能指标
        latencies = [r.get("latency_ms", 0) for r in results]
        ret_latencies = [
            r.get("retrieval_latency_ms", 0) for r in results if r.get("retrieval_latency_ms")
        ]
        gen_latencies = [r.get("gen_latency_ms", 0) for r in results if r.get("gen_latency_ms")]
        judge_latencies = [
            r.get("judge_latency_ms", 0) for r in results if r.get("judge_latency_ms")
        ]

        def _avg(vals):
            return round(sum(vals) / len(vals), 2) if vals else 0

        def _percentile(vals, p):
            """百分位计算 — p=50/90/95"""
            if not vals:
                return 0.0
            s = sorted(vals)
            k = (len(s) - 1) * p / 100
            f = int(k)
            c = f + 1
            if c >= len(s):
                return s[-1]
            return s[f] + (k - f) * (s[c] - s[f])

        summary["performance"] = {
            "avg_latency_ms": _avg(latencies),
            "max_latency_ms": max(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            # 延迟百分位 — 比 avg 更能反映真实用户体验
            "p50_latency_ms": round(_percentile(latencies, 50), 2),
            "p90_latency_ms": round(_percentile(latencies, 90), 2),
            "p95_latency_ms": round(_percentile(latencies, 95), 2),
            # 拆分统计 — 检索耗时才是系统真实性能
            "avg_retrieval_latency_ms": _avg(ret_latencies),
            "avg_gen_latency_ms": _avg(gen_latencies),
            "avg_judge_latency_ms": _avg(judge_latencies),
            # Judge 的百分位(评估开销可见, 便于优化 LLM Judge 成本)
            "p50_judge_latency_ms": round(_percentile(judge_latencies, 50), 2),
            "p90_judge_latency_ms": round(_percentile(judge_latencies, 90), 2),
        }

        # Token 用量汇总 — 以全局快照为准(生产链路 + Judge),
        # 与 by_model 保持一致(此前 total 只统计 Judge 的 case 增量, 链路调用被漏计)
        token_snapshot = self._snapshot_token_usage()
        token_costs = [r.get("token_cost", {}) for r in results if r.get("token_cost")]
        total_tokens = token_snapshot.get("total_tokens", 0)
        summary["token_usage"] = {
            "total_tokens": total_tokens,
            "total_cost_usd": round(token_snapshot.get("total_cost_usd", 0.0), 6),
            "total_calls": token_snapshot.get("total_calls", 0),
            "avg_tokens_per_case": (
                round(total_tokens / len(token_costs), 2) if token_costs else 0
            ),
            "by_model": token_snapshot.get("by_model", {}),
        }

        return summary


# ------------------------------------------------------------------ #
#  测试用例集 (带 ground truth)
# ------------------------------------------------------------------ #

SEARCH_TEST_CASES = [
    {
        "type": "search",
        "query": "蓝牙耳机",
        # Ground truth: 人工标注 — 查询"蓝牙耳机"时, 包含"蓝牙"且品类为"手机数码"的商品为相关
        # 使用关键词锚定法: 向量检索 Top-3 作为高置信相关, 关键词包含"蓝牙"的作为相关
        "relevant_ids": ["auto"],  # "auto" = 运行时从向量检索 Top-3 + 关键词匹配动态构建
        "relevant_keywords": ["蓝牙", "耳机", "无线"],
        "description": "搜索蓝牙耳机",
    },
    {
        "type": "search",
        "query": "500元以下手机",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["手机", "智能手机"],
        "description": "价格过滤搜索",
    },
    {
        "type": "search",
        "query": "婴儿纸尿裤",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["纸尿裤", "婴儿", "宝宝"],
        "description": "母婴商品搜索",
    },
    {
        "type": "search",
        "query": "运动鞋",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["运动鞋", "跑鞋", "篮球鞋"],
        "description": "服装鞋包搜索",
    },
    {
        "type": "search",
        "query": "面膜护肤",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["面膜", "护肤", "精华"],
        "description": "美妆护肤搜索",
    },
    {
        "type": "search",
        "query": "笔记本电脑",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["笔记本", "电脑", "laptop"],
        "description": "手机数码搜索",
    },
    {
        "type": "search",
        "query": "食品零食",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["零食", "食品", "坚果"],
        "description": "食品生鲜搜索",
    },
    {
        "type": "search",
        "query": "宠物狗粮",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["狗粮", "宠物", "猫粮"],
        "description": "宠物用品搜索",
    },
    {
        "type": "search",
        "query": "家用吸尘器",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["吸尘器", "除尘", "家用"],
        "description": "家用电器搜索",
    },
    {
        "type": "search",
        "query": "户外帐篷",
        "relevant_ids": ["auto"],
        "relevant_keywords": ["帐篷", "户外", "露营"],
        "description": "运动户外搜索",
    },
]

CS_TEST_CASES = [
    {
        "type": "cs",
        "query": "怎么退货？",
        "relevant_keywords": ["7天", "退换货", "申请", "退款"],
        "description": "退货政策",
    },
    {
        "type": "cs",
        "query": "退款多久到账？",
        "relevant_keywords": ["3-7", "工作日", "原路", "退款"],
        "description": "退款时效",
    },
    {
        "type": "cs",
        "query": "运费谁出？",
        "relevant_keywords": ["运费", "买家", "卖家", "质量"],
        "description": "运费责任",
    },
    {
        "type": "cs",
        "query": "发货时间",
        "relevant_keywords": ["48小时", "发货", "预售"],
        "description": "发货时效",
    },
    {
        "type": "cs",
        "query": "怎么开发票？",
        "relevant_keywords": ["发票", "电子", "下单"],
        "description": "发票政策",
    },
]


# ------------------------------------------------------------------ #
#  主入口
# ------------------------------------------------------------------ #


def main():
    """运行 RAG 评估"""
    print("=" * 60)
    print("  RAG 评估系统 — 检索质量 + 生成质量")
    print("=" * 60)

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.1,
        max_tokens=1024,
    )

    # 初始化 Agent
    from orchestration.registry import create_neo4j_driver, register_all_agents

    neo4j_driver = None
    try:
        neo4j_driver = create_neo4j_driver()
    except Exception:
        pass

    # 生产链路用包装 LLM 记录真实 token 用量; Judge 用原始 LLM(其自身记录)
    pipeline_llm = TokenRecordingLLM(llm)
    agents = register_all_agents(neo4j_driver=neo4j_driver, llm=pipeline_llm)

    require_assets(
        os.path.join(settings.faiss_index_path, "products.index"),
        os.path.join(settings.faiss_index_path, "product_ids.npy"),
        os.path.join(settings.faiss_index_path, "faq", "index.faiss"),
        os.path.join(settings.faiss_index_path, "faq", "index.pkl"),
    )

    evaluator = RAGEvaluator(
        llm=llm,
        search_agent=agents.get("catalog_agent").search_agent,
        cs_agent=agents.get("cs_agent"),
        kg_qa_agent=agents.get("kg_qa_agent"),
    )

    # 运行评估
    all_cases = SEARCH_TEST_CASES + CS_TEST_CASES
    print(
        f"\n共 {len(all_cases)} 个测试用例 (搜索 {len(SEARCH_TEST_CASES)} + 客服 {len(CS_TEST_CASES)})\n"
    )

    result = evaluator.run_full_evaluation(all_cases)

    # 打印汇总
    print(f"\n{'=' * 60}")
    print("  评估汇总")
    print(f"{'=' * 60}")

    summary = result.get("summary", {})

    if "retrieval" in summary:
        print("\n📊 检索指标:")
        for metric, value in summary["retrieval"].items():
            print(f"  {metric:20s}: {value:.4f}")

    if "generation" in summary:
        print("\n📝 生成指标:")
        gen = summary["generation"]
        if not gen.get("valid", True):
            print(f"  ⚠️  生成指标无效: {gen.get('error_count', 0)} 个用例评估失败")
            print(f"  ⚠️  首个错误: {gen.get('first_error', '未知')[:120]}")
        for metric, value in gen.items():
            if metric in ("valid", "error_count", "first_error"):
                continue
            print(f"  {metric:20s}: {value:.4f}")

    if "performance" in summary:
        print("\n⚡ 性能指标:")
        perf = summary["performance"]
        print(f"  {'avg_latency_ms':20s}: {perf.get('avg_latency_ms', 0)}")
        print(f"  {'p50_latency_ms':20s}: {perf.get('p50_latency_ms', 0)}")
        print(f"  {'p90_latency_ms':20s}: {perf.get('p90_latency_ms', 0)}")
        print(f"  {'p95_latency_ms':20s}: {perf.get('p95_latency_ms', 0)}")
        print(f"  {'min_latency_ms':20s}: {perf.get('min_latency_ms', 0)}")
        print(f"  {'max_latency_ms':20s}: {perf.get('max_latency_ms', 0)}")
        print(f"  {'avg_retrieval_ms':20s}: {perf.get('avg_retrieval_latency_ms', 0)}")
        print(f"  {'avg_gen_ms':20s}: {perf.get('avg_gen_latency_ms', 0)}")
        print(f"  {'avg_judge_ms':20s}: {perf.get('avg_judge_latency_ms', 0)}  (LLM Judge, 不计入系统延迟)")
        print(f"  {'p50_judge_ms':20s}: {perf.get('p50_judge_latency_ms', 0)}")
        print(f"  {'p90_judge_ms':20s}: {perf.get('p90_judge_latency_ms', 0)}")

    if "token_usage" in summary:
        tu = summary["token_usage"]
        print("\n💰 Token 用量 (LLM Judge):")
        print(f"  {'total_tokens':20s}: {tu.get('total_tokens', 0)}")
        print(f"  {'total_cost_usd':20s}: ${tu.get('total_cost_usd', 0):.6f}")
        print(f"  {'avg_tokens_per_case':20s}: {tu.get('avg_tokens_per_case', 0)}")
        by_model = tu.get("by_model", {})
        if by_model:
            print(f"  {'by_model':20s}:")
            for model, stats in by_model.items():
                print(f"    {model}: calls={stats.get('calls',0)}, tokens={stats.get('total_tokens',0)}")

    print(f"\n{'=' * 60}\n")

    # 检索策略 baseline 对比
    print(f"\n{'=' * 60}")
    print("  检索策略对比（同一 GT 下评估）")
    print(f"{'=' * 60}")

    ret_baseline = evaluator.evaluate_retrieval_strategies(all_cases)
    strategy_names = list(ret_baseline["summary"].keys())
    header = f"{'metric':<14s}" + "".join(f"{s:>14s}" for s in strategy_names)
    print(header)
    baseline_metrics = [
        "recall@1",
        "recall@3",
        "recall@5",
        "precision@5",
        "mrr",
        "ndcg@5",
        "hit_rate@5",
    ]
    for m in baseline_metrics:
        row = f"{m:<14s}" + "".join(
            f"{ret_baseline['summary'][s][m]:>14.4f}" for s in strategy_names
        )
        print(row)

    result["retrieval_baseline"] = ret_baseline

    # 保存报告
    output_path = os.path.join(PROJECT_ROOT, "tests", "rag_evaluation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
