"""
端到端真实跑通评估 — 替代 evaluate.py 的静态手写答案

此前 evaluate.py 用 5 条静态手写答案做 LLM Judge, 不能反映真实链路质量。
本评估:
1. 真实调用 ECommerceOrchestrator.invoke() 跑端到端流程
2. 记录真实延迟(p50/p90/p95)、路由结果、降级情况
3. 复用 AgentEvaluator(LLM Judge) 评估答案质量
4. 检测 fallback 触发(图查询失败降级、规则降级等)

运行方式:
    python tests/e2e_eval.py

依赖:
    - DeepSeek API (LLM Judge + 端到端链路中的 LLM 调用)
    - FAISS 索引 (搜索链路)
    - Neo4j / MySQL 可选(不可用时自动降级,记录降级事件)
"""

import json
import os
import sys
import time
from typing import List

# 必须在 import 任何 sentence-transformers/huggingface 相关模块前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 强制离线模式 — 模型已本地缓存, 避免运行时网络检查 adapter_config.json 等小文件超时
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
from tests.evaluate import AgentEvaluator

# ------------------------------------------------------------------ #
#  端到端测试用例 — 覆盖 8 个意图, 每条标注 expected_intent 用于路由准确率
# ---------------------------------------------------------------- #

E2E_TEST_CASES = [
    # search
    {"query": "有没有500元以下的蓝牙耳机？", "expected_intent": "search", "category": "search"},
    {"query": "搜索无线鼠标", "expected_intent": "search", "category": "search"},
    {"query": "找一双跑步鞋", "expected_intent": "search", "category": "search"},

    # recommend
    {"query": "推荐一些适合户外的装备", "expected_intent": "recommend", "category": "recommend"},
    {"query": "预算200以内买什么母婴用品好", "expected_intent": "recommend", "category": "recommend"},
    {"query": "帮我选个礼物", "expected_intent": "recommend", "category": "recommend"},

    # classify
    {"query": "分类: 纸尿裤", "expected_intent": "classify", "category": "classify"},

    # kg_qa
    {"query": "吸尘器属于什么类别", "expected_intent": "kg_qa", "category": "kg_qa"},

    # order (无订单号, 系统应提示输入)
    {"query": "查询我的订单", "expected_intent": "order", "category": "order"},

    # customer_service
    {"query": "怎么退货", "expected_intent": "customer_service", "category": "cs"},
    {"query": "退款多久到账", "expected_intent": "customer_service", "category": "cs"},
    {"query": "运费谁出", "expected_intent": "customer_service", "category": "cs"},

    # analytics
    {"query": "最近什么品类卖得最好", "expected_intent": "analytics", "category": "analytics"},

    # chitchat
    {"query": "你好", "expected_intent": "chitchat", "category": "chitchat"},
    {"query": "谢谢", "expected_intent": "chitchat", "category": "chitchat"},
]


# ------------------------------------------------------------------ #
#  延迟百分位计算
# ---------------------------------------------------------------- #


def percentile(values: List[float], p: float) -> float:
    """
    计算百分位数

    Args:
        values: 数值列表
        p: 百分位 (0-100), 如 50/90/95
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * p / 100
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


# ------------------------------------------------------------------ #
#  端到端评估器
# ---------------------------------------------------------------- #


class E2EEvaluator:
    """端到端真实链路评估器"""

    def __init__(self, orchestrator, judge_llm: ChatOpenAI):
        """
        Args:
            orchestrator: ECommerceOrchestrator 实例
            judge_llm: 用于 LLM Judge 评分的 LLM
        """
        self.orchestrator = orchestrator
        self.judge = AgentEvaluator(judge_llm)

    def evaluate_one(self, case: dict) -> dict:
        """评估单条用例"""
        query = case["query"]
        expected_intent = case["expected_intent"]

        # 真实跑端到端
        start = time.time()
        try:
            answer, state = self.orchestrator.invoke(query)
            error = None
        except Exception as e:
            answer = f"[执行异常] {type(e).__name__}: {e}"
            state = {"intent": "unknown", "agent": "unknown"}
            error = str(e)
        latency_ms = (time.time() - start) * 1000

        # 路由准确率
        actual_intent = state.get("intent", "unknown")
        intent_correct = actual_intent == expected_intent

        # 降级检测
        fallback = self._detect_fallback(answer, state)

        # LLM Judge 评估答案质量
        judge_result = self.judge.evaluate(
            question=query,
            answer=answer,
            context={
                "agent": state.get("agent", ""),
                "intent": actual_intent,
                "expected_intent": expected_intent,
            },
        )

        return {
            "query": query,
            "answer": answer[:500],
            "expected_intent": expected_intent,
            "actual_intent": actual_intent,
            "intent_correct": intent_correct,
            "fallback": fallback,
            "latency_ms": round(latency_ms, 2),
            "judge": {
                "overall_score": judge_result["overall_score"],
                "status": judge_result.get("status", "ok"),
                "error": judge_result.get("error"),
            },
            "error": error,
        }

    @staticmethod
    def _detect_fallback(answer: str, state: dict) -> dict:
        """
        检测降级事件

        降级信号:
        - 路由失败(intent=unknown)
        - 执行异常
        - 答案包含降级关键词
        """
        fallbacks = []

        if state.get("intent") == "unknown":
            fallbacks.append("route_failed")

        if "执行异常" in answer or "[ERROR]" in answer:
            fallbacks.append("execution_error")

        # 答案中包含常见降级话术
        degradation_markers = [
            "暂无推荐", "未找到结果", "无法判断", "请稍后",
            "降级", "不可用", "失败",
        ]
        for marker in degradation_markers:
            if marker in answer:
                fallbacks.append(f"answer_marker:{marker}")
                break

        return {
            "triggered": len(fallbacks) > 0,
            "events": fallbacks,
        }

    def evaluate(self, test_cases: List[dict] = None) -> dict:
        """运行端到端评估"""
        test_cases = test_cases or E2E_TEST_CASES
        results = []

        # 评估前重置 token 统计,本次评估独立计数
        obs.tokens.reset()

        print(f"\n  端到端用例 {len(test_cases)} 条\n")
        for i, case in enumerate(test_cases, 1):
            result = self.evaluate_one(case)
            results.append(result)
            status = "OK" if not result["error"] else "ERR"
            fb = " [FALLBACK]" if result["fallback"]["triggered"] else ""
            ic = "OK" if result["intent_correct"] else "MISS"
            print(
                f"  [{i:2d}/{len(test_cases)}] {status} intent={ic} "
                f"score={result['judge']['overall_score']:.1f} "
                f"latency={result['latency_ms']:.0f}ms{fb} "
                f"| {case['query'][:30]}"
            )

        summary = self._compute_summary(results)
        return {
            "total": len(results),
            "results": results,
            "summary": summary,
        }

    @staticmethod
    def _compute_summary(results: list) -> dict:
        """计算汇总指标"""
        if not results:
            return {}

        latencies = [r["latency_ms"] for r in results]
        # 质量分只统计"有可用分数"的样本（ok / partial）；judge 彻底失败时 evaluate.py
        # 返回 0 分，直接计入平均会把质量分系统性拉低（此前 15 条里 2 条失败 → 6.43，剔除后 7.42）。
        # 注意 partial（抢救到部分维度）要计入——它有可用分数，只是标注出来。
        valid_scores = [
            r["judge"]["overall_score"]
            for r in results
            if r["judge"].get("status", "ok") != "failed"
        ]
        all_scores = [r["judge"]["overall_score"] for r in results]
        scores = valid_scores or all_scores
        intent_correct = sum(1 for r in results if r["intent_correct"])
        fallback_count = sum(1 for r in results if r["fallback"]["triggered"])
        error_count = sum(1 for r in results if r["error"])
        judge_errors = sum(1 for r in results if r["judge"].get("error"))

        # 降级事件统计
        fallback_events = []
        for r in results:
            fallback_events.extend(r["fallback"]["events"])

        from collections import Counter
        event_counts = dict(Counter(fallback_events))

        # 按意图拆分准确率
        intent_acc = {}
        for r in results:
            exp = r["expected_intent"]
            if exp not in intent_acc:
                intent_acc[exp] = {"correct": 0, "total": 0}
            intent_acc[exp]["total"] += 1
            if r["intent_correct"]:
                intent_acc[exp]["correct"] += 1
        for k in intent_acc:
            v = intent_acc[k]
            intent_acc[k]["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] else 0

        # token 用量
        token_summary = obs.get_usage_summary()

        return {
            "accuracy": {
                "overall": round(intent_correct / len(results), 4),
                "by_intent": intent_acc,
            },
            "quality": {
                "avg_score": round(sum(scores) / len(scores), 2),
                "min_score": min(scores),
                "max_score": max(scores),
                "valid_count": len(valid_scores),
                "partial_count": sum(
                    1 for r in results if r["judge"].get("status") == "partial"
                ),
                "failed_count": sum(
                    1 for r in results if r["judge"].get("status") == "failed"
                ),
                # 保留"含失败样本"的口径，方便对比与排查
                "avg_score_including_failures": round(sum(all_scores) / len(all_scores), 2),
                "judge_error_count": judge_errors,
            },
            "latency": {
                "p50_ms": round(percentile(latencies, 50), 2),
                "p90_ms": round(percentile(latencies, 90), 2),
                "p95_ms": round(percentile(latencies, 95), 2),
                "avg_ms": round(sum(latencies) / len(latencies), 2),
                "max_ms": round(max(latencies), 2),
                "min_ms": round(min(latencies), 2),
            },
            "fallback": {
                "triggered_count": fallback_count,
                "triggered_rate": round(fallback_count / len(results), 4),
                "event_counts": event_counts,
            },
            "errors": {
                "execution_error_count": error_count,
            },
            "token_usage": {
                "total_tokens": token_summary.get("total_tokens", 0),
                "total_cost_usd": token_summary.get("total_cost_usd", 0),
                "by_model": token_summary.get("by_model", {}),
            },
        }


# ------------------------------------------------------------------ #
#  报告打印
# ---------------------------------------------------------------- #


def print_report(result: dict):
    """打印可读报告"""
    s = result["summary"]

    print(f"\n{'=' * 60}")
    print("  端到端评估汇总")
    print(f"{'=' * 60}")

    acc = s["accuracy"]
    print(f"\n  🎯 路由准确率: {acc['overall']:.4f}")
    print("    按意图:")
    for intent, stats in acc["by_intent"].items():
        print(f"      {intent:<18s}: {stats['accuracy']:.4f}  ({stats['correct']}/{stats['total']})")

    q = s["quality"]
    print("\n  📝 答案质量 (LLM Judge):")
    print(
        f"    平均分: {q['avg_score']}/10  (min={q['min_score']}, max={q['max_score']})"
        f"  [有效 {q.get('valid_count', 0)} 条，其中部分解析 {q.get('partial_count', 0)} 条；"
        f"彻底失败 {q.get('failed_count', 0)} 条；含失败样本口径 {q.get('avg_score_including_failures')}]"
    )
    if q["judge_error_count"]:
        print(f"    ⚠️  Judge 失败 {q['judge_error_count']} 次")

    lat = s["latency"]
    print("\n  ⚡ 延迟 (真实端到端,含 LLM 调用):")
    print(f"    p50: {lat['p50_ms']:.0f}ms  p90: {lat['p90_ms']:.0f}ms  p95: {lat['p95_ms']:.0f}ms")
    print(f"    avg: {lat['avg_ms']:.0f}ms  min: {lat['min_ms']:.0f}ms  max: {lat['max_ms']:.0f}ms")

    fb = s["fallback"]
    print("\n  🔄 降级事件:")
    print(f"    触发次数: {fb['triggered_count']}  触发率: {fb['triggered_rate']:.4f}")
    if fb["event_counts"]:
        print("    事件分布:")
        for event, count in fb["event_counts"].items():
            print(f"      {event}: {count}")

    err = s["errors"]
    if err["execution_error_count"]:
        print(f"\n  ❌ 执行异常: {err['execution_error_count']} 次")

    tu = s["token_usage"]
    print("\n  💰 Token 用量 (端到端链路 + Judge):")
    print(f"    total_tokens: {tu['total_tokens']}")
    print(f"    total_cost_usd: ${tu['total_cost_usd']:.6f}")
    if tu.get("by_model"):
        for model, stats in tu["by_model"].items():
            print(f"    {model}: calls={stats.get('calls',0)}, tokens={stats.get('total_tokens',0)}")


# ------------------------------------------------------------------ #
#  主入口
# ---------------------------------------------------------------- #


def main():
    print("=" * 60)
    print("  端到端真实跑通评估 (E2E Evaluation)")
    print("=" * 60)

    os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    # LLM 实例 — 端到端链路 + LLM Judge 共用
    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.1,
        # 该实例同时用于链路与 LLM Judge；judge 要输出 5 个维度带理由的 JSON，
        # 1024 会把输出截断（实测 15 条里 2 条解析失败），这里放宽到 2048
        max_tokens=2048,
    )

    # 初始化 Agent 与编排器
    from orchestration.graph import ECommerceOrchestrator
    from orchestration.registry import create_neo4j_driver, register_all_agents
    from orchestration.router import RouterAgent

    neo4j_driver = None
    try:
        neo4j_driver = create_neo4j_driver()
    except Exception as e:
        print(f"  Neo4j 不可用, 相关链路将降级: {e}")

    # 生产链路用包装 LLM 记录真实 token 用量; Judge 用原始 LLM(其自身记录)
    pipeline_llm = TokenRecordingLLM(llm)
    agents = register_all_agents(neo4j_driver=neo4j_driver, llm=pipeline_llm)

    require_assets(
        settings.classify_model_path,
        os.path.join(settings.classify_model_path, "model.safetensors"),
        os.path.join(settings.classify_model_path, "labels.txt"),
        settings.ner_model_path,
        os.path.join(settings.ner_model_path, "model.safetensors"),
        os.path.join(settings.faiss_index_path, "products.index"),
        os.path.join(settings.faiss_index_path, "product_ids.npy"),
        os.path.join(settings.faiss_index_path, "faq", "index.faiss"),
    )

    router = RouterAgent(llm=pipeline_llm)
    orchestrator = ECommerceOrchestrator(llm=pipeline_llm, agents=agents, router=router)

    evaluator = E2EEvaluator(orchestrator=orchestrator, judge_llm=llm)
    result = evaluator.evaluate()

    print_report(result)

    output_path = os.path.join(PROJECT_ROOT, "tests", "e2e_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
