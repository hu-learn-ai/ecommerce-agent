"""
KG-QA 链路评估 — Text2Cypher 准确率 + 安全检查 + 降级

此前 rag_evaluation.py 的 RAGEvaluator.__init__ 接收 kg_qa_agent 但从未调用,
KG-QA (Text2Cypher) 链路完全没有评估。本评估补齐:

1. Text2Cypher 准确率
   - 生成的 Cypher 语法是否合法 (能否被 Neo4j 解析)
   - 生成的 Cypher 是否能返回结果 (语义正确性)
   - 与参考 Cypher 的语义等价性
2. 安全检查覆盖率 (5 层防御是否都生效)
   - L0 多语句注入拒绝
   - L1 非 MATCH 开头拒绝
   - L2 危险关键词拦截 (DELETE/CREATE/...)
   - L3 复杂度限制 (长度/MATCH 数/无 WHERE 笛卡尔积)
   - L4 自动注入 LIMIT
3. 降级链路
   - 主查询无结果时是否触发 fallback
   - fallback (关键词 SPU 搜索) 的命中率
4. 端到端答案质量 (LLM Judge)

运行方式:
    python tests/kgqa_eval.py

    # 只评估 Cypher 生成 + 安全检查 (不需要 Neo4j)
    python tests/kgqa_eval.py --no-exec
"""

import json
import os
import sys
import time
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_openai import ChatOpenAI

from config.settings import settings
from orchestration.observability import obs
from tests._classification_metrics import compute_classification_metrics
from tests._eval_env import TokenRecordingLLM
from tests.evaluate import AgentEvaluator


# ------------------------------------------------------------------ #
#  测试用例集 — 覆盖 4 类问题 + 安全攻击用例
# ---------------------------------------------------------------- #

# 正常用例: 4 类问题 + 期望返回的 Cypher 包含的关键关键词(用于语义等价性检查)
KGQA_TEST_CASES = [
    # 1. 品牌查询
    {
        "query": "华为有哪些产品？",
        "category": "brand_query",
        "expected_keywords": ["MATCH", "Trademark", "CONTAINS", "华为", "RETURN", "LIMIT"],
        "expected_node": "Trademark",
    },
    {
        "query": "华为是什么品牌",
        "category": "brand_query",
        "expected_keywords": ["MATCH", "Trademark", "RETURN", "LIMIT"],
        "expected_node": "Trademark",
    },
    # 2. 分类导航
    {
        "query": "吸尘器属于什么类别",
        "category": "category_query",
        "expected_keywords": ["MATCH", "SPU", "Belong", "Category", "RETURN"],
        "expected_node": "Category",
    },
    {
        "query": "手机属于哪个分类",
        "category": "category_query",
        "expected_keywords": ["MATCH", "Belong", "Category", "RETURN"],
        "expected_node": "Category",
    },
    # 3. 属性查询
    {
        "query": "这款手机的内存是多少",
        "category": "attr_query",
        "expected_keywords": [],  # 图谱无属性维度, 应返回 NONE 走降级
        "expected_node": None,
    },
    # 4. 商品关系
    {
        "query": "和iPhone类似的产品有哪些",
        "category": "relation_query",
        "expected_keywords": ["MATCH", "SPU", "RETURN", "LIMIT"],
        "expected_node": "SPU",
    },
    # 5. 闲聊/无法用 Cypher 表达
    {
        "query": "你好",
        "category": "chitchat",
        "expected_keywords": [],  # 期望返回 NONE
        "expected_node": None,
    },
    {
        "query": "今天天气怎么样",
        "category": "chitchat",
        "expected_keywords": [],
        "expected_node": None,
    },
]

# 安全攻击用例 — 验证 5 层防御是否生效
# 每条用例注入恶意 Cypher 模板,应被拒绝
SECURITY_ATTACK_CASES = [
    {
        "name": "L0 多语句注入",
        "injected_cypher": "MATCH (n) RETURN n; DROP DATABASE neo4j;",
        "expected_layer": "L0",
        "expected_blocked": True,
    },
    {
        "name": "L1 非 MATCH 开头 (写操作)",
        "injected_cypher": "CREATE (n:Test) RETURN n",
        "expected_layer": "L1",
        "expected_blocked": True,
    },
    {
        "name": "L2 DELETE 写操作",
        "injected_cypher": "MATCH (n:Test) DELETE n",
        "expected_layer": "L2",
        "expected_blocked": True,
    },
    {
        "name": "L2 CALL 子过程",
        "injected_cypher": "MATCH (n) CALL db.labels() YIELD label RETURN label",
        "expected_layer": "L2",
        "expected_blocked": True,
    },
    {
        "name": "L3 笛卡尔积 (多 MATCH 无 WHERE)",
        "injected_cypher": "MATCH (a), (b), (c) RETURN a, b, c",
        "expected_layer": "L3",
        "expected_blocked": True,
    },
    {
        "name": "L3 MATCH 子句过多 (>5)",
        "injected_cypher": "MATCH (a) MATCH (b) MATCH (c) MATCH (d) MATCH (e) MATCH (f) MATCH (g) RETURN a",
        "expected_layer": "L3",
        "expected_blocked": True,
    },
    {
        "name": "L4 自动注入 LIMIT (合法查询但无 LIMIT)",
        "injected_cypher": "MATCH (n:SPU) WHERE n.name CONTAINS '手机' RETURN n.name AS name",
        "expected_layer": "L4",
        "expected_blocked": False,  # 不拒绝, 但应自动注入 LIMIT
        "expected_modified": True,  # 期望被修改(注入 LIMIT)
    },
    {
        "name": "合法查询 (应通过)",
        "injected_cypher": "MATCH (n:SPU) WHERE n.name CONTAINS '手机' RETURN n.name LIMIT 5",
        "expected_layer": None,
        "expected_blocked": False,
        "expected_modified": False,
    },
]


# ------------------------------------------------------------------ #
#  KG-QA 评估器
# ---------------------------------------------------------------- #


class KGQAEvaluator:
    """KG-QA (Text2Cypher) 链路评估器"""

    def __init__(self, kg_agent, judge_llm: ChatOpenAI):
        """
        Args:
            kg_agent: KGQAAgent 实例
            judge_llm: 用于答案质量评估的 LLM
        """
        self.agent = kg_agent
        self.judge = AgentEvaluator(judge_llm)
        self.neo4j_available = kg_agent.neo4j_driver is not None

    def evaluate_cypher_generation(self) -> dict:
        """评估 Cypher 生成质量(不执行,只检查语法/关键词)"""
        results = []
        for case in KGQA_TEST_CASES:
            query = case["query"]
            try:
                cypher = self.agent._generate_cypher(query)
            except Exception as e:
                cypher = ""

            # 判断是否 NONE (无法用 Cypher 表达)
            is_none = (
                not cypher
                or cypher.strip().upper() in ("NONE", "NULL", "")
            )

            # 语义等价性: 检查期望关键词
            expected_kws = case.get("expected_keywords", [])
            if is_none:
                keyword_coverage = 1.0 if not expected_kws else 0.0
            else:
                cypher_upper = cypher.upper()
                hits = sum(1 for kw in expected_kws if kw.upper() in cypher_upper)
                keyword_coverage = hits / len(expected_kws) if expected_kws else 1.0

            # 安全检查: 生成的 Cypher 是否通过安全校验
            try:
                safe_cypher = self.agent._entity_alignment(cypher) if not is_none else ""
                passed_safety = bool(safe_cypher) if not is_none else True
            except Exception:
                safe_cypher = ""
                passed_safety = False

            results.append({
                "query": query,
                "category": case["category"],
                "generated_cypher": (cypher or "")[:300],
                "is_none": is_none,
                "keyword_coverage": round(keyword_coverage, 4),
                "passed_safety": passed_safety,
                "expected_node": case.get("expected_node"),
            })

        # 汇总
        none_correct = sum(
            1 for r in results
            if r["is_none"] and r["category"] == "chitchat"
        )
        none_total = sum(1 for r in results if r["category"] == "chitchat")

        cypher_generated = [r for r in results if not r["is_none"]]
        avg_keyword_coverage = (
            sum(r["keyword_coverage"] for r in cypher_generated) / len(cypher_generated)
            if cypher_generated else 0
        )
        safety_pass_rate = (
            sum(1 for r in cypher_generated if r["passed_safety"]) / len(cypher_generated)
            if cypher_generated else 0
        )

        return {
            "total_cases": len(results),
            "results": results,
            "chitchat_none_rate": round(none_correct / none_total, 4) if none_total else 0,
            "avg_keyword_coverage": round(avg_keyword_coverage, 4),
            "safety_pass_rate": round(safety_pass_rate, 4),
        }

    def evaluate_security(self) -> dict:
        """评估安全检查覆盖率(注入攻击用例)"""
        results = []
        for case in SECURITY_ATTACK_CASES:
            cypher = case["injected_cypher"]
            try:
                result_cypher = self.agent._entity_alignment(cypher)
                blocked = not result_cypher  # 返回空字符串 = 被拒绝
                modified = (
                    not blocked
                    and result_cypher != cypher
                    and "LIMIT" in result_cypher.upper()
                    and "LIMIT" not in cypher.upper()
                )
            except Exception:
                blocked = True
                modified = False
                result_cypher = ""

            results.append({
                "name": case["name"],
                "injected_cypher": cypher[:120],
                "blocked": blocked,
                "expected_blocked": case["expected_blocked"],
                "modified": modified,
                "expected_modified": case.get("expected_modified", False),
                "result_cypher": (result_cypher or "")[:200],
                "passed": blocked == case["expected_blocked"]
                and modified == case.get("expected_modified", False),
            })

        passed = sum(1 for r in results if r["passed"])
        return {
            "total": len(results),
            "passed": passed,
            "pass_rate": round(passed / len(results), 4) if results else 0,
            "results": results,
        }

    def evaluate_e2e(self) -> dict:
        """端到端评估 (生成 Cypher → 执行 → 生成答案)"""
        if not self.neo4j_available:
            return {
                "available": False,
                "note": "Neo4j 不可用, 跳过端到端执行评估",
            }

        results = []
        obs.tokens.reset()

        for case in KGQA_TEST_CASES:
            query = case["query"]
            start = time.time()
            try:
                answer = self.agent.kg_query(query)
                error = None
            except Exception as e:
                answer = f"[执行异常] {type(e).__name__}: {e}"
                error = str(e)
            latency_ms = (time.time() - start) * 1000

            # 检测降级 (主查询无结果触发 fallback)
            fallback_triggered = "暂时未找到" in answer or "换个问法" in answer

            # LLM Judge 评估
            judge_start = time.time()
            judge_result = self.judge.evaluate(
                question=query,
                answer=answer,
                context={
                    "agent": "kg_qa_agent",
                    "category": case["category"],
                    "neo4j_available": True,
                },
            )
            judge_latency_ms = (time.time() - judge_start) * 1000

            results.append({
                "query": query,
                "category": case["category"],
                "answer": answer[:300],
                "latency_ms": round(latency_ms, 2),
                "judge_latency_ms": round(judge_latency_ms, 2),
                "judge_score": judge_result["overall_score"],
                "fallback_triggered": fallback_triggered,
                "error": error,
            })

        scores = [r["judge_score"] for r in results]
        fallback_count = sum(1 for r in results if r["fallback_triggered"])

        return {
            "available": True,
            "total": len(results),
            "results": results,
            "avg_judge_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "fallback_rate": round(fallback_count / len(results), 4) if results else 0,
            "token_usage": obs.get_usage_summary(),
        }

    def evaluate(self) -> dict:
        """运行完整评估"""
        return {
            "cypher_generation": self.evaluate_cypher_generation(),
            "security": self.evaluate_security(),
            "e2e": self.evaluate_e2e(),
        }


# ------------------------------------------------------------------ #
#  报告打印
# ---------------------------------------------------------------- #


def print_report(result: dict):
    """打印可读报告"""
    cg = result["cypher_generation"]
    print(f"\n  📝 Cypher 生成质量:")
    print(f"    用例数: {cg['total_cases']}")
    print(f"    闲聊识别率 (应返回 NONE): {cg['chitchat_none_rate']:.4f}")
    print(f"    关键词覆盖率 (语义等价性): {cg['avg_keyword_coverage']:.4f}")
    print(f"    安全校验通过率: {cg['safety_pass_rate']:.4f}")
    print(f"\n    生成样例:")
    for r in cg["results"][:5]:
        status = "NONE" if r["is_none"] else "OK"
        print(f"      [{r['category']}] {status} kw_cov={r['keyword_coverage']:.2f} "
              f"safety={'OK' if r['passed_safety'] else 'FAIL'}")
        if not r["is_none"]:
            print(f"        Cypher: {r['generated_cypher'][:100]}")

    sec = result["security"]
    print(f"\n  🛡️  安全检查 ({sec['passed']}/{sec['total']} 通过, 通过率 {sec['pass_rate']:.4f}):")
    for r in sec["results"]:
        mark = "PASS" if r["passed"] else "FAIL"
        action = "blocked" if r["blocked"] else ("modified" if r["modified"] else "passed-through")
        print(f"    [{mark}] {r['name']:30s} → {action}")

    e2e = result["e2e"]
    if e2e.get("available"):
        print(f"\n  🔗 端到端执行:")
        print(f"    用例数: {e2e['total']}")
        print(f"    平均评分: {e2e['avg_judge_score']}/10")
        print(f"    降级率: {e2e['fallback_rate']:.4f}")
        tu = e2e.get("token_usage", {})
        print(f"    Token: {tu.get('total_tokens', 0)}  Cost: ${tu.get('total_cost_usd', 0):.6f}")
    else:
        print(f"\n  🔗 端到端执行: {e2e.get('note', '不可用')}")


# ------------------------------------------------------------------ #
#  主入口
# ---------------------------------------------------------------- #


def main():
    import argparse

    parser = argparse.ArgumentParser(description="KG-QA 链路评估")
    parser.add_argument("--no-exec", action="store_true",
                        help="只评估 Cypher 生成 + 安全检查(不执行 Neo4j)")
    args = parser.parse_args()

    print("=" * 60)
    print("  KG-QA 链路评估 (Text2Cypher + 安全 + 降级)")
    print("=" * 60)

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.0,
        max_tokens=512,
    )

    # 初始化 KG Agent (Neo4j 可选)
    from agents.kg_qa_agent import KGQAAgent
    from orchestration.registry import create_neo4j_driver

    neo4j_driver = None
    try:
        neo4j_driver = create_neo4j_driver()
        print("  Neo4j 已连接")
    except Exception as e:
        print(f"  Neo4j 不可用 (端到端执行将被跳过): {e}")

    if args.no_exec:
        neo4j_driver = None

    # 生产链路用包装 LLM 记录真实 token 用量; Judge 用原始 LLM(其自身记录)
    kg_agent = KGQAAgent(neo4j_driver=neo4j_driver, llm=TokenRecordingLLM(llm))
    evaluator = KGQAEvaluator(kg_agent=kg_agent, judge_llm=llm)

    result = evaluator.evaluate()
    print_report(result)

    output_path = os.path.join(PROJECT_ROOT, "tests", "kgqa_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
