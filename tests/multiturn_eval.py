"""
多轮对话连贯性评估

此前评估都是单轮 (一条 query → 一条 answer), 无法评估:
1. 上下文继承 — 用户后续问题能否引用前文
2. 指代消解 — "它/这个/那个款" 是否正确指代前文实体
3. 话题切换 — 切换话题时不应混淆
4. 跨轮记忆 — 用户在轮 1 说的偏好, 轮 3 是否还记得

设计: 每个 scenario 是一个多轮对话脚本, 标注每轮的预期关键事实
- 上下文继承率: 后续轮能否引用前轮实体
- 指代消解成功率: 代词是否被正确解析
- 跨轮记忆命中率: 早期信息是否被记住

运行方式:
    python tests/multiturn_eval.py
"""

import json
import os
import sys
import time
from typing import List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import TokenRecordingLLM, ensure_project_root, require_assets

ensure_project_root()

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from langchain_openai import ChatOpenAI

from config.settings import settings
from orchestration.observability import obs
from tests.evaluate import AgentEvaluator

# ------------------------------------------------------------------ #
#  多轮对话测试用例
# ---------------------------------------------------------------- #

MULTITURN_SCENARIOS = [
    {
        "name": "商品咨询 + 追问规格",
        "turns": [
            {
                "user": "有没有蓝牙耳机推荐",
                "expected_keywords": ["蓝牙", "耳机"],  # 轮1 应提到蓝牙耳机
                "evaluation": "relevance",
            },
            {
                "user": "第一款多少钱",
                "expected_keywords": [],  # 关键: 应理解"第一款"指代上一轮提到的第一个商品
                "evaluation": "coreference",
                "needs_context": True,
            },
            {
                "user": "有降噪功能吗",
                "expected_keywords": ["降噪"],
                "evaluation": "context_inherit",
                "needs_context": True,
            },
        ],
    },
    {
        "name": "偏好记忆 + 后续推荐",
        "turns": [
            {
                "user": "我对价格敏感,预算500以内",
                "expected_keywords": ["500"],
                "evaluation": "relevance",
            },
            {
                "user": "推荐个手机",
                "expected_keywords": [],
                "evaluation": "long_term_recall",
                "needs_context": True,
                # 关键: 推荐应受 500 预算约束 (这一轮的答案若价格 >500 视为遗忘偏好)
            },
            {
                "user": "再推荐一款平板",
                "expected_keywords": [],
                "evaluation": "long_term_recall",
                "needs_context": True,
            },
        ],
    },
    {
        "name": "话题切换",
        "turns": [
            {
                "user": "你好",
                "expected_keywords": [],
                "evaluation": "relevance",
            },
            {
                "user": "搜索无线鼠标",
                "expected_keywords": ["鼠标"],
                "evaluation": "topic_switch",
                "needs_context": False,  # 话题从闲聊切到搜索, 不应继承前文
            },
            {
                "user": "怎么退货",
                "expected_keywords": ["退货"],
                "evaluation": "topic_switch",
                "needs_context": False,
            },
        ],
    },
    {
        "name": "对比咨询",
        "turns": [
            {
                "user": "iPhone 15 和华为 Mate60 哪个好",
                "expected_keywords": ["iPhone", "华为"],
                "evaluation": "relevance",
            },
            {
                "user": "前者的电池怎么样",
                "expected_keywords": ["电池"],
                "evaluation": "coreference",
                "needs_context": True,  # "前者" 应指 iPhone
            },
            {
                "user": "后者呢",
                "expected_keywords": [],
                "evaluation": "coreference",
                "needs_context": True,  # "后者" 应指华为
            },
        ],
    },
]


# ------------------------------------------------------------------ #
#  多轮对话评估器
# ---------------------------------------------------------------- #


class MultiTurnEvaluator:
    """多轮对话连贯性评估器"""

    def __init__(self, orchestrator, judge_llm: ChatOpenAI):
        """
        Args:
            orchestrator: 支持 session_id 的 orchestrator (ReActOrchestrator)
            judge_llm: 用于评估答案质量的 LLM
        """
        self.orchestrator = orchestrator
        self.judge = AgentEvaluator(judge_llm)

    def evaluate_scenario(self, scenario: dict, session_id: str) -> dict:
        """评估单个多轮场景"""
        turns_results = []

        for i, turn in enumerate(scenario["turns"], 1):
            user_input = turn["user"]
            start = time.time()
            try:
                answer, state = self.orchestrator.invoke(
                    user_input, session_id=session_id
                )
                error = None
            except Exception as e:
                answer = f"[执行异常] {type(e).__name__}: {e}"
                state = {"intent": "unknown"}
                error = str(e)
            latency_ms = (time.time() - start) * 1000

            # 检查预期关键词
            expected_kws = turn.get("expected_keywords", [])
            kw_hits = sum(1 for kw in expected_kws if kw in answer)
            kw_coverage = kw_hits / len(expected_kws) if expected_kws else 1.0

            # LLM Judge 评估 (含上下文是否被引用)
            judge_context = {
                "turn": i,
                "total_turns": len(scenario["turns"]),
                "evaluation_type": turn.get("evaluation", ""),
                "needs_context": turn.get("needs_context", False),
            }
            judge_start = time.time()
            judge_result = self.judge.evaluate(
                question=user_input,
                answer=answer,
                context=judge_context,
            )
            judge_latency_ms = (time.time() - judge_start) * 1000

            turns_results.append({
                "turn": i,
                "user": user_input,
                "answer": answer[:400],
                "expected_keywords": expected_kws,
                "keyword_coverage": round(kw_coverage, 4),
                "judge_score": judge_result["overall_score"],
                "latency_ms": round(latency_ms, 2),
                "judge_latency_ms": round(judge_latency_ms, 2),
                "intent": state.get("intent", "unknown"),
                "evaluation_type": turn.get("evaluation", ""),
                "needs_context": turn.get("needs_context", False),
                "error": error,
            })

        return {
            "scenario_name": scenario["name"],
            "total_turns": len(scenario["turns"]),
            "turns": turns_results,
            "session_id": session_id,
        }

    def evaluate(self, scenarios: List[dict] = None) -> dict:
        """运行所有多轮场景"""
        scenarios = scenarios or MULTITURN_SCENARIOS
        results = []
        obs.tokens.reset()

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n  场景 {i}/{len(scenarios)}: {scenario['name']}")
            session_id = f"multiturn_eval_{i}_{int(time.time())}"
            result = self.evaluate_scenario(scenario, session_id)
            results.append(result)
            # 打印每轮
            for t in result["turns"]:
                status = "OK" if not t["error"] else "ERR"
                print(f"    [{status}] turn{t['turn']} score={t['judge_score']:.1f} "
                      f"kw={t['keyword_coverage']:.2f} "
                      f"latency={t['latency_ms']:.0f}ms | {t['user'][:30]}")

        summary = self._compute_summary(results)
        return {
            "scenarios": results,
            "summary": summary,
        }

    @staticmethod
    def _compute_summary(results: list) -> dict:
        all_turns = [t for r in results for t in r["turns"]]
        scores = [t["judge_score"] for t in all_turns]
        kw_coverages = [t["keyword_coverage"] for t in all_turns]
        latencies = [t["latency_ms"] for t in all_turns]

        # 按评估类型拆分
        by_type = {}
        for t in all_turns:
            etype = t["evaluation_type"]
            if etype not in by_type:
                by_type[etype] = {"scores": [], "count": 0}
            by_type[etype]["scores"].append(t["judge_score"])
            by_type[etype]["count"] += 1
        for etype, d in by_type.items():
            d["avg_score"] = round(sum(d["scores"]) / len(d["scores"]), 2) if d["scores"] else 0
            del d["scores"]

        def _pct(vals, p):
            if not vals:
                return 0.0
            s = sorted(vals)
            k = (len(s) - 1) * p / 100
            f = int(k)
            c = f + 1
            if c >= len(s):
                return s[-1]
            return s[f] + (k - f) * (s[c] - s[f])

        return {
            "total_scenarios": len(results),
            "total_turns": len(all_turns),
            "avg_judge_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "avg_keyword_coverage": round(sum(kw_coverages) / len(kw_coverages), 4) if kw_coverages else 0,
            "by_evaluation_type": by_type,
            "latency": {
                "p50_ms": round(_pct(latencies, 50), 2),
                "p90_ms": round(_pct(latencies, 90), 2),
                "p95_ms": round(_pct(latencies, 95), 2),
                "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0,
            },
            "token_usage": obs.get_usage_summary(),
        }


# ------------------------------------------------------------------ #
#  报告打印
# ---------------------------------------------------------------- #


def print_report(result: dict):
    s = result["summary"]
    print(f"\n{'=' * 60}")
    print("  多轮对话连贯性评估汇总")
    print(f"{'=' * 60}")

    print(f"\n  场景数: {s['total_scenarios']}  总轮数: {s['total_turns']}")
    print(f"  平均 Judge 评分: {s['avg_judge_score']}/10")
    print(f"  关键词覆盖率: {s['avg_keyword_coverage']:.4f}")

    print("\n  按评估类型拆分:")
    print(f"    {'type':<20s}{'turns':>8s}{'avg_score':>12s}")
    for etype, d in s["by_evaluation_type"].items():
        print(f"    {etype:<20s}{d['count']:>8d}{d['avg_score']:>12.2f}")

    lat = s["latency"]
    print("\n  延迟:")
    print(f"    p50: {lat['p50_ms']:.0f}ms  p90: {lat['p90_ms']:.0f}ms  p95: {lat['p95_ms']:.0f}ms")
    print(f"    avg: {lat['avg_ms']:.0f}ms")

    tu = s["token_usage"]
    print("\n  Token 用量:")
    print(f"    total_tokens: {tu.get('total_tokens', 0)}")
    print(f"    total_cost_usd: ${tu.get('total_cost_usd', 0):.6f}")

    # 关键场景详查
    print("\n  关键场景详情:")
    for r in result["scenarios"]:
        print(f"\n  [{r['scenario_name']}] ({r['total_turns']} 轮)")
        for t in r["turns"]:
            needs_ctx = "需上下文" if t["needs_context"] else "独立"
            print(f"    turn{t['turn']} [{t['evaluation_type']}] {needs_ctx}")
            print(f"      user: {t['user']}")
            print(f"      ans:  {t['answer'][:100]}")
            print(f"      score={t['judge_score']:.1f} kw_cov={t['keyword_coverage']:.2f}")


# ------------------------------------------------------------------ #
#  主入口
# ---------------------------------------------------------------- #


def main():
    print("=" * 60)
    print("  多轮对话连贯性评估 (Multi-turn Evaluation)")
    print("=" * 60)

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.1,
        max_tokens=1024,
    )

    # 使用 ReactOrchestrator (支持 session_id 和记忆)
    from orchestration.react_orchestrator import ReactOrchestrator
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
    orchestrator = ReactOrchestrator(llm=pipeline_llm, agents=agents, router=router)

    evaluator = MultiTurnEvaluator(orchestrator=orchestrator, judge_llm=llm)
    result = evaluator.evaluate()

    print_report(result)

    output_path = os.path.join(PROJECT_ROOT, "tests", "multiturn_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
