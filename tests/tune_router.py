"""路由规则调参：把"泛词"从决定性强关键词降级为弱信号，并按门槛验证。

背景（2026-09-16 路由扩样到 209 条后的发现）：
  26 条误路由里 24 条出在关键词层——"买/卖/找/价格"这类**泛词**被当成确定性命中，
  把 recommend、order、analytics 请求抢成 search（单"买"字就误判 15 次）；
  而 LLM 语义兜底层准确率 98.2%。所以候选改动是：命中弱词时**不直接定意图**，继续走 LLM。

准入流程（与 tests/tune_rrf.py 一致）：留出集必须同向提升 + 4 折交叉验证 ≥3/4 折占优。
⚠️ 路由决策含 LLM 调用，为避免"每次评估都重新调一次模型"导致不可比，本脚本把
   每个查询的 LLM 路由结果缓存到 tests/router_llm_cache.json，所有配置共用同一批预测。

用法:
    python tests/tune_router.py            # 跑基线 / 弱词方案对比 + 留出/交叉验证
    python tests/tune_router.py --refresh  # 忽略缓存,重新请求 LLM

输出:
    tests/router_tuning_report.json
"""

import argparse
import collections
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root  # noqa: E402

ensure_project_root()

from langchain_openai import ChatOpenAI  # noqa: E402

from config.settings import settings  # noqa: E402
from orchestration.router import RouterAgent  # noqa: E402
from tests.router_eval import accuracy_ci, load_all_test_cases  # noqa: E402

CACHE = os.path.join(PROJECT_ROOT, "tests", "router_llm_cache.json")
REPORT = os.path.join(PROJECT_ROOT, "tests", "router_tuning_report.json")

# 候选弱词表（按证据构造，不凭直觉）：
#   V1   = 只降级"实测造成误判"的词
#   V1+  = V1 再加"问候语混说"规则（"你好,我想买耳机"不该判闲聊）
#   V2   = V1 再加同类的泛动词（有没有/多少钱/查）
VARIANTS = {
    "baseline（现状）": set(),
    "V1 降级实证误判词": {"买", "卖", "找", "价格"},
    "V1+混说规则": {"买", "卖", "找", "价格"},
    "V2 V1 + 同类泛词": {"买", "卖", "找", "价格", "有没有", "多少钱", "查"},
}
CHITCHAT_MIX_VARIANTS = {"V1+混说规则"}


def keyword_matches(query: str) -> list:
    """按规则优先级列出该查询命中的 (intent, keyword)，用于离线评估任意弱词表。"""
    text = query.lower().strip()
    hits = []
    for intent in RouterAgent.KEYWORD_PRIORITY:
        for kw in RouterAgent.KEYWORD_RULES.get(intent, []):
            if kw in text:
                hits.append((intent, kw))
    return hits


def get_llm_predictions(cases: list, refresh: bool) -> dict:
    """批量请求 LLM 路由（带缓存 + 并发），返回 {query: intent}。"""
    cache = {}
    if os.path.exists(CACHE) and not refresh:
        with open(CACHE, encoding="utf-8") as handle:
            cache = json.load(handle)
    todo = sorted({c["query"] for c in cases} - set(cache))
    if todo:
        print(f"[TuneRouter] 需要请求 LLM 的查询: {len(todo)} 条（缓存命中 {len(cache)} 条）")
        llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.1,
            max_tokens=32,
        )
        router = RouterAgent(llm=llm)
        started = time.time()

        def one(query: str) -> tuple:
            raw = router._llm_route(query)
            return query, (router._normalize_intent(raw) if raw else None)

        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, (query, intent) in enumerate(pool.map(one, todo), 1):
                cache[query] = intent
                if i % 40 == 0:
                    print(f"    {i}/{len(todo)}…")
        print(f"[TuneRouter] LLM 路由完成，用时 {time.time() - started:.1f}s")
        with open(CACHE, "w", encoding="utf-8") as handle:
            json.dump(cache, handle, ensure_ascii=False, indent=1)
    return cache


def predict(case: dict, weak: set, matches: list, llm_pred: dict, mix_rule: bool = False) -> str:
    """模拟改造后的 route()：跳过弱词；若没有强词命中则用 LLM 预测。"""
    has_weak = False
    for intent, kw in matches:
        if kw in weak:
            has_weak = True
            continue
        if mix_rule and intent == "chitchat" and has_weak:
            continue
        return intent
    return llm_pred.get(case["query"]) or (matches[0][0] if matches else "chitchat")


def evaluate(cases: list, weak: set, matches: dict, llm_pred: dict, mix_rule: bool = False) -> dict:
    correct = 0
    errors = []
    for case in cases:
        pred = predict(case, weak, matches[case["query"]], llm_pred, mix_rule)
        if pred == case["expected"]:
            correct += 1
        else:
            errors.append({"query": case["query"], "expected": case["expected"], "predicted": pred})
    return {"accuracy": round(correct / len(cases), 4) if cases else 0,
            "correct": correct, "total": len(cases), "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="路由弱词降级调参")
    parser.add_argument("--holdout", type=float, default=0.2, help="留出集比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    cases = load_all_test_cases()
    matches = {c["query"]: keyword_matches(c["query"]) for c in cases}
    llm_pred = get_llm_predictions(cases, args.refresh)
    print(f"[TuneRouter] 用例 {len(cases)} 条；LLM 路由覆盖 {sum(1 for c in cases if llm_pred.get(c['query']))} 条")

    # 按「来源 + 意图」分层切留出集，保证两边分布一致
    rng = random.Random(args.seed)
    strata = collections.defaultdict(list)
    for case in cases:
        strata[(case.get("source", "handwritten"), case["expected"])].append(case)
    holdout, train = [], []
    for items in strata.values():
        rng.shuffle(items)
        n_hold = max(1, round(len(items) * args.holdout))
        holdout.extend(items[:n_hold])
        train.extend(items[n_hold:])
    rng.shuffle(train)
    rng.shuffle(holdout)
    folds = [train[i::4] for i in range(4)]

    results = {}
    print(f"\n{'方案':<20}{'全量':>9}{'留出集':>10}")
    for name, weak in VARIANTS.items():
        mix = name in CHITCHAT_MIX_VARIANTS
        full = evaluate(cases, weak, matches, llm_pred, mix)
        hold = evaluate(holdout, weak, matches, llm_pred, mix)
        # 规则层命中率 = 有强关键词决定意图的用例占比（越低说明越依赖 LLM，延迟成本越高）
        rule_hits = sum(
            1 for c in cases
            if any(kw not in weak for _, kw in matches[c["query"]])
        )
        results[name] = {
            "weak_keywords": sorted(weak),
            "chitchat_mix_rule": mix,
            "full": full,
            "holdout": hold,
            "folds": [evaluate(f, weak, matches, llm_pred, mix) for f in folds],
            "rule_hit_rate": round(rule_hits / len(cases), 4),
        }
        print(f"{name:<20}{full['accuracy']:>9.4f}{hold['accuracy']:>10.4f}")

    base = results["baseline（现状）"]
    print("\n准入判定（门槛：留出集同向提升 + 4 折 ≥3/4 折占优）:")
    verdict = {}
    for name, info in results.items():
        if name == "baseline（现状）":
            continue
        wins = sum(1 for a, b in zip(info["folds"], base["folds"])
                   if a["accuracy"] > b["accuracy"])
        ok = info["holdout"]["accuracy"] > base["holdout"]["accuracy"] and wins >= 3
        verdict[name] = {"holdout_delta": round(info["holdout"]["accuracy"] - base["holdout"]["accuracy"], 4),
                         "fold_wins": wins, "accepted": ok}
        print(f"  {name}: 留出集 {info['holdout']['accuracy']:.4f} vs 基线 {base['holdout']['accuracy']:.4f} "
              f"({verdict[name]['holdout_delta']:+.4f})｜{wins}/4 折占优 → {'✅ 通过' if ok else '❌ 不通过'}")
        for i, (a, b) in enumerate(zip(info["folds"], base["folds"])):
            print(f"      fold{i}: {a['accuracy']:.4f} vs {base['folds'][i]['accuracy']:.4f}")

    print(f"\n判定通过的方案: {[k for k, v in verdict.items() if v['accepted']] or '无'}")

    with open(REPORT, "w", encoding="utf-8") as handle:
        json.dump({"cases": len(cases), "holdout": len(holdout), "train": len(train),
                   "results": results, "verdict": verdict,
                   "accuracy_ci_baseline": accuracy_ci(base["full"]["correct"], base["full"]["total"])},
                  handle, ensure_ascii=False, indent=1)
    print(f"报告已保存: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
