"""RRF 融合参数调优（k 常数 + 两路权重 + 关键词分支深度）。

背景：`agents/search_agent.py` 的 RRF 用固定 k=60、两路等权。在 16 用例人工标注 GT 上
发现有 3 个用例"融合后反而低于单路最优"，说明参数可再调。

方法：
1. 一次性取出每个查询的两路候选（向量 Top-N / BM25 Top-N）与人工 GT；
2. 在参数网格上计算融合排序的 NDCG@5 / MRR / P@5 / HitRate@5；
3. **留出验证**：12 个查询用于选参、4 个查询完全不用来选参，检查最优配置在留出集上是否仍然更好，
   避免"在 16 个用例上过拟合"。

用法:
    python tests/tune_rrf.py
    python tests/tune_rrf.py --top 15          # 多打印几个配置

输出:
    tests/rrf_tuning_report.json
"""

import argparse
import json
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root, require_assets  # noqa: E402

ensure_project_root()

from config.settings import settings  # noqa: E402
from tests.rag_evaluation import SEARCH_TEST_CASES, RetrievalMetrics  # noqa: E402

DEPTH = 20  # 两路各取 Top-20 参与融合

# 自适应加权候选：(属性/规格约束型的权重, 意图型的权重)，plain 恒为 (1,1)
ADAPTIVE_VARIANTS = [
    ((0.7, 1.3), (1.3, 0.7)),
    ((0.6, 1.4), (1.4, 0.6)),
    ((0.8, 1.2), (1.2, 0.8)),
    ((0.5, 1.5), (1.5, 0.5)),
    ((1.0, 1.5), (1.5, 1.0)),
]


def parse_args():
    parser = argparse.ArgumentParser(description="RRF 参数调优")
    parser.add_argument("--top", type=int, default=10, help="打印前 N 个配置")
    parser.add_argument("--holdout", type=int, default=4, help="留出验证的查询数")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fuse(vec_ids, kw_ids, k, w_vec, w_kw):
    """RRF 融合：score = w_vec/(k+rank_v) + w_kw/(k+rank_k)"""
    scores = {}
    for rank, doc_id in enumerate(vec_ids, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_vec / (k + rank)
    for rank, doc_id in enumerate(kw_ids, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_kw / (k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def evaluate(config, cases, top_k=10):
    """返回给定配置在 cases 上的平均指标"""
    k, w_vec, w_kw, kw_depth = config
    totals = {m: 0.0 for m in ("ndcg@5", "mrr", "precision@5", "hit_rate@5", "recall@5")}
    for case in cases:
        metrics = per_query_metrics(config, case, top_k)
        for name in totals:
            totals[name] += metrics[name]
    n = max(1, len(cases))
    return {name: value / n for name, value in totals.items()}


def per_query_metrics(config, case, top_k=10):
    """单查询指标（供逐用例对比与交叉验证使用）"""
    k, w_vec, w_kw, kw_depth = config
    ranked = fuse(case["vec"][:DEPTH], case["kw"][:kw_depth], k, w_vec, w_kw)
    return RetrievalMetrics.compute_all(ranked[:top_k], case["gt"])


def adaptive_metrics(variant, case, k=60, kw_depth=10, top_k=10):
    """按查询类型自适应加权的单查询指标。"""
    from agents.search_agent import ProductSearchAgent

    structured_w, intent_w = variant
    kind = ProductSearchAgent.classify_query_type(case["query"])
    w_vec, w_kw = structured_w if kind == "structured" else (
        intent_w if kind == "intent" else (1.0, 1.0)
    )
    ranked = fuse(case["vec"][:DEPTH], case["kw"][:kw_depth], k, w_vec, w_kw)
    return RetrievalMetrics.compute_all(ranked[:top_k], case["gt"])


def evaluate_adaptive(variant, cases, **kwargs):
    """自适应策略在 cases 上的平均指标 + 各类型分组均值。"""
    totals = {m: 0.0 for m in ("ndcg@5", "mrr", "precision@5", "hit_rate@5", "recall@5")}
    by_kind = {}
    for case in cases:
        metrics = adaptive_metrics(variant, case, **kwargs)
        for name in totals:
            totals[name] += metrics[name]
        from agents.search_agent import ProductSearchAgent

        kind = ProductSearchAgent.classify_query_type(case["query"])
        by_kind.setdefault(kind, []).append(metrics["ndcg@5"])
    n = max(1, len(cases))
    return {name: value / n for name, value in totals.items()}, {
        kind: sum(values) / len(values) for kind, values in by_kind.items()
    }


def main() -> dict:
    args = parse_args()
    require_assets(
        os.path.join(settings.faiss_index_path, "products.index"),
        os.path.join(settings.faiss_index_path, "product_ids.npy"),
    )
    labeled = json.load(
        open(os.path.join(PROJECT_ROOT, "tests", "retrieval_gt_labeled.json"), encoding="utf-8")
    )

    from agents.search_agent import ProductSearchAgent

    agent = ProductSearchAgent(
        embedding_model_name=settings.embedding_model,
        neo4j_driver=None,
        faiss_index_path=settings.faiss_index_path,
    )

    cases = []
    for case in SEARCH_TEST_CASES:
        query = case["query"]
        if query not in labeled:
            continue
        cases.append(
            {
                "query": query,
                "gt": [str(i) for i in labeled[query]],
                "vec": [str(r["id"]) for r in agent._vector_search(query, DEPTH)],
                "kw": [str(r["id"]) for r in agent.keyword_search(query, DEPTH)],
            }
        )
    print(f"参与调参的查询: {len(cases)} 个（均有人员标注 GT）")

    grid = []
    for k in (5, 10, 20, 30, 60, 100):
        for w_vec, w_kw in ((1, 1), (1, 0.5), (1, 2), (2, 1), (1, 0.3), (0.3, 1), (1, 3)):
            for depth_mult in (0.5, 1.0):
                grid.append((k, w_vec, w_kw, max(5, int(10 * depth_mult))))

    results = [(cfg, evaluate(cfg, cases)) for cfg in grid]
    results.sort(key=lambda item: item[1]["ndcg@5"], reverse=True)

    baseline = (60, 1, 1, 10)
    baseline_metrics = evaluate(baseline, cases)
    print(f"\n当前线上配置 (k=60, 等权, 关键词深度=10): "
          f"NDCG@5={baseline_metrics['ndcg@5']:.4f} MRR={baseline_metrics['mrr']:.4f} "
          f"P@5={baseline_metrics['precision@5']:.4f}")

    print(f"\n{'k':>4} {'w_vec':>6} {'w_kw':>5} {'kw深度':>6} "
          f"{'NDCG@5':>8} {'MRR':>7} {'P@5':>7} {'Hit@5':>7}")
    print("-" * 60)
    for cfg, metrics in results[: args.top]:
        k, w_vec, w_kw, kw_depth = cfg
        print(f"{k:>4} {w_vec:>6} {w_kw:>5} {kw_depth:>6} "
              f"{metrics['ndcg@5']:>8.4f} {metrics['mrr']:>7.4f} "
              f"{metrics['precision@5']:>7.4f} {metrics['hit_rate@5']:>7.4f}")

    # 留出验证：只在训练子集上选参，再看留出子集的表现
    rng = random.Random(args.seed)
    indices = list(range(len(cases)))
    rng.shuffle(indices)
    holdout_idx = set(indices[: args.holdout])
    train = [c for i, c in enumerate(cases) if i not in holdout_idx]
    holdout = [c for i, c in enumerate(cases) if i in holdout_idx]

    train_ranked = sorted(grid, key=lambda cfg: evaluate(cfg, train)["ndcg@5"], reverse=True)
    best_train = train_ranked[0]
    print(f"\n留出验证（{len(train)} 训练 / {len(holdout)} 留出）:")
    print(f"  训练集最优配置: k={best_train[0]} w_vec={best_train[1]} w_kw={best_train[2]} "
          f"kw深度={best_train[3]}")
    train_metrics = evaluate(best_train, train)
    hold_metrics = evaluate(best_train, holdout)
    base_hold = evaluate(baseline, holdout)
    print(f"  该配置 训练集 NDCG@5={train_metrics['ndcg@5']:.4f}"
          f"（线上配置 {evaluate(baseline, train)['ndcg@5']:.4f}）")
    print(f"  该配置 留出集 NDCG@5={hold_metrics['ndcg@5']:.4f}"
          f"（线上配置 {base_hold['ndcg@5']:.4f}）"
          f"  → {'✅ 留出集同样更好' if hold_metrics['ndcg@5'] > base_hold['ndcg@5'] else '⚠️ 留出集未提升'}")
    print(f"  留出查询: {', '.join(c['query'] for c in holdout)}")

    report = {
        "cases": len(cases),
        "grid_size": len(grid),
        "baseline": {"config": list(baseline), "metrics": baseline_metrics},
        "top": [{"config": list(cfg), "metrics": m} for cfg, m in results[: args.top]],
        "holdout_validation": {
            "train_size": len(train),
            "holdout_size": len(holdout),
            "best_on_train": list(best_train),
            "train_metrics": train_metrics,
            "holdout_metrics": hold_metrics,
            "holdout_metrics_baseline": base_hold,
            "holdout_queries": [c["query"] for c in holdout],
        },
    }

    # 逐用例对比 + K 折交叉验证：判断"整体最优配置"是否只是少数用例的收益
    candidate = results[0][0]
    print(f"\n逐用例对比（候选配置 k={candidate[0]} w_vec={candidate[1]} w_kw={candidate[2]} "
          f"vs 线上 k=60 等权）:")
    better = worse = same = 0
    deltas = {}
    for case in cases:
        new = per_query_metrics(candidate, case)["ndcg@5"]
        old = per_query_metrics(baseline, case)["ndcg@5"]
        deltas[case["query"]] = round(new - old, 4)
        if new > old + 1e-9:
            better += 1
        elif new < old - 1e-9:
            worse += 1
        else:
            same += 1
    print(f"  提升 {better} 个 / 持平 {same} 个 / 下降 {worse} 个")
    for query, delta in sorted(deltas.items(), key=lambda kv: kv[1]):
        if abs(delta) > 1e-9:
            print(f"    {'↑' if delta > 0 else '↓'} {query:<16} Δ={delta:+.4f}")

    folds = 4
    rng_cv = random.Random(args.seed + 1)
    order = list(range(len(cases)))
    rng_cv.shuffle(order)
    fold_of = {idx: i % folds for i, idx in enumerate(order)}
    print(f"\n{'-' * 60}\n{fold_of and folds} 折交叉验证（候选 vs 线上配置，NDCG@5）:")
    cv_wins = 0
    for fold in range(folds):
        fold_cases = [c for i, c in enumerate(cases) if fold_of[i] == fold]
        new_avg = evaluate(candidate, fold_cases)["ndcg@5"]
        old_avg = evaluate(baseline, fold_cases)["ndcg@5"]
        win = new_avg > old_avg
        cv_wins += int(win)
        print(f"  fold{fold}: 候选 {new_avg:.4f} vs 线上 {old_avg:.4f}  {'✅' if win else '❌'}")
    print(f"  候选配置在 {cv_wins}/{folds} 折上优于线上配置"
          f"{'  → 可以考虑采用' if cv_wins >= 3 else '  → 不建议采用（提升不稳定）'}")

    report["per_query_delta_vs_baseline"] = deltas
    report["per_query_summary"] = {"better": better, "same": same, "worse": worse}
    report["cross_validation"] = {"folds": folds, "candidate_wins": cv_wins}

    # 按查询类型自适应加权：与固定权重调参分开评估（同样做留出 + 交叉验证）
    from agents.search_agent import ProductSearchAgent

    kind_count = {}
    for case in cases:
        kind = ProductSearchAgent.classify_query_type(case["query"])
        kind_count.setdefault(kind, []).append(case["query"])
    print(f"\n{'=' * 60}\n按查询类型自适应加权评估")
    print("  查询类型分布: " + " | ".join(f"{k} {len(v)} 个" for k, v in kind_count.items()))
    for kind, queries in kind_count.items():
        print(f"    {kind}: {'、'.join(queries)}")

    adaptive_rows = []
    for variant in ADAPTIVE_VARIANTS:
        full, by_kind = evaluate_adaptive(variant, cases)
        hold, _ = evaluate_adaptive(variant, holdout)
        adaptive_rows.append((variant, full, by_kind, hold))
    adaptive_rows.sort(key=lambda row: row[1]["ndcg@5"], reverse=True)

    print(f"\n  {'属性/规格型权重':<18}{'意图型权重':<14}{'全量NDCG@5':>11}"
          f"{'留出NDCG@5':>12}  {'分组(plain/structured/intent)':<30}")
    print("  " + "-" * 82)
    base_hold_ndcg = evaluate(baseline, holdout)["ndcg@5"]
    for variant, full, by_kind, hold in adaptive_rows:
        groups = "/".join(f"{by_kind.get(k, 0):.3f}" for k in ("plain", "structured", "intent"))
        print(f"  {str(variant[0]):<18}{str(variant[1]):<14}{full['ndcg@5']:>11.4f}"
              f"{hold['ndcg@5']:>12.4f}  {groups:<30}")
    print(f"  {'（线上固定等权基线）':<32}{baseline_metrics['ndcg@5']:>11.4f}"
          f"{base_hold_ndcg:>12.4f}")

    cv_summary = []
    for variant, full, by_kind, hold in adaptive_rows:
        wins = 0
        for fold in range(folds):
            fold_cases = [c for i, c in enumerate(cases) if fold_of[i] == fold]
            new_avg, _ = evaluate_adaptive(variant, fold_cases)
            old_avg = evaluate(baseline, fold_cases)["ndcg@5"]
            wins += int(new_avg["ndcg@5"] > old_avg)
        cv_summary.append({"variant": [list(variant[0]), list(variant[1])], "cv_wins": wins})
        print(f"  自适应 {variant} 在 {wins}/{folds} 折上优于基线")

    report["adaptive_weights"] = {
        "kind_distribution": {k: v for k, v in kind_count.items()},
        "variants": [
            {
                "structured_weights": list(variant[0]),
                "intent_weights": list(variant[1]),
                "full_metrics": full,
                "by_kind_ndcg@5": by_kind,
                "holdout_metrics": hold,
            }
            for variant, full, by_kind, hold in adaptive_rows
        ],
        "cross_validation": cv_summary,
        "baseline_holdout_ndcg@5": base_hold_ndcg,
    }
    out = os.path.join(PROJECT_ROOT, "tests", "rrf_tuning_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\n调参报告已保存: {out}")
    return report


if __name__ == "__main__":
    main()
