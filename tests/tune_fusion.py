"""融合策略对比：RRF / RRF+共识折扣 / 分数归一化线性融合。

背景：在 36 个标注查询上发现 5 个用例"融合后低于单路最优"（牛奶 hybrid 0.316 vs 向量 1.000、
运动时戴的耳机 0.384 vs 0.723 等），原因是**单路失效时它的候选仍会挤占融合结果**。
本脚本对比三类融合，并用留出集 + 4 折交叉验证判断改动是否真的可泛化。

融合策略：
1. `rrf`：现行方案 score = w_v/(k+rank_v) + w_k/(k+rank_k)
2. `rrf_consensus`：在 1 的基础上，**只被一路召回的候选**乘以折扣 s（共识候选相对上升）
3. `score_norm`：两路分数各自 min-max 归一化后加权求和（单路缺失记 0 分）

用法:
    python tests/tune_fusion.py

输出:
    tests/fusion_tuning_report.json
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

DEPTH = 20
K = 60  # RRF 常数（此前网格搜索显示不敏感）


def parse_args():
    parser = argparse.ArgumentParser(description="融合策略对比")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--holdout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _weights(kind, policy):
    """按查询类型给两路权重。"""
    if policy == "equal":
        return (1.0, 1.0)
    if kind == "structured":
        return (0.5, 1.5)
    if kind == "intent":
        return (1.5, 0.5)
    return (1.0, 1.0)


def fuse_rrf(vec, kw, w_vec, w_kw, discount=1.0):
    """RRF；discount<1 时对"只被一路召回"的候选打折（共识感知）。"""
    scores = {}
    in_vec = {d["id"] for d in vec}
    in_kw = {d["id"] for d in kw}
    for rank, doc in enumerate(vec, 1):
        doc_id = doc["id"]
        factor = 1.0 if doc_id in in_kw else discount
        scores[doc_id] = scores.get(doc_id, 0.0) + w_vec * factor / (K + rank)
    for rank, doc in enumerate(kw, 1):
        doc_id = doc["id"]
        factor = 1.0 if doc_id in in_vec else discount
        scores[doc_id] = scores.get(doc_id, 0.0) + w_kw * factor / (K + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def _normalize(items):
    """把一路结果按分数 min-max 归一化到 [0,1]（全同分时记 1.0）。"""
    if not items:
        return {}
    values = [float(d.get("score", 0.0)) for d in items]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return {d["id"]: 1.0 for d in items}
    return {d["id"]: (float(d.get("score", 0.0)) - lo) / (hi - lo) for d in items}


def fuse_score(vec, kw, w_vec, w_kw):
    """分数归一化线性融合。"""
    nv, nk = _normalize(vec), _normalize(kw)
    scores = {}
    for doc_id, value in nv.items():
        scores[doc_id] = scores.get(doc_id, 0.0) + w_vec * value
    for doc_id, value in nk.items():
        scores[doc_id] = scores.get(doc_id, 0.0) + w_kw * value
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def rank_for(case, config):
    """按配置产出排序结果。"""
    method, discount, policy = config
    kind = case["kind"]
    w_vec, w_kw = _weights(kind, policy)
    if method == "rrf":
        return fuse_rrf(case["vec"], case["kw"], w_vec, w_kw, discount)
    return fuse_score(case["vec"], case["kw"], w_vec, w_kw)


def evaluate(config, cases, top_k=10):
    totals = {m: 0.0 for m in ("ndcg@5", "mrr", "precision@5", "hit_rate@5", "recall@5")}
    for case in cases:
        metrics = RetrievalMetrics.compute_all(rank_for(case, config)[:top_k], case["gt"])
        for name in totals:
            totals[name] += metrics[name]
    n = max(1, len(cases))
    return {name: value / n for name, value in totals.items()}


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
        gt = [str(i) for i in labeled.get(query, [])]
        if not gt:  # 人工标注为空集（库内无对应商品）→ 不参与指标
            continue
        cases.append(
            {
                "query": query,
                "gt": gt,
                "kind": ProductSearchAgent.classify_query_type(query),
                "vec": [
                    {"id": str(r["id"]), "score": float(r.get("score", 0.0))}
                    for r in agent._vector_search(query, DEPTH)
                ],
                "kw": [
                    {"id": str(r["id"]), "score": float(r.get("score", 0.0))}
                    for r in agent.keyword_search(query, DEPTH)
                ],
            }
        )
    print(f"参与对比的查询: {len(cases)} 个（已排除人工标注为空集的查询）")

    configs = [("rrf", 1.0, "adaptive")]  # 当前线上
    for discount in (0.85, 0.7, 0.5, 0.3):
        configs.append(("rrf", discount, "adaptive"))
        configs.append(("rrf", discount, "equal"))
    for policy in ("adaptive", "equal"):
        configs.append(("score", 1.0, policy))

    results = [(cfg, evaluate(cfg, cases)) for cfg in configs]
    results.sort(key=lambda item: item[1]["ndcg@5"], reverse=True)

    print(f"\n{'融合方式':<16}{'共识折扣':>9}{'权重策略':>10}{'NDCG@5':>10}{'MRR':>8}{'P@5':>8}{'Hit@5':>8}")
    print("-" * 72)
    for cfg, metrics in results[: args.top]:
        method, discount, policy = cfg
        name = {"rrf": "RRF", "score": "分数归一化"}[method]
        disc = f"{discount:.2f}" if method == "rrf" else "—"
        print(f"{name:<16}{disc:>9}{policy:>10}{metrics['ndcg@5']:>10.4f}"
              f"{metrics['mrr']:>8.4f}{metrics['precision@5']:>8.4f}{metrics['hit_rate@5']:>8.4f}")

    rng = random.Random(args.seed)
    order = list(range(len(cases)))
    rng.shuffle(order)
    holdout_idx = set(order[: args.holdout])
    train = [c for i, c in enumerate(cases) if i not in holdout_idx]
    holdout = [c for i, c in enumerate(cases) if i in holdout_idx]
    baseline = ("rrf", 1.0, "adaptive")

    print(f"\n留出验证（{len(train)} 训练 / {len(holdout)} 留出）:")
    print(f"  {'配置':<34}{'全量':>9}{'训练':>9}{'留出':>9}{'折数胜出':>10}")
    folds = 4
    cv_order = list(range(len(cases)))
    random.Random(args.seed + 1).shuffle(cv_order)
    fold_of = {idx: i % folds for i, idx in enumerate(cv_order)}
    rows = []
    for cfg, metrics in results:
        hold = evaluate(cfg, holdout)["ndcg@5"]
        train_ndcg = evaluate(cfg, train)["ndcg@5"]
        wins = 0
        for fold in range(folds):
            fold_cases = [c for i, c in enumerate(cases) if fold_of[i] == fold]
            wins += int(evaluate(cfg, fold_cases)["ndcg@5"] > evaluate(baseline, fold_cases)["ndcg@5"])
        rows.append({"config": list(cfg), "full": metrics["ndcg@5"], "train": train_ndcg,
                     "holdout": hold, "cv_wins": wins})
        method, discount, policy = cfg
        label = f"{method} s={discount:.2f} {policy}"
        print(f"  {label:<34}{metrics['ndcg@5']:>9.4f}{train_ndcg:>9.4f}{hold:>9.4f}{wins}/{folds:>8}")

    base_hold = evaluate(baseline, holdout)["ndcg@5"]
    print(f"\n  线上配置（RRF s=1.00 adaptive）留出集 NDCG@5 = {base_hold:.4f}")
    winners = [r for r in rows if r["holdout"] > base_hold + 1e-9 and r["cv_wins"] >= 3]
    print(f"  同时满足「留出集提升 + 交叉验证 ≥3/4 折」的候选: "
          f"{[r['config'] for r in winners] or '无 → 不建议改动'}")

    report = {
        "cases": len(cases),
        "depth": DEPTH,
        "rrf_k": K,
        "baseline": {"config": list(baseline), "metrics": evaluate(baseline, cases)},
        "ranking": rows,
        "holdout_queries": [c["query"] for c in holdout],
        "validated_candidates": [r["config"] for r in winners],
    }
    out = os.path.join(PROJECT_ROOT, "tests", "fusion_tuning_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\n对比报告已保存: {out}")
    return report


if __name__ == "__main__":
    main()
