"""
人工抽检 + Cohen kappa 一致性计算

对应 docs/微调项目执行计划.md 4.6：双盲抽 50-100 条，人工评分与 Judge 计算一致率（Cohen kappa），
最终结论以人工为准。

用法：
    # 1. 抽样生成人工评分表（CSV，四个维度留空待填）
    python finetune/scripts/human_eval.py --sample --count 50

    # 2. 人工填完 CSV 后计算与 Judge 的 kappa
    python finetune/scripts/human_eval.py --kappa --human-csv finetune/reports/human_scores.csv

输出：
    finetune/reports/human_scores.csv           # 人工评分表
    finetune/reports/human_eval_report.json     # kappa / 一致率
"""

import argparse
import csv
import json
import os
import random
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_dataset import load_jsonl

DIMS = ["correctness", "completeness", "tone", "safety"]


def cohen_kappa(y1, y2):
    """
    Cohen's kappa（类别型，多分类 1-5 分）
    kappa = (P_observed - P_expected) / (1 - P_expected)
    """
    labels = sorted(set(y1) | set(y2))
    n = len(y1)
    if n == 0 or len(labels) == 1:
        return 1.0 if n else None
    table = {(a, b): 0 for a in labels for b in labels}
    for a, b in zip(y1, y2):
        table[(a, b)] += 1
    p_obs = sum(table[(a, a)] for a in labels) / n
    p_exp = sum(
        (sum(table[(a, b)] for b in labels) / n) * (sum(table[(b, a)] for b in labels) / n)
        for a in labels
    )
    if p_exp >= 1:
        return 1.0
    return (p_obs - p_exp) / (1 - p_exp)


def sample_human_csv(args):
    records = load_jsonl(args.test)
    if args.count and len(records) > args.count:
        records = random.Random(args.seed).sample(records, args.count)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "human_scores.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "category", "question", "answer"] + DIMS + ["note"])
        for rec in records:
            q = rec["conversation"][0]["content"]
            a = rec["conversation"][-1]["content"]
            writer.writerow([rec["id"], rec["category"], q, a, "", "", "", "", ""])
    print(f"[HumanEval] 抽样 {len(records)} 条，评分表已生成: {csv_path}")
    print("[HumanEval] 请人工按 1-5 分填写四个维度后运行 --kappa")


def kappa_report(args):
    import evaluate_llm

    human_rows = []
    with open(args.human_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            scores = [row.get(d, "").strip() for d in DIMS]
            if all(s and s not in ("-", "") for s in scores):
                human_rows.append(
                    {"id": row["id"], "scores": {d: int(s) for d, s in zip(DIMS, scores)}}
                )
    if not human_rows:
        raise SystemExit("人工评分表没有有效评分，请先填写后重试")

    judge_path = os.path.join(args.output_dir, "judge", "finetuned.jsonl")
    if not os.path.exists(judge_path):
        raise SystemExit(f"未找到 Judge 结果: {judge_path}，请先运行 evaluate_llm.py")
    judge_rows = {r["id"]: r for r in load_jsonl(judge_path)}

    per_dim = {d: {"human": [], "judge": []} for d in DIMS}
    missing = 0
    for h in human_rows:
        j = judge_rows.get(h["id"])
        if not j:
            missing += 1
            continue
        for d in DIMS:
            js = (j.get("scores") or {}).get(d, {}).get("score")
            if isinstance(js, (int, float)) and h["scores"][d] in (1, 2, 3, 4, 5):
                per_dim[d]["human"].append(h["scores"][d])
                per_dim[d]["judge"].append(round(js))

    result = {"human_samples": len(human_rows), "missing_judge": missing, "per_dim": {}}
    for d in DIMS:
        y1, y2 = per_dim[d]["human"], per_dim[d]["judge"]
        kappa = cohen_kappa(y1, y2)
        agreement = round(sum(1 for a, b in zip(y1, y2) if a == b) / len(y1), 4) if y1 else None
        result["per_dim"][d] = {"kappa": round(kappa, 4) if kappa is not None else None, "agreement": agreement, "n": len(y1)}

    out_path = os.path.join(args.output_dir, "human_eval_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[HumanEval] 结果已写入: {out_path}")
    for d in DIMS:
        info = result["per_dim"][d]
        print(f"[HumanEval] {d}: kappa={info['kappa']} agreement={info['agreement']} n={info['n']}")
    print("[HumanEval] 最终结论以人工为准（kappa >= 0.6 视为 Judge 与人工一致性可接受）")


def main():
    parser = argparse.ArgumentParser(description="人工抽检与 kappa")
    parser.add_argument("--test", default=os.path.join(PROJECT_ROOT, "data", "processed", "test.jsonl"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "reports"))
    parser.add_argument("--sample", action="store_true", help="抽样生成人工评分表")
    parser.add_argument("--count", type=int, default=50, help="抽样条数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kappa", action="store_true", help="计算 kappa")
    parser.add_argument("--human-csv", default=os.path.join(PROJECT_ROOT, "reports", "human_scores.csv"))
    args = parser.parse_args()

    if args.kappa:
        kappa_report(args)
    elif args.sample:
        sample_human_csv(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
