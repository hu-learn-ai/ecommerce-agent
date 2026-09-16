"""README 数字一致性检查 —— 防止"文档与产物脱节"复发。

背景：本项目出现过 README 里的指标与 `tests/*_report.json` 不一致的情况
（分类准确率写 95.97% 而产物是 99.63%、RAG 表停留在旧版本）。本脚本把
"README 里的关键数字"与"报告 JSON 里的真实值"逐条比对，不一致就退出码非 0，
可直接挂到 CI。

分类：
- **strict（严格）**：确定性指标（检索/分类/路由/端到端准确率）→ 不一致即失败
- **volatile（提示）**：LLM Judge 打分的指标（faithfulness 等）→ 每次重跑都会变，
  只提示不失败，避免 CI 误报

用法:
    python tests/check_readme_consistency.py

输出:
    tests/readme_consistency_report.json
"""

import json
import os
import sys

# Windows GBK 控制台下打印 ✅/❌ 会抛 UnicodeEncodeError（检查跑完才崩、报告不落盘）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 - 流不支持 reconfigure 时忽略
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(PROJECT_ROOT, "README.md")


def load_json(rel_path: str):
    path = os.path.join(PROJECT_ROOT, rel_path)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # noqa: BLE001
        return None


def get(data, path: list):
    """按路径取值，任一层缺失返回 None。"""
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int) and key < len(current):
            current = current[key]
        else:
            return None
    return current


# (说明, README 行匹配串, 报告文件, 取值路径, 展示格式, 是否严格)
CHECKS = [
    ("RAG hybrid NDCG@5", "RAG 检索（**人工标注 GT**",
     "tests/rag_evaluation_report.json", ["summary", "retrieval", "ndcg@5"], "f4", True),
    ("三策略 hybrid", "RAG 检索·三策略",
     "tests/rag_evaluation_report.json", ["retrieval_baseline", "summary", "hybrid", "ndcg@5"], "f4", True),
    ("三策略 vector_only", "RAG 检索·三策略",
     "tests/rag_evaluation_report.json", ["retrieval_baseline", "summary", "vector_only", "ndcg@5"], "f4", True),
    ("三策略 keyword_only", "RAG 检索·三策略",
     "tests/rag_evaluation_report.json", ["retrieval_baseline", "summary", "keyword_only", "ndcg@5"], "f4", True),
    ("BM25 NDCG@5", "RAG 检索·BM25 对照",
     "tests/bm25_baseline_report.json", ["summary", "bm25", "ndcg@5"], "f4", True),
    ("向量 NDCG@5（BM25 对照）", "RAG 检索·BM25 对照",
     "tests/bm25_baseline_report.json", ["summary", "vector", "ndcg@5"], "f4", True),
    ("分类同分布验证集 acc", "商品分类 | 验证集 Accuracy",
     "models/best/training_config.json", ["eval_result", "eval_accuracy"], "pct", True),
    ("分类同分布验证集 macro-F1", "商品分类 | 验证集 Accuracy",
     "models/best/training_config.json", ["eval_result", "eval_f1"], "pct", True),
    ("分类去重干净子集 acc", "商品分类（鲁棒性对照）",
     "tests/classify_robustness_report.json", ["groups", "val_clean", "accuracy"], "pct", True),
    ("分类无模型关键词基线", "商品分类（无模型对照）",
     "tests/classify_robustness_report.json", ["groups", "keyword_rule", "accuracy"], "pct", True),
    ("分类真实短查询 acc", "分类评估·真实短查询",
     "tests/classify_eval_report.json", ["by_group", "short_query", "classification", "accuracy"], "f4", True),
    ("分类合并 acc", "分类评估·合并",
     "tests/classify_eval_report.json", ["by_group", "all", "classification", "accuracy"], "f4", True),
    ("OOD 召回", "分类评估·合并",
     "tests/classify_eval_report.json", ["by_group", "all", "ood_detection", "ood_recall"], "f4", True),
    ("OOD 域内接受率", "分类评估·合并",
     "tests/classify_eval_report.json", ["by_group", "all", "ood_detection", "indomain_accept_rate"], "f4", True),
    ("OOD AUROC", "分类评估·合并",
     "tests/classify_eval_report.json", ["by_group", "all", "ood_detection", "auroc"], "f4", True),
    # 路由评估自 2026-09-16 起含 LLM 语义兜底调用（temperature=0.1），
    # 跨运行会有极小幅波动，因此只提示不判失败
    ("路由准确率", "路由（209 用例）",
     "tests/router_eval_report.json", ["metrics", "accuracy"], "f4", False),
    ("端到端意图准确率", "端到端（15 用例）",
     "tests/e2e_eval_report.json", ["summary", "accuracy", "overall"], "f4", True),
    ("KGQA 注入拦截率", "KG-QA（8 用例）",
     "tests/kgqa_eval_report.json", ["security", "pass_rate"], "f4", True),
    ("RAG 生成 faithfulness", "RAG 生成",
     "tests/rag_evaluation_report.json", ["summary", "generation", "faithfulness"], "f4", False),
    ("RAG 生成 answer_relevance", "RAG 生成",
     "tests/rag_evaluation_report.json", ["summary", "generation", "answer_relevance"], "f4", False),
    ("FAQ 检索 NDCG@5", "FAQ 客服检索（**人工标注 GT**",
     "tests/rag_evaluation_report.json", ["summary", "retrieval_cs", "ndcg@5"], "f4", True),
    ("FAQ 检索 MRR", "FAQ 客服检索（**人工标注 GT**",
     "tests/rag_evaluation_report.json", ["summary", "retrieval_cs", "mrr"], "f4", True),
    ("FAQ 检索 Recall@3", "FAQ 客服检索（**人工标注 GT**",
     "tests/rag_evaluation_report.json", ["summary", "retrieval_cs", "recall@3"], "f4", True),
    ("FAQ 检索 HitRate@5", "FAQ 客服检索（**人工标注 GT**",
     "tests/rag_evaluation_report.json", ["summary", "retrieval_cs", "hit_rate@5"], "f4", True),
    ("推荐 NDCG@5（生产默认，无重排）", "推荐排序（**人工标注 GT**",
     "tests/recommendation_eval_report.json",
     ["baseline_comparison", "graph+query_aware", "ranking", "ndcg@5"], "f4", True),
    ("推荐 HitRate@5（生产默认，无重排）", "推荐排序（**人工标注 GT**",
     "tests/recommendation_eval_report.json",
     ["baseline_comparison", "graph+query_aware", "ranking", "hit_rate@5"], "f4", True),
    # LLM 重排是可选增强且跨运行波动，只提示不判失败（volatile）
    ("推荐 NDCG@5（可选 LLM 重排）", "推荐排序·可选 LLM 重排",
     "tests/recommendation_eval_report.json",
     ["baseline_comparison", "graph+query_aware+llm_rerank", "ranking", "ndcg@5"], "f4", False),
    ("自适应加权全量 NDCG@5", "自适应 (0.5,1.5)",
     "tests/rrf_tuning_report.json", ["adaptive_weights", "variants", 0, "full_metrics", "ndcg@5"], "f4", True),
]


def format_value(value, style: str) -> str:
    if style == "pct":
        return f"{float(value) * 100:.2f}%"
    return f"{float(value):.4f}"


def main() -> int:
    if not os.path.exists(README):
        print("未找到 README.md")
        return 1
    lines = open(README, encoding="utf-8").read().splitlines()

    results, failures, warnings = [], [], []
    for name, marker, report_path, path, style, strict in CHECKS:
        data = load_json(report_path)
        if data is None:
            results.append({"name": name, "status": "report_missing", "report": report_path})
            (failures if strict else warnings).append(f"{name}: 报告缺失 {report_path}")
            continue
        value = get(data, path)
        if value is None:
            results.append({"name": name, "status": "metric_missing", "path": path, "report": report_path})
            (failures if strict else warnings).append(f"{name}: 报告中找不到 {path}")
            continue
        expected = format_value(value, style)
        matched = [line for line in lines if marker in line]
        found = any(expected in line for line in matched)
        results.append({
            "name": name,
            "status": "ok" if found else ("mismatch" if strict else "volatile_diff"),
            "expected": expected,
            "marker": marker,
            "matched_rows": len(matched),
            "report": report_path,
        })
        if not found and not matched:
            (failures if strict else warnings).append(f"{name}: README 中找不到匹配行「{marker}」")
        elif not found:
            message = f"{name}: README 未包含报告值 {expected}（行「{marker}」）"
            (failures if strict else warnings).append(message)

    print("=" * 66)
    print("  README 数字一致性检查")
    print("=" * 66)
    for item in results:
        status = item["status"]
        icon = {"ok": "✅", "mismatch": "❌", "volatile_diff": "⚠️",
                "report_missing": "❌", "metric_missing": "❌"}.get(status, "?")
        detail = item.get("expected", item.get("report", ""))
        print(f"  {icon} {item['name']:<24}{detail}")

    if warnings:
        print("\n  提示（LLM Judge 指标，每次重跑会变，不判失败）:")
        for message in warnings:
            print(f"    ⚠️ {message}")
    if failures:
        print("\n  不一致（需修正 README 或重跑评估）:")
        for message in failures:
            print(f"    ❌ {message}")

    report = {"checks": results, "failures": failures, "warnings": warnings}
    out = os.path.join(PROJECT_ROOT, "tests", "readme_consistency_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out}")
    print(f"  通过 {sum(1 for r in results if r['status'] == 'ok')}/{len(results)}，"
          f"不一致 {len(failures)}，提示 {len(warnings)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
