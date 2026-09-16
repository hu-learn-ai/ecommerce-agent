"""
BM25 Baseline 对比评估 — 关键词检索 vs 向量语义检索

目的:
    补齐 RAG 检索质量的 baseline 对比（算法岗面试最看重的相对提升数据）。
    用中立的关键词匹配 ground truth（对两种方法公平），对比:
    1. bm25   : 传统关键词检索（jieba 分词 + BM25Okapi）
    2. vector : 现有 FAISS + BGE 向量语义检索

使用方法:
    python tests/bm25_baseline.py

输出:
    tests/bm25_baseline_report.json — 对比明细与汇总
    tests/bm25_baseline_report.md  — 可读对比报告
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root, require_assets

ensure_project_root()

import jieba

from tests.rag_evaluation import SEARCH_TEST_CASES, RetrievalMetrics, load_labeled_gt

# ------------------------------------------------------------------ #
#  BM25 检索器
# ------------------------------------------------------------------ #


class BM25Retriever:
    """基于 jieba 分词 + BM25Okapi 的中文关键词检索器"""

    def __init__(self, products: list):
        self.products = products
        self.idx_by_id = {str(p["id"]): p for p in products}
        # 检索文本: 名称 + 分类 + 品牌 + 描述（描述中含价格/销量等干扰信息，权重降低）
        self.corpus = [
            f"{p.get('name', '')} {p.get('category', '')} {p.get('brand', '')}" for p in products
        ]
        self.tokenized = [list(jieba.cut(doc)) for doc in self.corpus]

        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self.tokenized)

    def search(self, query: str, top_k: int) -> list:
        """返回 [{id, name, category, score}, ...]"""
        tokens = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for rank, idx in enumerate(ranked[:top_k]):
            if scores[idx] <= 0:
                continue  # 无任何词命中，跳过
            p = self.products[idx]
            results.append(
                {
                    "id": str(p["id"]),
                    "name": p.get("name", ""),
                    "category": p.get("category", ""),
                    "score": float(scores[idx]),
                    "rank": rank,
                }
            )
        return results


# ------------------------------------------------------------------ #
#  中立 Ground Truth
# ------------------------------------------------------------------ #


def build_neutral_ground_truth(products: list, relevant_keywords: list) -> set:
    """
    构建对 vector / bm25 都公平的 ground truth:
    商品名或分类包含任一查询关键词 → 相关。
    不依赖任何检索方法的结果，避免向某一方倾斜。
    """
    gt = set()
    for p in products:
        name = (p.get("name", "") or "").lower()
        category = (p.get("category", "") or "").lower()
        text = f"{name} {category}"
        if any(kw.lower() in text for kw in relevant_keywords):
            gt.add(str(p["id"]))
    return gt


# ------------------------------------------------------------------ #
#  评估主体
# ------------------------------------------------------------------ #


def evaluate() -> dict:
    print("=" * 60)
    print("  BM25 Baseline vs 向量检索 对比评估")
    print("=" * 60)

    # 1. 加载商品数据
    with open(
        os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json"),
        "r",
        encoding="utf-8",
    ) as f:
        products = json.load(f)
    print(f"商品数: {len(products)}")

    # 2. 初始化 BM25
    print("构建 BM25 索引 (jieba 分词)...")
    bm25_retriever = BM25Retriever(products)

    # 3. 初始化向量检索（复用 search_agent）
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from agents.search_agent import ProductSearchAgent
    from config.settings import settings

    require_assets(
        os.path.join(settings.faiss_index_path, "products.index"),
        os.path.join(settings.faiss_index_path, "product_ids.npy"),
    )

    search_agent = ProductSearchAgent(
        embedding_model_name=settings.embedding_model,
        neo4j_driver=None,
        faiss_index_path=settings.faiss_index_path,
    )

    # 4. 逐用例评估（GT 优先用人工标注，缺失时回落关键词匹配）
    labeled_gt = load_labeled_gt()
    human_gt_cases = 0
    results = []
    for case in SEARCH_TEST_CASES:
        query = case["query"]
        keywords = case["relevant_keywords"]
        if query in labeled_gt:
            gt = {str(i) for i in labeled_gt[query]}
            gt_source = "human"
            human_gt_cases += 1
        else:
            gt = build_neutral_ground_truth(products, keywords)
            gt_source = "keyword"

        # BM25 检索
        bm25_hits = bm25_retriever.search(query, 10)
        bm25_ids = [r["id"] for r in bm25_hits]

        # 向量检索
        vec_hits = search_agent._vector_search(query, 10)
        vec_ids = [r["id"] for r in vec_hits]

        # 指标
        bm25_metrics = RetrievalMetrics.compute_all(bm25_ids, list(gt))
        vec_metrics = RetrievalMetrics.compute_all(vec_ids, list(gt))

        results.append(
            {
                "query": query,
                "ground_truth_size": len(gt),
                "gt_source": gt_source,
                "bm25": {"ids": bm25_ids, "metrics": bm25_metrics},
                "vector": {"ids": vec_ids, "metrics": vec_metrics},
            }
        )

        print(
            f"  [{query}] gt={len(gt):3d} | "
            f"bm25 ndcg@5={bm25_metrics['ndcg@5']:.3f} vs "
            f"vec  ndcg@5={vec_metrics['ndcg@5']:.3f}"
        )

    # 5. 汇总（只统计 ground truth 非空的用例；gt=0 说明合成商品名中
    #    不含该细粒度品类词，无法建立中立相关集，与检索方法优劣无关）
    valid_results = [r for r in results if r["ground_truth_size"] > 0]
    skipped = len(results) - len(valid_results)
    if skipped:
        skipped_queries = [r["query"] for r in results if r["ground_truth_size"] == 0]
        print(f"  跳过（人工标注为商品库内无相关商品）: {'、'.join(skipped_queries)}")

    metrics_keys = [
        "recall@1",
        "recall@3",
        "recall@5",
        "precision@5",
        "mrr",
        "ndcg@5",
        "hit_rate@5",
    ]
    summary = {
        "bm25": {},
        "vector": {},
        "delta": {},
        "valid_cases": len(valid_results),
        "skipped_cases": skipped,
    }
    for key in metrics_keys:
        bm25_avg = sum(r["bm25"]["metrics"][key] for r in valid_results) / len(valid_results)
        vec_avg = sum(r["vector"]["metrics"][key] for r in valid_results) / len(valid_results)
        summary["bm25"][key] = round(bm25_avg, 4)
        summary["vector"][key] = round(vec_avg, 4)
        summary["delta"][key] = round(vec_avg - bm25_avg, 4)

    # 按 GT 来源分组：人工标注才是无偏口径，关键词 GT 与它不能混在一起解读
    summary["by_gt_source"] = {}
    for source in ("human", "keyword"):
        group = [r for r in valid_results if r.get("gt_source") == source]
        if not group:
            continue
        summary["by_gt_source"][source] = {"cases": len(group)}
        for key in metrics_keys:
            summary["by_gt_source"][source][f"bm25_{key}"] = round(
                sum(r["bm25"]["metrics"][key] for r in group) / len(group), 4
            )
            summary["by_gt_source"][source][f"vector_{key}"] = round(
                sum(r["vector"]["metrics"][key] for r in group) / len(group), 4
            )

    # 6. 输出汇总表
    print(f"\n{'=' * 60}")
    print(f"  汇总对比 ({len(valid_results)} 个有效用例, 跳过 {skipped} 个 gt=0 用例)")
    print(f"{'=' * 60}")
    print(f"  {'指标':<14}{'BM25':<12}{'向量检索':<12}{'Δ (向量-BM25)':<12}")
    print("  " + "-" * 50)
    for key in metrics_keys:
        print(
            f"  {key:<14}{summary['bm25'][key]:<12.4f}"
            f"{summary['vector'][key]:<12.4f}{summary['delta'][key]:<12.4f}"
        )

    if summary.get("by_gt_source"):
        print(f"\n{'=' * 60}")
        print("  按 GT 来源拆分（人工标注 = 无偏口径；关键词 GT 偏向 BM25）")
        print(f"{'=' * 60}")
        for source, data in summary["by_gt_source"].items():
            label = "人工标注" if source == "human" else "关键词匹配"
            print(f"  [{label}] {data['cases']} 个用例")
            for key in ("ndcg@5", "mrr", "precision@5", "hit_rate@5"):
                print(
                    f"    {key:<12} bm25={data[f'bm25_{key}']:.4f}  "
                    f"vector={data[f'vector_{key}']:.4f}"
                )

    # 7. 保存报告
    report = {"summary": summary, "results": results}
    report["summary"]["human_gt_cases"] = human_gt_cases
    json_path = os.path.join(PROJECT_ROOT, "tests", "bm25_baseline_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n对比报告已保存: {json_path}")

    # 8. 生成 Markdown 报告
    md_path = os.path.join(PROJECT_ROOT, "tests", "bm25_baseline_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# BM25 Baseline vs 向量检索 对比报告\n\n")
        f.write(
            f"> 有效用例 {len(valid_results)} 个（跳过 {skipped} 个 ground truth 为空的用例："
            "合成商品名不含该细粒度品类词，无法建立中立相关集）\n\n"
        )
        f.write("| 指标 | BM25 | 向量检索 | Δ (向量-BM25) | 相对提升 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for key in metrics_keys:
            delta = summary["delta"][key]
            rel = f"{delta / summary['bm25'][key] * 100:+.1f}%" if summary["bm25"][key] > 0 else "—"
            f.write(
                f"| {key} | {summary['bm25'][key]:.4f} | "
                f"{summary['vector'][key]:.4f} | "
                f"{delta:+.4f} | {rel} |\n"
            )
        if human_gt_cases:
            f.write(
                f"\n> Ground truth 为**人工标注**（{human_gt_cases}/{len(SEARCH_TEST_CASES)} 个用例，"
                "来源 `tests/retrieval_gt_labeled.json`），其余用例回落为关键词匹配。\n"
            )
        if summary.get("by_gt_source"):
            f.write("\n### 按 GT 来源拆分（人工标注 = 无偏口径）\n\n")
            f.write("| GT 来源 | 用例数 | 指标 | BM25 | 向量检索 |\n|---|---:|---|---:|---:|\n")
            for source, data in summary["by_gt_source"].items():
                label = "人工标注" if source == "human" else "关键词匹配"
                for key in ("ndcg@5", "mrr", "precision@5", "hit_rate@5"):
                    f.write(
                        f"| {label} | {data['cases']} | {key} | "
                        f"{data[f'bm25_{key}']:.4f} | {data[f'vector_{key}']:.4f} |\n"
                    )
        else:
            f.write(
                "\n> Ground truth 为关键词匹配（商品名/分类包含查询关键词）。"
                "注意：该口径天然偏向 BM25，人工标注后重跑可消除此偏置"
                "（见 `python tests/build_retrieval_gt.py`）。\n"
            )
    print(f"Markdown 报告已保存: {md_path}")

    return report


if __name__ == "__main__":
    evaluate()
