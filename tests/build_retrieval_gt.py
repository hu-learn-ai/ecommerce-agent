"""构建检索评估的人工标注 Ground Truth（TREC pooling 流程）。

背景：现有检索 GT 有两套偏置 —— `rag_evaluation.py` 把向量 Top-3 直接塞进 GT（自举，
MRR 恒为 1.000），`bm25_baseline.py` 用关键词匹配（天然偏向 BM25）。两者都无法给出
可信的绝对数值。

本脚本产出 pooling 候选池，交给人工判定相关性，再回填为 `retrieval_gt_labeled.json`：
    1. 对每个搜索查询，取 BM25 Top-10 ∪ 向量 Top-10 作为候选池
    2. 输出候选池（含商品名/分类，便于人工判读）
    3. 人工在 JSON 的 "relevant" 字段填写 1/0（或只保留相关项）
    4. `rag_evaluation.py` / `bm25_baseline.py` 检测到该文件后自动改用人工 GT

用法:
    python tests/build_retrieval_gt.py            # 生成候选池（已存在则不覆盖）
    python tests/build_retrieval_gt.py --force    # 强制重新生成

输出:
    tests/retrieval_gt_candidates.json  — 候选池（可编辑，填 relevant 字段）
    tests/retrieval_gt_candidates.md    — 同内容的可读版，便于人工评审
"""

import argparse
import json
import os
import re
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root, require_assets  # noqa: E402

ensure_project_root()

import jieba  # noqa: E402

from config.settings import settings  # noqa: E402
from tests._faq_corpus import (  # noqa: E402
    FAQ_LABELED_PATH,
    load_faq_items,  # noqa: E402
)
from tests._faq_corpus import id_by_text as faq_id_by_text
from tests.rag_evaluation import CS_TEST_CASES, SEARCH_TEST_CASES  # noqa: E402

POOL_K = 10
CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "tests", "retrieval_gt_candidates.json")
MARKDOWN_PATH = os.path.join(PROJECT_ROOT, "tests", "retrieval_gt_candidates.md")
LABELED_PATH = os.path.join(PROJECT_ROOT, "tests", "retrieval_gt_labeled.json")
FAQ_CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "tests", "retrieval_gt_candidates_faq.json")
FAQ_MARKDOWN_PATH = os.path.join(PROJECT_ROOT, "tests", "retrieval_gt_candidates_faq.md")

# Markdown 评审格式：
#   ## 查询名（候选 N 条）
#   - [ ] `JD001565` [手机数码] 商品名（来源 bm25+vector）
# 约定（方框内允许空格，兼容 [ 1] / [1 ] / [ x ] 等对齐写法）：
#   相关   : [x] [X] [1] [√] [是] [y]
#   不相关 : [0] [否] [n]
#   未标注 : [ ]（留空）
MD_LINE_RE = re.compile(r"^\s*-\s*\[([^\]]*)\]\s*`([^`]+)`")
MD_HEADER_RE = re.compile(r"^##\s+(.+?)(?:（候选.*）)?\s*$")
RELEVANT_FLAGS = {"x", "1", "√", "是", "y"}
IRRELEVANT_FLAGS = {"0", "否", "n"}


def import_from_markdown(md_path: str) -> dict:
    """解析 Markdown 里的勾选，生成人工标注 GT。

    规则：某个查询只要有 ≥1 处显式标注（[x]/[1]/[0]），即视为已标注，
    未标注的候选按"不相关"处理；完全没有标注的查询会被跳过（回落到原伪标注）。
    """
    relevant: dict = {}
    marked_count: dict = {}
    pool_count: dict = {}
    current = None

    with open(md_path, encoding="utf-8") as handle:
        for line in handle:
            header = MD_HEADER_RE.match(line)
            if header:
                current = header.group(1).strip()
                relevant.setdefault(current, [])
                marked_count.setdefault(current, 0)
                pool_count.setdefault(current, 0)
                continue
            match = MD_LINE_RE.match(line)
            if not match or current is None:
                continue
            flag = match.group(1).strip().lower()
            product_id = match.group(2).strip()
            pool_count[current] += 1
            if flag in RELEVANT_FLAGS:
                relevant[current].append(product_id)
                marked_count[current] += 1
            elif flag in IRRELEVANT_FLAGS:
                marked_count[current] += 1
            # 其它（含空方框）= 未标注

    labeled = {q: ids for q, ids in relevant.items() if marked_count.get(q)}
    print(f"解析完成: {len(labeled)}/{len(pool_count)} 个查询已标注")
    for query, ids in labeled.items():
        print(
            f"  {query:<16} 相关 {len(ids):>2} / 候选 {pool_count[query]:>2} "
            f"（标注 {marked_count[query]:>2} 条）"
        )
    unlabeled = [q for q in pool_count if not marked_count.get(q)]
    if unlabeled:
        print(f"\n  ⚠️ 未标注（将回落为伪标注 GT）: {', '.join(unlabeled)}")
    return labeled


def main() -> dict:
    parser = argparse.ArgumentParser(description="构建检索评估 pooling 候选池")
    parser.add_argument("--force", action="store_true", help="已存在时强制覆盖")
    parser.add_argument(
        "--import-md",
        action="store_true",
        help="解析 tests/retrieval_gt_candidates.md 的勾选，生成 retrieval_gt_labeled.json",
    )
    parser.add_argument("--md", type=str, default=MARKDOWN_PATH, help="待解析的 Markdown 路径")
    parser.add_argument(
        "--faq-md",
        type=str,
        default=FAQ_MARKDOWN_PATH,
        help="待解析的客服 FAQ Markdown 路径",
    )
    args = parser.parse_args()

    if args.import_md:
        if not os.path.exists(args.md):
            raise SystemExit(f"未找到 Markdown: {args.md}")
        labeled = import_from_markdown(args.md)
        if not labeled:
            raise SystemExit("没有任何查询被标注，未生成文件（请先在 md 里勾选 [x]）")
        with open(LABELED_PATH, "w", encoding="utf-8") as handle:
            json.dump(labeled, handle, ensure_ascii=False, indent=2)
        print(f"\n人工标注 GT 已生成: {LABELED_PATH}")
        # 客服 FAQ 的标注（若存在对应 Markdown）
        if os.path.exists(args.faq_md):
            faq_labeled = import_from_markdown(args.faq_md)
            if faq_labeled:
                with open(FAQ_LABELED_PATH, "w", encoding="utf-8") as handle:
                    json.dump(faq_labeled, handle, ensure_ascii=False, indent=2)
                print(f"客服 FAQ 人工标注 GT 已生成: {FAQ_LABELED_PATH}")
        print("下一步重跑评估（两个脚本会自动改用人工 GT）:")
        print("  python tests/rag_evaluation.py")
        print("  python tests/bm25_baseline.py")
        return labeled

    if os.path.exists(CANDIDATES_PATH) and not args.force:
        print(f"候选池已存在，跳过（--force 覆盖）: {CANDIDATES_PATH}")
        return json.load(open(CANDIDATES_PATH, encoding="utf-8"))

    require_assets(
        os.path.join(settings.faiss_index_path, "products.index"),
        os.path.join(settings.faiss_index_path, "product_ids.npy"),
    )
    products = json.load(
        open(
            os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json"),
            encoding="utf-8",
        )
    )
    by_id = {str(p["id"]): p for p in products}

    # BM25 检索器（与 bm25_baseline.py 同口径）
    from rank_bm25 import BM25Okapi

    corpus = [
        f"{p.get('name', '')} {p.get('category', '')} {p.get('brand', '')}" for p in products
    ]
    tokenized = [list(jieba.cut(doc)) for doc in corpus]
    bm25 = BM25Okapi(tokenized)

    # 向量检索（复用生产链路）
    from agents.search_agent import ProductSearchAgent

    search_agent = ProductSearchAgent(
        embedding_model_name=settings.embedding_model,
        neo4j_driver=None,
        faiss_index_path=settings.faiss_index_path,
    )

    pools = []
    md_lines = ["# 检索 GT 候选池（人工标注用）", ""]
    md_lines.append("> 标注方法：把**相关**的商品前面的方框从 `[ ]` 改成 `[x]`（`[1]` 同义），")
    md_lines.append("> 不相关的留空即可（也可显式写 `[0]`）。某个查询只要有一处标注就算已标注。")
    md_lines.append("> 标完在项目根目录执行：")
    md_lines.append("> ```bash")
    md_lines.append("> python tests/build_retrieval_gt.py --import-md")
    md_lines.append("> ```")
    md_lines.append("> 会自动生成 `tests/retrieval_gt_labeled.json`，两个评估脚本随即改用人工 GT。")
    md_lines.append(">")
    md_lines.append("> 注意：已标注过的查询会按上一轮结果**预填**（`[x]` 相关 / `[0]` 不相关），")
    md_lines.append("> 你只需标注仍为 `[ ]` 的查询块。")
    md_lines.append("")

    # 已标注结果：用于预填，避免重复人工标注
    prev_labeled = {}
    if os.path.exists(LABELED_PATH):
        try:
            prev_labeled = json.load(open(LABELED_PATH, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"读取既有标注失败（将不预填）: {exc}")

    coverage_rows = []
    for case in SEARCH_TEST_CASES:
        query = case["query"]
        expected_category = case.get("description", "").split("·")[0].strip() or "未指定"
        constraint = case.get("constraint", "")
        # 与生产链路一致：解析查询里的预算约束；价格未知的商品视为不满足约束
        price_range = search_agent._parse_price_constraints(query)
        min_price, max_price = price_range if price_range else (None, None)

        def price_ok(product_id: str) -> bool:
            if min_price is None and max_price is None:
                return True
            raw = by_id.get(product_id, {}).get("price")
            if raw in (None, ""):
                return False
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return False
            if min_price is not None and value < min_price:
                return False
            if max_price is not None and value > max_price:
                return False
            return True

        scores = bm25.get_scores(list(jieba.cut(query)))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm25_hits = [
            str(products[i]["id"])
            for i in ranked[: POOL_K * 5]
            if scores[i] > 0
        ]
        bm25_hits = [pid for pid in bm25_hits if price_ok(pid)][:POOL_K]
        vec_hits = [
            str(r["id"]) for r in search_agent._vector_search(query, POOL_K * 5) if price_ok(str(r["id"]))
        ][:POOL_K]

        pool_ids = list(dict.fromkeys(bm25_hits + vec_hits))
        # 预填：只把「上一轮判为相关、且仍在本次候选池内」的商品填成 [x]，其余留空。
        # 这样候选池发生变化（如新增价格过滤）的查询会自动回到"待人工标注"，
        # 而不会把新进入候选池的商品误标为不相关。
        prefilled = {}
        if query in prev_labeled:
            relevant_ids = {str(i) for i in prev_labeled[query]}
            if relevant_ids:
                prefilled = {pid: "x" for pid in pool_ids if pid in relevant_ids}
            else:
                # 上一轮已判定"库内无相关商品"（空集）：显式标 [0]，
                # 否则重新导入时会因为没有勾选被当成"未标注"、回落伪标注
                prefilled = {pid: "0" for pid in pool_ids}

        same_cat = sum(1 for pid in pool_ids if by_id.get(pid, {}).get("category") == expected_category)
        constraint_hits = sum(
            1 for pid in pool_ids if constraint and constraint in by_id.get(pid, {}).get("name", "")
        )
        coverage_rows.append(
            (
                expected_category,
                query,
                len(pool_ids),
                same_cat,
                query in prev_labeled,
                constraint,
                constraint_hits,
                f"{min_price or 0:g}-{max_price or '∞'}" if price_range else "",
            )
        )
        pool = [
            {
                "id": pid,
                "name": by_id.get(pid, {}).get("name", ""),
                "category": by_id.get(pid, {}).get("category", ""),
                "price": by_id.get(pid, {}).get("price", ""),
                "from": (
                    ("bm25" if pid in bm25_hits else "")
                    + ("+vector" if pid in vec_hits else "")
                ).strip("+"),
                "relevant": 1 if prefilled.get(pid) == "x" else None,  # 人工填写 1 / 0
            }
            for pid in pool_ids
        ]
        pools.append(
            {
                "query": query,
                "description": case.get("description", ""),
                "keywords_hint": case.get("relevant_keywords", []),
                "pool_size": len(pool),
                "candidates": pool,
                "relevant": [],  # 人工填写：相关商品 ID 列表
            }
        )
        md_lines.append(f"## {query}（候选 {len(pool)} 条）")
        if prefilled:
            md_lines.append("")
            md_lines.append("_（本查询上一轮已标注，勾选已预填，可直接跳过）_")
        elif constraint:
            md_lines.append("")
            md_lines.append(
                f"> 约束：本查询带筛选条件「{constraint}」，建议只把名称/属性体现该条件的商品判为相关，"
                f"其余标 `[0]`（池中符合条件约 {constraint_hits}/{len(pool)} 条）。"
            )
        md_lines.append("")
        for item in pool:
            flag = prefilled.get(item["id"], " ")
            price_text = (
                f" ¥{item['price']}"
                if item.get("price") not in (None, "")
                else " 价格未知"
            )
            md_lines.append(
                f"- [{flag}] `{item['id']}` [{item['category']}]{price_text} {item['name']}"
                f"（来源 {item['from']}）"
            )
        md_lines.append("")

    report = {
        "note": "人工标注候选池：请在 relevant 字段填写相关商品 ID；"
        "填写完成后另存为 retrieval_gt_labeled.json（结构：{query: [id, ...]}）",
        "pool_k": POOL_K,
        "queries": pools,
    }
    with open(CANDIDATES_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines))

    print(f"候选池已生成: {CANDIDATES_PATH}")
    print(f"可读版: {MARKDOWN_PATH}")
    print(f"共 {len(pools)} 个查询，候选合计 {sum(p['pool_size'] for p in pools)} 条")
    print()
    print("覆盖度自检（同类目=0 说明库内无对应商品；带约束查询另列「条件命中/不命中」，两者都有才有区分度）：")
    print(
        "  %-10s %-14s %5s %6s %12s %s"
        % ("期望类目", "查询", "候选", "同类目", "条件命中/不命中", "状态")
    )
    for cat, query, total, same_cat, done, constraint, chits, price_label in coverage_rows:
        if done:
            status = "已标注·预填"
        elif same_cat < 3:
            status = "❌ 库内覆盖不足"
        else:
            status = "⚠️ 需人工标注"
        if price_label:
            mix = f"¥{price_label}"
        elif constraint:
            mix = f"{chits}/{total - chits}"
        else:
            mix = "—"
        print(
            "  %-10s %-14s %5d %6d %12s %s"
            % (cat, query, total, same_cat, mix, status)
        )
    weak = [row for row in coverage_rows if row[3] < 3 and not row[4]]
    if weak:
        print(f"\n  ⚠️ 以下查询在商品库内同类目候选 < 3，标注后可能仍为空集：{[r[1] for r in weak]}")
    need = [row for row in coverage_rows if not row[4]]
    print(f"\n  待人工标注查询 {len(need)} 个：{'、'.join(r[1] for r in need) or '无（已全部标注）'}")
    print("\n标注完成后保存为 tests/retrieval_gt_labeled.json，例如：")
    print('  {"蓝牙耳机": ["JD001565", "JD002845"], "相机": ["JD000123"]}')

    build_faq_pools()
    return report


def build_faq_pools() -> None:
    """客服（FAQ）链路的中立候选池：BM25(问题) Top-10 ∪ 向量检索 Top-10。"""
    faq_items = load_faq_items()
    if not faq_items:
        print("\n[FAQ] 未找到 faq_data.json，跳过客服候选池")
        return

    from agents.cs_agent import CustomerServiceAgent

    cs_agent = CustomerServiceAgent(
        faiss_index_path=os.path.join(settings.faiss_index_path, "faq"),
        embedding_model_name=settings.embedding_model,
    )
    cs_agent._ensure_loaded()
    id_map = faq_id_by_text(faq_items)
    by_id = {item["id"]: item for item in faq_items}

    from rank_bm25 import BM25Okapi

    faq_bm25 = BM25Okapi([list(jieba.cut(item["question"])) for item in faq_items])

    prev = {}
    if os.path.exists(FAQ_LABELED_PATH):
        try:
            prev = json.load(open(FAQ_LABELED_PATH, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}

    md_lines = [
        "# 客服（FAQ）检索 GT 候选池（人工标注用）",
        "",
        "> 标注方法：把**能回答该问题**的 FAQ 前面的方框改成 `[x]`，其余留空（或写 `[0]`）。",
        "> 标完在项目根目录执行：",
        "> ```bash",
        "> python tests/build_retrieval_gt.py --import-md",
        "> ```",
        "> 会同时生成 `tests/retrieval_gt_labeled_faq.json`，RAG 评估随即计算客服检索指标。",
        "",
    ]
    pools = []
    for case in CS_TEST_CASES:
        query = case["query"]
        scores = faq_bm25.get_scores(list(jieba.cut(query)))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        bm_hits = [faq_items[i]["id"] for i in ranked[:POOL_K] if scores[i] > 0]
        docs = cs_agent._retrieve(query, POOL_K)
        vec_hits = [id_map.get(d.page_content.strip()) for d in docs]
        vec_hits = [i for i in vec_hits if i]

        pool_ids = list(dict.fromkeys(bm_hits + vec_hits))
        if query in prev:
            prev_relevant = {str(i) for i in prev[query]}
            prefilled = (
                {pid: "x" for pid in pool_ids if pid in prev_relevant}
                if prev_relevant
                else {pid: "0" for pid in pool_ids}
            )
        else:
            prefilled = {}
        pool = [
            {
                "id": pid,
                "question": by_id[pid]["question"],
                "answer": by_id[pid]["answer"],
                "category": by_id[pid]["category"],
                "from": (
                    ("bm25" if pid in bm_hits else "")
                    + ("+vector" if pid in vec_hits else "")
                ).strip("+"),
                "relevant": 1 if pid in prefilled else None,
            }
            for pid in pool_ids
        ]
        pools.append(
            {
                "query": query,
                "pool_size": len(pool),
                "candidates": pool,
                "relevant": [],
            }
        )
        md_lines.append(f"## {query}（候选 {len(pool)} 条）")
        if prefilled:
            md_lines.append("")
            md_lines.append("_（上一轮已标注，勾选已预填，可直接跳过）_")
        md_lines.append("")
        for item in pool:
            flag = prefilled.get(item["id"], " ")
            md_lines.append(
                f"- [{flag}] `{item['id']}` [{item['category']}] {item['question']}"
                f" → {item['answer']}（来源 {item['from']}）"
            )
        md_lines.append("")

    with open(FAQ_CANDIDATES_PATH, "w", encoding="utf-8") as handle:
        json.dump({"pool_k": POOL_K, "queries": pools}, handle, ensure_ascii=False, indent=2)
    with open(FAQ_MARKDOWN_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines))

    print(f"\n客服 FAQ 候选池已生成: {FAQ_CANDIDATES_PATH}")
    print(f"  共 {len(pools)} 个查询，候选合计 {sum(p['pool_size'] for p in pools)} 条")
    for pool in pools:
        general = sum(1 for c in pool["candidates"] if c["category"] != "商品质量")
        print(f"    {pool['query']:<14} 候选 {pool['pool_size']:>2} 条（非商品质量类 {general} 条）")
    print(f"  可读版: {FAQ_MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
