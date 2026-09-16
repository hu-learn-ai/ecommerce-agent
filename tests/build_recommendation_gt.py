"""推荐评估的人工标注 GT（TREC pooling 流程）。

背景：推荐评估的 GT 原为"关键词匹配"，覆盖商品库 33%（平均 2285/7000），
导致 popularity 与完整推荐管线都能拿满分，指标失去区分度。
本脚本把**参与对比的各策略输出并集**作为候选池，交给人工判定相关性：

    1. 对每个用例取各策略 Top-10 的并集（popularity / graph_only / item_cf_only /
       query_aware / graph+query_aware，可选加 LLM 重排）
    2. 输出候选池（含商品名/类目/价格，便于判读）
    3. 人工在 md 里把相关商品前面的方框改成 [x]（或写 [1]）
    4. `python tests/build_recommendation_gt.py --import-md` 生成
       `tests/recommendation_gt_labeled.json`，`recommendation_eval.py` 随即改用人工 GT

用法:
    python tests/build_recommendation_gt.py                 # 生成候选池（离线）
    python tests/build_recommendation_gt.py --force         # 覆盖已有候选池
    python tests/build_recommendation_gt.py --with-llm      # 候选池额外纳入 LLM 重排结果（需联网）
    python tests/build_recommendation_gt.py --import-md     # 解析标注生成 GT

输出:
    tests/recommendation_gt_candidates.json / .md
    tests/recommendation_gt_labeled.json（人工标注后）
"""

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root, require_assets  # noqa: E402

ensure_project_root()

from config.settings import settings  # noqa: E402
from tests.build_retrieval_gt import import_from_markdown  # noqa: E402

CANDIDATES_PATH = os.path.join(PROJECT_ROOT, "tests", "recommendation_gt_candidates.json")
MARKDOWN_PATH = os.path.join(PROJECT_ROOT, "tests", "recommendation_gt_candidates.md")
LABELED_PATH = os.path.join(PROJECT_ROOT, "tests", "recommendation_gt_labeled.json")
POOL_K = 10


def normalize_price(value):
    """把 None / "" / "None" / "nan" 统一成 None，避免 md 里出现 "¥None"。"""
    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "None", "nan", "NaN", "null"):
        return None
    return text


def price_text(value) -> str:
    normalized = normalize_price(value)
    return f" ¥{normalized}" if normalized else " 价格未知"


def parse_args():
    parser = argparse.ArgumentParser(description="构建推荐评估 pooling 候选池")
    parser.add_argument("--force", action="store_true", help="已存在时强制覆盖")
    parser.add_argument("--with-llm", action="store_true", help="候选池额外纳入 LLM 重排结果")
    parser.add_argument("--import-md", action="store_true", help="解析 md 标注生成人工 GT")
    parser.add_argument("--md", type=str, default=MARKDOWN_PATH, help="待解析的 Markdown 路径")
    return parser.parse_args()


def build_agent():
    """构造推荐 Agent（与 recommendation_eval 一致：走注册中心，复用图/MySQL/搜索能力）。"""
    from langchain_openai import ChatOpenAI

    from orchestration.registry import create_neo4j_driver, register_all_agents

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.1,
        max_tokens=512,
    )
    driver = None
    try:
        driver = create_neo4j_driver()
    except Exception as exc:  # noqa: BLE001
        print(f"[RecommendGT] Neo4j 不可用（相关策略将为空）: {exc}")
    agents = register_all_agents(neo4j_driver=driver, llm=llm)
    return agents.get("catalog_agent").recommend_agent, llm


def main() -> dict:
    args = parse_args()

    if args.import_md:
        if not os.path.exists(args.md):
            raise SystemExit(f"未找到 Markdown: {args.md}")
        labeled = import_from_markdown(args.md)
        if not labeled:
            raise SystemExit("没有任何用例被标注，未生成文件（请先在 md 里勾选 [x]）")
        with open(LABELED_PATH, "w", encoding="utf-8") as handle:
            json.dump(labeled, handle, ensure_ascii=False, indent=2)
        print(f"\n推荐人工标注 GT 已生成: {LABELED_PATH}")
        print("下一步重跑推荐评估：python tests/recommendation_eval.py")
        return labeled

    require_assets(
        os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json"),
        os.path.join(settings.faiss_index_path, "products.index"),
    )

    from tests.recommendation_eval import generate_test_cases

    agent, llm = build_agent()
    products = json.load(
        open(
            os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json"),
            encoding="utf-8",
        )
    )
    by_id = {str(p["id"]): p for p in products}

    prev = {}
    if os.path.exists(LABELED_PATH) and not args.force:
        print(f"人工 GT 已存在，跳过生成（--force 覆盖）: {LABELED_PATH}")
        return json.load(open(LABELED_PATH, encoding="utf-8"))
    if os.path.exists(LABELED_PATH):
        try:
            prev = json.load(open(LABELED_PATH, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}

    pools = []
    md_lines = [
        "# 推荐评估 GT 候选池（人工标注用）",
        "",
        "> 标注方法：把**符合该用户需求**的商品前面的方框改成 `[x]`（`[1]` 同义），",
        "> 其余留空（或写 `[0]`）。",
        ">",
        "> **判断标准（重要）**：",
        "> 1. 主要看**品类与查询意图是否匹配**（如「推荐牛奶和水果」→ 是真牛奶/真水果就判相关）；",
        "> 2. **「价格未知」≠ 不相关**——本商品库只有手机数码类目有价格，其它类目基本无价格数据，",
        ">    不要因为看不到价格就判 0；",
        "> 3. 只有当候选**有明确价格且明显超出预算**时才判 `[0]`；",
        "> 4. **不要把整段全部填 0**：那样该用例的相关集为空，会被判定为「库内无相关商品」并整体排除出指标。",
        "> 标完执行：`python tests/build_recommendation_gt.py --import-md`",
        "",
    ]

    for case in generate_test_cases():
        query = case.query
        profile = case.user_profile
        sources = {
            "popularity": agent._get_popular_products(POOL_K),
            "graph_only": agent._graph_based_recommend(query, profile, POOL_K),
            "item_cf_only": agent._item_cf_recommend(query, profile, POOL_K),
            "query_aware": agent._query_aware_candidates(query, profile, POOL_K),
        }
        if args.with_llm:
            merged = agent._merge_candidates(
                sources["query_aware"], sources["graph_only"], limit=POOL_K * 2
            )
            if merged:
                sources["llm_rerank"] = agent._llm_rerank(query, merged, profile, POOL_K)

        seen, pool = set(), []
        skipped_outside_catalog = 0
        for source_name, items in sources.items():
            for item in items:
                item_id = str(item.get("id", ""))
                if not item_id or item_id in seen:
                    if item_id in seen:
                        for entry in pool:  # 记录该候选被哪些策略召回
                            if entry["id"] == item_id and source_name not in entry["from"]:
                                entry["from"].append(source_name)
                    continue
                if item_id not in by_id:
                    # Item-CF 走 MySQL 订单库（P* ID），与商品库 ID 空间不重合：
                    # 这类候选无法标注、也无法与其它策略比较，直接排除并计数
                    skipped_outside_catalog += 1
                    continue
                seen.add(item_id)
                meta = by_id.get(item_id, {})
                pool.append(
                    {
                        "id": item_id,
                        "name": item.get("product") or meta.get("name", ""),
                        "category": item.get("category") or meta.get("category", ""),
                        "price": normalize_price(item.get("price") or meta.get("price", "")),
                        "from": [source_name],
                    }
                )

        prev_relevant = {str(i) for i in prev.get(query, [])}
        priced_count = sum(1 for entry in pool if normalize_price(entry["price"]))
        prefilled = (
            {entry["id"]: "x" for entry in pool if entry["id"] in prev_relevant}
            if prev_relevant
            else ({entry["id"]: "0" for entry in pool} if query in prev else {})
        )
        for entry in pool:
            entry["relevant"] = 1 if entry["id"] in prefilled else None

        pools.append(
            {
                "query": query,
                "user_profile": profile,
                "pool_size": len(pool),
                "skipped_outside_catalog": skipped_outside_catalog,
                "candidates": pool,
            }
        )

        md_lines.append(
            f"## {query}（候选 {len(pool)} 条，画像: {json.dumps(profile, ensure_ascii=False)}）"
        )
        budget = profile.get("预算")
        if budget and priced_count < len(pool) / 2:
            md_lines.append("")
            md_lines.append(
                f"> ⚠️ 本用例画像含预算「{budget}」，但候选中仅 {priced_count}/{len(pool)} 条有价格"
                "（商品库仅手机数码类目有价格）——**预算约束仅作参考，请按品类/查询意图判断相关性**，"
                "不要因为价格未知就判 0。"
            )
        if prefilled and prev_relevant:
            md_lines.append("")
            md_lines.append("_（上一轮已标注，勾选已预填，可直接跳过）_")
        md_lines.append("")
        for entry in pool:
            flag = prefilled.get(entry["id"], " ")
            md_lines.append(
                f"- [{flag}] `{entry['id']}` [{entry['category']}]{price_text(entry['price'])} "
                f"{entry['name']}（来源 {'+'.join(entry['from'])}）"
            )
        md_lines.append("")

    with open(CANDIDATES_PATH, "w", encoding="utf-8") as handle:
        json.dump({"pool_k": POOL_K, "cases": pools}, handle, ensure_ascii=False, indent=2)
    with open(MARKDOWN_PATH, "w", encoding="utf-8") as handle:
        handle.write("\n".join(md_lines))

    print(f"\n推荐候选池已生成: {CANDIDATES_PATH}")
    print(f"可读版: {MARKDOWN_PATH}")
    print(f"共 {len(pools)} 个用例，候选合计 {sum(p['pool_size'] for p in pools)} 条")
    print("\n  用例 / 候选数 / 各策略贡献:")
    for pool in pools:
        from collections import Counter

        counts = Counter(s for entry in pool["candidates"] for s in entry["from"])
        detail = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"    {pool['query'][:20]:<22} {pool['pool_size']:>2} 条 | {detail}")
    print("\n标注完成后执行: python tests/build_recommendation_gt.py --import-md")
    return {"cases": pools}


if __name__ == "__main__":
    main()
