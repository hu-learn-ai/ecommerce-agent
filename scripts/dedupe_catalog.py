"""商品库去重：把同款多 ID 的重复商品合并，消除检索/推荐指标的虚高。

背景（README「已知问题」）：`products_for_faiss.json` 7000 件商品里存在大量同款重复，
如华为 MatePad 有 210 个 ID、Apple iPhone 130 个、iPad 79 个（同一型号的不同颜色/容量/渠道
被拆成多个 ID）。后果：
- 检索 Recall 虚高：命中任一同款 ID 都算命中，指标不代表真实检索质量；
- NDCG/MRR 的"命中"含义变弱；
- Item-CF / 推荐把同一型号的不同 SKU 当成不同商品，共现信号被稀释。

去重规则（确定性）：
1. 去标题头部的装饰段（【包邮】/（特价）等）；
2. 去全部标点/空白；
3. 去渠道/营销/品相词（官方旗舰店/自营/全新/二手/套装…）；
4. 去型号尾部的规格/颜色（容量 GB、尺寸 寸/升、颜色词…）——仅 `--merge-specs` 开启，
   因为"1TB 硬盘 vs 2TB 硬盘"这类容量即本质的商品会因此被误合并；
5. 按 (品牌, 类目, 归一化标题) 聚组，每组保留 1 件（优先真实价格，其次最短标题）。

用法
----
    python scripts/dedupe_catalog.py --dry-run            # 只看去重统计（默认，不写文件）
    python scripts/dedupe_catalog.py --write              # 写回（自动备份）
    python scripts/dedupe_catalog.py --write --merge-specs # 连同规格变体一起合并（更激进）

写回后需重建下游产物（本脚本不自动重建）：
    python scripts/build_faiss_index.py        # FAISS 索引
    python scripts/rebuild_neo4j.py            # Neo4j 图谱
    python scripts/generate_catalog_orders.py  # 订单明细（若已生成过）
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime

# Windows GBK 控制台打印中文会抛 UnicodeEncodeError，先切换为可替换模式
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATALOG = os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json")

# 标题头部的装饰段：【包邮】/（特价）/ [热卖] …
_LEADING_DECOR = re.compile(r"^[【\[(（][^】\]）)]*[】\]）)]")

# 渠道 / 营销 / 品相词（去掉后不影响"是否同一型号"判断）
_MARKETING = (
    "官方旗舰店", "官方旗舰", "旗舰店", "专卖店", "专营店", "自营", "直营",
    "全新", "正品", "包邮", "特价", "热卖", "新款", "现货", "厂家", "原装", "进口",
    "简装", "盒装", "袋装", "套装", "礼盒", "组合装",
    "国行", "港版", "美版", "日版", "欧版", "二手", "翻新", "拆机",
)

# 型号尾部的规格（容量/尺寸/重量/功率…）——仅 --merge-specs 时剥掉
_SPEC_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(?:GB|TB|MB|G|T|英寸|寸|厘米|CM|毫米|MM|米|M|斤|公斤|KG|克|G|升|L|ML|毫升|匹|核|瓦|W|万|毫安|MAH)"
    r"(?:版|款)?",
    re.IGNORECASE,
)

# 常见颜色词（含苹果/华为等品牌命名色）
_COLOR_RE = re.compile(
    r"(?:深空灰|星光色|午夜色|远峰蓝|苍岭绿|石墨色|曜石黑|月光白|晨曦金|流光银|幻夜黑|"
    r"玫瑰金|薄荷绿|香槟金|亮黑|亮白|暗夜|深蓝|浅蓝|深灰|浅灰|银灰|"
    r"棕|橙|青|紫|粉|红|蓝|绿|黄|金|银|灰|黑|白)(?:色)?"
)


def normalize_title(name: str, merge_specs: bool) -> str:
    """归一化标题，用于判断是否同一型号（确定性）。"""
    t = _LEADING_DECOR.sub("", (name or "").strip())
    for word in _MARKETING:
        t = t.replace(word, "")
    if merge_specs:
        # 反复剥尾部的规格/颜色 token（可能有 "128G 蓝色" 两段）
        for _ in range(3):
            new = _SPEC_RE.sub("", t)
            new = _COLOR_RE.sub("", new)
            if new == t:
                break
            t = new
    # 去剩余标点/空白，拉丁统一小写
    t = re.sub(r"[\W_]+", "", t)
    return t.lower()


def core_key(product: dict, merge_specs: bool) -> tuple:
    """去重主键：(品牌, 类目, 归一化标题)。"""
    brand = (product.get("brand") or "").strip().lower()
    category = (product.get("category") or "").strip()
    title = normalize_title(product.get("name") or "", merge_specs)
    return (brand, category, title)


def _rank(product: dict) -> tuple:
    """组内保留优先级：真实价格 > 模拟价格 > 无价格；其次标题更短；再次 ID 更小。"""
    source = product.get("price_source")
    price_rank = 0 if source == "real" else (1 if source == "simulated" else 2)
    return (price_rank, len(product.get("name") or ""), product.get("id", ""))


def dedupe(products: list[dict], merge_specs: bool, max_keep: int = 1) -> tuple[list, list]:
    """返回 (去重后商品列表, 被合并的重复组列表)。"""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in products:
        groups[core_key(p, merge_specs)].append(p)

    kept: list[dict] = []
    removed_groups: list[list[dict]] = []
    for items in groups.values():
        items.sort(key=_rank)
        kept.extend(items[:max_keep])
        if len(items) > max_keep:
            removed_groups.append(items)

    # 保持确定性输出：按 id 排序
    kept.sort(key=lambda p: p["id"])
    removed_groups.sort(key=lambda g: -len(g))
    return kept, removed_groups


def _report(products: list[dict], kept: list[dict], removed: list[list[dict]]) -> None:
    print("=" * 66)
    print("  商品库去重报告")
    print("=" * 66)
    print(f"  去重前: {len(products)} 件")
    print(f"  去重后: {len(kept)} 件（减少 {len(products) - len(kept)} 件）")
    print(f"  重复组: {len(removed)} 组（每组保留 1 件，共移除 "
          f"{sum(len(g) - 1 for g in removed)} 件）")
    print()
    print("  最大的重复组（同一型号被拆成的 ID 数）:")
    for g in removed[:10]:
        rep = g[0]
        brand = rep.get("brand") or "无品牌"
        print(f"    {len(g):>3} 个ID  [{rep.get('category','')}] {brand} "
              f"{rep.get('name','')[:40]}")
    print()
    by_cat_before: dict[str, int] = defaultdict(int)
    by_cat_after: dict[str, int] = defaultdict(int)
    for p in products:
        by_cat_before[p.get("category", "")] += 1
    for p in kept:
        by_cat_after[p.get("category", "")] += 1
    print("  各类目去重前后:")
    for cat in sorted(set(by_cat_before) | set(by_cat_after)):
        print(f"    {cat}: {by_cat_before[cat]} -> {by_cat_after.get(cat, 0)}")


def main() -> dict:
    ap = argparse.ArgumentParser(description="商品库去重")
    ap.add_argument("--dry-run", action="store_true", help="只看统计，不写文件（默认）")
    ap.add_argument("--write", action="store_true", help="写回 products_for_faiss.json")
    ap.add_argument("--merge-specs", action="store_true",
                    help="连同规格/颜色变体一起合并（更激进，可能合并容量不同的商品）")
    ap.add_argument("--max-keep", type=int, default=1, help="每组最多保留件数（默认 1）")
    args = ap.parse_args()

    with open(CATALOG, encoding="utf-8") as f:
        products = json.load(f)

    kept, removed = dedupe(products, merge_specs=args.merge_specs, max_keep=args.max_keep)
    _report(products, kept, removed)

    if args.write:
        stamp = datetime.now().strftime("%Y%m%d")
        backup = f"{CATALOG}.bak_dedupe_{stamp}"
        if not os.path.exists(backup):
            shutil.copy2(CATALOG, backup)
            print(f"\n  已备份原文件: {backup}")
        with open(CATALOG, "w", encoding="utf-8") as f:
            json.dump(kept, f, ensure_ascii=False, indent=1)
        print(f"  已写回: {CATALOG}")
        print("\n  下一步（下游产物需重建，本脚本不自动执行）:")
        print("    python scripts/build_faiss_index.py")
        print("    python scripts/rebuild_neo4j.py")
        print("    python scripts/generate_catalog_orders.py")
    else:
        print("\n  （--dry-run：未写文件。确认后加 --write 写回）")

    return {"before": len(products), "after": len(kept), "groups": len(removed)}


if __name__ == "__main__":
    main()
