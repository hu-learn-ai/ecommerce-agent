"""为商品库补齐缺失的价格（模拟值，确定性可复现）。

背景：`products_for_faiss.json` 中 7000 件商品只有 3000 件（KD 前缀、手机数码）带真实价格，
其余 4000 件（JD 前缀，4 个类目各 1000）没有价格。后果：
- 搜索链路的"500元以下"这类价格约束对这 4000 件商品失效（无价格 → 被过滤掉或无法判断）；
- 推荐链路的用户画像预算（如"预算200以内"）无法执行；
- 价格约束类评估用例无法构造。

原始数据里没有可用价格（JD 分类数据只有 `一级,一级@二级,标题`；天猫商品表无价格列），
因此这里按**类目分布 + 标题线索**生成模拟价格：

1. 类目基准：对数正态分布，手机数码的 μ/σ 由该类的 3000 条真实价格拟合
   （几何均值 2816、log σ 0.80），其余类目取行业常识量级（见 CATEGORY_BASE）；
2. 品类线索覆盖：主机与配件价差 1~2 个数量级，只按类目分布会把手机壳算成手机价
   （实测"kindle保护套"被算成 ¥3609），因此标题命中配件/耳机/手表/相机等线索时
   整体替换基准（见 TYPE_BASE）；
3. 标题调制（有界乘子）：品牌档次、二手、套装礼盒、规格（匹数/斤数/容量）等；
4. 确定性：随机种子由商品 ID 的哈希决定，重复运行结果一致；
5. 每件商品写 `price_source`：`real`（原有真实价格）/ `simulated`（本次生成）。

⚠️ 模拟价格**只用于打通价格过滤链路与离线评估**，不代表真实市场价；README 与报告中已标注。

用法:
    python scripts/fill_catalog_prices.py                # 补齐并写回（自动备份）
    python scripts/fill_catalog_prices.py --dry-run      # 只看统计，不写文件
    python scripts/fill_catalog_prices.py --regenerate   # 按当前规则重算模拟价格
    python scripts/fill_catalog_prices.py --sync-neo4j   # 同时把价格同步到 Neo4j SKU
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
from datetime import datetime

# Windows GBK 控制台下打印 ￥/emoji 会抛 UnicodeEncodeError（价格写回后才崩、看起来像失败）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 - 流不支持 reconfigure 时忽略
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CATALOG = os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json")

# (几何均值, log σ, 下限, 上限) —— 手机数码由真实价格拟合，其余为行业量级假设
CATEGORY_BASE = {
    "手机数码": (2816.0, 0.80, 30.0, 60000.0),
    "家用电器": (1500.0, 1.00, 50.0, 80000.0),
    "食品生鲜": (60.0, 0.90, 5.0, 3000.0),
    "医药保健": (120.0, 0.90, 8.0, 5000.0),
}

# 标题线索 → 基准价（比类目基准优先）。
# 原因：同一类目里主机与配件价差 1~2 个数量级，只用类目分布会把手机壳算成手机价
# （实测"kindle保护套"被算成 ¥3609）。命中即整体替换类目基准。
TYPE_BASE = {
    "手机数码": (
        # 配件（保护套/壳/膜/线/支架/耳塞套/镜头盖…）
        (("保护套", "保护壳", "贴膜", "数据线", "支架", "耳塞套", "耳帽", "耳翼", "耳套",
          "镜头盖", "转接", "防尘塞", "收纳包", "收纳盒", "替换", "配件", "适用",
          "保护膜", "膜", "壳"),
         (45.0, 0.60, 5.0, 500.0)),
        # 存储/电芯配件
        (("内存卡", "存储卡", "u盘", "硬盘", "读卡器", "移动电源", "充电宝", "充电器", "电池"),
         (300.0, 0.80, 20.0, 3000.0)),
        # 音频设备
        (("耳机", "耳麦", "音箱", "音响", "麦克风", "话筒"), (400.0, 0.75, 30.0, 6000.0)),
        # 穿戴
        (("手表", "手环"), (700.0, 0.70, 60.0, 8000.0)),
        # 影像
        (("相机", "单反", "微单", "镜头"), (4500.0, 0.70, 300.0, 40000.0)),
        # 电脑/平板
        (("笔记本", "平板", "电脑", "主机", "显示器"), (5000.0, 0.60, 800.0, 40000.0)),
        (("手机", "智能手机"), (2200.0, 0.75, 200.0, 20000.0)),
    ),
    "家用电器": (
        (("刀头", "刀网", "滤芯", "替换", "配件", "适用", "底座", "杯垫"),
         (120.0, 0.60, 15.0, 800.0)),
        # 小家电（水壶/锅具/茶具/取暖器…）
        (("水壶", "热水壶", "锅", "茶具", "杯", "刀", "风扇", "取暖", "剃须", "加湿", "挂烫"),
         (200.0, 0.70, 30.0, 2000.0)),
        # 大家电
        (("空调", "冰箱", "洗衣机", "电视", "热水器", "油烟机", "洗碗机"),
         (2800.0, 0.70, 600.0, 30000.0)),
    ),
}

# 品牌档次（乘子）；命中即相乘，未命中用 1.0
PREMIUM_BRANDS = ("apple", "苹果", "戴森", "dyson", "sony", "索尼", "华为", "huawei",
                  "西门子", "bosch", "博世", "thinkpad", "外星人", "alienware")
CHEAP_HINTS = ("适用", "通用", "配件", "替换", "二手", "拆机", "翻新", "简装")
HIGH_HINTS = ("套装", "礼盒", "组合", "旗舰", "顶配", "官方旗舰", "进口")

SPEC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(匹|斤|kg|公斤|升|L|GB|TB|寸|英寸|片|袋|盒|瓶|台)")


def _seed_of(item_id: str) -> int:
    return int(hashlib.md5(item_id.encode("utf-8")).hexdigest()[:8], 16)


def simulate_price(item: dict) -> float:
    """按类目 + 标题线索生成模拟价格（确定性）。"""
    category = item.get("category", "")
    mu, sigma, low, high = CATEGORY_BASE.get(category, (200.0, 1.0, 5.0, 20000.0))
    name = (item.get("name") or "").lower()
    for keywords, base in TYPE_BASE.get(category, ()):
        if any(keyword in name for keyword in keywords):
            mu, sigma, low, high = base
            break

    rng = random.Random(_seed_of(str(item.get("id", ""))))
    price = math.exp(rng.gauss(math.log(mu), sigma))

    if any(brand in name for brand in PREMIUM_BRANDS):
        price *= 1.6
    if any(hint in name for hint in CHEAP_HINTS):
        price *= 0.55
    if any(hint in name for hint in HIGH_HINTS):
        price *= 1.35

    # 规格调制：有界（0.7~1.6），避免个别规格把价格打到离谱
    multiplier = 1.0
    for value_text, unit in SPEC_RE.findall(name)[:3]:
        value = float(value_text)
        if unit == "匹":
            multiplier *= min(1.3, 0.75 + 0.15 * value)
        elif unit in ("斤", "kg", "公斤"):
            multiplier *= min(1.25, 0.9 + 0.05 * value)
        elif unit in ("升", "L"):
            multiplier *= min(1.2, 0.95 + 0.02 * value)
        elif unit in ("GB", "TB"):
            multiplier *= min(1.25, 0.95 + 0.03 * value)
    price *= max(0.7, min(1.6, multiplier))

    return round(max(low, min(high, price)), 2)


def main() -> dict:
    parser = argparse.ArgumentParser(description="补齐商品库价格（模拟值）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不写文件")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="重算已有的模拟价格（真实价格不动）；默认只补缺失值，可重复执行",
    )
    parser.add_argument("--sync-neo4j", action="store_true", help="同步价格到 Neo4j SKU")
    args = parser.parse_args()

    with open(CATALOG, encoding="utf-8") as handle:
        products = json.load(handle)

    filled = []
    for item in products:
        raw = item.get("price")
        is_simulated = item.get("price_source") == "simulated"
        if raw not in (None, "", "None") and not (args.regenerate and is_simulated):
            item["price"] = float(raw)
            # 重跑时必须保留已有的 price_source：否则上一轮生成的模拟价格会被
            # 重新标成 real，进而被检索链路当成真实价格做硬过滤（静默的数据污染）
            if item.get("price_source") not in ("real", "simulated"):
                item["price_source"] = "real"
            continue
        item["price"] = simulate_price(item)
        item["price_source"] = "simulated"
        filled.append(item)

    after_priced = sum(1 for p in products if p.get("price") not in (None, ""))
    print("=" * 62)
    print("  商品库价格补齐（模拟值）")
    print("=" * 62)
    real_count = sum(1 for p in products if p.get("price_source") == "real")
    sim_count = sum(1 for p in products if p.get("price_source") == "simulated")
    print(
        f"  商品总数: {len(products)}｜真实价格: {real_count}｜模拟价格: {sim_count}"
        f"｜本次生成: {len(filled)}｜无价格: {len(products) - after_priced}"
    )
    print()
    by_cat = collections.defaultdict(list)
    for item in products:
        by_cat[item.get("category", "")].append(float(item["price"]))
    print("  %-8s %6s %10s %10s %10s %10s" % ("类目", "数量", "P5", "中位数", "均值", "P95"))
    for category, values in sorted(by_cat.items()):
        values.sort()

        def q(p: float) -> float:
            return values[int(p * (len(values) - 1))]

        print("  %-8s %6d %10.1f %10.1f %10.1f %10.1f"
              % (category, len(values), q(0.05), statistics.median(values),
                 statistics.fmean(values), q(0.95)))

    print("\n  模拟价格样例:")
    for item in filled[:6]:
        # 用全角￥：半角 ¥(U+00A5) 在 GBK 控制台会抛 UnicodeEncodeError
        print(f"    ￥{item['price']:<9.2f} [{item['category']}] {item['name'][:44]}")

    if args.dry_run:
        print("\n  （--dry-run：未写文件）")
        return {"filled": len(filled)}

    stamp = datetime.now().strftime("%Y%m%d")
    backup = f"{CATALOG}.bak_{stamp}"
    if not os.path.exists(backup):
        shutil.copy2(CATALOG, backup)
        print(f"\n  已备份原文件: {backup}")

    with open(CATALOG, "w", encoding="utf-8") as handle:
        json.dump(products, handle, ensure_ascii=False, indent=1)
    print(f"  已写回: {CATALOG}")

    if args.sync_neo4j:
        from orchestration.registry import create_neo4j_driver

        driver = create_neo4j_driver()
        updated = 0
        with driver.session() as session:
            for item in products:
                result = session.run(
                    "MATCH (p:SPU {id: $spu})-[:Have]->(sku:SKU) "
                    "WHERE sku.price IS NULL "
                    "SET sku.price = $price, sku.price_source = $source "
                    "RETURN count(sku) AS n",
                    spu=str(item["id"]),
                    price=float(item["price"]),
                    source=item.get("price_source", "unknown"),
                ).single()
                updated += int(result["n"] or 0)
        print(f"  已同步 Neo4j SKU 价格: {updated} 条（仅补齐原本无价格的 SKU）")
        driver.close()

    return {"filled": len(filled), "total": len(products)}


if __name__ == "__main__":
    main()
