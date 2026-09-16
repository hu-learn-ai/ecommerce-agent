"""生成与商品库对齐的订单明细（order_detail），修复 Item-CF 协同过滤恒为 0 的问题。

背景 / 根因
------------
推荐链路里 Item-CF（agents/recommend_agent.py::_item_cf_recommend）做
"购买了该商品的用户还购买了" 的共现推荐，查询 MySQL 的 order_detail 表：

    SELECT DISTINCT product_id, product_name FROM order_detail
    WHERE product_name LIKE %<preferred_category>%  -- 种子商品
    → 找到种子商品 → 找买过种子的用户 → 统计这些用户还买了什么

而旧的 order_detail（import_taobao_data.py 从合成淘宝 CSV 生成）用的是另一套商品：
  - product_id 是 ``P000001`` 这类 2000 个合成淘宝商品 ID（15 个类目）
  - product_name 是 ``中信出版社图书音像商品781``（旧 15 类目名）

真实商品库（data/processed/products_for_faiss.json → Neo4j/FAISS）是：
  - product_id 是 ``JD000001`` / ``KD000001``（7000 件，京东 4 类目）
  - 类目只有 医药保健 / 家用电器 / 手机数码 / 食品生鲜

因此 Item-CF 即使命中种子，共现出来的 product_id 也全是 ``P...``，在商品库 /
Neo4j / FAISS 里根本不存在，推荐结果无法映射到任何真实商品，评估 NDCG 恒为 0。

修复
----
本脚本按真实商品库重新生成 order_detail：
  - product_id 直接取自商品库（JD/KD），保证推荐结果可映射回真实商品；
  - product_name 采用 ``{品牌}{类目}商品{序号}`` 的合成名（与 recommend_agent 的
    _pretty_name 约定一致，且保证 ``LIKE %类目%`` 能命中种子商品）；
  - 引入共现结构：用户有 1~3 个偏好品牌，多次下单偏好品牌/同类的热门商品，
    使"买了 A 的用户也买了 B"具有真实的协同信号（而非随机噪声）；
  - 确定性：固定随机种子，重复运行结果一致。

用法
----
    python scripts/generate_catalog_orders.py                # 默认 2000 用户，生成 SQL + JSON
    python scripts/generate_catalog_orders.py --users 5000   # 自定义用户数
    python scripts/generate_catalog_orders.py --seed 7       # 自定义随机种子

生成文件
--------
    data/processed/mysql_catalog_orders.sql   DROP+CREATE+INSERT order_detail（替换旧表）
    data/processed/catalog_orders.json        原始订单明细（审计 / 测试用）

加载到 MySQL（order_detail 之外的表沿用 import_taobao_data.py 的产物）:
    mysql -u root -p gmall < data/processed/mysql_catalog_orders.sql
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime

# Windows GBK 控制台打印中文/￥ 会抛 UnicodeEncodeError，先切换到可替换模式
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

CATALOG = os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json")
OUTPUT_SQL = os.path.join(PROJECT_ROOT, "data", "processed", "mysql_catalog_orders.sql")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "catalog_orders.json")

# 与 config/settings.py 的 get_mysql_config 一致，类目来自 models/best/labels.txt
VALID_CATEGORIES = ("医药保健", "家用电器", "手机数码", "食品生鲜")


def _escape_sql(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _product_number(product_id: str) -> int:
    """JD000123 / KD000456 → 123 / 456（商品名的 序号 部分）。"""
    return int(product_id[2:])


def _synthetic_name(product: dict) -> str:
    """按 recommend_agent._pretty_name 约定生成合成名：{品牌}{类目}商品{序号}。"""
    brand = (product.get("brand") or "").strip()
    category = product.get("category", "")
    return f"{brand}{category}商品{_product_number(product['id'])}"


def load_catalog() -> list[dict]:
    with open(CATALOG, encoding="utf-8") as f:
        return json.load(f)


def build_category_pools(products: list[dict]) -> dict[str, list[dict]]:
    """按类目分组商品，并按类目内热度（Zipf 风格权重）排序，供加权抽样。

    用确定性伪热度制造"爆款"：同一类目里少数商品被大量用户购买，这些爆款
    构成共现的锚点（买了爆款 A 的用户通常也买了同品牌/同类的 B）。
    """
    pools: dict[str, list[dict]] = defaultdict(list)
    for p in products:
        pools[p.get("category", "")].append(p)

    weighted: dict[str, list[dict]] = {}
    for cat, items in pools.items():
        # 按 ID 排序保证确定性，再按 Zipf 权重抽样（排名越靠前越常被买）
        items = sorted(items, key=lambda p: p["id"])
        weighted[cat] = items
    return dict(weighted)


def _zipf_weighted_choice(rng: random.Random, items: list[dict]) -> dict:
    """按 1/rank 权重从 items 中抽一个（排序越靠前概率越高）。"""
    total = sum(1.0 / (i + 1) for i in range(len(items)))
    r = rng.random() * total
    acc = 0.0
    for i, item in enumerate(items):
        acc += 1.0 / (i + 1)
        if r <= acc:
            return item
    return items[-1]


def generate_orders(products: list[dict], n_users: int, seed: int = 42) -> list[dict]:
    """生成 catalog-aligned 订单明细。

    Returns:
        list[dict] 每行 order_id / user_id / product_id / product_name /
                    quantity / unit_price / amount
    """
    rng = random.Random(seed)
    pools = build_category_pools(products)

    # 类目 → 品牌列表（去空品牌），用于给用户分配偏好品牌
    brands_by_cat: dict[str, list[str]] = {}
    for cat, items in pools.items():
        brands = sorted({p["brand"].strip() for p in items if p.get("brand", "").strip()})
        brands_by_cat[cat] = brands

    # 每个类目的商品池按品牌再分组，供"偏好品牌内"抽样
    by_brand: dict[str, dict[str, list[dict]]] = {}
    for cat, items in pools.items():
        grouped: dict[str, list[dict]] = defaultdict(list)
        for p in items:
            brand = (p.get("brand") or "").strip()
            key = brand if brand else f"<{cat}无品牌>"
            grouped[key].append(p)
        by_brand[cat] = grouped

    categories = [c for c in VALID_CATEGORIES if c in pools]

    orders: list[dict] = []
    order_seq = 0

    for u in range(1, n_users + 1):
        user_id = f"C{u:06d}"
        # 用户画像：主类目 + 1~3 个偏好品牌
        home_cat = rng.choice(categories)
        fav_brands = rng.sample(
            brands_by_cat[home_cat],
            k=min(rng.randint(1, 3), len(brands_by_cat[home_cat])),
        )
        fav_pool = [
            p
            for p in pools[home_cat]
            if (p.get("brand") or "").strip() in fav_brands
        ] or pools[home_cat]

        # 每个用户 2~5 个订单，每单 1~3 件商品
        n_orders = rng.randint(2, 5)
        for _ in range(n_orders):
            order_seq += 1
            order_id = f"O{order_seq:08d}"
            n_items = rng.randint(1, 3)
            seen_in_order: set[str] = set()
            for _ in range(n_items):
                # 60% 偏好品牌内 / 30% 主类目内 / 10% 跨类目噪声
                roll = rng.random()
                if roll < 0.60:
                    product = _zipf_weighted_choice(rng, fav_pool)
                elif roll < 0.90:
                    product = _zipf_weighted_choice(rng, pools[home_cat])
                else:
                    other_cat = rng.choice([c for c in categories if c != home_cat] or categories)
                    product = _zipf_weighted_choice(rng, pools[other_cat])
                if product["id"] in seen_in_order:
                    continue
                seen_in_order.add(product["id"])

                quantity = rng.randint(1, 3)
                try:
                    unit_price = float(product.get("price") or 0.0)
                except (TypeError, ValueError):
                    unit_price = 0.0
                amount = round(unit_price * quantity, 2)

                orders.append(
                    {
                        "order_id": order_id,
                        "user_id": user_id,
                        "product_id": product["id"],
                        "product_name": _synthetic_name(product),
                        "quantity": quantity,
                        "unit_price": round(unit_price, 2),
                        "amount": amount,
                    }
                )

    return orders


def write_json(orders: list[dict]) -> None:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(orders, f, ensure_ascii=False, indent=1)
    print(f"[CatalogOrders] 已生成 JSON: {OUTPUT_JSON} ({len(orders)} 行)")


def write_sql(orders: list[dict]) -> None:
    lines = [
        "-- ============================================",
        "-- order_detail 订单明细（与商品库对齐，供 Item-CF 协同过滤）",
        f"-- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "-- 来源: scripts/generate_catalog_orders.py",
        "-- 仅替换 order_detail 表；order_info / user_info 等沿用旧脚本产物",
        "-- ============================================",
        "USE gmall;",
        "",
        "DROP TABLE IF EXISTS order_detail;",
        """CREATE TABLE order_detail (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id        VARCHAR(32) NOT NULL COMMENT '订单号',
    user_id         VARCHAR(32) NOT NULL COMMENT '用户ID',
    product_id      VARCHAR(32) NOT NULL COMMENT '商品ID',
    product_name    VARCHAR(200) DEFAULT NULL COMMENT '商品名称',
    quantity        INT DEFAULT 1 COMMENT '购买数量',
    unit_price      DECIMAL(12,2) DEFAULT 0 COMMENT '单价',
    amount          DECIMAL(12,2) DEFAULT 0 COMMENT '金额',
    KEY idx_order_id (order_id),
    KEY idx_product_id (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单明细表（商品库对齐）';""",
        "",
    ]
    for o in orders:
        lines.append(
            "INSERT INTO order_detail (order_id, user_id, product_id, product_name, "
            "quantity, unit_price, amount) VALUES "
            f"('{o['order_id']}', '{o['user_id']}', '{o['product_id']}', "
            f"'{_escape_sql(o['product_name'])}', {o['quantity']}, "
            f"{o['unit_price']:.2f}, {o['amount']:.2f});"
        )
    lines.append("")
    lines.append(f"-- 共 {len(orders)} 行 order_detail")

    with open(OUTPUT_SQL, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[CatalogOrders] 已生成 SQL: {OUTPUT_SQL} ({len(orders)} 行)")


def _report(products: list[dict], orders: list[dict]) -> None:
    """打印统计，验证共现结构 + 商品对齐。"""
    catalog_ids = {p["id"] for p in products}
    bad_ids = [o for o in orders if o["product_id"] not in catalog_ids]
    name_cats = [
        o for o in orders
        if not any(c in o["product_name"] for c in VALID_CATEGORIES)
    ]

    # 共现：某商品被多少"不同用户"购买（>=2 即有协同信号）
    users_per_product = defaultdict(set)
    for o in orders:
        users_per_product[o["product_id"]].add(o["user_id"])
    co_occurring = sum(1 for u in users_per_product.values() if len(u) >= 2)

    print("=" * 62)
    print("  订单明细与商品库对齐 — 生成报告")
    print("=" * 62)
    print(f"  商品库: {len(catalog_ids)} 件 | 订单明细: {len(orders)} 行")
    print(f"  product_id 不在商品库: {len(bad_ids)} 行")
    print(f"  product_name 不含合法类目: {len(name_cats)} 行")
    print(f"  被 ≥2 个用户共同购买的商品（共现锚点）: {co_occurring} 件")
    print()
    cat_rows = Counter(
        next((c for c in VALID_CATEGORIES if c in o["product_name"]), "?")
        for o in orders
    )
    print("  order_detail 类目分布:")
    for cat, n in sorted(cat_rows.items(), key=lambda kv: -kv[1]):
        print(f"    {cat}: {n} 行")
    print()
    if bad_ids:
        print("  ⚠️ 存在未对齐 product_id（应重跑）")
    if name_cats:
        print("  ⚠️ 存在不含类目的 product_name（应重跑）")


def main() -> dict:
    parser = argparse.ArgumentParser(description="生成商品库对齐的订单明细")
    parser.add_argument("--users", type=int, default=2000, help="合成用户数（默认 2000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--no-sql", action="store_true", help="只写 JSON，不写 SQL")
    parser.add_argument("--dry-run", action="store_true", help="只统计，不写文件")
    args = parser.parse_args()

    products = load_catalog()
    orders = generate_orders(products, n_users=args.users, seed=args.seed)

    if args.dry_run:
        _report(products, orders)
        return {"rows": len(orders)}

    write_json(orders)
    if not args.no_sql:
        write_sql(orders)
    _report(products, orders)
    return {"rows": len(orders)}


if __name__ == "__main__":
    main()
