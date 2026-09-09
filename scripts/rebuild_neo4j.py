"""重建 Neo4j 产品知识图谱（真实京东数据）。

清空旧合成图（User/SPU/SKU/Trademark/Category3），导入:
  - SPU: data/processed/products_for_faiss.json 的 4000 条真实商品
  - Trademark: 提取到的品牌（按出现频次过滤，>= --min_brand_count）
  - Category3: 4 个大类（沿用旧标签名，保证现有 Cypher 兼容）
  - SKU: 每个 SPU 一个（无价格，保持 Have 关系与查询兼容）
  - 关系: (SPU)-[:Belong]->(Category3), (SPU)-[:Have]->(Trademark), (SPU)-[:Have]->(SKU)

用法:
    python scripts/rebuild_neo4j.py                 # 默认从 .env 读取连接，品牌阈值 3
    python scripts/rebuild_neo4j.py --min-brand-count 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

PRODUCTS_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json")


def main() -> None:
    ap = argparse.ArgumentParser(description="重建 Neo4j 产品知识图谱")
    ap.add_argument("--uri", default=os.getenv("NEO4J_URI", "neo4j://localhost:7687"))
    ap.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", ""))
    ap.add_argument("--min-brand-count", type=int, default=3, help="品牌出现次数阈值")
    args = ap.parse_args()

    from neo4j import GraphDatabase

    with open(PRODUCTS_JSON, encoding="utf-8") as f:
        products = json.load(f)
    print(f"[Rebuild] 商品数: {len(products)}")

    # 品牌统计（过滤）
    brand_counter = Counter(p.get("brand", "").strip() for p in products if p.get("brand"))
    keep_brands = {b for b, c in brand_counter.items() if c >= args.min_brand_count}
    print(f"[Rebuild] 品牌总数: {len(brand_counter)}，保留(>= {args.min_brand_count}次): {len(keep_brands)}")

    categories = sorted({p["category"] for p in products})
    print(f"[Rebuild] 类目: {categories}")

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    try:
        with driver.session() as session:
            # 1. 清空旧图
            print("[Rebuild] 清空旧图...")
            session.run("MATCH (n) DETACH DELETE n")

            # 2. 建类目 + 品牌
            print("[Rebuild] 创建类目/品牌节点...")
            session.run("UNWIND $cats AS c CREATE (:Category3 {name: c})", cats=categories)
            session.run("UNWIND $brands AS b CREATE (:Trademark {name: b})", brands=sorted(keep_brands))

            # 3. 批量导入商品（分批）
            BATCH = 500
            for i in range(0, len(products), BATCH):
                batch = products[i : i + BATCH]
                session.run(
                    """
                    UNWIND $rows AS row
                    CREATE (p:SPU {id: row.id, name: row.name})
                    WITH p, row
                    MATCH (c:Category3 {name: row.category})
                    CREATE (p)-[:Belong]->(c)
                    CREATE (p)-[:Have]->(:SKU {id: row.id, price: row.price})
                    WITH p, row
                    FOREACH (_ IN CASE WHEN row.brand IN $keep_brands THEN [1] ELSE [] END |
                        MERGE (t:Trademark {name: row.brand})
                        CREATE (p)-[:Have]->(t)
                    )
                    """,
                    rows=[
                        {
                            "id": p["id"],
                            "name": p["name"],
                            "category": p["category"],
                            "brand": p.get("brand", ""),
                            "price": p.get("price"),
                        }
                        for p in batch
                    ],
                    keep_brands=list(keep_brands),
                )
                print(f"  进度: {min(i + BATCH, len(products))}/{len(products)}")

            # 4. 校验
            counts = session.run(
                "MATCH (n) RETURN labels(n)[0] AS l, count(*) AS c ORDER BY c DESC"
            ).data()
            print("\n[Rebuild] 重建后节点分布:")
            for row in counts:
                print(f"  {row['l']}: {row['c']}")
            rel = session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY c DESC").data()
            print("[Rebuild] 关系分布:")
            for row in rel:
                print(f"  {row['t']}: {row['c']}")
    finally:
        driver.close()
    print("\n[Rebuild] [OK] Neo4j 产品图重建完成")


if __name__ == "__main__":
    main()
