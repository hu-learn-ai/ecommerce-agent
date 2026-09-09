"""从京东真实数据生成商品库 products_for_faiss.json。

数据源: data/processed/classify_train_real.csv（京东 4 类真实标题）
输出:   data/processed/products_for_faiss.json

用法:
    python scripts/build_product_catalog.py                # 默认 60000 条（每类 15000）
    python scripts/build_product_catalog.py --max 30000    # 自定义总量
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "classify_train_real.csv")
OUTPUT_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "products_for_faiss.json")
EXTRA_DIGITAL_TSV = os.path.join(PROJECT_ROOT, "data", "raw", "jddata", "data6.txt")

# 补充真实数码产品（kkllffyo/jddata: 手机/电脑/平板），各取 N 条，带价格
EXTRA_DIGITAL_SAMPLE = {"手机": 1500, "电脑": 1000, "平板": 500}

# 去掉标题开头的【包邮】/（特价）等装饰段，再取品牌候选
_LEADING_DECOR = re.compile(r"^[【\[(（][^】\]）)]*[】\]）)]")


def extract_brand(title: str) -> str:
    """从京东标题中启发式提取品牌（取第一个分隔符前的片段）。"""
    t = _LEADING_DECOR.sub("", title.strip())
    m = re.match(r"^([^\s（(【\[\|/,，:：·]+)", t)
    if not m:
        return ""
    cand = m.group(1).strip().strip("·:：-—")
    if not cand or len(cand) > 12:
        return ""
    # 中文品牌 + 拉丁型号连写: 苹果iPhone15 -> 苹果
    m2 = re.match(r"^([\u4e00-\u9fa5]{2,6})(?=[A-Za-z0-9])", cand)
    if m2:
        return m2.group(1)
    # 拉丁品牌 + 中文品类连写: OPPO手机 -> OPPO
    m3 = re.match(r"^([A-Za-z]{2,8})(?=[\u4e00-\u9fa5])", cand)
    if m3:
        return m3.group(1)
    if re.fullmatch(r"[\d\W_]+", cand):
        return ""
    # 明显是营销词而非品牌
    if cand.lower() in {"正品", "官方", "旗舰", "全新", "热卖", "包邮", "特价", "新款", "厂家", "现货"}:
        return ""
    return cand


def build_catalog(max_products: int) -> None:
    random.seed(42)
    rows_by_label: dict[str, list[tuple[str, str]]] = {}
    with open(INPUT_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                rows_by_label.setdefault(label, []).append((text, label))

    # 分层抽样：每类平均分配
    labels = sorted(rows_by_label)
    per_class = max(1, max_products // len(labels))
    sampled: list[tuple[str, str]] = []
    for label in labels:
        rows = rows_by_label[label]
        sampled += random.sample(rows, min(per_class, len(rows)))
    random.shuffle(sampled)
    sampled = sampled[:max_products]

    products = []
    for i, (title, label) in enumerate(sampled, 1):
        products.append(
            {
                "id": f"JD{i:06d}",
                "name": title,
                "description": "",
                "category": label,
                "brand": extract_brand(title),
            }
        )

    # 补充真实手机/电脑/平板（京东数码域缺手机，kkllffyo/jddata 补齐）
    if os.path.isfile(EXTRA_DIGITAL_TSV):
        extra = []
        for cat, n in EXTRA_DIGITAL_SAMPLE.items():
            rows = []
            with open(EXTRA_DIGITAL_TSV, encoding="utf-8") as f:
                header = f.readline().rstrip("\r\n").split("\t")
                idx = {name: i for i, name in enumerate(header)}
                for line in f:
                    parts = line.rstrip("\r\n").split("\t")
                    if len(parts) <= max(idx.values()):
                        continue
                    if parts[idx["分类"]] == cat:
                        title = parts[idx["标题"]].strip()
                        brand = parts[idx["品牌"]].strip()
                        price = parts[idx["优惠价"]].strip() or parts[idx["原价"]].strip()
                        rows.append({"title": title, "brand": brand, "price": price})
            random.shuffle(rows)
            for r in rows[:n]:
                extra.append(r)
        random.shuffle(extra)
        for i, r in enumerate(extra, 1):
            products.append(
                {
                    "id": f"KD{i:06d}",
                    "name": r["title"],
                    "description": "",
                    "category": "手机数码",
                    "brand": r["brand"],
                    "price": r["price"],
                }
            )
        print(f"[Catalog] 补充真实数码产品: {len(extra)} 条（含价格）")
    else:
        print(f"[Catalog] 未找到补充数据: {EXTRA_DIGITAL_TSV}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False, indent=1)

    dist = Counter(p["category"] for p in products)
    brands = sum(1 for p in products if p["brand"])
    print(f"[Catalog] 商品库已生成: {OUTPUT_JSON}")
    print(f"[Catalog] 商品数: {len(products)}")
    print(f"[Catalog] 类目分布: {dict(dist)}")
    print(f"[Catalog] 提取到品牌: {brands} ({brands / len(products):.1%})")


def main() -> None:
    ap = argparse.ArgumentParser(description="生成京东真实商品库")
    ap.add_argument("--max", type=int, default=60000, help="商品总数（默认 60000）")
    args = ap.parse_args()
    build_catalog(args.max)


if __name__ == "__main__":
    main()
