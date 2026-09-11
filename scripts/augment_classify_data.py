"""短查询数据增强：为 4 类分类模型生成真实风格的短查询变体。

问题背景：训练数据为京东完整标题（几乎总带品类关键词），导致模型对
"华为Mate60 Pro""车厘子"这类不带品类词的短查询高置信度误判。

本脚本：
1. 从完整标题生成多种短变体（品牌+型号 / 截断片段 / 核心词）
2. 按类别目标量分层采样，缓解类别不平衡（家用电器 42 万 vs 医药保健 1.6 万）
3. 注入已知难例（实测误判的查询）确保回归

输出: data/processed/classify_train_real_aug_train.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import re
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "classify_train_real_train.csv")
OUTPUT_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "classify_train_real_aug_train.csv")

# 目标样本数（按类别，缓解不平衡）
TARGET_PER_CLASS = {
    "家用电器": 220_000,
    "手机数码": 220_000,
    "食品生鲜": 160_000,
    "医药保健": 160_000,
}

MARKETING_WORDS = (
    "包邮", "正品", "官方", "旗舰", "热卖", "特价", "新款", "现货", "促销",
    "秒杀", "清仓", "优惠", "补贴", "免税", "同款", "通用", "适用", "专用",
    "智能", "家用", "便携", "迷你", "大容量", "多功能", "高颜值", "耐用",
)

# 实测失败/易错查询（注入训练集确保回归）
HARD_CASES = [
    ("华为Mate60 Pro", "手机数码"),
    ("华为Mate60 Pro 手机", "手机数码"),
    ("红米Note13", "手机数码"),
    ("红米Note13 手机", "手机数码"),
    ("华为平板", "手机数码"),
    ("iPhone 15 Pro Max", "手机数码"),
    ("iPhone 15 Pro Max 钛金属", "手机数码"),
    ("iPhone 15 Pro Max 手机", "手机数码"),
    ("车厘子", "食品生鲜"),
    ("五常大米", "食品生鲜"),
    ("安溪铁观音茶叶", "食品生鲜"),
    ("智利车厘子2斤装", "食品生鲜"),
    ("纯牛奶", "食品生鲜"),
    ("鱼跃血压计", "医药保健"),
    ("汤臣倍健维生素C", "医药保健"),
    ("小米手环", "手机数码"),
    ("电动牙刷", "家用电器"),
    ("空气炸锅", "家用电器"),
    ("咖啡机", "家用电器"),
]

# 医药保健品类补充样本（训练数据里只有按摩/足浴类，覆盖血压计/保健品等）
MEDICAL_BRANDS = [
    "鱼跃", "欧姆龙", "九安", "三诺", "松下", "飞利浦", "博朗",
    "汤臣倍健", "善存", "钙尔奇", "同仁堂", "云南白药", "仁和", "哈药",
]
MEDICAL_PRODUCTS = [
    "血压计", "血糖仪", "体温计", "血氧仪", "雾化器", "制氧机",
    "维生素C", "维生素片", "钙片", "鱼油", "蛋白粉", "感冒药",
    "创可贴", "医用口罩", "棉签", "碘伏", "益生菌", "叶酸",
    "护膝", "颈椎按摩仪", "理疗灯",
]

_DECOR = re.compile(r"^[【\[（(][^】\]）)]*[】\]）)]")


def _clean(title: str) -> str:
    t = _DECOR.sub("", title.strip())
    # 去掉重复空格与常见营销词片段
    for w in MARKETING_WORDS:
        t = t.replace(w, "")
    return re.sub(r"\s+", " ", t).strip(" ，,。、|/")


def _variants(title: str) -> set[str]:
    """从一条完整标题生成短查询变体集合。"""
    t = _clean(title)
    if not t:
        return set()
    out = set()

    # 1. 品牌 + 型号片段：中文品牌后跟字母数字型号
    m = re.match(
        r"^([\u4e00-\u9fa5A-Za-z·]{2,10}?)[（(]?([A-Za-z0-9][A-Za-z0-9\s\-./]*?){1,20}",
        t,
    )
    if m:
        brand, model = m.group(1), m.group(2)
        if model.strip():
            out.add(f"{brand} {model.strip()}")
            out.add(model.strip()[:30])

    # 2. 截断变体：按标点切第一段，取 6/10/14/18 字符
    first = re.split(r"[，。|/！!？?；;]", t)[0]
    for n in (6, 10, 14, 18):
        if len(first) > n:
            out.add(first[:n])
        else:
            out.add(first)

    # 3. 品牌 + 前 4 字（近似"品牌+品类"）
    m2 = re.match(r"^([\u4e00-\u9fa5A-Za-z·]{2,10})", t)
    if m2 and len(t) >= 6:
        out.add(m2.group(1) + t[len(m2.group(1)) : len(m2.group(1)) + 4])

    out.add(t[:40])
    return {v for v in out if 2 <= len(v) <= 40}


def main() -> None:
    ap = argparse.ArgumentParser(description="短查询数据增强")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    rows_by_label: dict[str, list[str]] = {}
    with open(INPUT_CSV, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                rows_by_label.setdefault(label, []).append(text)

    # 生成变体（去重）
    pool: dict[str, set[str]] = {lab: set() for lab in rows_by_label}
    for lab, titles in rows_by_label.items():
        for t in titles:
            pool[lab] |= _variants(t)

    # 原始标题也保留（先放池子再统一抽样）
    for lab, titles in rows_by_label.items():
        pool[lab] |= set(titles)

    # 医药保健补充样本：品牌×品类 组合（真实电商查询风格）
    for brand in MEDICAL_BRANDS:
        for prod in MEDICAL_PRODUCTS:
            pool["医药保健"].add(f"{brand} {prod}")
            pool["医药保健"].add(prod)
            pool["医药保健"].add(f"{brand} {prod} 家用")
            pool["医药保健"].add(f"{brand}{prod}")
    print(f"[Augment] 医药保健补充组合样本: 品牌 {len(MEDICAL_BRANDS)} × 品类 {len(MEDICAL_PRODUCTS)}")

    print("[Augment] 变体数量:")
    for lab in sorted(pool):
        print(f"  {lab}: {len(pool[lab])}")

    # 分层抽样到目标量
    sampled: list[tuple[str, str]] = []
    for lab in sorted(TARGET_PER_CLASS):
        items = sorted(pool[lab])
        n = min(TARGET_PER_CLASS[lab], len(items))
        chosen = random.sample(items, n)
        sampled += [(t, lab) for t in chosen]
        print(f"[Augment] {lab}: 抽样 {n}")

    # 注入难例（去重）
    seen = set()
    hard_added = 0
    for t, lab in HARD_CASES:
        key = (t, lab)
        if key not in seen:
            seen.add(key)
            sampled.append(key)
            hard_added += 1

    random.shuffle(sampled)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label"])
        w.writerows(sampled)

    dist = Counter(lbl for _, lbl in sampled)
    print(f"\n[Augment] 输出: {OUTPUT_CSV}")
    print(f"[Augment] 总样本: {len(sampled)}（含难例 {hard_added} 条）")
    print(f"[Augment] 分布: {dict(dist)}")


if __name__ == "__main__":
    main()
