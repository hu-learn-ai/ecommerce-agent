"""4 类分类模型缺陷检查（真实短查询 + 边界 + 域外 + 混淆分析）。

用法:
    python scripts/probe_classify_model.py                 # 默认模型 models/best
    python scripts/probe_classify_model.py --model_path <目录>
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch

from models.persistence import load_classification_model

# (查询, 期望类别或 None=域外)
CASES = [
    # 手机数码（短查询，无品类词）
    ("iPhone 15 Pro Max", "手机数码"),
    ("iPhone 15 Pro Max 钛金属", "手机数码"),
    ("华为Mate60 Pro", "手机数码"),
    ("Mate60 Pro", "手机数码"),
    ("红米Note13", "手机数码"),
    ("小米14", "手机数码"),
    ("荣耀90", "手机数码"),
    ("华为平板", "手机数码"),
    ("索尼相机", "手机数码"),
    ("蓝牙耳机", "手机数码"),
    ("苹果手表", "手机数码"),
    ("小米手环", "手机数码"),
    # 家用电器
    ("海尔冰箱", "家用电器"),
    ("格力空调", "家用电器"),
    ("美的电饭煲", "家用电器"),
    ("戴森吸尘器", "家用电器"),
    ("扫地机器人", "家用电器"),
    ("微波炉", "家用电器"),
    ("空气炸锅", "家用电器"),
    ("咖啡机", "家用电器"),
    ("电动牙刷", "家用电器"),
    ("冲牙器", "家用电器"),
    # 食品生鲜
    ("车厘子", "食品生鲜"),
    ("五常大米", "食品生鲜"),
    ("纯牛奶", "食品生鲜"),
    ("牛排", "食品生鲜"),
    ("安溪铁观音茶叶", "食品生鲜"),
    ("智利车厘子2斤装", "食品生鲜"),
    ("阳澄湖大闸蟹", "食品生鲜"),
    ("现摘草莓", "食品生鲜"),
    # 医药保健
    ("松下按摩椅", "医药保健"),
    ("鱼跃血压计", "医药保健"),
    ("汤臣倍健维生素C", "医药保健"),
    ("四季沐歌泡脚桶", "医药保健"),
    ("眼部按摩仪", "医药保健"),
    ("艾灸仪", "医药保健"),
    ("家用雾化器", "医药保健"),
    # 边界/歧义（标注合理预期）
    ("小米充电宝", "手机数码"),
    ("kindle电子书阅读器", "手机数码"),
    # 域外（4 类模型没有的类）
    ("红色连衣裙", None),
    ("口红", None),
    ("纸尿裤", None),
    ("篮球鞋", None),
    ("猫粮", None),
    ("汽车机油", None),
    ("黄金项链", None),
    ("实木书桌", None),
    ("考研数学教材", None),
    ("鲜花礼盒", None),
    # 边界输入
    ("", None),
    ("abc123", None),
    ("123456", None),
    ("😀😀😀", None),
    ("a" * 300, None),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "best"),
    )
    args = ap.parse_args()

    tok, model, labels, dev = load_classification_model(args.model_path)
    in_domain = [(q, e) for q, e in CASES if e is not None]
    [(q, e) for q, e in CASES if e is None]

    def predict(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt").to(dev)
        with torch.no_grad():
            return torch.softmax(model(**enc).logits, -1)

    print(f"\n{'查询':<26}{'预测':<9}{'置信度':<9}{'期望':<9}判定")
    print("-" * 70)
    wrong = 0
    low_conf = 0
    confusion = Counter()
    for i in range(0, len(CASES), 16):
        batch = CASES[i : i + 16]
        probs = predict([q for q, _ in batch])
        for (q, exp), p in zip(batch, probs):
            idx = p.argmax().item()
            pred, conf = labels[idx], p[idx].item()
            if conf < 0.60:
                pred = "无法判断"
                low_conf += 1
            if exp is None:
                mark = "域外" if pred == "无法判断" else f"域外->{pred}"
                if pred != "无法判断":
                    confusion[(exp or "域外", pred)] += 1
            elif pred == exp:
                mark = "OK"
            else:
                mark = "FAIL"
                wrong += 1
                confusion[(exp, pred)] += 1
            print(f"{q[:26]:<26}{pred:<9}{conf:6.1%}  {str(exp or '-'):<9}{mark}")

    acc = (len(in_domain) - wrong) / len(in_domain)
    print(f"\n域内准确率: {acc:.1%} ({len(in_domain)-wrong}/{len(in_domain)})")
    print(f"低置信度(<60%)触发'无法判断': {low_conf} 条")
    print("混淆对（期望->预测）:")
    for (e, p), c in confusion.most_common():
        print(f"  {e or '域外'} -> {p}: {c}")


if __name__ == "__main__":
    main()
