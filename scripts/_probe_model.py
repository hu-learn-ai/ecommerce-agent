"""临时脚本：4 类模型真实查询鲁棒性测试（用后即删）。"""
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

import torch

from models.persistence import load_classification_model

tok, model, labels, dev = load_classification_model("models/best")

cases = [
    ("iPhone 15 Pro Max", "手机数码"),
    ("iPhone 15 Pro Max 钛金属", "手机数码"),
    ("华为Mate60 Pro", "手机数码"),
    ("小米14", "手机数码"),
    ("蓝牙耳机", "手机数码"),
    ("红米Note13", "手机数码"),
    ("华为平板", "手机数码"),
    ("索尼相机", "手机数码"),
    ("海尔冰箱", "家用电器"),
    ("格力空调", "家用电器"),
    ("美的电饭煲", "家用电器"),
    ("戴森吸尘器", "家用电器"),
    ("扫地机器人", "家用电器"),
    ("微波炉", "家用电器"),
    ("车厘子", "食品生鲜"),
    ("五常大米", "食品生鲜"),
    ("纯牛奶", "食品生鲜"),
    ("牛排", "食品生鲜"),
    ("安溪铁观音茶叶", "食品生鲜"),
    ("智利车厘子2斤装", "食品生鲜"),
    ("松下按摩椅", "医药保健"),
    ("鱼跃血压计", "医药保健"),
    ("汤臣倍健维生素C", "医药保健"),
    ("四季沐歌泡脚桶", "医药保健"),
    ("眼部按摩仪", "医药保健"),
    ("艾灸仪", "医药保健"),
    ("小米手环", "手机数码"),
    ("电动牙刷", "家用电器"),
    ("空气炸锅", "家用电器"),
    ("咖啡机", "家用电器"),
    ("冲牙器", "家用电器"),
    ("红色连衣裙", None),
    ("口红", None),
    ("纸尿裤", None),
    ("篮球鞋", None),
    ("猫粮", None),
]

enc = tok([c[0] for c in cases], padding=True, truncation=True, max_length=64, return_tensors="pt")
with torch.no_grad():
    probs = torch.softmax(model(**enc).logits, -1)

in_domain = [c for c in cases if c[1] is not None]
out_domain = [c for c in cases if c[1] is None]
wrong = 0
print(f"{'查询':<24}{'预测':<8}{'置信度':<9}{'期望':<8}{'判定'}")
for (q, exp), p in zip(cases, probs):
    idx = p.argmax().item()
    pred, conf = labels[idx], p[idx].item()
    if exp is None:
        mark = "(域外)"
    elif pred == exp:
        mark = "OK"
    else:
        mark = "FAIL"
        wrong += 1
    print(f"{q:<24}{pred:<8}{conf:6.1%}  {str(exp or '-'):<8}{mark}")

acc = (len(in_domain) - wrong) / len(in_domain)
print(f"\n域内 {len(in_domain)} 条准确率: {acc:.1%} ({len(in_domain)-wrong}/{len(in_domain)})")
from collections import Counter
print("域外分布:", dict(Counter(labels[p.argmax().item()] for _, p in zip(out_domain, probs[len(in_domain):]))))
