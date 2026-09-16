"""商品分类模型鲁棒性核查 — 回答"验证集 99.6% 是不是数据泄漏/任务太简单"。

三组对照（同一模型，仅换评估子集）：
1. balanced   : 类别均衡子集（1000/类）→ 排除类别不平衡导致的虚高
2. dup / clean: 按"是否与训练集共享 ≥12 字归一化前缀"分成近重复组与干净组
                → 若两者差距很小，说明高分不是靠记住训练样本
3. keyword    : 无模型的关键词规则基线 → 若远低于模型，说明任务不是关键词映射

用法:
    python tests/classify_robustness.py            # 默认每组 2000 条
    python tests/classify_robustness.py --sample 500   # 快速冒烟（CPU 上更快）

输出:
    tests/classify_robustness_report.json
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root, require_assets  # noqa: E402

ensure_project_root()

from config.settings import settings  # noqa: E402

TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "classify_train_real_train.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "classify_train_real_val.csv")
BALANCED_VAL_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "classify_train_real_cpu_val.csv")
PREFIX_LEN = 12

# 无模型的关键词规则（用于"任务是否可被关键词解"的对照）
KEYWORDS = {
    "手机数码": ["手机", "电脑", "笔记本", "平板", "耳机", "相机", "数码", "充电", "蓝牙", "智能",
                 "显示器", "键盘", "鼠标", "路由器", "音箱", "摄像", "存储", "显卡", "手环", "手表"],
    "家用电器": ["冰箱", "洗衣机", "空调", "电视", "电饭", "微波", "吸尘", "电扇", "加湿", "净水",
                 "烤箱", "电磁炉", "热水器", "吹风", "剃须", "家电", "电器", "破壁", "豆浆", "扫地"],
    "食品生鲜": ["零食", "坚果", "水果", "牛奶", "大米", "粮油", "茶", "海鲜", "饼干", "巧克力",
                 "面包", "饮料", "咖啡", "蜂蜜", "熟食", "蔬菜", "鸡蛋", "生鲜", "食品", "乳品"],
    "医药保健": ["药", "维生素", "钙", "血压", "血糖", "口罩", "创可贴", "保健", "鱼油", "益生菌",
                 "胶囊", "颗粒", "口服液", "体温", "膏", "器械", "医用", "营养", "蛋白", "理疗"],
}


def load_rows(path: str) -> list:
    with open(path, encoding="utf-8-sig") as handle:
        return [(r["text"].strip(), r["label"].strip()) for r in csv.DictReader(handle)]


def normalize_prefix(text: str) -> str:
    """去掉非字母数字字符后取前 N 个字符，作为"同款商品"的近似判据。"""
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", text)[:PREFIX_LEN]


def keyword_predict(title: str) -> str:
    best, best_hits = "", 0
    for label, words in KEYWORDS.items():
        hits = sum(1 for word in words if word in title)
        if hits > best_hits:
            best, best_hits = label, hits
    return best or "无命中"


def main() -> dict:
    parser = argparse.ArgumentParser(description="商品分类模型鲁棒性核查")
    parser.add_argument("--sample", type=int, default=2000, help="每组抽样条数")
    args = parser.parse_args()

    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = settings.classify_model_path
    require_assets(
        model_path,
        os.path.join(model_path, "model.safetensors"),
        os.path.join(model_path, "labels.txt"),
    )
    labels = [
        line.strip()
        for line in open(os.path.join(model_path, "labels.txt"), encoding="utf-8")
        if line.strip()
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    def predict(rows: list) -> list:
        texts = [t for t, _ in rows]
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                batch = tokenizer(
                    texts[i : i + 64], padding=True, truncation=True,
                    max_length=128, return_tensors="pt",
                )
                preds += [labels[j] for j in model(**batch).logits.argmax(-1).tolist()]
        return preds

    def score(rows: list) -> dict:
        gold = [label for _, label in rows]
        preds = predict(rows)
        return {
            "n": len(rows),
            "accuracy": round(float(accuracy_score(gold, preds)), 4),
            "macro_f1": round(float(f1_score(gold, preds, average="macro")), 4),
        }

    report = {"prefix_len": PREFIX_LEN, "sample_per_group": args.sample, "groups": {}}

    # 1. 类别均衡子集
    if os.path.exists(BALANCED_VAL_CSV):
        balanced = load_rows(BALANCED_VAL_CSV)
        report["groups"]["balanced"] = score(balanced[: args.sample * 2])

    # 2. 近重复组 vs 干净组（相对训练集）
    train_prefixes = Counter(normalize_prefix(t) for t, _ in load_rows(TRAIN_CSV))
    val_rows = load_rows(VAL_CSV)
    dup = [r for r in val_rows if train_prefixes[normalize_prefix(r[0])] > 0]
    clean = [r for r in val_rows if train_prefixes[normalize_prefix(r[0])] == 0]
    report["dup_ratio"] = round(len(dup) / len(val_rows), 4)
    random.seed(42)
    report["groups"]["val_near_dup"] = score(random.sample(dup, min(args.sample, len(dup))))
    report["groups"]["val_clean"] = score(random.sample(clean, min(args.sample, len(clean))))

    # 3. 关键词规则基线（无模型）
    sample = random.sample(val_rows, min(args.sample, len(val_rows)))
    gold = [label for _, label in sample]
    rule_preds = [keyword_predict(text) for text, _ in sample]
    report["groups"]["keyword_rule"] = {
        "n": len(sample),
        "accuracy": round(float(accuracy_score(gold, rule_preds)), 4),
        "macro_f1": round(float(f1_score(gold, rule_preds, average="macro")), 4),
        "no_hit_ratio": round(
            sum(1 for p in rule_preds if p == "无命中") / len(rule_preds), 4
        ),
    }

    print("=" * 60)
    print("  商品分类模型鲁棒性核查")
    print("=" * 60)
    print(f"  验证集中与训练集共享 ≥{PREFIX_LEN} 字前缀的比例: {report['dup_ratio']:.2%}")
    for name, metrics in report["groups"].items():
        extra = (
            f"  无命中率={metrics['no_hit_ratio']:.2%}"
            if "no_hit_ratio" in metrics
            else ""
        )
        print(
            f"  [{name:14s}] n={metrics['n']:5d}  acc={metrics['accuracy']:.4f}  "
            f"macroF1={metrics['macro_f1']:.4f}{extra}"
        )

    out = os.path.join(PROJECT_ROOT, "tests", "classify_robustness_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存: {out}")
    return report


if __name__ == "__main__":
    main()
