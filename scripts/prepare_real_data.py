"""准备真实数据：京东商品分类映射 + 电商 NER 清洗。

产出（写入 data/processed/）：
  classify_train_real.csv              全部真实样本 (text,label)
  classify_train_real_train.csv         训练集（京东 sample_train，去重）
  classify_train_real_val.csv          验证集（京东 sample_test，剔除与训练集重叠）
  jd_category_mapping.json             映射表与统计
  ner/train.txt|dev.txt|test.txt       清洗后的 BIO 序列标注
  ner/train.jsonl|dev.jsonl|test.jsonl 实体级格式（供后续 NER 训练/评估）
  data_report.txt                      生成摘要
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_JD = PROJECT_ROOT / "data" / "raw" / "jd_product_categories"
RAW_NER = PROJECT_ROOT / "data" / "raw" / "NER"
OUT = PROJECT_ROOT / "data" / "processed"
NER_OUT = OUT / "ner"

JD_DOMAINS = ["digital products", "fresh", "household appliances"]
JD_FILES = ["sample_train.txt", "sample_test.txt"]

# 京东一级类目 -> 项目 15 类（labels.txt 中的标准名称）
JD_L1_MAP = {
    # 数码产品域
    "影音娱乐": "手机数码",
    "摄影摄像": "手机数码",
    "数码配件": "手机数码",
    "智能设备": "手机数码",
    "电子教育": "手机数码",
    # 生鲜域
    "乳品冷饮": "食品生鲜",
    "水果": "食品生鲜",
    "海鲜水产": "食品生鲜",
    "猪牛羊肉": "食品生鲜",
    "禽肉蛋品": "食品生鲜",
    "蔬菜": "食品生鲜",
    "速食熟食": "食品生鲜",
    "面点烘焙": "食品生鲜",
    # 家用电器域
    "厨卫大电": "家用电器",
    "厨房小电": "家用电器",
    "商用电器": "家用电器",
    "大家电": "家用电器",
    "家电服务": "家用电器",
    "家电配件": "家用电器",
    "生活电器": "家用电器",
    "视听影音": "家用电器",
}

# 个护健康：按摩/理疗类归医药保健（labels.txt 医药保健含"按摩仪/护具"），其余归家用电器
HEALTH_KW = re.compile(r"按摩|理疗|足浴|泡脚|足蒸|热敷|艾灸|牵引|护颈|护腰|拔罐")

PROJECT_LABELS = [
    "医药保健",
    "图书音像",
    "宠物用品",
    "家居家装",
    "家用电器",
    "手机数码",
    "服装鞋包",
    "母婴用品",
    "汽车用品",
    "珠宝首饰",
    "礼品鲜花",
    "箱包配饰",
    "美妆护肤",
    "运动户外",
    "食品生鲜",
]


def map_jd_category(l1: str, title: str) -> str:
    """京东一级类目 -> 15 类；个护健康按关键词细分。"""
    if l1 == "个护健康":
        return "医药保健" if HEALTH_KW.search(title) else "家用电器"
    return JD_L1_MAP.get(l1, "")


def build_jd_dataset() -> dict:
    """读取京东 6 个文件，映射标签，去重并拆分 train/val。"""
    records: list[dict] = []  # text, label, jd_l1, jd_l2, domain, split
    for domain in JD_DOMAINS:
        for fname in JD_FILES:
            split = "train" if "train" in fname else "val"
            path = RAW_JD / domain / fname
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    # 格式: 一级,一级@二级,商品标题
                    parts = line.split(",", 2)
                    if len(parts) != 3:
                        continue
                    l1, l1l2, title = parts
                    label = map_jd_category(l1, title)
                    if not label or not title.strip():
                        continue
                    l2 = l1l2.split("@", 1)[1] if "@" in l1l2 else ""
                    records.append(
                        {
                            "text": title.strip(),
                            "label": label,
                            "jd_l1": l1,
                            "jd_l2": l2,
                            "domain": domain,
                            "split": split,
                        }
                    )

    # 全局去重（按标题），并记录顺序
    seen: set[str] = set()
    unique = []
    for rec in records:
        key = rec["text"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(rec)

    train = [r for r in unique if r["split"] == "train"]
    val = [r for r in unique if r["split"] == "val"]
    # 剔除验证集中与训练集标题重复的样本，避免泄漏
    train_texts = {r["text"] for r in train}
    val = [r for r in val if r["text"] not in train_texts]

    def write_csv(rows, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            import csv

            writer = csv.writer(f)
            writer.writerow(["text", "label"])
            for r in rows:
                writer.writerow([r["text"], r["label"]])

    write_csv(unique, OUT / "classify_train_real.csv")
    write_csv(train, OUT / "classify_train_real_train.csv")
    write_csv(val, OUT / "classify_train_real_val.csv")

    stats = {
        "total_raw": len(records),
        "total_unique": len(unique),
        "train": len(train),
        "val": len(val),
        "label_dist": dict(Counter(r["label"] for r in unique)),
        "domain_dist": dict(Counter(r["domain"] for r in unique)),
        "jd_l1_dist": dict(Counter(r["jd_l1"] for r in unique)),
    }
    with open(OUT / "jd_category_mapping.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mapping": {"个护健康": "按摩/理疗类->医药保健，其余->家用电器", **JD_L1_MAP},
                "stats": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return stats


def clean_ner() -> dict:
    """清洗 NER 的 BIO 文件，并生成实体级 JSONL。"""
    stats: dict = {}
    for split in ["train", "dev", "test"]:
        src = RAW_NER / f"{split}.txt"
        sentences: list[list[tuple[str, str]]] = []
        malformed = 0
        cur: list[tuple[str, str]] = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    if cur:
                        sentences.append(cur)
                        cur = []
                    continue
                parts = line.split()
                if len(parts) != 2:
                    malformed += 1
                    continue
                tok, lab = parts
                if not tok or not lab:
                    malformed += 1
                    continue
                cur.append((tok, lab))
        if cur:
            sentences.append(cur)

        # 写清洗后的 BIO
        NER_OUT.mkdir(parents=True, exist_ok=True)
        with open(NER_OUT / f"{split}.txt", "w", encoding="utf-8") as f:
            for sent in sentences:
                for tok, lab in sent:
                    f.write(f"{tok} {lab}\n")
                f.write("\n")

        # 写实体级 JSONL：{text, entities:[{label,text,start,end}]}
        entity_counter = Counter()
        with open(NER_OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for sent in sentences:
                text = "".join(t for t, _ in sent)
                entities = []
                i = 0
                while i < len(sent):
                    tok, lab = sent[i]
                    if lab.startswith("B-") or (lab.startswith("I-") and (i == 0 or not sent[i - 1][1].startswith(("B-", "I-")))):
                        ent_type = lab.split("-", 1)[1]
                        start = len("".join(t for t, _ in sent[:i]))
                        j = i + 1
                        while j < len(sent) and sent[j][1] == f"I-{ent_type}":
                            j += 1
                        ent_text = text[start : start + sum(len(t) for _, t in sent[i:j])]
                        entities.append(
                            {
                                "label": ent_type,
                                "text": ent_text,
                                "start": start,
                                "end": start + len(ent_text),
                            }
                        )
                        entity_counter[ent_type] += 1
                        i = j
                    else:
                        i += 1
                f.write(json.dumps({"text": text, "entities": entities}, ensure_ascii=False) + "\n")

        stats[split] = {
            "sentences": len(sentences),
            "tokens": sum(len(s) for s in sentences),
            "malformed_lines_removed": malformed,
            "entities": dict(entity_counter),
        }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="准备京东分类 + NER 真实数据")
    ap.add_argument("--skip-jd", action="store_true", help="跳过京东数据")
    ap.add_argument("--skip-ner", action="store_true", help="跳过 NER 数据")
    args = ap.parse_args()

    report: dict = {"labels_15": PROJECT_LABELS}
    if not args.skip_jd:
        report["jd"] = build_jd_dataset()
    if not args.skip_ner:
        report["ner"] = clean_ner()

    with open(OUT / "data_report.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[Done] 输出目录: {OUT}")


if __name__ == "__main__":
    main()
