"""构建分类模型域外拒识参考统计（逐类 Mahalanobis 距离 + 阈值）。

原理：BERT 最后一层 [CLS] 输出在高维空间里，训练分布内的文本距离本类中心
（按本类协方差计算的马氏距离）较小；域外文本（模型 4 类覆盖不到的品类）
距离所有类中心都显著更远。推理时取「最近类的马氏距离」作为 OOD 分数，
超过阈值即拒识，弥补 softmax 置信度恒高的缺陷。

用法:
    python scripts/build_ood_stats.py
    python scripts/build_ood_stats.py --sample 6000 --output models/best/ood_stats.npz

输出:
    ood_stats.npz — means / precs / labels / threshold
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# 固定域外探针（模型 4 类不覆盖的品类），用于阈值校验
OOD_PROBES = [
    "跑步机 家用电动 静音",
    "儿童玩具 积木 益智",
    "保温杯 316不锈钢 500ml",
    "雨伞 晴雨两用 自动",
    "拖鞋 浴室防滑 家用",
    "宜家简约书桌 实木",
    "口红 哑光丝绒",
    "红色连衣裙 夏季新款",
    "猫粮 全价成猫粮 10kg",
    "考研数学教材 高数",
    "黄金项链 足金",
    "纸尿裤 超薄透气 大码",
    "吉他 民谣单板 初学者",
    "山地自行车 21速 变速",
    "汽车机油 全合成 5W-30",
    "小提琴 成人 入门级",
]

# 短查询域内参照（真实用户/LLM 常用输入风格，用于校准小类协方差与阈值）
SHORT_QUERY_REFS = {
    "手机数码": [
        "华为Mate60 Pro", "iPhone 15 Pro Max", "小米14", "红米Note13",
        "蓝牙耳机", "索尼相机", "华为平板", "充电宝 20000mAh",
        "华为Mate60 Pro 12+512G 5G手机", "iPhone 15 Pro Max 256GB 钛金属",
        "红米Note13 8+256G", "小米蓝牙耳机 降噪版 黑色",
    ],
    "家用电器": [
        "海尔冰箱", "格力空调", "美的电饭煲", "戴森吸尘器",
        "扫地机器人", "微波炉", "空气炸锅", "咖啡机",
        "戴森 V12 Detect Slim 无线吸尘器", "海尔冰箱 一级能效 双开门",
        "格力空调 1.5匹 新一级能效",
    ],
    "食品生鲜": [
        "车厘子", "五常大米", "纯牛奶", "牛排",
        "安溪铁观音茶叶", "阳澄湖大闸蟹", "现摘草莓",
        "新鲜智利车厘子JJ级 2斤装", "安溪铁观音茶叶 500g",
    ],
    "医药保健": [
        "松下按摩椅", "鱼跃血压计", "汤臣倍健维生素C",
        "四季沐歌泡脚桶", "眼部按摩仪", "艾灸仪",
        "鱼跃血压计 家用上臂式", "四季沐歌泡脚桶 恒温按摩",
        "汤臣倍健维生素C 咀嚼片",
    ],
}


def parse_args():
    ap = argparse.ArgumentParser(description="构建分类模型域外拒识统计")
    ap.add_argument(
        "--data",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "processed", "classify_train_real_train.csv"),
        help="训练数据 CSV（text,label）",
    )
    ap.add_argument("--sample", type=int, default=6000, help="抽样条数")
    ap.add_argument("--max_length", type=int, default=64, help="编码最大长度（特征用）")
    ap.add_argument("--batch", type=int, default=128, help="编码批次大小")
    ap.add_argument(
        "--encode_batch",
        type=int,
        default=1,
        help="特征编码批次（默认 1：与运行时单条推理一致，避免动态量化批内缩放差异）",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "best", "ood_stats.npz"),
        help="输出 npz 路径",
    )
    return ap.parse_args()


def load_sample(csv_path: str, sample: int, seed: int = 42):
    """读取 CSV 并随机抽样，返回 (texts, labels)。"""
    if not os.path.isfile(csv_path):
        raise SystemExit(f"[OOD] 训练数据不存在: {csv_path}")
    texts, labels = [], []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(row["label"].strip())
    # 分层抽样：每个类别最多 sample/len(classes) 条，保证小类（医药保健）充分覆盖
    rng = random.Random(seed)
    per_class = max(200, sample // len(set(labels)))
    by_label: dict = {}
    for t, lbl in zip(texts, labels):
        by_label.setdefault(lbl, []).append(t)
    sampled = []
    for lbl, ts in by_label.items():
        picked = rng.sample(ts, min(per_class, len(ts)))
        sampled.extend((t, lbl) for t in picked)
    print(f"[OOD] 加载 {len(texts)} 条，抽样 {len(sampled)} 条")
    print(f"[OOD] 分布: {dict(Counter(lbl for _, lbl in sampled))}")
    return [t for t, _ in sampled], [lbl for _, lbl in sampled]


def main() -> None:
    args = parse_args()
    import numpy as np
    import torch

    from models.persistence import load_classification_model

    texts, labels = load_sample(args.data, args.sample)
    label_list = sorted(set(labels))
    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    tok, model, model_labels, device = load_classification_model(
        os.path.dirname(args.output)
    )
    # 输出目录内 labels.txt 为准，保证与线上一致
    label_list = list(model_labels)
    label2id = {lbl: i for i, lbl in enumerate(label_list)}

    print(f"[OOD] 模型类: {label_list}")
    model.eval()

    # 追加短查询域内参照（真实用户输入风格），并记录索引用于阈值校准
    ref_indices: list = []
    for cat, qs in SHORT_QUERY_REFS.items():
        if cat not in label2id:
            continue
        ref_indices.extend(range(len(texts), len(texts) + len(qs)))
        texts = texts + qs
        labels = labels + [cat] * len(qs)

    def encode_with_logits(texts_batch: list):
        # 动态量化的激活缩放与批内组成有关；为保证与运行时（单条推理）一致，默认逐条编码
        if args.encode_batch == 1:
            feats, logits = [], []
            for t in texts_batch:
                enc = tok(
                    [t],
                    max_length=args.max_length,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                feat = out.hidden_states[-1][:, 0].detach().cpu().numpy().astype(np.float32)
                feat /= np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
                feats.append(feat)
                logits.append(out.logits.detach().cpu().numpy().astype(np.float32))
            return np.vstack(feats), np.vstack(logits)
        enc = tok(
            texts_batch,
            max_length=args.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        feat = out.hidden_states[-1][:, 0].detach().cpu().numpy().astype(np.float32)
        feat /= np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-8
        logits = out.logits.detach().cpu().numpy().astype(np.float32)
        return feat, logits

    # 1. 训练样本特征
    print("[OOD] 编码训练样本特征...")
    feats, logits, ys = [], [], []
    for i in range(0, len(texts), args.batch):
        f, lgt = encode_with_logits(texts[i : i + args.batch])
        feats.append(f)
        logits.append(lgt)
        ys.extend(label2id[lbl] for lbl in labels[i : i + args.batch])
    feats = np.vstack(feats)
    logits = np.vstack(logits)
    ys = np.asarray(ys, dtype=np.int64)

    # 2. 逐类均值与协方差（正则化收缩，保证可逆）
    dim = feats.shape[1]
    means = np.zeros((len(label_list), dim), dtype=np.float32)
    precs = np.zeros((len(label_list), dim, dim), dtype=np.float32)
    for c in range(len(label_list)):
        Xc = feats[ys == c].astype(np.float64)
        mu = Xc.mean(axis=0)
        means[c] = mu
        centered = Xc - mu
        cov = (centered.T @ centered) / max(1, len(Xc))
        # 收缩正则：λ = 1e-2 * trace/dim，保证小类（医药保健仅百级样本）可逆
        lam = 1e-2 * (np.trace(cov) / dim)
        cov_reg = cov + lam * np.eye(dim)
        precs[c] = np.linalg.inv(cov_reg)

    def maha_to_class(X: np.ndarray, cls_ids: np.ndarray) -> np.ndarray:
        """返回每个样本到其预测类的马氏距离（平方根）。"""
        out = np.zeros(len(X), dtype=np.float64)
        for c in range(len(label_list)):
            mask = cls_ids == c
            if not mask.any():
                continue
            d = X[mask].astype(np.float64) - means[c]
            q = np.einsum("bi,ij,bj->b", d, precs[c], d)
            out[mask] = np.sqrt(np.maximum(q, 0.0))
        return out

    # 3. 域内样本到自身预测类的距离分布（分位阈值）
    pred_ids = logits.argmax(axis=-1)
    in_scores = maha_to_class(feats, pred_ids)
    per_class_thr = {}
    for c in range(len(label_list)):
        d = in_scores[pred_ids == c]
        if len(d) == 0:
            per_class_thr[c] = 60.0
            continue
        p995 = float(np.quantile(d, 0.995))
        # 下限 55：避免小类协方差过紧导致真实短查询被误拒
        per_class_thr[c] = max(p995, 55.0)
    print("[OOD] 逐类 p99.5 距离 / 采用阈值:")
    for c in range(len(label_list)):
        d = in_scores[pred_ids == c]
        print(
            f"  {label_list[c]}: p50={np.quantile(d, 0.5):.2f} "
            f"p95={np.quantile(d, 0.95):.2f} p99.5={np.quantile(d, 0.995):.2f} "
            f"max={d.max():.2f} -> thr={per_class_thr[c]:.2f}"
        )

    # 短查询参照距离（保证真实短查询不被误拒）
    ref_dist_by_class = {c: 0.0 for c in range(len(label_list))}
    for i in ref_indices:
        c = int(pred_ids[i])
        ref_dist_by_class[c] = max(ref_dist_by_class[c], float(in_scores[i]))
    for c in range(len(label_list)):
        # 阈值至少覆盖该类所有短查询参照（×1.05 余量）
        per_class_thr[c] = max(per_class_thr[c], ref_dist_by_class[c] * 1.05)

    print("[OOD] 短查询参照最大距离 / 修正后阈值:")
    for c in range(len(label_list)):
        print(f"  {label_list[c]}: ref_max={ref_dist_by_class[c]:.2f} thr={per_class_thr[c]:.2f}")

    # 4. 域外探针 OOD 分数（预测类 -> 该类马氏距离，逐类阈值判定）
    ood_feats, ood_logits = encode_with_logits(OOD_PROBES)
    ood_pred = ood_logits.argmax(axis=-1)
    ood_scores = maha_to_class(ood_feats, ood_pred)
    ood_gate_reject = 0
    print("[OOD] 域外探针（预测类 -> 该类马氏距离 / 是否触发门控）:")
    for q, s, c in zip(OOD_PROBES, ood_scores, ood_pred):
        hit = s > per_class_thr[c]
        ood_gate_reject += int(hit)
        print(f"  {q[:24]:<26} {label_list[c]} {s:7.2f} / thr={per_class_thr[c]:.2f} {'REJECT' if hit else 'pass'}")
    print(f"[OOD] 域外探针门控拒识: {ood_gate_reject}/{len(OOD_PROBES)}")

    in_reject = float((in_scores > np.array([per_class_thr[c] for c in pred_ids])).mean())
    print(f"[OOD] 域内误拒率（校准集，含短查询）: {in_reject:.2%}")

    thresholds = np.array([per_class_thr[c] for c in range(len(label_list))], dtype=np.float32)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(
        args.output,
        means=means,
        precs=precs,
        labels=np.asarray(label_list),
        thresholds=thresholds,
    )
    with open(args.output + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "per_class_mahalanobis",
                "sample": len(texts),
                "max_length": args.max_length,
                "thresholds": {label_list[c]: float(thresholds[c]) for c in range(len(label_list))},
                "in_domain_reject_rate": in_reject,
                "ood_probe_gate_reject": ood_gate_reject,
                "labels": label_list,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[OOD] [OK] 已保存: {args.output}")


if __name__ == "__main__":
    main()
