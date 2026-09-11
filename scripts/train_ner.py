"""电商 NER 模型训练脚本（bert-base-chinese + TokenClassification）。

数据: data/processed/ner/{train,dev,test}.txt（BIO 字符级标注）
实体: HPPX=品牌, HCCX=商品, XH=款式, MISC=规格/其他
输出: models/ner/best_model/

用法:
    python scripts/train_ner.py                         # 全量训练（CPU 较慢，建议 GPU）
    python scripts/train_ner.py --epochs 2 --batch_size 16
    python scripts/train_ner.py --quick                 # 小样本冒烟测试
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

# HuggingFace 镜像 + 离线模式 (模型已本地缓存)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NER_DATA = PROJECT_ROOT / "data" / "processed" / "ner"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "ner" / "best_model"


def load_bio(path) -> list[list[tuple[str, str]]]:
    """读取 BIO 文件 -> [[(token, label), ...], ...]"""
    sentences = []
    cur = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) == 2:
                cur.append((parts[0], parts[1]))
    if cur:
        sentences.append(cur)
    return sentences


def build_label_vocab(train_path) -> tuple[list[str], dict, dict]:
    """从训练集构建标签表（排序固定）。"""
    labels = set()
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                labels.add(parts[1])
    label_list = sorted(labels)
    label2id = {lbl: i for i, lbl in enumerate(label_list)}
    id2label = {i: lbl for i, lbl in enumerate(label_list)}
    return label_list, label2id, id2label


def tokenize_and_align(
    sentences: list[list[tuple[str, str]]],
    tokenizer,
    label2id: dict,
    max_length: int,
):
    """句子级字符标注 -> BERT token 级标签（非首个子词置 -100）。"""
    input_ids_list, attn_list, labels_list = [], [], []
    for sent in sentences:
        text = "".join(t for t, _ in sent)
        char_labels = [label2id.get(lbl, 0) for _, lbl in sent]
        enc = tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0]
        labels = []
        for start, end in offsets.tolist():
            if start == end:  # CLS / SEP / padding
                labels.append(-100)
            elif start < len(char_labels):
                labels.append(char_labels[start])
            else:
                labels.append(-100)
        input_ids_list.append(enc["input_ids"][0])
        attn_list.append(enc["attention_mask"][0])
        labels_list.append(labels)
    return input_ids_list, attn_list, labels_list


def decode_entities(tokens: list[str], label_ids: list[int], id2label: dict) -> list[dict]:
    """BIO 序列 -> 实体列表 [{label, text}]"""
    entities = []
    cur_type, cur_text = None, []
    for tok, lid in zip(tokens, label_ids):
        lab = id2label.get(lid, "O")
        if lab.startswith("B-"):
            if cur_type:
                entities.append({"label": cur_type, "text": "".join(cur_text)})
            cur_type = lab[2:]
            cur_text = [tok]
        elif lab.startswith("I-") and cur_type == lab[2:]:
            cur_text.append(tok)
        else:
            if cur_type:
                entities.append({"label": cur_type, "text": "".join(cur_text)})
            cur_type, cur_text = None, []
    if cur_type:
        entities.append({"label": cur_type, "text": "".join(cur_text)})
    return entities


def decode_spans(label_ids: list[int], id2label: dict) -> set[tuple]:
    """BIO 标签序列 -> {(label, start, end)}，用于无文本情况下的 span 级匹配。"""
    spans = set()
    cur_type, start = None, None
    for i, lid in enumerate(label_ids):
        lab = id2label.get(lid, "O")
        if lab.startswith("B-"):
            if cur_type is not None:
                spans.add((cur_type, start, i))
            cur_type, start = lab[2:], i
        elif lab.startswith("I-") and cur_type == lab[2:]:
            continue
        else:
            if cur_type is not None:
                spans.add((cur_type, start, i))
            cur_type, start = None, None
    if cur_type is not None:
        spans.add((cur_type, start, len(label_ids)))
    return spans


def span_f1(gold_sents, pred_sents, id2label) -> dict:
    """实体级别 precision/recall/F1（按 (label, text) 匹配）。"""
    tp = fp = fn = 0
    for g_sent, p_sent in zip(gold_sents, pred_sents):
        g_ents = {(e["label"], e["text"]) for e in decode_entities(*g_sent, id2label)}
        p_ents = {(e["label"], e["text"]) for e in decode_entities(*p_sent, id2label)}
        tp += len(g_ents & p_ents)
        fp += len(p_ents - g_ents)
        fn += len(g_ents - p_ents)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser(description="电商 NER 训练")
    ap.add_argument("--data_dir", type=str, default=str(NER_DATA))
    ap.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT))
    ap.add_argument("--model_name", type=str, default="bert-base-chinese")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--quick", action="store_true", help="小样本冒烟测试")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    print("[NER] 加载数据...")
    train_sents = load_bio(data_dir / "train.txt")
    dev_sents = load_bio(data_dir / "dev.txt")
    test_sents = load_bio(data_dir / "test.txt")
    if args.quick:
        train_sents = train_sents[:300]
        dev_sents = dev_sents[:100]
    print(f"[NER] train={len(train_sents)} dev={len(dev_sents)} test={len(test_sents)}")

    label_list, label2id, id2label = build_label_vocab(data_dir / "train.txt")
    print(f"[NER] 标签数: {len(label_list)} -> {label_list}")

    import numpy as np
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    class NerDataset(Dataset):
        def __init__(self, input_ids, attn, labels):
            self.input_ids = input_ids
            self.attn = attn
            self.labels = labels

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attn[idx],
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    print(f"[NER] 加载预训练模型: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[NER] 使用设备: {device}")

    train_ids, train_attn, train_labels = tokenize_and_align(
        train_sents, tokenizer, label2id, args.max_length
    )
    dev_ids, dev_attn, dev_labels = tokenize_and_align(
        dev_sents, tokenizer, label2id, args.max_length
    )
    test_ids, test_attn, test_labels = tokenize_and_align(
        test_sents, tokenizer, label2id, args.max_length
    )
    train_ds = NerDataset(train_ids, train_attn, train_labels)
    dev_ds = NerDataset(dev_ids, dev_attn, dev_labels)

    # 类别权重（O 占绝大多数，降权）
    flat = [lbl for seq in train_labels for lbl in seq if lbl != -100]
    counts = Counter(flat)
    weights = np.zeros(len(label_list), dtype=np.float32)
    total = sum(counts.values())
    for cls_id in range(len(label_list)):
        c = counts.get(cls_id, 1)
        weights[cls_id] = total / (len(label_list) * c)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    class WeightedNerTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                weight=class_weights,
                ignore_index=-100,
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        elif len(eval_pred) == 3:  # trainer.predict 返回 (predictions, label_ids, metrics)
            logits, labels, _ = eval_pred
        else:
            logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        tp = fp = fn = 0
        for sent_pred, sent_labels in zip(preds, labels):
            valid_gold = [lbl for lbl in sent_labels if lbl != -100]
            valid_pred = [p for p, lbl in zip(sent_pred, sent_labels) if lbl != -100]
            g_spans = decode_spans(valid_gold, id2label)
            p_spans = decode_spans(valid_pred, id2label)
            tp += len(g_spans & p_spans)
            fp += len(p_spans - g_spans)
            fn += len(g_spans - p_spans)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"eval_f1": f1, "eval_precision": prec, "eval_recall": rec}

    ta_kwargs = dict(
        output_dir=str(PROJECT_ROOT / "models" / "ner" / "checkpoint"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    import inspect

    sig = inspect.signature(TrainingArguments.__init__)
    if "warmup_ratio" in sig.parameters:
        ta_kwargs["warmup_ratio"] = 0.1
    eval_param = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"
    ta_kwargs[eval_param] = "epoch"

    trainer = WeightedNerTrainer(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=compute_metrics,
    )
    print(f"\n[NER] 开始训练 (epochs={args.epochs}, batch={args.batch_size})...")
    trainer.train()

    print("[NER] 验证集评估...")
    eval_result = trainer.evaluate()
    print(f"[NER] dev: {eval_result}")

    # 测试集评估
    print("[NER] 测试集评估...")
    def evaluate_dataset(ids, attn, lab) -> dict:
        """手动批量评估（与 trainer.evaluate 同款对齐逻辑，避免 predict 返回格式差异）。"""
        tp = fp = fn = 0
        model.eval()
        with torch.no_grad():
            for i in range(0, len(ids), args.batch_size * 2):
                b_ids = torch.stack(ids[i : i + args.batch_size * 2]).to(device)
                b_attn = torch.stack(attn[i : i + args.batch_size * 2]).to(device)
                logits = model(input_ids=b_ids, attention_mask=b_attn).logits
                for b in range(logits.size(0)):
                    valid_gold = [lbl for lbl in lab[i + b] if lbl != -100]
                    valid_pred = [
                        p for p, lbl in zip(logits[b].argmax(-1).tolist(), lab[i + b]) if lbl != -100
                    ]
                    g_spans = decode_spans(valid_gold, id2label)
                    p_spans = decode_spans(valid_pred, id2label)
                    tp += len(g_spans & p_spans)
                    fp += len(p_spans - g_spans)
                    fn += len(g_spans - p_spans)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"eval_f1": f1, "eval_precision": prec, "eval_recall": rec}

    test_metrics = evaluate_dataset(test_ids, test_attn, test_labels)
    print(f"[NER] test: {test_metrics}")

    # 保存
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    with open(out / "labels.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(label_list))
    config = {
        "model_name": args.model_name,
        "num_labels": len(label_list),
        "labels": label_list,
        "entity_types": {"HPPX": "品牌", "HCCX": "商品", "XH": "款式", "MISC": "规格/其他"},
        "train_samples": len(train_sents),
        "dev_samples": len(dev_sents),
        "test_samples": len(test_sents),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "eval_result": {k: float(v) for k, v in eval_result.items()},
        "test_result": {k: float(v) for k, v in test_metrics.items()},
    }
    with open(out / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n[NER] 模型已保存到: {out}")
    print(f"[NER] 测试集 F1: {test_metrics['eval_f1']:.4f}")


if __name__ == "__main__":
    main()
