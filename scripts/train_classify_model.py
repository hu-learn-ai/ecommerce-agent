"""
BERT 商品分类模型训练脚本

使用 classify_train.csv (由 import_taobao_data.py 生成) 训练
基于 bert-base-chinese + 微调

使用方法:
    python scripts/train_classify_model.py

可选参数:
    --epochs 5          训练轮数 (默认 5)
    --batch_size 16     批大小 (默认 16)
    --lr 2e-5           学习率 (默认 2e-5)
    --max_length 128    最大序列长度 (默认 128)

输出:
    models/best/  — 模型权重（与 .env 的 CLASSIFY_MODEL_PATH=./models/best 对齐）
    models/best/labels.txt  — 标签文件
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter

# HuggingFace 镜像 + 离线模式 (模型已本地缓存)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="BERT 商品分类模型训练")
    parser.add_argument(
        "--data",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "processed", "classify_train.csv"),
        help="训练数据 CSV 路径",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="外部验证集 CSV 路径（可选；不提供时按 --val_ratio 从训练集切分）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "best"),
        help="模型输出目录",
    )
    parser.add_argument(
        "--model_name", type=str, default="bert-base-chinese", help="预训练模型名称"
    )
    parser.add_argument("--epochs", type=int, default=5, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="批大小")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    return parser.parse_args()


def load_data(csv_path):
    """加载训练数据"""
    print(f"[Train] 加载数据: {csv_path}")
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        texts = []
        labels = []
        for row in reader:
            texts.append(row["text"].strip())
            labels.append(row["label"].strip())

    # 统计
    label_dist = Counter(labels)
    print(f"[Train] 总样本: {len(texts)}")
    print(f"[Train] 类别数: {len(label_dist)}")
    for k, v in label_dist.most_common():
        print(f"  {k}: {v}")

    return texts, labels


def build_label_list(labels):
    """构建标签列表 (排序后固定顺序)"""
    label_list = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def split_data(texts, labels, val_ratio=0.2, seed=42):
    """划分训练集和验证集"""
    random.seed(seed)
    indices = list(range(len(texts)))
    random.shuffle(indices)

    val_size = int(len(texts) * val_ratio)
    val_indices = set(indices[:val_size])
    train_indices = indices[val_size:]

    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    print(f"[Train] 训练集: {len(train_texts)}, 验证集: {len(val_texts)}")
    return train_texts, train_labels, val_texts, val_labels


def train():
    args = parse_args()

    # 设置 HF 镜像
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    # 1. 加载数据
    texts, labels = load_data(args.data)
    label_list, label2id, id2label = build_label_list(labels)

    if args.val_data:
        val_texts, val_labels = load_data(args.val_data)
        missing_labels = set(val_labels) - set(label_list)
        if missing_labels:
            raise SystemExit(
                f"[Train] 验证集包含训练集中不存在的标签: {missing_labels}，请检查 --val_data"
            )
        train_texts, train_labels = texts, labels
        print(f"[Train] 使用外部验证集: {args.val_data}")
    else:
        train_texts, train_labels, val_texts, val_labels = split_data(
            texts, labels, val_ratio=args.val_ratio
        )

    # 2. 加载 tokenizer 和模型
    print(f"\n[Train] 加载预训练模型: {args.model_name}")
    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
    )

    # EarlyStoppingCallback 在 transformers>=4.0 可用，低版本降级处理
    try:
        from transformers import EarlyStoppingCallback

        _has_early_stopping = True
    except ImportError:
        _has_early_stopping = False
        print("[Train] 当前 transformers 版本不支持 EarlyStoppingCallback，跳过早停")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[Train] 使用设备: {device}")

    # 3. 构建数据集
    from torch.utils.data import Dataset

    class TextDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = [label2id[label] for label in labels]
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            encoding = self.tokenizer(
                self.texts[idx],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long),
            }

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, args.max_length)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, args.max_length)

    # 4. 计算类别权重 (处理类别不平衡)
    from collections import Counter as CCounter

    train_label_ids = [label2id[label] for label in train_labels]
    label_counts = CCounter(train_label_ids)
    num_samples = len(train_label_ids)
    num_classes = len(label_list)

    # 逆频率权重: weight[i] = total / (num_classes * count[i])
    # 少数类获得更高权重, 多数类权重降低
    class_weights = np.zeros(num_classes, dtype=np.float32)
    for cls_id in range(num_classes):
        count = label_counts.get(cls_id, 1)
        class_weights[cls_id] = num_samples / (num_classes * count)

    # 归一化: 使权重均值为 1
    class_weights = class_weights / class_weights.mean()

    print("\n[Train] 类别权重 (处理不平衡):")
    for cls_id in range(num_classes):
        print(
            f"  {id2label[cls_id]}: count={label_counts.get(cls_id, 0)}, weight={class_weights[cls_id]:.3f}"
        )

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # 5. 评估指标
    from sklearn.metrics import accuracy_score, f1_score

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="macro")
        return {"accuracy": acc, "f1": f1}

    # 5. 训练参数 (兼容不同 transformers 版本)
    # 旧版 transformers 不支持 warmup_ratio / eval_strategy / dataloader_num_workers
    import inspect

    ta_kwargs = dict(
        output_dir=os.path.join(PROJECT_ROOT, "models", "product_classification", "checkpoint"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=3,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # warmup: 新版用 warmup_ratio, 旧版用 warmup_steps
    ta_sig = inspect.signature(TrainingArguments.__init__)
    if "warmup_ratio" in ta_sig.parameters:
        ta_kwargs["warmup_ratio"] = 0.1
    else:
        steps_per_epoch = max(1, len(train_texts) // args.batch_size)
        ta_kwargs["warmup_steps"] = int(steps_per_epoch * args.epochs * 0.1)

    # eval strategy: 新版 eval_strategy, 旧版 evaluation_strategy
    eval_param = "eval_strategy" if "eval_strategy" in ta_sig.parameters else "evaluation_strategy"
    ta_kwargs[eval_param] = "epoch"

    # dataloader_num_workers: 不支持则跳过
    if "dataloader_num_workers" in ta_sig.parameters:
        ta_kwargs["dataloader_num_workers"] = 4 if torch.cuda.is_available() else 0

    training_args = TrainingArguments(**ta_kwargs)

    # 6. 训练 (带早停 + 加权交叉熵处理类别不平衡)
    print(f"\n[Train] 开始训练 (epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr})")
    print("[Train] 使用加权交叉熵损失 (class weights normalized, mean=1.0)")

    # 自定义 Trainer: 使用加权 CrossEntropyLoss 处理类别不平衡
    from transformers import Trainer as BaseTrainer

    class WeightedTrainer(BaseTrainer):
        """使用加权交叉熵损失的 Trainer — 处理类别不平衡"""

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.get("logits")

            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
            loss = loss_fct(logits, labels)

            return (loss, outputs) if return_outputs else loss

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    if _has_early_stopping:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=3)]

    trainer = WeightedTrainer(**trainer_kwargs)

    trainer.train()

    # 7. 评估
    print("\n[Train] 训练完成，评估验证集...")
    eval_result = trainer.evaluate()
    print(f"[Train] 验证集结果: {eval_result}")

    # 8. 保存最佳模型
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 保存标签文件
    labels_file = os.path.join(args.output_dir, "labels.txt")
    with open(labels_file, "w", encoding="utf-8") as f:
        for label in label_list:
            f.write(label + "\n")

    # 保存训练配置
    config = {
        "model_name": args.model_name,
        "num_labels": len(label_list),
        "labels": label_list,
        "train_samples": len(train_texts),
        "val_samples": len(val_texts),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "loss_function": "weighted_cross_entropy",
        "class_weights": {id2label[i]: float(class_weights[i]) for i in range(num_classes)},
        "label_distribution": {id2label[i]: label_counts.get(i, 0) for i in range(num_classes)},
        "eval_result": {k: float(v) for k, v in eval_result.items()},
    }
    config_file = os.path.join(args.output_dir, "training_config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n[Train] 模型已保存到: {args.output_dir}")
    print(f"[Train] 标签文件: {labels_file}")
    print(f"[Train] 训练配置: {config_file}")
    print(f"[Train] 验证集准确率: {eval_result.get('eval_accuracy', 0):.4f}")
    print(f"[Train] 验证集 F1: {eval_result.get('eval_f1', 0):.4f}")

    # 9. 模型完整性校验（固化后自检，防止不完整模型上线）
    try:
        from models.persistence import describe_model, validate_model_dir

        missing_after = validate_model_dir(args.output_dir)
        if missing_after:
            raise RuntimeError(f"模型保存后校验失败，缺失文件: {missing_after}")
        model_info = describe_model(args.output_dir)
        print(
            f"[Train] 模型完整性校验通过: {len(model_info.get('labels', []))} 类, "
            f"{model_info.get('total_bytes', 0) / 1e6:.1f} MB"
        )
    except ImportError:
        # models/persistence 未随包上传时跳过（模型已保存，不阻断后续任务）
        print("[Train] 跳过完整性校验（models.persistence 未安装，不影响模型文件）")
    except Exception as exc:  # noqa: BLE001
        print(f"[Train] 模型完整性校验失败: {exc}")
        raise
    print()
    print("=" * 60)
    print("  训练完成! 请在 .env 中设置:")
    print(f"  CLASSIFY_MODEL_PATH={args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    train()
