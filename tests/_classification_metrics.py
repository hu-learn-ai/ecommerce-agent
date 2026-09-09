"""共享: 多分类指标计算器

被 router_eval.py / classify_eval.py 复用,避免代码重复。
"""

from collections import defaultdict
from typing import List


def compute_classification_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str],
) -> dict:
    """
    计算多分类指标

    Returns:
        {
            "accuracy": float,
            "macro_precision": ..., "macro_recall": ..., "macro_f1": ...,
            "micro_precision": ..., "micro_recall": ..., "micro_f1": ...,
            "per_class": {label: {"precision", "recall", "f1", "support", "errors"}},
            "confusion_matrix": {true_label: {pred_label: count}},
        }
    """
    assert len(y_true) == len(y_pred), "y_true 和 y_pred 长度不一致"

    n = len(y_true)
    if n == 0:
        return {"accuracy": 0, "macro_f1": 0}

    # 混淆矩阵
    confusion = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        confusion[t][p] += 1

    # 每类指标
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        support = sum(confusion[label].values())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        errors = {
            other: confusion[label][other]
            for other in labels
            if other != label and confusion[label][other] > 0
        }

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "errors": dict(errors),
        }

    valid_classes = [c for c in labels if per_class[c]["support"] > 0]
    macro_p = sum(per_class[c]["precision"] for c in valid_classes) / len(valid_classes) if valid_classes else 0
    macro_r = sum(per_class[c]["recall"] for c in valid_classes) / len(valid_classes) if valid_classes else 0
    macro_f1 = sum(per_class[c]["f1"] for c in valid_classes) / len(valid_classes) if valid_classes else 0

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / n

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(accuracy, 4),
        "micro_recall": round(accuracy, 4),
        "micro_f1": round(accuracy, 4),
        "per_class": per_class,
        "confusion_matrix": {t: dict(preds) for t, preds in confusion.items()},
    }


def compute_ece(confidences: List[float], correct: List[bool], n_bins: int = 10) -> dict:
    """
    Expected Calibration Error (ECE)

    衡量模型置信度与实际准确率的偏差:
    - 把预测按置信度分到 n_bins 个桶(0-0.1, 0.1-0.2, ..., 0.9-1.0)
    - 每个桶内: |平均置信度 - 实际准确率| * (桶样本数 / 总样本数)
    - ECE = 加权平均

    ECE=0 完美校准; ECE 越大, 模型越"过度自信"或"不够自信"

    Args:
        confidences: 每条预测的置信度(0-1)
        correct: 每条预测是否正确
        n_bins: 分桶数

    Returns:
        {"ece": float, "bins": [{bin_low, bin_high, count, avg_conf, acc, gap}]}
    """
    assert len(confidences) == len(correct)
    n = len(confidences)
    if n == 0:
        return {"ece": 0.0, "bins": []}

    bins = []
    for i in range(n_bins):
        low = i / n_bins
        high = (i + 1) / n_bins
        if i == n_bins - 1:
            idx = [j for j, c in enumerate(confidences) if low <= c <= high]
        else:
            idx = [j for j, c in enumerate(confidences) if low <= c < high]

        if not idx:
            bins.append(
                {
                    "bin_low": round(low, 2),
                    "bin_high": round(high, 2),
                    "count": 0,
                    "avg_conf": 0.0,
                    "acc": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        avg_conf = sum(confidences[j] for j in idx) / len(idx)
        acc = sum(1 for j in idx if correct[j]) / len(idx)
        bins.append(
            {
                "bin_low": round(low, 2),
                "bin_high": round(high, 2),
                "count": len(idx),
                "avg_conf": round(avg_conf, 4),
                "acc": round(acc, 4),
                "gap": round(abs(avg_conf - acc), 4),
            }
        )

    ece = sum(b["gap"] * b["count"] / n for b in bins)
    return {"ece": round(ece, 4), "bins": bins}
