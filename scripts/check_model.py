"""模型完整性审查脚本。

用法:
    python scripts/check_model.py                 # 审查默认模型 models/best
    python scripts/check_model.py --model_path <目录>
    python scripts/check_model.py --load-test     # 额外做一次真实加载 + 预测测试
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.persistence import (  # noqa: E402
    describe_model,
    load_classification_model,
    validate_model_dir,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="模型完整性审查")
    ap.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "models", "best"),
        help="模型目录",
    )
    ap.add_argument("--load-test", action="store_true", help="额外执行加载 + 预测测试")
    args = ap.parse_args()

    mp = args.model_path
    print(f"[Check] 模型目录: {mp}")
    missing = validate_model_dir(mp)
    if missing:
        print(f"[Check] [FAIL] 模型不完整，缺失文件: {missing}")
        sys.exit(1)
    print("[Check] [OK] 必需文件齐全")

    info = describe_model(mp)
    print(f"[Check] 类别数: {info.get('num_labels')}")
    print(f"[Check] 标签: {info.get('labels')}")
    print(f"[Check] 架构: {info.get('architectures')} | 模型类型: {info.get('model_type')}")
    if info.get("training"):
        t = info["training"]
        print(f"[Check] 训练: {t.get('train_samples')} 条 / 验证 {t.get('val_samples')} 条 / "
              f"{t.get('epochs')} 轮 / batch {t.get('batch_size')} / lr {t.get('learning_rate')}")
        er = t.get("eval_result") or {}
        print(f"[Check] 验证集: accuracy={er.get('eval_accuracy', 0):.4f} f1={er.get('eval_f1', 0):.4f}")
    print(f"[Check] 文件总大小: {info.get('total_bytes', 0) / 1e6:.1f} MB")

    if args.load_test:
        print("\n[Check] 执行加载 + 预测测试...")
        tok, model, labels, device = load_classification_model(mp)
        samples = [
            "苹果iPhone15 Pro Max 256G 手机",
            "海尔510升十字对开门冰箱一级能效",
            "新鲜车厘子JJ级 2斤装 智利进口水果",
            "小米电动牙刷T302 声波震动 成人款",
        ]
        import torch

        enc = tok(samples, padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = logits.argmax(-1).tolist()
        for s, p in zip(samples, preds):
            print(f"  {s[:28]:30s} -> {labels[p]}")
        print("[Check] [OK] 加载与预测测试通过")


if __name__ == "__main__":
    main()
