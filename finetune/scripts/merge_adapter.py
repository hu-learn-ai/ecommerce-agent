"""
合并 LoRA adapter 到 base 模型，导出完整可推理模型

对应 docs/微调项目执行计划.md 5.1：训练完成后 merge adapter → 导出完整模型 → vLLM 启动。

用法：
    # 合并（默认：base 用 models/Qwen2.5-7B-Instruct，adapter 用 finetune/checkpoints/qwen-cs-7b）
    python finetune/scripts/merge_adapter.py \
        --base-model /root/autodl-tmp/models/Qwen2.5-7B-Instruct \
        --adapter finetune/checkpoints/qwen-cs-7b \
        --output /root/autodl-tmp/models/qwen-cs-7b-merged

注意事项：
- 合并时以 bf16 加载 base（不要量化），合并结果可直接被 vLLM / transformers 加载
- 7B bf16 约 14GB 显存，24G 单卡可完成
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.settings import settings

os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA adapter 到 base 模型")
    parser.add_argument("--base-model", required=True, help="base 模型路径或 HF 名称")
    parser.add_argument("--adapter", required=True, help="训练产出的 adapter 目录")
    parser.add_argument("--output", default=os.path.join(PROJECT_ROOT, "checkpoints", "qwen-cs-7b-merged"))
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16", help="导出精度")
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(f"[MergeAdapter] 加载 base: {args.base_model}（{args.dtype}）")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[MergeAdapter] 加载 adapter: {args.adapter}")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    print(f"[MergeAdapter] 合并完成，已导出: {args.output}")


if __name__ == "__main__":
    main()
