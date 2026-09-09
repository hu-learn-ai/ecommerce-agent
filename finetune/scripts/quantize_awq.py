"""
AWQ 4-bit 量化（部署对比档，对应 docs/微调项目执行计划.md 5.2）

用 AutoAWQ 对合并后的模型做激活感知量化，输出可在 T4/16G 卡上跑的 4-bit 权重。

用法：
    pip install autoawq
    python finetune/scripts/quantize_awq.py \
        --model /root/autodl-tmp/models/qwen-cs-7b-merged \
        --output /root/autodl-tmp/models/qwen-cs-7b-awq \
        --calib-data finetune/data/processed/train.jsonl \
        --calib-samples 128

注意事项：
- 量化需要 GPU；calib-data 使用训练数据中的用户问题做校准集
- 量化后请用量化对比表（BF16 vs AWQ 的 TTFT/吞吐/评估损失）填写结论
"""

import argparse
import json
import os

# HuggingFace 镜像 + 离线模式 (模型已本地缓存)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_calibration_texts(jsonl_path, sample_limit):
    """从训练数据抽取用户问题作为校准集"""
    texts = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["conversation"][0]["content"])
            if len(texts) >= sample_limit:
                break
    return texts


def main():
    parser = argparse.ArgumentParser(description="AWQ 4-bit 量化")
    parser.add_argument("--model", required=True, help="合并后的模型路径")
    parser.add_argument("--output", required=True, help="量化模型输出目录")
    parser.add_argument("--calib-data", required=True, help="校准数据 JSONL（train）")
    parser.add_argument("--calib-samples", type=int, default=128, help="校准样本数")
    parser.add_argument("--w-bit", type=int, default=4, help="权重量化位数")
    parser.add_argument("--q-group-size", type=int, default=128, help="量化分组大小")
    args = parser.parse_args()

    try:
        from awq import AutoAWQForCausalLM
    except ImportError:
        raise SystemExit("未安装 AutoAWQ，请先执行: pip install autoawq")

    from transformers import AutoTokenizer

    calib_texts = load_calibration_texts(args.calib_data, args.calib_samples)
    print(f"[QuantizeAWQ] 加载模型: {args.model}，校准样本 {len(calib_texts)} 条")
    model = AutoAWQForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }
    print(f"[QuantizeAWQ] 量化配置: {quant_config}")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)
    model.save_quantized(args.output, safetensors=True)
    print(f"[QuantizeAWQ] 量化完成，已保存: {args.output}")


if __name__ == "__main__":
    main()
