"""
Qwen2.5-7B-Instruct QLoRA 微调训练（配置驱动）

对应 docs/微调项目执行计划.md 3.2：
- 4-bit NF4 + double quant + bf16 compute（bitsandbytes）
- LoRA target_modules 全线性层，r=64 / alpha=128
- SFT 只对 assistant 回答计算 loss（DataCollatorForCompletionOnlyLM，关键面试点）
- 不 packing（标准对话数据集）
- gradient checkpointing + paged_adamw_8bit，4090 24G 单卡可跑
- 每 500 步保存 adapter，支持断点续跑

用法：
    # 正式训练（读取 finetune/configs/qlora_7b.yaml）
    python finetune/scripts/train_qlora.py --config finetune/configs/qlora_7b.yaml

    # smoke 训练（200 条 + 50 步，验证全链路）
    python finetune/scripts/train_qlora.py --config finetune/configs/qlora_7b.yaml \
        --max-steps 50 --data-subset 200

    # 断点续跑
    python finetune/scripts/train_qlora.py --config finetune/configs/qlora_7b.yaml \
        --resume-from-checkpoint finetune/checkpoints/qwen-cs-7b/checkpoint-500
"""

import argparse
import json
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.settings import settings

# 国内网络：默认走 hf-mirror（与项目 .env 一致）
os.environ.setdefault("HF_ENDPOINT", settings.hf_endpoint)

# Qwen2.5-Instruct 的 assistant 起始标记，用于 loss mask（只对 assistant 内容计算 loss）
ASSISTANT_TEMPLATE = "<|im_start|>assistant\n"


def load_chat_records(jsonl_path, limit=None):
    """读取 JSONL 对话数据"""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if limit:
        records = records[:limit]
    return records


def build_text_dataset(records, tokenizer, max_seq_len):
    """对话列表 → HF Dataset（text 列，走 chat template，不 packing）"""
    from datasets import Dataset

    texts = []
    for rec in records:
        text = tokenizer.apply_chat_template(
            rec["conversation"], tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return Dataset.from_dict({"text": texts})


def build_collator(tokenizer, max_seq_len):
    """assistant-only loss mask：只有 assistant 回答参与 loss 计算"""
    from trl import DataCollatorForCompletionOnlyLM

    return DataCollatorForCompletionOnlyLM(
        response_template=ASSISTANT_TEMPLATE,
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-7B QLoRA 微调")
    parser.add_argument("--config", required=True, help="训练配置 YAML")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖训练步数（smoke 用）")
    parser.add_argument("--data-subset", type=int, default=None, help="每个 split 只取前 N 条（smoke 用）")
    parser.add_argument("--output-dir", default=None, help="覆盖输出目录")
    parser.add_argument("--resume-from-checkpoint", default=None, help="断点目录（如 .../checkpoint-500）")
    parser.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_path = os.path.join(REPO_ROOT, cfg["data"]["train_path"])
    val_path = os.path.join(REPO_ROOT, cfg["data"]["val_path"])
    output_dir = args.output_dir or os.path.join(PROJECT_ROOT, cfg["output_dir"])
    os.makedirs(output_dir, exist_ok=True)

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTConfig, SFTTrainer

    model_name = cfg["model_name"]
    quant_cfg = cfg["quantization"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]
    cfg["data"]

    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"[TrainQLoRA] GPU: {torch.cuda.get_device_name(0)}，显存 {total_mem / 2**30:.1f}GB，空闲 {free_mem / 2**30:.1f}GB")
    else:
        raise RuntimeError("未检测到 GPU，QLoRA 训练必须在 CUDA 环境运行（建议 AutoDL 4090 24G）")

    print(f"[TrainQLoRA] 加载模型: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # QLoRA 前置：gradient checkpointing + 输入 dtype 对齐
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_cfg["gradient_checkpointing"])
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"[TrainQLoRA] 加载数据: {train_path}")
    train_records = load_chat_records(train_path, args.data_subset)
    val_records = load_chat_records(val_path, args.data_subset) if os.path.exists(val_path) else []
    train_dataset = build_text_dataset(train_records, tokenizer, train_cfg["max_seq_len"])
    eval_dataset = build_text_dataset(val_records, tokenizer, train_cfg["max_seq_len"]) if val_records else None
    print(f"[TrainQLoRA] train={len(train_dataset)}，val={len(eval_dataset) if eval_dataset else 0}，"
          f"max_seq_len={train_cfg['max_seq_len']}，packing={train_cfg['packing']}")

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        num_train_epochs=train_cfg["num_train_epochs"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        optim="paged_adamw_8bit",
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg["save_steps"],
        save_total_limit=3,
        logging_strategy=train_cfg["logging_strategy"],
        logging_steps=train_cfg["logging_steps"],
        report_to=train_cfg["report_to"],
        seed=args.seed or train_cfg["seed"],
        max_steps=args.max_steps if args.max_steps is not None else -1,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=train_cfg["save_steps"],
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        dataset_text_field="text",
        max_seq_length=train_cfg["max_seq_len"],
        packing=train_cfg["packing"],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=build_collator(tokenizer, train_cfg["max_seq_len"]),
    )

    print(f"[TrainQLoRA] 开始训练（输出: {output_dir}）")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # adapter + tokenizer 落盘（adapter_only，支持后续 merge）
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[TrainQLoRA] 完成，adapter 已保存: {output_dir}")
    if torch.cuda.is_available():
        print(f"[TrainQLoRA] 峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")


if __name__ == "__main__":
    main()
