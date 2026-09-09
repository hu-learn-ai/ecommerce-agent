"""
客服指令数据集构建流水线（全流程可复现）

流程（对应 docs/微调项目执行计划.md 2.4 / 2.5）：
    1. 合并人工种子数据 → data/raw/seed_data.jsonl
    2. （可选 --expand）DeepSeek 扩写：按 8 类场景目标比例分批生成，缓存可断点续跑
    3. 清洗：格式校验 + 去重（文本/BGE 向量）+ 质量过滤（复用 clean_dataset）
    4. 分层划分：train/val/test = 90/5/5（按 category 分层抽样，固定 seed）
    5. 防泄漏：test 与 train 做 embedding 相似度去重（> 0.9 移除）
    6. 输出 data/processed/{train,val,test}.jsonl + dataset_report.json

用法：
    # 仅种子数据（免费）
    python finetune/scripts/build_dataset.py --seeds-only

    # 全量扩写至 15k
    python finetune/scripts/build_dataset.py --expand --target 15000

    # 冒烟扩写（验证链路）
    python finetune/scripts/build_dataset.py --expand --target 30 --smoke
"""

import argparse
import collections
import json
import os
import random
import sys
import time

# HuggingFace 镜像 + 离线模式 (模型已本地缓存)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from clean_dataset import (
    MAX_ANSWER_LEN,
    MAX_USER_LEN,
    VALID_CATEGORIES,
    clean_dataset,
    distribution,
    load_jsonl,
    load_policy_library,
    save_jsonl,
    validate_record,
)

CATEGORY_RATIOS = {
    "售前咨询": 0.20,
    "售后退换货": 0.20,
    "物流发货": 0.10,
    "退款发票": 0.10,
    "投诉安抚": 0.10,
    "政策规则": 0.10,
    "域外拒答": 0.10,
    "闲聊": 0.10,
}

SCENES = [
    "单轮_直接", "单轮_含情绪", "多轮_追问", "多轮_指代", "多轮_情绪升级",
    "单轮_情绪爆发", "单轮_无关话题", "单轮_敏感内容", "单轮_比价外链",
    "单轮_超范围", "单轮_诱导违规", "单轮_问候", "单轮_感谢", "单轮_寒暄", "单轮_道别",
]
EMOTIONS = ["中性", "不满", "着急", "疑惑", "愤怒", "感谢", "伤心"]

DEEPSEEK_INPUT_CNY_PER_M = 2.0  # 官方价可能变化，仅用于估算，报告中标注以官方为准
DEEPSEEK_OUTPUT_CNY_PER_M = 8.0


def load_env():
    """加载仓库根目录 .env（DEEPSEEK_API_KEY 等）"""
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))


def get_client():
    """构造 DeepSeek OpenAI 兼容客户端"""
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    if not api_key or api_key.startswith("sk-your"):
        raise RuntimeError("未检测到有效的 DEEPSEEK_API_KEY，请在仓库根目录 .env 中配置")
    return OpenAI(api_key=api_key, base_url=base_url)


def category_targets(target):
    """按 8 类目标比例计算各类别扩写目标"""
    return {cat: max(1, round(target * ratio)) for cat, ratio in CATEGORY_RATIOS.items()}


def build_expansion_prompt(category, policies, examples, count):
    """构造扩写 prompt：政策库 + 种子示例 + 输出约束"""
    policy_text = "\n".join(
        f"- {p['policy_id']}（{p['title']}）: {p['policy']}" for p in policies
    )
    examples_text = "\n".join(json.dumps(e, ensure_ascii=False) for e in examples)
    system = (
        "你是电商客服数据集生成工程师。根据给定的政策库和种子样本，生成新的客服对话训练样本。\n"
        "硬性要求：\n"
        "1. 只引用政策库中的条款和数字，禁止编造政策库之外的规则、金额、时效、天数；\n"
        "2. 用户问法必须多样（换说法、换商品、换场景、多轮追问/指代），贴近真实用户口语；\n"
        "3. 客服回答必须基于政策库内容，语气专业、友好，多轮场景要承接上文；\n"
        f"4. 严格输出 JSON 数组（不要输出任何其他文字），数组元素字段：id, category, scene, emotion, conversation, policy_ref, answer_entities；\n"
        f"5. conversation 为 user/assistant 交替数组，必须以 assistant 结尾，用户消息不超过 {MAX_USER_LEN} 字，回答不超过 {MAX_ANSWER_LEN} 字；\n"
        f"6. category 必须为「{category}」；\n"
        f"7. scene 只能取以下枚举之一：{json.dumps(SCENES, ensure_ascii=False)}；\n"
        f"8. emotion 只能取以下枚举之一：{json.dumps(EMOTIONS, ensure_ascii=False)}；\n"
        "9. policy_ref 只能引用政策库中给出的 policy_id，answer_entities 只能填写政策库实体字段对应的值；\n"
        "10. 生成的样本不得与给出的示例重复。"
    )
    user = (
        f"【类别】{category}\n\n"
        f"【政策库】\n{policy_text}\n\n"
        f"【种子示例】\n{examples_text}\n\n"
        f"请生成 {count} 条该类别的新样本。"
    )
    return system, user


def parse_json_array(content):
    """解析模型输出的 JSON 数组，容忍 markdown 代码块与前后噪声"""
    if not content:
        return None
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:]
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start, end = content.find("["), content.rfind("]")
        if start != -1 and end > start:
            try:
                return json.loads(content[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def normalize_expanded(records, category, allowed_policy_ids, start_index):
    """扩写结果规范化：重写 id、校验字段合法性，返回 (valid_records, invalid_count)"""
    valid, invalid = [], 0
    mini_library = {pid: {"policy_id": pid} for pid in allowed_policy_ids}
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            invalid += 1
            continue
        rec["id"] = f"cs_{category}_{start_index + i:05d}"
        rec["category"] = category
        ok, errors = validate_record(rec, mini_library)
        if not ok:
            invalid += 1
            continue
        if rec.get("scene") not in SCENES or rec.get("emotion") not in EMOTIONS:
            invalid += 1
            continue
        valid.append(rec)
    return valid, invalid


def expand_category(client, model, category, policies, seeds, target, cache_path, max_calls, per_call):
    """
    扩写单个类别：读取已有缓存续跑，直到达到目标或超出调用上限

    返回 (records, usage_tokens, call_count)
    """
    records = load_jsonl(cache_path) if os.path.exists(cache_path) else []
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    call_count = 0
    examples = random.sample(seeds, min(4, len(seeds)))
    allowed_ids = [p["policy_id"] for p in policies]
    start_index = len(records) + 1
    consecutive_failures = 0

    while len(records) < target and call_count < max_calls:
        remaining = target - len(records)
        batch = min(per_call, remaining)
        system, user = build_expansion_prompt(category, policies, examples, batch)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.8,
                max_tokens=4096,
            )
        except Exception as e:
            consecutive_failures += 1
            print(f"[BuildDataset] 扩写调用失败（{category}）: {e}")
            if consecutive_failures >= 5:
                print(f"[BuildDataset] {category} 连续失败 {consecutive_failures} 次，停止该类别扩写")
                break
            time.sleep(2)
            continue

        consecutive_failures = 0
        call_count += 1
        if resp.usage:
            usage["prompt_tokens"] += resp.usage.prompt_tokens or 0
            usage["completion_tokens"] += resp.usage.completion_tokens or 0

        parsed = parse_json_array(resp.choices[0].message.content)
        if parsed is None:
            print(f"[BuildDataset] 解析失败（{category}），跳过该批")
            continue
        valid, invalid = normalize_expanded(parsed, category, allowed_ids, start_index)
        records.extend(valid)
        start_index += len(valid)
        print(f"[BuildDataset] {category}: 新增 {len(valid)} 条（无效 {invalid}），累计 {len(records)}/{target}")
        # 每批落盘：崩溃/中断时最多丢失一批（默认 10 条），断点续跑更稳
        save_jsonl(cache_path, records)

    save_jsonl(cache_path, records)
    return records, usage, call_count


def stratified_split(records, val_ratio=0.05, test_ratio=0.05, seed=42):
    """按 category 分层抽样划分 train/val/test"""
    rng = random.Random(seed)
    by_category = collections.defaultdict(list)
    for rec in records:
        by_category[rec["category"]].append(rec)

    train, val, test = [], [], []
    for cat, group in by_category.items():
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_ratio)) if len(group) >= 20 else 0
        n_val = max(1, round(len(group) * val_ratio)) if len(group) >= 20 else 0
        test.extend(group[:n_test])
        val.extend(group[n_test : n_test + n_val])
        train.extend(group[n_test + n_val :])
    return train, val, test


def anti_leak_dedup(test_records, train_records, model_path, threshold=0.9):
    """
    防泄漏：test 与 train 的首轮问题 embedding 相似度 > threshold 的 test 样本移除

    返回 (kept_test, removed_ids)
    """
    if not test_records or not train_records:
        return test_records, []
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)
    train_texts = [rec["conversation"][0]["content"] for rec in train_records]
    test_texts = [rec["conversation"][0]["content"] for rec in test_records]
    train_embs = model.encode(train_texts, normalize_embeddings=True)
    test_embs = model.encode(test_texts, normalize_embeddings=True)

    import numpy as np

    sims = np.asarray(test_embs) @ np.asarray(train_embs).T
    max_sim = sims.max(axis=1)
    kept, removed = [], []
    for rec, sim in zip(test_records, max_sim):
        if sim > threshold:
            removed.append({"id": rec["id"], "max_sim": float(sim)})
        else:
            kept.append(rec)
    return kept, removed


def main():
    parser = argparse.ArgumentParser(description="构建客服指令数据集")
    parser.add_argument("--seeds-dir", default=os.path.join(PROJECT_ROOT, "data", "raw", "seeds"))
    parser.add_argument("--policy-library", default=os.path.join(PROJECT_ROOT, "data", "raw", "policy_library.json"))
    parser.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data", "processed"))
    parser.add_argument("--raw-out", default=os.path.join(PROJECT_ROOT, "data", "raw", "seed_data.jsonl"))
    parser.add_argument("--expand", action="store_true", help="启用 DeepSeek 扩写")
    parser.add_argument("--seeds-only", action="store_true", help="仅使用种子数据（默认行为，无需扩写时使用）")
    parser.add_argument("--target", type=int, default=15000, help="扩写目标总量")
    parser.add_argument("--smoke", action="store_true", help="冒烟模式：限制单类别扩写调用次数")
    parser.add_argument("--expand-max-calls", type=int, default=4000, help="扩写调用上限（防失控）")
    parser.add_argument("--per-call", type=int, default=5, help="每次调用生成条数")
    parser.add_argument("--embed-model", default=os.path.join(REPO_ROOT, "models", "bge-base-zh-v1.5"))
    parser.add_argument("--skip-embed", action="store_true", help="跳过 embedding 去重与防泄漏")
    parser.add_argument("--dedup-threshold", type=float, default=0.95)
    parser.add_argument("--leak-threshold", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    policy_library = load_policy_library(args.policy_library)
    seeds = []
    for path in sorted(os.listdir(args.seeds_dir)):
        if not path.endswith(".jsonl"):
            continue
        seeds.extend(load_jsonl(os.path.join(args.seeds_dir, path)))
    print(f"[BuildDataset] 种子数据 {len(seeds)} 条")

    records = list(seeds)
    expansion_report = None

    if args.expand:
        load_env()
        client = get_client()
        model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        targets = category_targets(args.target)
        expanded_dir = os.path.join(PROJECT_ROOT, "data", "raw", "expanded")
        os.makedirs(expanded_dir, exist_ok=True)

        total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
        total_calls = 0
        per_category = {}
        max_calls = 1 if args.smoke else args.expand_max_calls

        for category in VALID_CATEGORIES:
            policies = [p for p in policy_library.values() if p["category"] == category]
            cat_seeds = [s for s in seeds if s["category"] == category]
            if not cat_seeds or not policies:
                print(f"[BuildDataset] {category} 无种子或政策，跳过扩写")
                continue
            target = targets[category]
            cache_path = os.path.join(expanded_dir, f"{category}.jsonl")
            cat_records, usage, calls = expand_category(
                client,
                model,
                category,
                policies,
                cat_seeds,
                target,
                cache_path,
                max_calls=max_calls,
                per_call=args.per_call,
            )
            records.extend(cat_records)
            total_usage["prompt_tokens"] += usage["prompt_tokens"]
            total_usage["completion_tokens"] += usage["completion_tokens"]
            total_calls += calls
            per_category[category] = {"cached_or_generated": len(cat_records), "target": target}

        cost_est = (
            total_usage["prompt_tokens"] / 1e6 * DEEPSEEK_INPUT_CNY_PER_M
            + total_usage["completion_tokens"] / 1e6 * DEEPSEEK_OUTPUT_CNY_PER_M
        )
        expansion_report = {
            "target": args.target,
            "total_calls": total_calls,
            "usage_tokens": total_usage,
            "estimated_cost_cny": round(cost_est, 2),
            "cost_note": "按输入 ¥2/M、输出 ¥8/M 估算，官方价可能变化，以实际账单为准",
            "per_category": per_category,
        }
        print(f"[BuildDataset] 扩写完成：调用 {total_calls} 次，累计 token {total_usage}，估算 ¥{cost_est:.2f}")

    # 合并后的原始数据落盘（种子合并文件）
    save_jsonl(args.raw_out, records)

    # 清洗
    passed, review, clean_stats = clean_dataset(
        records,
        policy_library,
        embed_model_path=args.embed_model,
        dedup_threshold=args.dedup_threshold,
        use_embedding=not args.skip_embed,
    )
    print(f"[BuildDataset] 清洗后 {len(passed)} 条（格式非法 {clean_stats['invalid']}，"
          f"文本重复 {clean_stats['dup_exact']}，向量重复 {clean_stats['dup_embedding']}，质量复核 {len(review)} 条）")

    if review:
        save_jsonl(os.path.join(PROJECT_ROOT, "data", "raw", "quality_review.jsonl"), review)

    # 分层划分
    train, val, test = stratified_split(passed, seed=args.seed)

    # 防泄漏
    if args.skip_embed:
        leak_removed = []
    else:
        test, leak_removed = anti_leak_dedup(test, train, args.embed_model, args.leak_threshold)
    print(f"[BuildDataset] 划分 train={len(train)} val={len(val)} test={len(test)}，防泄漏移除 {len(leak_removed)} 条")

    save_jsonl(os.path.join(args.out_dir, "train.jsonl"), train)
    save_jsonl(os.path.join(args.out_dir, "val.jsonl"), val)
    save_jsonl(os.path.join(args.out_dir, "test.jsonl"), test)

    report = {
        "seed_count": len(seeds),
        "raw_count": len(records),
        "expansion": expansion_report,
        "clean_stats": dict(clean_stats),
        "cleaned_count": len(passed),
        "quality_review_count": len(review),
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "leak_removed": leak_removed,
        "leak_threshold": args.leak_threshold,
        "train_category_distribution": distribution(train),
        "val_category_distribution": distribution(val),
        "test_category_distribution": distribution(test),
        "scene_distribution": distribution(passed, key="scene"),
        "emotion_distribution": distribution(passed, key="emotion"),
        "params": {
            "dedup_threshold": args.dedup_threshold,
            "seed": args.seed,
            "embed_model": args.embed_model,
            "skip_embed": args.skip_embed,
        },
    }
    report_path = os.path.join(args.out_dir, "dataset_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[BuildDataset] 报告已写入: {report_path}")


if __name__ == "__main__":
    main()
