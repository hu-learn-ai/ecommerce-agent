"""
客服指令数据集清洗器

流程（对应 docs/微调项目执行计划.md 2.5）：
    1. 格式校验：JSON 合法、role 序列合法（user/assistant 交替）、内容非空、长度范围
    2. 去重：文本完全一致去重 + BGE embedding 相似度 >= 阈值合并
    3. 质量过滤：事实类场景校验回答是否包含政策库实体（不包含 → 写入人工复核文件）
    4. 统计报告：输入/丢弃/保留数量与类别分布

用法：
    python finetune/scripts/clean_dataset.py --input data/raw/seed_data.jsonl
"""

import argparse
import collections
import json
import os
import re
import sys

# HuggingFace 镜像 + 离线模式 (模型已本地缓存)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

VALID_CATEGORIES = [
    "售前咨询",
    "售后退换货",
    "物流发货",
    "退款发票",
    "投诉安抚",
    "政策规则",
    "域外拒答",
    "闲聊",
]

# 事实类场景：回答必须覆盖政策实体，可自动校验；其余场景走人工复核
FACTUAL_CATEGORIES = {"售前咨询", "售后退换货", "物流发货", "退款发票", "政策规则"}

# 可自动校验的实体类型（具体事实，回答中必须出现）；
# rule/flow 属于行为指引/流程描述，允许多样表达，不做逐字匹配，避免误杀
CHECKABLE_ENTITY_TYPES = {"duration", "amount", "condition", "exception", "time"}

MAX_USER_LEN = 200
MAX_ANSWER_LEN = 500


def load_policy_library(path):
    """加载政策库，返回 {policy_id: policy}"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["policy_id"]: p for p in data["policies"]}


def load_jsonl(path):
    """读取 JSONL 文件"""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_jsonl(path, records, max_retry=5):
    """写入 JSONL 文件（Windows 文件锁时自动重试）"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import time
    for attempt in range(1, max_retry + 1):
        try:
            with open(path, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return
        except PermissionError:
            if attempt == max_retry:
                raise
            time.sleep(2 * attempt)


def validate_record(rec, policy_library):
    """
    校验单条记录格式，返回 (ok, errors)

    规则：
    - 必填字段齐全（id/category/scene/emotion/conversation）
    - category 属于 8 类场景
    - conversation 为 user/assistant 交替、以 assistant 结尾、内容非空
    - 用户消息 <= 200 字，回答 <= 500 字
    - policy_ref 引用的 policy_id 必须存在于政策库
    """
    errors = []
    if not isinstance(rec, dict):
        return False, ["记录不是 JSON 对象"]

    for key in ("id", "category", "scene", "emotion", "conversation"):
        if key not in rec or rec[key] in (None, ""):
            errors.append(f"缺少字段 {key}")

    if rec.get("category") not in VALID_CATEGORIES:
        errors.append(f"category 非法: {rec.get('category')}")

    conv = rec.get("conversation")
    if not isinstance(conv, list) or not conv:
        errors.append("conversation 为空")
    else:
        expected = "user"
        for i, turn in enumerate(conv):
            if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
                errors.append(f"第 {i} 轮 role 非法")
                break
            content = turn.get("content", "")
            if not content.strip():
                errors.append(f"第 {i} 轮内容为空")
                break
            if turn.get("role") != expected:
                errors.append(f"第 {i} 轮 role 序列非法（应 {expected}）")
                break
            expected = "assistant" if expected == "user" else "user"
            if turn["role"] == "user" and len(content) > MAX_USER_LEN:
                errors.append(f"用户消息超长 {len(content)} > {MAX_USER_LEN}")
            if turn["role"] == "assistant" and len(content) > MAX_ANSWER_LEN:
                errors.append(f"回答超长 {len(content)} > {MAX_ANSWER_LEN}")
        if conv and isinstance(conv[-1], dict) and conv[-1].get("role") != "assistant":
            errors.append("conversation 必须以 assistant 结尾")

    refs = rec.get("policy_ref", [])
    if not isinstance(refs, list):
        errors.append("policy_ref 必须是数组")
    elif policy_library:
        for rid in refs:
            if rid not in policy_library:
                errors.append(f"policy_ref 引用了不存在的政策: {rid}")

    return not errors, errors


def dedup_exact(records):
    """按首轮问题 + 末轮回答做文本规范化完全去重"""
    seen = set()
    kept, dropped = [], 0
    for rec in records:
        first_user = rec["conversation"][0]["content"]
        last_answer = rec["conversation"][-1]["content"]
        key = re.sub(r"\s+", "", first_user + "\x00" + last_answer)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(rec)
    return kept, dropped


_EMBED_MODEL_CACHE = {}


def _get_embedder(model_path):
    """懒加载 BGE 模型（复用现有 models/bge-base-zh-v1.5）"""
    if model_path not in _EMBED_MODEL_CACHE:
        from sentence_transformers import SentenceTransformer

        print(f"[CleanDataset] 加载嵌入模型: {model_path}")
        _EMBED_MODEL_CACHE[model_path] = SentenceTransformer(model_path)
    return _EMBED_MODEL_CACHE[model_path]


def dedup_embedding(records, model_path, threshold=0.95, batch_size=64):
    """
    按首轮问题 embedding 余弦相似度去重（>= threshold 视为重复，保留先出现的）

    贪心实现：新样本与所有已保留样本的向量点积均 < threshold 才保留，
    复杂度 O(n*k)，15k 规模在 numpy 下可接受。
    """
    import numpy as np

    model = _get_embedder(model_path)
    texts = [rec["conversation"][0]["content"] for rec in records]
    print(f"[CleanDataset] 计算 {len(texts)} 条 embedding...")
    embs = np.asarray(model.encode(texts, normalize_embeddings=True, batch_size=batch_size))

    kept, kept_embs, dropped = [], [], 0
    for i, emb in enumerate(embs):
        if kept_embs:
            sims = np.asarray(kept_embs) @ emb
            if sims.max() >= threshold:
                dropped += 1
                continue
        kept.append(records[i])
        kept_embs.append(emb)
    return kept, dropped


def _entity_values(policy_library, policy_refs):
    """
    从引用政策提取可判定的实体值（仅事实类实体，长度 >= 2，去掉空白）

    回答包含任一实体值即视为覆盖政策实体；"rule/flow" 类实体允许多样表达，不参与校验。
    """
    values = set()
    for rid in policy_refs:
        policy = policy_library.get(rid)
        if not policy:
            continue
        for etype, value in policy.get("entities", {}).items():
            if etype not in CHECKABLE_ENTITY_TYPES:
                continue
            vals = value if isinstance(value, list) else [value]
            for v in vals:
                v = re.sub(r"\s+", "", str(v))
                if len(v) >= 2:
                    values.add(v)
    return values


def check_entity_coverage(rec, policy_library):
    """事实类场景：回答是否覆盖引用政策的实体片段"""
    if rec.get("category") not in FACTUAL_CATEGORIES:
        return True
    values = _entity_values(policy_library, rec.get("policy_ref", []))
    if not values:
        return True
    answer = re.sub(
        r"\s+", "", "".join(t["content"] for t in rec["conversation"] if t["role"] == "assistant")
    )
    return any(value in answer for value in values)


def clean_dataset(records, policy_library, embed_model_path=None, dedup_threshold=0.95, use_embedding=True):
    """
    全流程清洗，返回 (passed, quality_review, stats)

    - passed: 通过校验 + 去重 + 质量过滤的记录
    - quality_review: 格式合法但实体覆盖不足的记录（人工复核）
    - stats: 各环节丢弃统计
    """
    stats = collections.Counter()

    valid = []
    for rec in records:
        ok, errors = validate_record(rec, policy_library)
        if not ok:
            stats["invalid"] += 1
            continue
        valid.append(rec)
    stats["valid"] = len(valid)

    cleaned, dup_exact = dedup_exact(valid)
    stats["dup_exact"] = dup_exact

    if use_embedding and embed_model_path and os.path.isdir(embed_model_path):
        cleaned, dup_embed = dedup_embedding(cleaned, embed_model_path, dedup_threshold)
        stats["dup_embedding"] = dup_embed
    else:
        stats["dup_embedding"] = "skipped"

    passed, review = [], []
    for rec in cleaned:
        if check_entity_coverage(rec, policy_library):
            passed.append(rec)
        else:
            review.append(rec)
    stats["quality_fail"] = len(review)
    stats["passed"] = len(passed)
    return passed, review, stats


def distribution(records, key="category"):
    """统计字段分布"""
    return dict(collections.Counter(rec.get(key, "未知") for rec in records))


def main():
    parser = argparse.ArgumentParser(description="清洗客服指令数据集")
    parser.add_argument("--input", required=True, help="输入 JSONL")
    parser.add_argument("--output", default=None, help="输出 JSONL（默认 <input 目录>/cleaned.jsonl）")
    parser.add_argument("--quality-review", default=None, help="质量复核输出 JSONL")
    parser.add_argument("--policy-library", default=os.path.join(PROJECT_ROOT, "data", "raw", "policy_library.json"))
    parser.add_argument("--embed-model", default=os.path.join(REPO_ROOT, "models", "bge-base-zh-v1.5"))
    parser.add_argument("--threshold", type=float, default=0.95, help="embedding 去重相似度阈值")
    parser.add_argument("--skip-embed", action="store_true", help="跳过 embedding 去重")
    parser.add_argument("--report", default=None, help="统计报告 JSON 输出路径")
    args = parser.parse_args()

    policy_library = load_policy_library(args.policy_library)
    records = load_jsonl(args.input)
    print(f"[CleanDataset] 输入 {len(records)} 条")

    passed, review, stats = clean_dataset(
        records,
        policy_library,
        embed_model_path=args.embed_model,
        dedup_threshold=args.threshold,
        use_embedding=not args.skip_embed,
    )

    output = args.output or os.path.join(os.path.dirname(args.input), "cleaned.jsonl")
    save_jsonl(output, passed)
    review_path = args.quality_review or os.path.join(os.path.dirname(args.input), "quality_review.jsonl")
    if review:
        save_jsonl(review_path, review)

    report = {
        "input": len(records),
        "stats": dict(stats),
        "passed": len(passed),
        "quality_review": len(review),
        "category_distribution": distribution(passed),
        "output": output,
        "quality_review_path": review_path if review else None,
    }
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[CleanDataset] 通过 {len(passed)} 条，格式非法 {stats['invalid']}，"
          f"文本重复 {stats['dup_exact']}，向量重复 {stats['dup_embedding']}，质量复核 {len(review)} 条")


if __name__ == "__main__":
    main()
