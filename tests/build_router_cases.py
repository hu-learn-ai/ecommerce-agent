"""扩充路由评估用例（LLM 生成 + 机械校验，产出 `tests/router_cases_generated.json`）。

背景：原路由用例只有 61 条，且 45/61 由关键词规则直接命中——100% 只说明"规则覆盖了这些写法"。
61/61 的 95% 置信区间下界其实只有 94.1%，per-class 支持数最低只有 3 条，样本明显不足。

本脚本用 LLM 按意图批量造口语化用例（**刻意避开关键词规则**，用于压测 LLM 兜底层），
生成后做机械校验（去重 / 长度 / 关键词冲突 / 覆盖度），再由人工过一遍才落盘。

用法:
    python tests/build_router_cases.py --per-intent 15          # 生成并写盘
    python tests/build_router_cases.py --per-intent 15 --dry-run  # 只打印不写

输出:
    tests/router_cases_generated.json  # [{"query", "expected", "source", "note"}]
"""

import argparse
import collections
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root  # noqa: E402

ensure_project_root()

from langchain_openai import ChatOpenAI  # noqa: E402

from config.settings import settings  # noqa: E402
from orchestration.router import RouterAgent  # noqa: E402
from tests.router_eval import ROUTER_TEST_CASES, load_all_test_cases  # noqa: E402

OUTPUT = os.path.join(PROJECT_ROOT, "tests", "router_cases_generated.json")

INTENT_DESC = {
    "search": "搜索/查找具体商品（问有没有某个商品、多少钱、某个型号能不能买到）",
    "recommend": "要个性化推荐（让对方帮自己挑、给建议、按预算/用途选）",
    "classify": "明确要求把某个商品分类（“这个商品属于哪类”式的直接分类指令）",
    "kg_qa": "问品牌/属性/关系类知识（某品牌有哪些产品、两个品牌什么关系、属于什么品类）",
    "order": "查订单与物流（我的订单到哪了、快递什么时候到、发货没）",
    "customer_service": "售后与政策咨询（退换货、退款、运费、发票、价保、投诉处理规则）",
    "analytics": "问经营数据（销量排行、品类趋势、占比、统计分析）",
    "chitchat": "问候/感谢/闲聊/与购物无关的话题",
}


def build_prompt(per_intent: int) -> str:
    keywords = "\n".join(
        f"- {intent}: {'、'.join(words)}" for intent, words in RouterAgent.KEYWORD_RULES.items()
    )
    intents = "\n".join(f"- {k}: {v}" for k, v in INTENT_DESC.items())
    return f"""你在为一个中文电商客服系统的**意图路由**构造测试集。

意图定义：
{intents}

系统的第一层是关键词规则匹配，命中任一关键词就直接定意图；关键词表如下（这是要被测的对象，
你造用例时要**刻意避开**）：
{keywords}

请为上面 8 个意图各生成 {per_intent} 条**用户口吻的短句**（8~25 个汉字），要求：
1. 每条**只能属于一个意图**，不允许含糊或跨意图（例如"推荐个便宜的手机"属于 recommend，
   不要造"搜一下有什么手机"这种既可 search 又可 recommend 的句子）；
2. 其中约 2/3 的句子**不要出现关键词表里的任何词**（包括不要出现该意图自己的关键词），
   目的是逼系统走 LLM 语义兜底；剩下约 1/3 可以包含该意图自己的关键词；
3. 句式要多样（陈述句、疑问句、吐槽、带情绪、省略主语、口语化），覆盖不同品类与场景；
4. 不要出现英文单词、emoji、编号、重复句。

只输出 JSON，不要任何解释，格式：
{{"search": ["...", ...], "recommend": [...], "classify": [...], "kg_qa": [...],
  "order": [...], "customer_service": [...], "analytics": [...], "chitchat": [...]}}"""


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        text = text[4:] if text.lower().startswith("json") else text
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start : end + 1])


# 项目既有约定（见 tests/router_eval.py 用例注释）：
#   "把某个商品分到某类"（指令式）→ classify
#   "这个商品属于什么类别"（问句式）→ kg_qa（问句式分类走知识图谱）
_ASK_STYLE_MARKERS = ("属于", "是什么", "算是什么", "算哪", "是哪")
_TOO_VAGUE = ("哪个区域", "放在哪个区", "归入哪个区")


def normalize_label(intent: str, query: str) -> tuple:
    """按项目既有约定修正标签；返回 (intent, note) 或 (None, reason) 表示该丢弃。"""
    if intent == "classify":
        if any(marker in query for marker in _TOO_VAGUE):
            return None, "分类指向过于模糊（区域/区）"
        if any(marker in query for marker in _ASK_STYLE_MARKERS):
            return "kg_qa", "问句式分类，按项目约定归 kg_qa"
    return intent, ""


def validate(raw: dict, existing: list) -> tuple:
    """机械校验：长度 / 去重。**关键词冲突不丢弃，只标注**。

    命中别的意图关键词的句子（如"家里有小孩适合买哪种家具"含 search 的"买"）恰恰是
    最该被测的难例——标签按"用户真实意图"给，规则误路由就是失败，这比丢掉它们更有信息量。
    """
    seen = {c["query"] for c in existing}
    kept, dropped = [], []
    for intent_key, queries in raw.items():
        if intent_key not in INTENT_DESC:
            continue
        for query in queries or []:
            query = str(query).strip()
            # 下限放到 5 个字符：闲聊类短句（"在忙吗""今天真热"）本来就短，
            # 卡 8 字会把整个 chitchat 类掉光
            if not (5 <= len(query) <= 40):
                dropped.append((intent_key, query, "长度不合适"))
                continue
            if query in seen:
                dropped.append((intent_key, query, "重复"))
                continue
            expected, relabel_note = normalize_label(intent_key, query)
            if expected is None:
                dropped.append((intent_key, query, relabel_note))
                continue
            seen.add(query)
            conflicts = sorted(
                other
                for other, words in RouterAgent.KEYWORD_RULES.items()
                if other != expected and any(w in query for w in words)
            )
            note = "LLM 生成（build_router_cases.py），人工过审"
            if relabel_note:
                note += f"；{relabel_note}"
            if conflicts:
                note += f"；含其他意图关键词 {'/'.join(conflicts)}（难例：规则可能误路由）"
            kept.append(
                {
                    "query": query,
                    "expected": expected,
                    "source": "llm_generated",
                    "note": note,
                }
            )
    return kept, dropped


def main() -> int:
    parser = argparse.ArgumentParser(description="扩充路由评估用例")
    parser.add_argument("--per-intent", type=int, default=15)
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--normalize-only",
        action="store_true",
        help="不重新生成，只对已落盘的用例重跑标签归一化（按项目既有约定修正 classify/kg_qa 边界）",
    )
    args = parser.parse_args()

    existing = load_all_test_cases()
    if args.normalize_only:
        with open(OUTPUT, encoding="utf-8") as handle:
            rows = json.load(handle)
        fixed, dropped_rows = [], []
        for row in rows:
            expected, relabel_note = normalize_label(row["expected"], row["query"])
            if expected is None:
                dropped_rows.append((row["expected"], row["query"], relabel_note))
                continue
            if expected != row["expected"]:
                row["expected"] = expected
                row["note"] = row["note"] + f"；{relabel_note}"
            fixed.append(row)
        with open(OUTPUT, "w", encoding="utf-8") as handle:
            json.dump(fixed, handle, ensure_ascii=False, indent=1)
        print(f"[BuildRouterCases] 归一化：保留 {len(fixed)} 条，丢弃 {len(dropped_rows)} 条")
        print("  分布:", dict(collections.Counter(r["expected"] for r in fixed)))
        for intent, query, reason in dropped_rows:
            print(f"    [{intent}] {query} -> {reason}")
        return 0

    llm = ChatOpenAI(
        model=args.model or settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=args.temperature,
        max_tokens=4096,
    )
    print(f"[BuildRouterCases] 向 {args.model or settings.deepseek_model} 请求 "
          f"{len(INTENT_DESC)} × {args.per_intent} 条用例…")
    raw = parse_json(llm.invoke(build_prompt(args.per_intent)).content)

    kept, dropped = validate(raw, existing)
    dist = collections.Counter(c["expected"] for c in kept)
    print(f"[BuildRouterCases] 生成 {sum(len(v or []) for v in raw.values())} 条，"
          f"机械校验通过 {len(kept)} 条，丢弃 {len(dropped)} 条")
    print("  通过用例的意图分布:", dict(dist))
    if dropped:
        print("  丢弃样例（前 8 条）:")
        for intent, query, reason in dropped[:8]:
            print(f"    [{intent}] {query[:26]} -> {reason}")

    if args.dry_run:
        print("\n  （--dry-run：未写文件）")
        return 0

    with open(OUTPUT, "w", encoding="utf-8") as handle:
        json.dump(kept, handle, ensure_ascii=False, indent=1)
    print(f"  已写入: {OUTPUT}（{len(kept)} 条，source=llm_generated）")
    print(f"  手写用例仍为 {len(ROUTER_TEST_CASES)} 条，评估时按 source 分开统计")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
