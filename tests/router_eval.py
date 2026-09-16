"""
意图路由器评估 — Router Agent 准确率与错误分析

路由器是系统入口,8 个意图决定后续 Agent 调用,但此前全项目零路由评估。
本评估补齐以下指标:
1. 整体 Accuracy / Macro-F1 / Micro-F1
2. 每类 Precision / Recall / F1
3. 混淆矩阵(哪个意图最易误判,误判到哪)
4. 路由分层评估:
   - Level 1 (keyword): 命中率 / 准确率 / 误路由分布
   - Level 2 (LLM): fallback 触发率 / LLM 准确率
5. 误路由代价分析(误路由到错误 Agent 的下游影响)

运行方式:
    # 仅 Level 1 (keyword, 零依赖, 无需 LLM/网络)
    python tests/router_eval.py --level keyword

    # Level 1 + Level 2 (需要 DeepSeek API)
    python tests/router_eval.py

    # 在代码中调用
    from tests.router_eval import RouterEvaluator
    evaluator = RouterEvaluator()
    result = evaluator.evaluate(level="both")
"""

import json
import os
import sys
from typing import List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_openai import ChatOpenAI

from config.settings import settings
from orchestration.router import RouterAgent

# 复用共享多分类指标计算器
from tests._classification_metrics import compute_classification_metrics

# ------------------------------------------------------------------ #
#  8 个意图 — 与 RouterAgent.IntentType 保持一致
# ---------------------------------------------------------------- #

INTENT_LABELS = [
    "kg_qa",
    "search",
    "classify",
    "recommend",
    "order",
    "customer_service",
    "analytics",
    "chitchat",
]


# ------------------------------------------------------------------ #
#  测试用例集 — 人工标注 expected_intent,覆盖 8 类意图 + 边界 case
# ---------------------------------------------------------------- #

ROUTER_TEST_CASES = [
    # --- search (商品搜索) ---
    {"query": "有没有500元以下的蓝牙耳机？", "expected": "search", "note": "价格+品类"},
    {"query": "搜索无线鼠标", "expected": "search", "note": "显式搜索词"},
    {"query": "卖iPhone 15吗", "expected": "search", "note": "口语化搜索"},
    {"query": "找一双跑步鞋", "expected": "search", "note": "找字"},
    {"query": "多少钱一台戴森吸尘器", "expected": "search", "note": "价格询问"},
    {"query": "有没有适合油皮的洗面奶", "expected": "search", "note": "条件搜索"},

    # --- recommend (个性化推荐) ---
    {"query": "推荐一些适合户外的装备", "expected": "recommend", "note": "显式推荐"},
    {"query": "预算200以内买什么母婴用品好", "expected": "recommend", "note": "预算+买什么"},
    {"query": "帮我选个礼物", "expected": "recommend", "note": "选什么"},
    {"query": "求推荐一款降噪耳机", "expected": "recommend", "note": "求推荐"},
    {"query": "有什么好的手机推荐", "expected": "recommend", "note": "有什么好"},
    {"query": "我最近喜欢户外运动,适合买什么", "expected": "recommend", "note": "适合"},

    # --- classify (直接分类请求) ---
    {"query": "分类: 纸尿裤", "expected": "classify", "note": "显式分类"},
    {"query": "把这个商品分到什么类", "expected": "classify", "note": "什么类"},
    {"query": "这个商品是什么类别", "expected": "kg_qa", "note": "问句式分类 → kg_qa"},

    # --- kg_qa (知识图谱问答 — 问句式分类/品牌属性) ---
    {"query": "纸尿裤属于什么类别", "expected": "kg_qa", "note": "属于什么"},
    {"query": "Apple有哪些产品", "expected": "kg_qa", "note": "有哪些产品"},
    {"query": "Nike是什么品牌", "expected": "kg_qa", "note": "什么牌子"},
    {"query": "华为和小米是什么关系", "expected": "kg_qa", "note": "什么关系"},
    {"query": "手机和路由器属于哪个分类", "expected": "kg_qa", "note": "哪个分类"},

    # --- order (订单/物流) ---
    {"query": "查询我的订单 ORD123456", "expected": "order", "note": "订单号"},
    {"query": "我的订单到哪了", "expected": "order", "note": "订单"},
    {"query": "快递什么时候到", "expected": "order", "note": "快递"},
    {"query": "我的物流信息", "expected": "order", "note": "物流"},
    {"query": "发货了吗", "expected": "order", "note": "发货"},

    # --- customer_service (售后/政策) ---
    {"query": "怎么退货", "expected": "customer_service", "note": "退货"},
    {"query": "退款多久到账", "expected": "customer_service", "note": "退款"},
    {"query": "运费谁出", "expected": "customer_service", "note": "运费"},
    {"query": "怎么开发票", "expected": "customer_service", "note": "发票"},
    {"query": "我要投诉", "expected": "customer_service", "note": "投诉"},
    {"query": "你们的售后政策是什么", "expected": "customer_service", "note": "售后"},

    # --- analytics (数据分析) ---
    {"query": "最近什么品类卖得最好", "expected": "analytics", "note": "销量+top"},
    {"query": "销量排行", "expected": "analytics", "note": "销量+排行"},
    {"query": "销售趋势如何", "expected": "analytics", "note": "趋势"},
    {"query": "给我看下数据统计", "expected": "analytics", "note": "数据统计"},
    {"query": "美妆护肤的销量占比", "expected": "analytics", "note": "占比"},

    # --- chitchat (闲聊/问候) ---
    {"query": "你好", "expected": "chitchat", "note": "问候"},
    {"query": "早上好", "expected": "chitchat", "note": "问候"},
    {"query": "谢谢", "expected": "chitchat", "note": "感谢"},
    {"query": "再见", "expected": "chitchat", "note": "告别"},
    {"query": "今天天气不错", "expected": "chitchat", "note": "闲聊"},
    {"query": "你是谁", "expected": "chitchat", "note": "身份询问"},

    # --- 边界 case (歧义/混淆) ---
    {"query": "推荐一款耳机", "expected": "recommend", "note": "推荐 vs 搜索"},
    {"query": "有没有耳机推荐", "expected": "recommend", "note": "推荐词在后"},
    {"query": "你好,我想买耳机", "expected": "search", "note": "问候+搜索混合"},
    {"query": "这个耳机多少钱", "expected": "search", "note": "价格询问,但无品类词触发"},
    {"query": "纸尿裤多少钱", "expected": "search", "note": "品类词但询问价格"},

    # --- 口语化 / 换说法：**故意不命中 Level-1 关键词**，用于真正测 LLM 兜底层 ---
    # 旧用例（含上面 47 条）与 Level-1 规则同源，100% 只反映规则覆盖；
    # 这一组不含任何规则词，全部会落到 Level-2，得出的才是 LLM 路由的真实准确率。
    {"query": "我上周下的那个东西，怎么到现在还没动静？", "expected": "order",
     "note": "口语化: 无 订单/物流/快递 字样"},
    {"query": "包裹现在在哪儿了", "expected": "order", "note": "口语化: 包裹"},
    {"query": "我想知道包裹啥时候能到", "expected": "order", "note": "口语化: 啥时候能到"},
    {"query": "收到的货有瑕疵，我要走什么流程？", "expected": "customer_service", "note": "口语化: 瑕疵"},
    {"query": "这个能七天无理由吗？", "expected": "customer_service", "note": "口语化: 七天无理由"},
    {"query": "上个月哪个品类表现最好？", "expected": "analytics", "note": "口语化: 表现最好"},
    {"query": "近期的经营情况怎么样？", "expected": "analytics", "note": "口语化: 经营情况"},
    {"query": "小米和红米是一家的吗？", "expected": "kg_qa", "note": "口语化: 关系问法"},
    {"query": "耐克是做什么的", "expected": "kg_qa", "note": "口语化: 问品牌"},
    {"query": "帮我判断一下这个商品该归到哪一类", "expected": "classify", "note": "口语化: 不用 分类/类别 字样"},
    {"query": "我想给女朋友挑个礼物", "expected": "recommend", "note": "口语化: 挑个"},
    {"query": "iphone 15 pro max 钛金属", "expected": "search", "note": "纯商品词,无搜索动词"},
    {"query": "在忙吗", "expected": "chitchat", "note": "口语化寒暄"},
    {"query": "哈哈你们真逗", "expected": "chitchat", "note": "口语化闲聊"},
]


# ------------------------------------------------------------------ #
#  分类指标计算 — 复用 tests._classification_metrics
# ---------------------------------------------------------------- #

# 保留 ClassificationMetrics 类名作为旧入口,内部委托给共享函数
class ClassificationMetrics:
    """多分类指标计算器(委托给 tests._classification_metrics)"""

    @staticmethod
    def compute(y_true: List[str], y_pred: List[str], labels: List[str]) -> dict:
        return compute_classification_metrics(y_true, y_pred, labels)


# ------------------------------------------------------------------ #
#  准确率的置信区间（n 小的时候,必须报区间而不是只报一个点）
# ---------------------------------------------------------------- #


def _betacf(a: float, b: float, x: float, itmax: int = 200, eps: float = 3e-12) -> float:
    """连分式展开（Numerical Recipes 的 betacf），用于正则化不完全 Beta。"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """正则化不完全 Beta 函数 I_x(a, b)（纯标准库实现,避免引入 scipy 依赖）。"""
    import math

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Beta 分布分位数的二分求根（用于 Clopper-Pearson 精确区间）。"""
    lo, hi = 0.0, 1.0
    for _ in range(120):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def accuracy_ci(correct: int, total: int, alpha: float = 0.05) -> dict:
    """准确率的 95% 置信区间：Wilson（常规）与 Clopper-Pearson（精确,保守）。

    61/61 = 100% 看起来很漂亮，但精确区间的下界只有 94.1%——
    小样本上"100%"必须带区间一起报，否则是过度声称。
    """
    import math

    if total == 0:
        return {}
    p = correct / total
    z = 1.959963985
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    wilson = ((center - margin) / denom, (center + margin) / denom)
    cp_low = 0.0 if correct == 0 else _beta_ppf(alpha / 2, correct, total - correct + 1)
    cp_high = 1.0 if correct == total else _beta_ppf(1 - alpha / 2, correct + 1, total - correct)
    return {
        "wilson_95": [round(wilson[0], 4), round(wilson[1], 4)],
        "clopper_pearson_95": [round(cp_low, 4), round(cp_high, 4)],
    }


def _group_by(rows: list, key: str) -> dict:
    """按字段分组（保持插入顺序）。"""
    grouped = {}
    for row in rows:
        grouped.setdefault(row.get(key), []).append(row)
    return grouped


GENERATED_CASES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "router_cases_generated.json")


def load_all_test_cases() -> list:
    """手写用例 + 生成用例（若 `router_cases_generated.json` 存在）。

    生成用例会带 `source: llm_generated` 标记，报告里按来源分开统计——
    手写用例与生成用例的可信度不同，不能混成一个数字对外说。
    """
    cases = [dict(c, source=c.get("source", "handwritten")) for c in ROUTER_TEST_CASES]
    if os.path.exists(GENERATED_CASES):
        try:
            with open(GENERATED_CASES, encoding="utf-8") as handle:
                extra = json.load(handle)
            cases += [
                {**c, "source": c.get("source", "llm_generated")}
                for c in extra
                if c.get("query") and c.get("expected")
            ]
        except Exception as exc:  # noqa: BLE001
            print(f"[RouterEval] 生成用例读取失败，仅用手写用例: {exc}")
    return cases


# ------------------------------------------------------------------ #
#  路由评估器
# ---------------------------------------------------------------- #


class RouterEvaluator:
    """
    Router Agent 准确率评估器

    分层评估:
    - Level 1 (keyword): 零依赖,纯规则匹配,评估关键词路由覆盖率与准确率
    - Level 2 (LLM): 关键词失败时 fallback 到 LLM,评估 LLM 路由准确率
    - full (Level 1 + 2): 真实生产链路,评估端到端路由准确率

    Args:
        llm: 用于 Level 2 路由的 LLM 实例;为 None 时只跑 Level 1
    """

    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm
        # router 在无 LLM 时也能跑 Level 1; 传 mock 避免初始化报错
        if llm is None:
            self.router = RouterAgent(llm=_MockLLM())
        else:
            self.router = RouterAgent(llm=llm)

    def evaluate(
        self,
        test_cases: List[dict] = None,
        level: str = "both",
    ) -> dict:
        """
        Args:
            test_cases: [{"query", "expected", "note"}]
            level: "keyword" | "llm" | "both"
                - keyword: 只评估 Level 1 关键词路由(零依赖)
                - llm: 强制走 Level 2 LLM 路由(忽略 Level 1 结果)
                - both: 真实生产链路(L1 失败时 fallback 到 L2)

        Returns:
            {
                "level": "both",
                "total": N,
                "metrics": {...ClassificationMetrics...},
                "level1": {"hit_rate": ..., "accuracy_on_hits": ...},
                "errors": [{query, expected, predicted, note}, ...],
            }
        """
        test_cases = test_cases or load_all_test_cases()
        y_true, y_pred = [], []
        errors = []
        rows = []
        level1_hits = 0
        level1_correct = 0

        for case in test_cases:
            query = case["query"]
            expected = case["expected"]

            if level == "keyword":
                # 只跑 Level 1, 不做 LLM fallback
                pred = self.router._keyword_route(query)
                if not pred:
                    pred = "chitchat"  # Level 1 无命中时默认走闲聊
                level1_hit = pred != "chitchat" or expected == "chitchat"
            else:
                if level == "llm":
                    # 强制走 Level 2: 跳过关键词匹配
                    pred_raw = self.router._llm_route(query)
                    pred = self.router._normalize_intent(pred_raw)
                    level1_hit = False
                else:  # "both" — 真实生产链路
                    # route() 内部先 L1 失败再 L2; 这里需检测 L1 是否命中
                    kw_pred = self.router._keyword_route(query)
                    level1_hit = kw_pred is not None
                    result = self.router.route(query)
                    pred = result["intent"]

            y_true.append(expected)
            y_pred.append(pred)

            if level1_hit:
                level1_hits += 1
                if pred == expected:
                    level1_correct += 1

            rows.append(
                {
                    "query": query,
                    "expected": expected,
                    "predicted": pred,
                    "correct": pred == expected,
                    "level1_hit": level1_hit,
                    "source": case.get("source", "handwritten"),
                    "note": case.get("note", ""),
                }
            )
            if pred != expected:
                errors.append(
                    {
                        "query": query,
                        "expected": expected,
                        "predicted": pred,
                        "note": case.get("note", ""),
                    }
                )

        metrics = ClassificationMetrics.compute(y_true, y_pred, INTENT_LABELS)

        result = {
            "level": level,
            "total": len(test_cases),
            "metrics": metrics,
            "errors": errors,
            "rows": rows,
            "accuracy_ci": accuracy_ci(sum(1 for r in rows if r["correct"]), len(rows)),
            # 按来源分组（手写用例 vs 生成用例），避免把两者混成一个数字
            "by_source": {
                source: {
                    "total": len(items),
                    "correct": sum(1 for r in items if r["correct"]),
                    "accuracy": round(sum(1 for r in items if r["correct"]) / len(items), 4),
                    "accuracy_ci": accuracy_ci(sum(1 for r in items if r["correct"]), len(items)),
                }
                for source, items in _group_by(rows, "source").items()
            },
            # 按“是否由关键词层命中”分组：这两层的能力完全不同，必须分开看
            "by_layer": {
                ("keyword" if hit else "llm"): {
                    "total": len(items),
                    "correct": sum(1 for r in items if r["correct"]),
                    "accuracy": round(sum(1 for r in items if r["correct"]) / len(items), 4),
                }
                for hit, items in _group_by(rows, "level1_hit").items()
            },
        }

        # Level 1 分层指标(只在 keyword/both 模式有意义)
        if level in ("keyword", "both"):
            result["level1"] = {
                "hit_rate": round(level1_hits / len(test_cases), 4),
                "accuracy_on_hits": round(level1_correct / level1_hits, 4) if level1_hits else 0,
                "hit_count": level1_hits,
                "total": len(test_cases),
            }

        return result

    def print_report(self, result: dict):
        """打印可读报告"""
        level = result["level"]
        m = result["metrics"]
        print(f"\n  路由模式: {level}")
        print(f"  测试用例: {result['total']}")
        ci = result.get("accuracy_ci") or {}
        ci_txt = ""
        if ci:
            ci_txt = "  95%%CI %.4f~%.4f（Wilson）/ %.4f~%.4f（精确）" % (
                ci["wilson_95"][0], ci["wilson_95"][1],
                ci["clopper_pearson_95"][0], ci["clopper_pearson_95"][1],
            )
        print(f"  Accuracy: {m['accuracy']:.4f}{ci_txt}")
        for source, info in (result.get("by_source") or {}).items():
            label = "手写用例" if source == "handwritten" else "生成用例"
            print(f"  [{label}] {info['correct']}/{info['total']} = {info['accuracy']:.4f}")
        for layer, info in (result.get("by_layer") or {}).items():
            label = "关键词层命中" if layer == "keyword" else "LLM 兜底层"
            print(f"  [{label}] {info['correct']}/{info['total']} = {info['accuracy']:.4f}")
        print(f"  Macro-F1: {m['macro_f1']:.4f}  (Macro-P={m['macro_precision']:.4f}, Macro-R={m['macro_recall']:.4f})")
        print(f"  Micro-F1: {m['micro_f1']:.4f}")

        if "level1" in result:
            l1 = result["level1"]
            print("\n  Level 1 (keyword) 分层:")
            print(f"    命中率: {l1['hit_rate']:.4f}  ({l1['hit_count']}/{l1['total']})")
            print(f"    命中时准确率: {l1['accuracy_on_hits']:.4f}")

        print("\n  每类指标:")
        print(f"    {'intent':<18s}{'P':>8s}{'R':>8s}{'F1':>8s}{'support':>10s}  errors")
        for label in INTENT_LABELS:
            c = m["per_class"].get(label, {})
            if c.get("support", 0) == 0:
                continue
            err_str = ", ".join(f"{k}={v}" for k, v in c.get("errors", {}).items()) or "—"
            print(
                f"    {label:<18s}{c['precision']:>8.4f}{c['recall']:>8.4f}"
                f"{c['f1']:>8.4f}{c['support']:>10d}  {err_str}"
            )

        if result["errors"]:
            print(f"\n  误路由样例 ({len(result['errors'])} 个):")
            for e in result["errors"][:10]:
                print(f"    [{e['expected']} → {e['predicted']}] {e['query']}")
                if e["note"]:
                    print(f"      note: {e['note']}")


class _MockLLM:
    """无 LLM 时的占位 — Level 1 不会调用它,避免初始化报错"""

    def invoke(self, *args, **kwargs):
        return type("Msg", (), {"content": "chitchat"})()


# ------------------------------------------------------------------ #
#  主入口
# ---------------------------------------------------------------- #


def main():
    import argparse

    parser = argparse.ArgumentParser(description="路由准确率评估")
    parser.add_argument(
        "--level",
        choices=["keyword", "llm", "both"],
        default="both",
        help="keyword=仅Level1(零依赖); llm=强制Level2; both=生产链路L1+L2",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  路由准确率评估 (Router Evaluation)")
    print("=" * 60)

    llm = None
    if args.level in ("llm", "both"):
        try:
            llm = ChatOpenAI(
                model=settings.deepseek_model,
                api_key=settings.deepseek_api_key,
                base_url=settings.deepseek_base_url,
                temperature=0.0,
                max_tokens=64,
            )
        except Exception as e:
            print(f"  LLM 初始化失败,降级为仅 Level 1: {e}")
            args.level = "keyword"

    evaluator = RouterEvaluator(llm=llm)
    result = evaluator.evaluate(level=args.level)

    evaluator.print_report(result)

    # 保存报告
    output_path = os.path.join(PROJECT_ROOT, "tests", "router_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
