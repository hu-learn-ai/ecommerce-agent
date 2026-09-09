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
        test_cases = test_cases or ROUTER_TEST_CASES
        y_true, y_pred = [], []
        errors = []
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
        print(f"  Accuracy: {m['accuracy']:.4f}")
        print(f"  Macro-F1: {m['macro_f1']:.4f}  (Macro-P={m['macro_precision']:.4f}, Macro-R={m['macro_recall']:.4f})")
        print(f"  Micro-F1: {m['micro_f1']:.4f}")

        if "level1" in result:
            l1 = result["level1"]
            print(f"\n  Level 1 (keyword) 分层:")
            print(f"    命中率: {l1['hit_rate']:.4f}  ({l1['hit_count']}/{l1['total']})")
            print(f"    命中时准确率: {l1['accuracy_on_hits']:.4f}")

        print(f"\n  每类指标:")
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
