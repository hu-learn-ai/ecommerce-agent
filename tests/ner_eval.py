"""
NER 实体抽取 Agent 评估 — entity-level P/R/F1

此前 model_regression.py 只断言"华为 in 品牌"这种弱检查, 缺:
1. Entity-level Precision/Recall/F1 (按实体文本完全匹配)
2. 按实体类型(品牌/商品/款式/规格)拆分的 P/R/F1
3. 微观 vs 宏观平均
4. 边界错误分析 (部分匹配、类型混淆、漏抽、多抽)

运行方式:
    python tests/ner_eval.py
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root, require_assets

ensure_project_root()

from config.settings import settings

# 4 种实体类型 — 与 NerAgent.ENTITY_NAMES 一致
ENTITY_TYPES = ["品牌", "商品", "款式", "规格"]


# ------------------------------------------------------------------ #
#  测试用例集 — 每条用例标注期望抽取的实体
# ---------------------------------------------------------------- #

NER_TEST_CASES = [
    # 品牌明确 + 商品明确
    {
        "query": "华为Mate60 Pro 5G手机",
        "expected": {"品牌": ["华为"], "商品": ["手机"], "款式": [], "规格": []},
    },
    {
        "query": "海尔冰箱 一级能效 双开门",
        "expected": {"品牌": ["海尔"], "商品": ["冰箱"], "款式": [], "规格": []},
    },
    {
        "query": "苹果iPhone 15 Pro Max 256GB",
        "expected": {"品牌": ["苹果"], "商品": ["iPhone"], "款式": [], "规格": []},
    },
    {
        "query": "小米蓝牙耳机 降噪版",
        "expected": {"品牌": ["小米"], "商品": ["蓝牙耳机"], "款式": [], "规格": []},
    },
    {
        "query": "戴森V12 Detect Slim 无线吸尘器",
        "expected": {"品牌": ["戴森"], "商品": ["吸尘器"], "款式": [], "规格": []},
    },
    {
        "query": "美的空调 1.5匹 挂机",
        "expected": {"品牌": ["美的"], "商品": ["空调"], "款式": [], "规格": []},
    },
    {
        "query": "苏泊尔电饭煲 4L 球釜",
        "expected": {"品牌": ["苏泊尔"], "商品": ["电饭煲"], "款式": [], "规格": []},
    },
    {
        "query": "飞利浦电动牙刷 HX6730",
        "expected": {"品牌": ["飞利浦"], "商品": ["电动牙刷"], "款式": [], "规格": []},
    },

    # 仅商品(无品牌)
    {
        "query": "蓝牙耳机无线降噪",
        "expected": {"品牌": [], "商品": ["蓝牙耳机"], "款式": [], "规格": []},
    },
    {
        "query": "户外帐篷 防水",
        "expected": {"品牌": [], "商品": ["帐篷"], "款式": [], "规格": []},
    },
    {
        "query": "连衣裙 红色 韩版",
        "expected": {"品牌": [], "商品": ["连衣裙"], "款式": ["红色", "韩版"], "规格": []},
    },
    {
        "query": "运动鞋 男款 透气",
        "expected": {"品牌": [], "商品": ["运动鞋"], "款式": [], "规格": []},
    },

    # 款式/规格
    {
        "query": "纸尿裤 L码 超薄",
        "expected": {"品牌": [], "商品": ["纸尿裤"], "款式": [], "规格": []},
    },
    {
        "query": "口红哑光丝绒 红色",
        "expected": {"品牌": [], "商品": ["口红"], "款式": ["红色"], "规格": []},
    },

    # 复合/多实体
    {
        "query": "小米手环8华为手表对比",
        "expected": {"品牌": ["小米", "华为"], "商品": ["手环", "手表"], "款式": [], "规格": []},
    },
    {
        "query": "iPhone和Android手机哪个好",
        "expected": {"品牌": [], "商品": ["手机"], "款式": [], "规格": []},
    },

    # 无实体 / 纯闲聊
    {
        "query": "你好",
        "expected": {"品牌": [], "商品": [], "款式": [], "规格": []},
    },
    {
        "query": "怎么退货",
        "expected": {"品牌": [], "商品": [], "款式": [], "规格": []},
    },

    # 边界 case
    {
        "query": "500元以下的蓝牙耳机",
        "expected": {"品牌": [], "商品": ["蓝牙耳机"], "款式": [], "规格": []},
    },
    {
        "query": "运动鞋篮球鞋跑步鞋推荐",
        "expected": {"品牌": [], "商品": ["运动鞋", "篮球鞋", "跑步鞋"], "款式": [], "规格": []},
    },
]


# ------------------------------------------------------------------ #
#  Entity-level 指标计算
# ---------------------------------------------------------------- #


class EntityMetrics:
    """NER entity-level 指标计算器(完全匹配 + 部分匹配)"""

    @staticmethod
    def compute(
        y_true: List[Dict[str, List[str]]],
        y_pred: List[Dict[str, List[str]]],
    ) -> dict:
        """
        计算 entity-level P/R/F1

        匹配规则: 实体文本完全一致(大小写敏感)且类型一致才算 TP

        Returns:
            {
                "micro": {"precision", "recall", "f1"},
                "macro": {"precision", "recall", "f1"},
                "per_type": {type: {"precision", "recall", "f1", "tp", "fp", "fn"}},
                "errors": [
                    {"query", "type", "true", "pred", "error_type"}, ...
                ]
            }
        """
        assert len(y_true) == len(y_pred)

        # 累计 TP/FP/FN
        per_type_tp = defaultdict(int)
        per_type_fp = defaultdict(int)
        per_type_fn = defaultdict(int)
        errors = []

        for case, true_ents, pred_ents in zip(NER_TEST_CASES, y_true, y_pred):
            query = case["query"]
            for etype in ENTITY_TYPES:
                true_set = set(true_ents.get(etype, []))
                pred_set = set(pred_ents.get(etype, []))

                tp = true_set & pred_set
                fp = pred_set - true_set  # 多抽(误报)
                fn = true_set - pred_set  # 漏抽

                per_type_tp[etype] += len(tp)
                per_type_fp[etype] += len(fp)
                per_type_fn[etype] += len(fn)

                # 记录错误样例
                for ent in fp:
                    errors.append({
                        "query": query,
                        "type": etype,
                        "true": None,
                        "pred": ent,
                        "error_type": "false_positive",
                    })
                for ent in fn:
                    errors.append({
                        "query": query,
                        "type": etype,
                        "true": ent,
                        "pred": None,
                        "error_type": "false_negative",
                    })

        # 每类指标
        per_type = {}
        for etype in ENTITY_TYPES:
            tp = per_type_tp[etype]
            fp = per_type_fp[etype]
            fn = per_type_fn[etype]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )
            per_type[etype] = {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "support": tp + fn,  # 该类真实实体总数
            }

        # Micro 平均
        total_tp = sum(per_type_tp.values())
        total_fp = sum(per_type_fp.values())
        total_fn = sum(per_type_fn.values())
        micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

        # Macro 平均 (只在有真实样本的类上平均)
        valid_types = [t for t in ENTITY_TYPES if per_type[t]["support"] > 0]
        macro_p = sum(per_type[t]["precision"] for t in valid_types) / len(valid_types) if valid_types else 0
        macro_r = sum(per_type[t]["recall"] for t in valid_types) / len(valid_types) if valid_types else 0
        macro_f1 = sum(per_type[t]["f1"] for t in valid_types) / len(valid_types) if valid_types else 0

        return {
            "micro": {
                "precision": round(micro_p, 4),
                "recall": round(micro_r, 4),
                "f1": round(micro_f1, 4),
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
            },
            "macro": {
                "precision": round(macro_p, 4),
                "recall": round(macro_r, 4),
                "f1": round(macro_f1, 4),
            },
            "per_type": per_type,
            "errors": errors,
        }


# ------------------------------------------------------------------ #
#  NER 评估器
# ---------------------------------------------------------------- #


class NerEvaluator:
    """NER Agent entity-level 评估器"""

    def __init__(self, agent):
        """
        Args:
            agent: NerAgent 实例
        """
        self.agent = agent

    def evaluate(self) -> dict:
        """运行评估"""
        y_true = []
        y_pred = []

        for case in NER_TEST_CASES:
            query = case["query"]
            expected = case["expected"]

            try:
                pred = self.agent.extract(query)
            except Exception as e:
                print(f"  [评估异常] {query}: {e}")
                pred = {t: [] for t in ENTITY_TYPES}

            # 规范化: 确保所有 4 个类型都存在
            pred_norm = {t: pred.get(t, []) for t in ENTITY_TYPES}

            y_true.append(expected)
            y_pred.append(pred_norm)

        metrics = EntityMetrics.compute(y_true, y_pred)
        return {
            "total_cases": len(NER_TEST_CASES),
            "metrics": metrics,
        }


# ------------------------------------------------------------------ #
#  报告打印
# ---------------------------------------------------------------- #


def print_report(result: dict):
    """打印可读报告"""
    print(f"\n  测试用例: {result['total_cases']}")
    m = result["metrics"]

    print("\n  📊 Entity-level 指标:")
    print(f"    Micro:  P={m['micro']['precision']:.4f}  R={m['micro']['recall']:.4f}  F1={m['micro']['f1']:.4f}")
    print(f"            (TP={m['micro']['tp']}  FP={m['micro']['fp']}  FN={m['micro']['fn']})")
    print(f"    Macro:  P={m['macro']['precision']:.4f}  R={m['macro']['recall']:.4f}  F1={m['macro']['f1']:.4f}")

    print("\n    按实体类型:")
    print(f"      {'type':<8s}{'P':>8s}{'R':>8s}{'F1':>8s}{'tp':>6s}{'fp':>6s}{'fn':>6s}{'support':>10s}")
    for etype in ENTITY_TYPES:
        c = m["per_type"][etype]
        if c["support"] == 0:
            continue
        print(
            f"      {etype:<8s}{c['precision']:>8.4f}{c['recall']:>8.4f}"
            f"{c['f1']:>8.4f}{c['tp']:>6d}{c['fp']:>6d}{c['fn']:>6d}{c['support']:>10d}"
        )

    errors = m["errors"]
    if errors:
        fp_errors = [e for e in errors if e["error_type"] == "false_positive"]
        fn_errors = [e for e in errors if e["error_type"] == "false_negative"]
        print(f"\n  ❌ 错误分析 ({len(errors)} 个):")
        print(f"    漏抽 (FN): {len(fn_errors)} 个")
        for e in fn_errors[:8]:
            print(f"      [{e['type']}] 期望='{e['true']}'  query='{e['query']}'")
        print(f"    多抽 (FP): {len(fp_errors)} 个")
        for e in fp_errors[:8]:
            print(f"      [{e['type']}] 实际='{e['pred']}'  query='{e['query']}'")


# ------------------------------------------------------------------ #
#  主入口
# ---------------------------------------------------------------- #


def main():
    print("=" * 60)
    print("  NER 实体抽取 Agent 评估 (entity-level P/R/F1)")
    print("=" * 60)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from agents.ner_agent import NerAgent

    require_assets(
        settings.ner_model_path,
        os.path.join(settings.ner_model_path, "model.safetensors"),
    )
    agent = NerAgent(model_path=settings.ner_model_path)
    evaluator = NerEvaluator(agent)

    print(f"\n  测试用例 {len(NER_TEST_CASES)} 条 (覆盖品牌/商品/款式/规格)")

    result = evaluator.evaluate()
    print_report(result)

    output_path = os.path.join(PROJECT_ROOT, "tests", "ner_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
