"""
商品分类 Agent 多分类指标 + 置信度校准评估

此前 model_regression.py 只测了"域内命中/域外拒识"二元判断,缺:
1. 多分类 Accuracy / Macro-F1 / Micro-F1
2. 每类 Precision/Recall/F1 + 混淆矩阵
3. 域外拒识的 AUROC / FPR@95%TPR (标准 OOD 指标)
4. 置信度校准 ECE (Expected Calibration Error)
   - 模型说 98.76% 置信, 实际准确率是不是真的接近 98.76%?
   - ECE=0 完美校准; ECE 越大模型越过度/不足自信

运行方式:
    python tests/classify_eval.py

    # 仅规则降级模式(无模型文件,零依赖)
    python tests/classify_eval.py --rule-only
"""

import json
import os
import sys
from typing import List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests._eval_env import ensure_project_root, require_assets

ensure_project_root()

from config.settings import settings
from tests._classification_metrics import (
    compute_classification_metrics,
    compute_ece,
)

# 4 类模型标签集 — 与 ClassifyAgent._default_labels() 一致
IN_DOMAIN_LABELS = ["医药保健", "家用电器", "手机数码", "食品生鲜"]
# 评估时把"无法判断"作为第 5 个类别
ALL_LABELS = IN_DOMAIN_LABELS + ["无法判断"]


# ------------------------------------------------------------------ #
#  测试用例集 — 域内 + 域外
# ---------------------------------------------------------------- #

# 域内用例: 4 类各 ~8 条,覆盖不同品牌/款式表达
IN_DOMAIN_CASES = [
    # 手机数码
    ("华为Mate60 Pro 12+512G 5G手机", "手机数码"),
    ("iPhone 15 Pro Max 256GB 钛金属", "手机数码"),
    ("小米14 Ultra 徕卡影像", "手机数码"),
    ("OPPO Find X7 天玑9300", "手机数码"),
    ("vivo X100 Pro 影像旗舰", "手机数码"),
    ("三星Galaxy S24 Ultra", "手机数码"),
    ("荣耀Magic6 至臻版", "手机数码"),
    ("红米K70 Pro 骁龙8Gen3", "手机数码"),
    ("realme GT5 240W快充", "手机数码"),
    ("一加12 2K屏幕", "手机数码"),

    # 家用电器
    ("戴森 V12 Detect Slim 无线吸尘器", "家用电器"),
    ("海尔冰箱 一级能效 双开门", "家用电器"),
    ("美的空调 1.5匹 挂机", "家用电器"),
    ("格力空调 变频冷暖", "家用电器"),
    ("苏泊尔电饭煲 4L 球釜", "家用电器"),
    ("九阳破壁机 静音", "家用电器"),
    ("飞利浦电动牙刷 HX6730", "家用电器"),
    ("松下微波炉 变频", "家用电器"),
    ("小米扫地机器人 扫拖一体", "家用电器"),
    ("追觅洗地机 无线", "家用电器"),

    # 医药保健
    ("鱼跃血压计 家用上臂式", "医药保健"),
    ("欧姆龙电子体温计 红外", "医药保健"),
    ("汤臣倍健维生素D3", "医药保健"),
    ("Swisse深海鱼油胶囊", "医药保健"),
    ("三九感冒灵颗粒", "医药保健"),
    ("同仁堂六味地黄丸", "医药保健"),
    ("云南白药气雾剂", "医药保健"),
    ("强生邦迪创可贴", "医药保健"),
    ("杜蕾斯避孕套 超薄", "医药保健"),
    ("稳健医用口罩 一次性", "医药保健"),

    # 食品生鲜
    ("新鲜智利车厘子JJ级 2斤装", "食品生鲜"),
    ("三只松鼠每日坚果750g", "食品生鲜"),
    ("良品铺子肉松饼 整箱", "食品生鲜"),
    ("百草味腰果 500g", "食品生鲜"),
    ("伊利金典纯牛奶 250ml*12", "食品生鲜"),
    ("蒙牛特仑苏 有机纯牛奶", "食品生鲜"),
    ("农夫山泉17.5°橙 5斤", "食品生鲜"),
    ("阳澄湖大闸蟹礼券 4两公3两母", "食品生鲜"),
    ("宁夏中宁枸杞 500g", "食品生鲜"),
    ("新疆和田大枣 一级", "食品生鲜"),
]

# 域外用例: 模型 4 类不覆盖的品类,期望被拒识为"无法判断"
OOD_CASES = [
    # 服装鞋包
    ("红色连衣裙 夏季新款", "连衣裙"),
    ("男士运动鞋 透气", "运动鞋"),
    ("羽绒服中长款 加厚", "羽绒服"),
    # 美妆护肤
    ("口红 哑光丝绒", "口红"),
    ("补水面膜 10片装", "面膜"),
    # 母婴
    ("婴儿纸尿裤 L码", "纸尿裤"),
    ("进口奶粉 3段", "奶粉"),
    # 图书文具
    ("考研数学教材 高数", "教材"),
    ("中性笔 0.5mm 黑色", "中性笔"),
    # 家居日用
    ("保温杯 316不锈钢 500ml", "保温杯"),
    ("雨伞 晴雨两用 自动", "雨伞"),
    # 运动户外
    ("户外帐篷 3-4人 防水", "帐篷"),
    ("哑铃 可调节 20kg", "哑铃"),
    # 宠物
    ("猫粮 10kg 全期", "猫粮"),
    ("狗牵引绳 中型犬", "牵引绳"),
    # 珠宝
    ("黄金项链 999足金", "项链"),
    # 乐器
    ("民谣吉他 单板 初学者", "吉他"),
]


# ------------------------------------------------------------------ #
#  分类评估器
# ---------------------------------------------------------------- #


class ClassifyEvaluator:
    """商品分类 Agent 多分类 + OOD + 校准评估"""

    def __init__(self, agent):
        """
        Args:
            agent: ClassifyAgent 实例
        """
        self.agent = agent

    @staticmethod
    def _parse_top_k(result: list) -> Tuple[str, float]:
        """
        解析 get_top_k 输出为 (预测类别, 置信度浮点)

        - 正常: [{"category":"手机数码", "confidence":"99.98%"}, ...] → ("手机数码", 0.9998)
        - 拒识: [{"category":"无法判断", "confidence":"N/A"}] → ("无法判断", 0.0)
        - 马氏距离拒识: confidence 形如 "马氏距离 12.34" → ("无法判断", 0.0)
        """
        if not result:
            return "无法判断", 0.0
        top = result[0]
        category = top.get("category", "无法判断")
        conf_str = top.get("confidence", "N/A")
        if conf_str == "N/A":
            return "无法判断", 0.0
        # 形如 "99.98%" 或 "马氏距离 12.34"
        if "马氏距离" in conf_str:
            return "无法判断", 0.0
        try:
            # 去掉 % 后转浮点
            return category, float(conf_str.rstrip("%")) / 100.0
        except (ValueError, AttributeError):
            return "无法判断", 0.0

    def evaluate(self) -> dict:
        """运行完整评估"""
        # --- 域内评估 ---
        in_domain_true = []
        in_domain_pred = []
        in_domain_conf = []
        in_domain_correct = []
        in_domain_titles = [c[0] for c in IN_DOMAIN_CASES]

        for title, expected in IN_DOMAIN_CASES:
            try:
                top_k = self.agent.get_top_k(title, k=1)
                pred, conf = self._parse_top_k(top_k)
            except Exception as e:
                pred, conf = "无法判断", 0.0
                print(f"  [评估异常] {title[:20]}: {e}")

            # 域内用例被拒识为"无法判断"视为误分类
            in_domain_true.append(expected)
            in_domain_pred.append(pred)
            in_domain_conf.append(conf)
            in_domain_correct.append(pred == expected)

        # --- 域外拒识评估 ---
        ood_titles = []
        ood_pred_labels = []  # 模型对 OOD 样本的 top1 预测
        ood_confs = []
        ood_rejected = []  # 是否被拒识为"无法判断"

        for title, _ in OOD_CASES:
            try:
                top_k = self.agent.get_top_k(title, k=1)
                pred, conf = self._parse_top_k(top_k)
            except Exception:
                pred, conf = "无法判断", 0.0

            ood_titles.append(title)
            ood_pred_labels.append(pred)
            ood_confs.append(conf)
            ood_rejected.append(pred == "无法判断")

        # --- 1. 多分类指标(域内 + 域外,统一以 ALL_LABELS 评估) ---
        # OOD 用例期望预测="无法判断"
        all_true = in_domain_true + ["无法判断"] * len(OOD_CASES)
        all_pred = in_domain_pred + ood_pred_labels
        cls_metrics = compute_classification_metrics(all_true, all_pred, ALL_LABELS)

        # --- 2. 置信度校准 ECE (仅域内样本) ---
        # 校准问的是: 模型对 top1 预测的置信度是否与真实准确率匹配
        ece_result = compute_ece(in_domain_conf, in_domain_correct, n_bins=10)

        # --- 3. OOD 检测能力 ---
        # 二分类: OOD(应拒识=1) vs InDomain(应接收=0)
        # 拒识置信度 = 1 - top1_conf (模型越不自信, 越倾向拒识)
        ood_true = [1] * len(OOD_CASES) + [0] * len(IN_DOMAIN_CASES)
        ood_pred_binary = [1 if r else 0 for r in ood_rejected] + [
            0 if ok else 1 for ok in in_domain_correct  # 域内误分类也视为"应拒识但被接收"
        ]
        ood_scores = [1.0 - c for c in ood_confs] + [1.0 - c for c in in_domain_conf]
        ood_metrics = self._compute_ood_metrics(ood_true, ood_pred_binary, ood_scores)

        return {
            "in_domain_total": len(IN_DOMAIN_CASES),
            "ood_total": len(OOD_CASES),
            "classification": cls_metrics,
            "calibration": ece_result,
            "ood_detection": ood_metrics,
            "in_domain_errors": [
                {"title": t, "expected": e, "predicted": p, "confidence": round(c, 4)}
                for t, e, p, c, ok in zip(
                    in_domain_titles,
                    in_domain_true,
                    in_domain_pred,
                    in_domain_conf,
                    in_domain_correct,
                )
                if not ok
            ],
            "ood_errors": [
                {
                    "title": t,
                    "expected": "无法判断",
                    "predicted": p,
                    "confidence": round(c, 4),
                }
                for t, p, c, r in zip(ood_titles, ood_pred_labels, ood_confs, ood_rejected)
                if not r  # 未被拒识的 OOD 样本
            ],
        }

    @staticmethod
    def _compute_ood_metrics(
        y_true: List[int], y_pred: List[int], scores: List[float]
    ) -> dict:
        """
        计算 OOD 检测指标

        y_true: 1=OOD(应拒识), 0=InDomain(应接收)
        y_pred: 1=拒识, 0=接收
        scores: 拒识置信度(越高越可能 OOD)

        Returns:
            {
                "ood_recall": OOD 样本被正确拒识的比例,
                "indomain_accept_rate": 域内样本被正确接收的比例,
                "fpr_at_95_tpr": 在 95% OOD 召回下, 域内被误拒的比例,
                "auroc": ROC 曲线下面积,
            }
        """
        n_ood = sum(1 for y in y_true if y == 1)
        n_in = sum(1 for y in y_true if y == 0)
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        ood_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        indomain_accept = tn / (tn + fp) if (tn + fp) > 0 else 0

        # FPR@95%TPR: 找一个阈值使 OOD 召回≥95%, 看此时的域内误拒率
        # 按 score 降序, 逐步降低阈值, 找到 TPR≥95% 的最小 FPR
        fpr_at_95 = ClassifyEvaluator._fpr_at_tpr(y_true, scores, target_tpr=0.95)
        auroc = ClassifyEvaluator._auroc(y_true, scores)

        return {
            "ood_recall": round(ood_recall, 4),
            "indomain_accept_rate": round(indomain_accept, 4),
            "fpr_at_95_tpr": round(fpr_at_95, 4),
            "auroc": round(auroc, 4),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

    @staticmethod
    def _fpr_at_tpr(y_true: List[int], scores: List[float], target_tpr: float) -> float:
        """计算指定 TPR 下的 FPR"""
        n_ood = sum(1 for y in y_true if y == 1)
        n_in = sum(1 for y in y_true if y == 0)
        if n_ood == 0 or n_in == 0:
            return 1.0

        # 按 score 降序排列
        paired = sorted(zip(y_true, scores), key=lambda x: -x[1])
        tp = fp = 0
        for threshold_idx in range(len(paired)):
            y, s = paired[threshold_idx]
            if y == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / n_ood
            if tpr >= target_tpr:
                return fp / n_in
        return 1.0

    @staticmethod
    def _auroc(y_true: List[int], scores: List[float]) -> float:
        """计算 AUROC(ROC 曲线下面积)"""
        n_ood = sum(1 for y in y_true if y == 1)
        n_in = sum(1 for y in y_true if y == 0)
        if n_ood == 0 or n_in == 0:
            return 0.5

        # Mann-Whitney U 统计量
        ood_scores = [s for y, s in zip(y_true, scores) if y == 1]
        in_scores = [s for y, s in zip(y_true, scores) if y == 0]

        correct_pairs = 0
        for ood_s in ood_scores:
            for in_s in in_scores:
                if ood_s > in_s:
                    correct_pairs += 1
                elif ood_s == in_s:
                    correct_pairs += 0.5

        return correct_pairs / (n_ood * n_in)


# ------------------------------------------------------------------ #
#  报告打印
# ---------------------------------------------------------------- #


def print_report(result: dict):
    """打印可读报告"""
    print(f"\n  域内用例: {result['in_domain_total']}  |  域外用例: {result['ood_total']}")

    cls = result["classification"]
    print(f"\n  📊 多分类指标 (4 类 + 无法判断):")
    print(f"    Accuracy: {cls['accuracy']:.4f}")
    print(f"    Macro-F1: {cls['macro_f1']:.4f}  (P={cls['macro_precision']:.4f}, R={cls['macro_recall']:.4f})")
    print(f"    Micro-F1: {cls['micro_f1']:.4f}")

    print(f"\n    每类指标:")
    print(f"      {'class':<12s}{'P':>8s}{'R':>8s}{'F1':>8s}{'support':>10s}  errors")
    for label in ALL_LABELS:
        c = cls["per_class"].get(label, {})
        if c.get("support", 0) == 0:
            continue
        err_str = ", ".join(f"{k}={v}" for k, v in c.get("errors", {}).items()) or "—"
        print(
            f"      {label:<12s}{c['precision']:>8.4f}{c['recall']:>8.4f}"
            f"{c['f1']:>8.4f}{c['support']:>10d}  {err_str}"
        )

    cal = result["calibration"]
    print(f"\n  🎯 置信度校准 (ECE):")
    print(f"    ECE = {cal['ece']:.4f}  (0=完美校准, 越大越过度/不足自信)")
    print(f"    分桶明细 (置信度区间 → 实际准确率):")
    for b in cal["bins"]:
        if b["count"] == 0:
            continue
        print(
            f"      [{b['bin_low']:.2f}, {b['bin_high']:.2f}]  "
            f"n={b['count']:>3d}  avg_conf={b['avg_conf']:.4f}  "
            f"acc={b['acc']:.4f}  gap={b['gap']:.4f}"
        )

    ood = result["ood_detection"]
    print(f"\n  🛡️  OOD 域外拒识:")
    print(f"    OOD Recall (域外样本被拒识比例): {ood['ood_recall']:.4f}")
    print(f"    InDomain Accept (域内被正确接收): {ood['indomain_accept_rate']:.4f}")
    print(f"    FPR@95%TPR (95% OOD 召回下的域内误拒率): {ood['fpr_at_95_tpr']:.4f}")
    print(f"    AUROC: {ood['auroc']:.4f}  (0.5=随机, 1.0=完美)")
    print(f"    混淆矩阵: TP={ood['confusion']['tp']} FP={ood['confusion']['fp']} "
          f"FN={ood['confusion']['fn']} TN={ood['confusion']['tn']}")

    if result["in_domain_errors"]:
        print(f"\n  ❌ 域内误分类 ({len(result['in_domain_errors'])} 个):")
        for e in result["in_domain_errors"][:10]:
            print(f"    [{e['expected']} → {e['predicted']}] conf={e['confidence']:.4f}  {e['title']}")

    if result["ood_errors"]:
        print(f"\n  ❌ OOD 未拒识 ({len(result['ood_errors'])} 个):")
        for e in result["ood_errors"][:10]:
            print(f"    {e['title']}  →  {e['predicted']}")


# ------------------------------------------------------------------ #
#  主入口
# ---------------------------------------------------------------- #


def main():
    import argparse

    parser = argparse.ArgumentParser(description="商品分类 Agent 评估")
    parser.add_argument(
        "--rule-only",
        action="store_true",
        help="仅使用规则降级模式(无模型文件)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  商品分类 Agent 评估 (多分类 + OOD + 校准)")
    print("=" * 60)

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    from agents.classify_agent import ClassifyAgent

    model_path = settings.classify_model_path
    labels_path = os.path.join(model_path, "labels.txt")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(model_path, "labels.txt")

    if args.rule_only:
        # 指定不存在路径,触发规则降级
        model_path = "/nonexistent/model"
        labels_path = "/nonexistent/labels.txt"
        print("  规则降级模式 (无模型文件)")
    else:
        # 模型缺失时直接报错退出, 避免在规则降级链路上产出误导性报告
        require_assets(
            model_path,
            os.path.join(model_path, "model.safetensors"),
            os.path.join(model_path, "labels.txt"),
        )

    agent = ClassifyAgent(model_path=model_path, labels_path=labels_path)
    evaluator = ClassifyEvaluator(agent)

    print(f"\n  域内用例 {len(IN_DOMAIN_CASES)} 条 (4 类各 ~10 条)")
    print(f"  域外用例 {len(OOD_CASES)} 条 (4 类不覆盖品类)")

    result = evaluator.evaluate()
    print_report(result)

    output_path = os.path.join(PROJECT_ROOT, "tests", "classify_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
