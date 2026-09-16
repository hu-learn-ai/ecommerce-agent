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

import csv
import json
import os
import sys
from typing import List, Tuple

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


# 真实短查询用例：用户实际输入 1-3 个词，而非完整商品标题。
# 标题型用例（IN_DOMAIN_CASES）已能拿到 ~99%，短查询才是线上真实分布，
# 也是 OOD 门控误拒的高发区。
SHORT_QUERY_CASES = [
    # 手机数码
    ("蓝牙耳机", "手机数码"),
    ("无线鼠标", "手机数码"),
    ("机械键盘", "手机数码"),
    ("充电宝", "手机数码"),
    ("数据线", "手机数码"),
    ("平板电脑", "手机数码"),
    ("显示器", "手机数码"),
    ("路由器", "手机数码"),
    ("智能手表", "手机数码"),
    ("固态硬盘", "手机数码"),
    ("蓝牙音箱", "手机数码"),
    ("游戏鼠标", "手机数码"),
    ("电竞显示器", "手机数码"),
    ("手机壳", "手机数码"),
    ("摄像头", "手机数码"),
    ("内存条", "手机数码"),
    # 家用电器
    ("电饭煲", "家用电器"),
    ("微波炉", "家用电器"),
    ("洗衣机", "家用电器"),
    ("冰箱", "家用电器"),
    ("空调", "家用电器"),
    ("电视机", "家用电器"),
    ("吸尘器", "家用电器"),
    ("扫地机器人", "家用电器"),
    ("破壁机", "家用电器"),
    ("豆浆机", "家用电器"),
    ("电水壶", "家用电器"),
    ("加湿器", "家用电器"),
    ("电风扇", "家用电器"),
    ("热水器", "家用电器"),
    ("电磁炉", "家用电器"),
    ("洗碗机", "家用电器"),
    ("吹风机", "家用电器"),
    ("剃须刀", "家用电器"),
    ("净水器", "家用电器"),
    ("挂烫机", "家用电器"),
    # 食品生鲜
    ("薯片", "食品生鲜"),
    ("坚果礼盒", "食品生鲜"),
    ("巧克力", "食品生鲜"),
    ("咖啡豆", "食品生鲜"),
    ("龙井茶", "食品生鲜"),
    ("蜂蜜", "食品生鲜"),
    ("牛肉干", "食品生鲜"),
    ("三文鱼", "食品生鲜"),
    ("鸡蛋", "食品生鲜"),
    ("苹果", "食品生鲜"),
    ("香蕉", "食品生鲜"),
    ("酸奶", "食品生鲜"),
    ("面包", "食品生鲜"),
    ("食用油", "食品生鲜"),
    ("挂面", "食品生鲜"),
    ("螺蛳粉", "食品生鲜"),
    ("速冻水饺", "食品生鲜"),
    ("大闸蟹", "食品生鲜"),
    ("火龙果", "食品生鲜"),
    ("午餐肉", "食品生鲜"),
    # 医药保健
    ("维生素C", "医药保健"),
    ("钙片", "医药保健"),
    ("血糖仪", "医药保健"),
    ("体温计", "医药保健"),
    ("医用口罩", "医药保健"),
    ("创可贴", "医药保健"),
    ("鱼油", "医药保健"),
    ("益生菌", "医药保健"),
    ("蛋白粉", "医药保健"),
    ("叶酸", "医药保健"),
    ("感冒药", "医药保健"),
    ("眼药水", "医药保健"),
    ("颈椎按摩仪", "医药保健"),
    ("血氧仪", "医药保健"),
    ("雾化器", "医药保健"),
    ("医用棉签", "医药保健"),
    ("碘伏", "医药保健"),
    ("退热贴", "医药保健"),
    ("护膝", "医药保健"),
    ("理疗灯", "医药保健"),
]

# 短查询形式的域外用例（4 类不覆盖的品类）
OOD_SHORT_CASES = [
    ("连衣裙", "连衣裙"),
    ("牛仔裤", "牛仔裤"),
    ("羽绒服", "羽绒服"),
    ("口红", "口红"),
    ("面膜", "面膜"),
    ("精华液", "精华液"),
    ("纸尿裤", "纸尿裤"),
    ("婴儿奶粉", "奶粉"),
    ("猫粮", "猫粮"),
    ("狗粮", "狗粮"),
    ("猫砂", "猫砂"),
    ("宠物牵引绳", "牵引绳"),
    ("户外帐篷", "帐篷"),
    ("哑铃", "哑铃"),
    ("瑜伽垫", "瑜伽垫"),
    ("民谣吉他", "吉他"),
    ("保温杯", "保温杯"),
    ("雨伞", "雨伞"),
    ("中性笔", "中性笔"),
    ("考研教材", "教材"),
    ("黄金项链", "项链"),
    ("鲜花礼盒", "鲜花"),
    ("儿童玩具", "玩具"),
    ("汽车脚垫", "脚垫"),
]


def load_training_texts() -> set:
    """加载增强训练集文本，用于判断短查询是否在训练集中出现过（区分"见过/没见过"）。

    增强数据由商品标题派生出大量短查询变体，若不区分，短查询准确率会被"训练集里见过"
    的样本拉高。缺失训练集时返回空集合（该维度不参与统计）。
    """
    for name in ("classify_train_real_aug_train.csv", "classify_train_real.csv"):
        path = os.path.join(PROJECT_ROOT, "data", "processed", name)
        if not os.path.exists(path):
            continue
        texts = set()
        with open(path, encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                texts.add(row["text"].strip())
            return texts
    print("  [提示] 未找到增强训练集，跳过'是否见过'维度")
    return set()


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
        """运行完整评估（分组：完整标题 / 真实短查询 / 短查询中训练集未见过）"""

        def run_in_domain(cases) -> list:
            rows = []
            for text, expected in cases:
                try:
                    top_k = self.agent.get_top_k(text, k=1)
                    pred, conf = self._parse_top_k(top_k)
                except Exception as exc:  # noqa: BLE001
                    pred, conf = "无法判断", 0.0
                    print(f"  [评估异常] {text[:20]}: {exc}")
                rows.append(
                    {
                        "title": text,
                        "expected": expected,
                        "predicted": pred,
                        "confidence": round(conf, 4),
                        "correct": pred == expected,
                    }
                )
            return rows

        def run_ood(cases) -> list:
            rows = []
            for text, _ in cases:
                try:
                    top_k = self.agent.get_top_k(text, k=1)
                    pred, conf = self._parse_top_k(top_k)
                except Exception:  # noqa: BLE001
                    pred, conf = "无法判断", 0.0
                rows.append(
                    {
                        "title": text,
                        "expected": "无法判断",
                        "predicted": pred,
                        "confidence": round(conf, 4),
                        "rejected": pred == "无法判断",
                    }
                )
            return rows

        def group_metrics(in_rows: list, ood_rows: list) -> dict:
            all_true = [r["expected"] for r in in_rows] + ["无法判断"] * len(ood_rows)
            all_pred = [r["predicted"] for r in in_rows] + [r["predicted"] for r in ood_rows]
            ood_true = [1] * len(ood_rows) + [0] * len(in_rows)
            ood_pred_binary = [1 if r["rejected"] else 0 for r in ood_rows] + [
                0 if r["correct"] else 1 for r in in_rows
            ]
            ood_scores = [1.0 - r["confidence"] for r in ood_rows] + [
                1.0 - r["confidence"] for r in in_rows
            ]
            return {
                "in_domain_total": len(in_rows),
                "ood_total": len(ood_rows),
                "classification": compute_classification_metrics(all_true, all_pred, ALL_LABELS),
                "calibration": compute_ece(
                    [r["confidence"] for r in in_rows],
                    [r["correct"] for r in in_rows],
                    n_bins=10,
                ),
                "ood_detection": self._compute_ood_metrics(
                    ood_true, ood_pred_binary, ood_scores
                ),
            }

        print("  [组 1/3] 完整商品标题 ...")
        long_in = run_in_domain(IN_DOMAIN_CASES)
        long_ood = run_ood(OOD_CASES)

        print("  [组 2/3] 真实短查询 ...")
        short_in = run_in_domain(SHORT_QUERY_CASES)
        short_ood = run_ood(OOD_SHORT_CASES)

        # 标注短查询是否在增强训练集中出现（增强数据由标题派生大量短查询变体）
        train_texts = load_training_texts()
        for row in short_in:
            row["seen_in_train"] = row["title"] in train_texts
        unseen_in = [r for r in short_in if not r["seen_in_train"]]
        seen_ratio = (
            sum(1 for r in short_in if r["seen_in_train"]) / len(short_in)
            if short_in
            else 0.0
        )

        print(f"  [组 3/3] 短查询中训练集未见过（{len(unseen_in)} 条）...")
        by_group = {
            "long_title": group_metrics(long_in, long_ood),
            "short_query": group_metrics(short_in, short_ood),
            "short_query_unseen": group_metrics(unseen_in, short_ood),
            "all": group_metrics(long_in + short_in, long_ood + short_ood),
        }

        # 顶层字段保持与原报告同口径（完整标题组），新增 by_group 供分组对比
        result = dict(by_group["long_title"])
        result.update(
            {
                "by_group": by_group,
                "short_query_seen_in_train_ratio": round(seen_ratio, 4),
                "in_domain_errors": [r for r in long_in + short_in if not r["correct"]],
                "ood_errors": [r for r in long_ood + short_ood if not r["rejected"]],
            }
        )
        return result

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
        sum(1 for y in y_true if y == 1)
        sum(1 for y in y_true if y == 0)
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
    groups = result.get("by_group") or {}
    if groups:
        print("\n  📊 分组结果（难易对比）:")
        print(
            f"    {'group':<28s}{'域内':>5s}{'域外':>5s}{'Accuracy':>10s}"
            f"{'MacroF1':>9s}{'OOD召回':>9s}{'域内接收':>9s}"
        )
        for key, label in (
            ("long_title", "完整标题"),
            ("short_query", "真实短查询"),
            ("short_query_unseen", "短查询·训练集未见"),
            ("all", "合并"),
        ):
            group = groups.get(key)
            if not group:
                continue
            group_cls = group["classification"]
            group_ood = group["ood_detection"]
            print(
                f"    {label:<28s}{group['in_domain_total']:>5d}{group['ood_total']:>5d}"
                f"{group_cls['accuracy']:>10.4f}{group_cls['macro_f1']:>9.4f}"
                f"{group_ood['ood_recall']:>9.4f}{group_ood['indomain_accept_rate']:>9.4f}"
            )
        print(
            f"    短查询中在增强训练集里出现过的比例: "
            f"{result.get('short_query_seen_in_train_ratio', 0):.2%}"
        )

    detail = groups.get("all") or result
    print(f"\n  明细口径: {detail['in_domain_total']} 条域内 + {detail['ood_total']} 条域外")

    cls = detail["classification"]
    print("\n  📊 多分类指标 (4 类 + 无法判断):")
    print(f"    Accuracy: {cls['accuracy']:.4f}")
    print(f"    Macro-F1: {cls['macro_f1']:.4f}  (P={cls['macro_precision']:.4f}, R={cls['macro_recall']:.4f})")
    print(f"    Micro-F1: {cls['micro_f1']:.4f}")

    print("\n    每类指标:")
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

    cal = detail["calibration"]
    print("\n  🎯 置信度校准 (ECE):")
    print(f"    ECE = {cal['ece']:.4f}  (0=完美校准, 越大越过度/不足自信)")
    print("    分桶明细 (置信度区间 → 实际准确率):")
    for b in cal["bins"]:
        if b["count"] == 0:
            continue
        print(
            f"      [{b['bin_low']:.2f}, {b['bin_high']:.2f}]  "
            f"n={b['count']:>3d}  avg_conf={b['avg_conf']:.4f}  "
            f"acc={b['acc']:.4f}  gap={b['gap']:.4f}"
        )

    ood = detail["ood_detection"]
    print("\n  🛡️  OOD 域外拒识:")
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
