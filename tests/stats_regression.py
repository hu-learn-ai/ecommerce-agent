"""
统计显著性检验 + 评估回归追踪

此前评估报告都是单次快照, 缺:
1. 统计显著性 — 策略 A vs B 的 0.02 差异是真改进还是噪声?
2. 回归追踪 — 指标随版本的变化趋势, 上次 0.85 这次 0.79 是回归还是波动?

本模块提供:
1. significance_test() — 配对/独立样本 t 检验
2. bootstrap_ci() — Bootstrap 置信区间
3. RegressionTracker — 追踪指标历史, 检测回归

运行方式:
    # 显著性检验示例 (对比两个策略的 per-case 分数)
    python tests/stats_regression.py --demo-significance

    # 记录一次评估结果
    python tests/stats_regression.py --track --name "rag_v1.2" --report tests/rag_evaluation_report.json

    # 查看回归历史
    python tests/stats_regression.py --history
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime
from typing import List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 回归追踪存储路径
REGRESSION_DB = os.path.join(PROJECT_ROOT, "tests", "regression_history.json")


# ------------------------------------------------------------------ #
#  统计显著性检验
# ---------------------------------------------------------------- #


def paired_t_test(
    a: List[float],
    b: List[float],
) -> dict:
    """
    配对样本 t 检验 (paired t-test)

    适用: 同一组测试用例在两个策略下的 per-case 分数对比
    - H0: 两个策略的均值无差异 (μ_a = μ_b)
    - H1: 两个策略的均值有差异 (μ_a ≠ μ_b)

    Args:
        a: 策略 A 的 per-case 分数
        b: 策略 B 的 per-case 分数 (长度需与 a 相同)

    Returns:
        {
            "mean_a", "mean_b", "mean_diff",
            "t_statistic", "p_value", "df",
            "significant": bool,  # p < 0.05
            "effect_size": "cohen's d",
            "interpretation": 文字说明,
        }
    """
    assert len(a) == len(b), f"配对样本长度不一致: a={len(a)}, b={len(b)}"
    n = len(a)
    if n < 2:
        return {"error": "样本数 < 2, 无法做 t 检验"}

    diffs = [ai - bi for ai, bi in zip(a, b)]
    mean_diff = statistics.mean(diffs)
    std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0

    if std_diff == 0:
        # 所有差值相同, 无方差
        return {
            "mean_a": round(statistics.mean(a), 4),
            "mean_b": round(statistics.mean(b), 4),
            "mean_diff": round(mean_diff, 4),
            "t_statistic": float("inf") if mean_diff != 0 else 0,
            "p_value": 0.0 if mean_diff != 0 else 1.0,
            "df": n - 1,
            "significant": mean_diff != 0,
            "effect_size": float("inf") if mean_diff != 0 else 0,
            "interpretation": "所有差值相同, 差异确定" if mean_diff != 0 else "无差异",
        }

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1

    # 精确 p-value (使用 scipy.stats.t.sf, 双尾)
    # 此前用正态近似, 在小样本(df<30)和临界值(p≈0.05)附近有偏差
    p_value = _exact_p_value_t(t_stat, df)

    # Cohen's d 效应量
    pooled_std = math.sqrt((statistics.variance(a) + statistics.variance(b)) / 2) if n > 1 else 1
    cohen_d = mean_diff / pooled_std if pooled_std > 0 else 0

    # p-value 显示: 极小值用科学计数法, 避免显示 0.0 造成"精确零"误解
    if p_value < 0.0001:
        p_display = f"{p_value:.2e}"
    else:
        p_display = f"{p_value:.4f}"

    if p_value < 0.05:
        interpretation = f"差异显著 (p={p_display}<0.05), 策略{'A' if mean_diff > 0 else 'B'}更优"
    else:
        interpretation = f"差异不显著 (p={p_display}≥0.05), 可能是噪声"

    return {
        "n": n,
        "mean_a": round(statistics.mean(a), 4),
        "mean_b": round(statistics.mean(b), 4),
        "mean_diff": round(mean_diff, 4),
        "std_diff": round(std_diff, 4),
        "t_statistic": round(t_stat, 4),
        "p_value": p_value,  # 保留原始精度, 不 round
        "p_display": p_display,  # 显示用
        "df": df,
        "significant": p_value < 0.05,
        "effect_size_cohen_d": round(cohen_d, 4),
        "interpretation": interpretation,
    }


def _exact_p_value_t(t: float, df: int) -> float:
    """
    精确计算双尾 p-value (使用 scipy.stats.t)

    Args:
        t: t 统计量
        df: 自由度

    Returns:
        双尾 p-value: P(|T| > |t|)
    """
    try:
        from scipy import stats as sp_stats
        # t.sf(|t|, df) = P(T > |t|), 双尾需 ×2
        return float(2 * sp_stats.t.sf(abs(t), df))
    except ImportError:
        # scipy 不可用时降级为近似
        return _approx_p_value_t(t, df)


def _approx_p_value_t(t: float, df: int) -> float:
    """
    近似计算双尾 p-value (无 scipy 依赖)

    df≥30 时用正态近似; df<30 用简单查表
    """
    abs_t = abs(t)
    if df >= 30:
        # 正态近似: P(|Z|>t) = 2 * (1 - Φ(t))
        # Φ(t) 近似公式
        z = abs_t
        phi = 1 - 0.5 * math.erfc(z / math.sqrt(2))
        p = 2 * (1 - phi)
    else:
        # 粗略近似: df 越小 t 分布越厚尾
        # 用经验公式: p ≈ 2 * (1 - Φ(t * sqrt(df/(df+2))))
        z = abs_t * math.sqrt(df / (df + 2))
        phi = 1 - 0.5 * math.erfc(z / math.sqrt(2))
        p = 2 * (1 - phi)
    return min(max(p, 0.0), 1.0)


def bootstrap_ci(
    data: List[float],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> dict:
    """
    Bootstrap 置信区间

    通过重采样估计均值的置信区间, 不假设分布

    Args:
        data: 样本数据
        confidence: 置信水平 (0.95 = 95% CI)
        n_bootstrap: 重采样次数

    Returns:
        {"mean", "ci_low", "ci_high", "n", "confidence"}
    """
    import random

    if not data:
        return {"mean": 0, "ci_low": 0, "ci_high": 0}

    random.seed(42)
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = [random.choice(data) for _ in range(n)]
        boot_means.append(statistics.mean(sample))

    boot_means.sort()
    alpha = 1 - confidence
    low_idx = int((alpha / 2) * n_bootstrap)
    high_idx = int((1 - alpha / 2) * n_bootstrap)

    return {
        "mean": round(statistics.mean(data), 4),
        "ci_low": round(boot_means[low_idx], 4),
        "ci_high": round(boot_means[high_idx], 4),
        "n": n,
        "confidence": confidence,
    }


# ------------------------------------------------------------------ #
#  回归追踪
# ---------------------------------------------------------------- #


class RegressionTracker:
    """
    评估指标回归追踪

    每次评估后调用 track() 记录, 后续可查看历史趋势和检测回归
    """

    def __init__(self, db_path: str = REGRESSION_DB):
        self.db_path = db_path
        self._records = self._load()

    def _load(self) -> list:
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, ensure_ascii=False, indent=2)

    def track(self, name: str, report_path: str, metadata: dict = None) -> dict:
        """
        记录一次评估结果

        Args:
            name: 版本名 (如 "rag_v1.2", "router_baseline")
            report_path: 评估报告 JSON 路径
            metadata: 额外元信息 (如 git_commit, model_version)

        Returns:
            本次记录 + 与上次的回归对比
        """
        if not os.path.exists(report_path):
            return {"error": f"报告文件不存在: {report_path}"}

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # 提取关键指标 (扁平化, 只保留数值型)
        metrics = self._extract_metrics(report)

        record = {
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "report_path": report_path,
            "metrics": metrics,
            "metadata": metadata or {},
        }
        self._records.append(record)
        self._save()

        # 与上一次同名记录对比
        prev = None
        for r in reversed(self._records[:-1]):
            if r["name"] == name:
                prev = r
                break

        regression = None
        if prev:
            regression = self._compare(prev["metrics"], metrics, prev["timestamp"])

        return {
            "tracked": record,
            "previous": prev["timestamp"] if prev else None,
            "regression": regression,
        }

    @staticmethod
    def _extract_metrics(report: dict, prefix: str = "") -> dict:
        """递归提取报告中的数值型指标 (扁平化)"""
        metrics = {}
        for k, v in report.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics[key] = v
            elif isinstance(v, dict):
                metrics.update(RegressionTracker._extract_metrics(v, key))
        return metrics

    @staticmethod
    def _compare(prev: dict, curr: dict, prev_time: str) -> dict:
        """对比两次指标, 检测回归"""
        regressions = []
        improvements = []
        unchanged = []

        # 回归阈值: 变化超过 5% 视为回归/改进
        THRESHOLD = 0.05

        for key in prev:
            if key not in curr:
                continue
            old, new = prev[key], curr[key]
            if old == 0:
                if new == 0:
                    unchanged.append(key)
                elif new > 0:
                    improvements.append({"key": key, "old": old, "new": new, "change": "new"})
                continue

            change_pct = (new - old) / abs(old)

            # 指标方向: 有些指标越高越好(accuracy/F1), 有些越低越好(latency/error_rate)
            higher_better = not any(
                kw in key.lower()
                for kw in ["latency", "error", "cost", "fail", "miss", "fn", "fp", "forget"]
            )

            if abs(change_pct) < THRESHOLD:
                unchanged.append(key)
            elif (change_pct > 0) == higher_better:
                improvements.append({
                    "key": key, "old": round(old, 4), "new": round(new, 4),
                    "change_pct": round(change_pct * 100, 2),
                })
            else:
                regressions.append({
                    "key": key, "old": round(old, 4), "new": round(new, 4),
                    "change_pct": round(change_pct * 100, 2),
                })

        return {
            "previous_time": prev_time,
            "regressions": regressions,
            "improvements": improvements,
            "unchanged_count": len(unchanged),
            "total_compared": len(prev),
            "regression_rate": round(len(regressions) / max(len(prev), 1), 4),
        }

    def history(self, name: Optional[str] = None) -> dict:
        """查看历史记录"""
        records = self._records
        if name:
            records = [r for r in records if r["name"] == name]

        if not records:
            return {"total": 0, "records": []}

        # 按 name 分组, 显示每个指标的时序变化
        by_name = {}
        for r in records:
            n = r["name"]
            if n not in by_name:
                by_name[n] = []
            by_name[n].append({
                "timestamp": r["timestamp"],
                "metrics": r["metrics"],
            })

        # 找出有多个记录的, 显示趋势
        trends = {}
        for n, entries in by_name.items():
            if len(entries) < 2:
                continue
            # 取第一个和最后一个的差值
            first_metrics = entries[0]["metrics"]
            last_metrics = entries[-1]["metrics"]
            trend = {}
            for k in first_metrics:
                if k in last_metrics:
                    trend[k] = {
                        "first": first_metrics[k],
                        "last": last_metrics[k],
                        "delta": round(last_metrics[k] - first_metrics[k], 4),
                    }
            trends[n] = {
                "records": len(entries),
                "first_time": entries[0]["timestamp"],
                "last_time": entries[-1]["timestamp"],
                "metric_trends": trend,
            }

        return {
            "total": len(records),
            "by_name": {n: len(rs) for n, rs in by_name.items()},
            "trends": trends,
        }


# ------------------------------------------------------------------ #
#  主入口
# ---------------------------------------------------------------- #


def main():
    import argparse

    parser = argparse.ArgumentParser(description="统计显著性 + 回归追踪")
    parser.add_argument("--demo-significance", action="store_true",
                        help="演示配对 t 检验")
    parser.add_argument("--demo-bootstrap", action="store_true",
                        help="演示 Bootstrap 置信区间")
    parser.add_argument("--track", action="store_true",
                        help="记录一次评估结果")
    parser.add_argument("--name", type=str, help="版本名")
    parser.add_argument("--report", type=str, help="评估报告 JSON 路径")
    parser.add_argument("--history", action="store_true",
                        help="查看回归历史")
    args = parser.parse_args()

    if args.demo_significance:
        print("=" * 60)
        print("  配对 t 检验演示 (策略 A vs B, 同 10 个用例)")
        print("=" * 60)

        # 模拟数据: 策略 A 平均 0.82, 策略 B 平均 0.79, 看是否显著
        a = [0.85, 0.80, 0.88, 0.82, 0.79, 0.84, 0.86, 0.81, 0.83, 0.85]
        b = [0.82, 0.77, 0.84, 0.80, 0.76, 0.81, 0.83, 0.78, 0.80, 0.82]
        result = paired_t_test(a, b)
        print(f"\n  策略 A 均值: {result['mean_a']}")
        print(f"  策略 B 均值: {result['mean_b']}")
        print(f"  均值差: {result['mean_diff']}")
        print(f"  t = {result['t_statistic']}, df = {result['df']}")
        print(f"  p-value = {result['p_display']}")
        print(f"  Cohen's d = {result['effect_size_cohen_d']}")
        print(f"  显著性: {'是' if result['significant'] else '否'}")
        print(f"  解读: {result['interpretation']}")

        print("\n  --- 对比: 差异很小 (噪声) ---")
        a2 = [0.82, 0.81, 0.83, 0.82, 0.80, 0.82, 0.83, 0.81, 0.82, 0.83]
        b2 = [0.81, 0.82, 0.82, 0.81, 0.81, 0.82, 0.82, 0.82, 0.81, 0.82]
        result2 = paired_t_test(a2, b2)
        print(f"  A 均值={result2['mean_a']}, B 均值={result2['mean_b']}, "
              f"p={result2['p_display']}, 显著: {'是' if result2['significant'] else '否'}")
        print(f"  解读: {result2['interpretation']}")

    if args.demo_bootstrap:
        print("=" * 60)
        print("  Bootstrap 置信区间演示")
        print("=" * 60)
        data = [0.85, 0.80, 0.88, 0.82, 0.79, 0.84, 0.86, 0.81, 0.83, 0.85]
        ci = bootstrap_ci(data, confidence=0.95)
        print(f"\n  样本: {data}")
        print(f"  均值: {ci['mean']}")
        print(f"  95% CI: [{ci['ci_low']}, {ci['ci_high']}]")
        print(f"  n={ci['n']}")
        print(f"  解读: 真实均值有 95% 概率落在 [{ci['ci_low']}, {ci['ci_high']}] 内")

    if args.track:
        if not args.name or not args.report:
            print("  用法: --track --name <版本名> --report <报告路径>")
            return
        tracker = RegressionTracker()
        result = tracker.track(args.name, args.report)
        print("=" * 60)
        print(f"  记录完成: {args.name}")
        print("=" * 60)
        print(f"\n  时间: {result['tracked']['timestamp']}")
        print(f"  指标数: {len(result['tracked']['metrics'])}")
        if result.get("previous"):
            print(f"  上次记录: {result['previous']}")
        if result.get("regression"):
            r = result["regression"]
            print(f"\n  回归分析:")
            print(f"    回归指标数: {len(r['regressions'])}")
            print(f"    改进指标数: {len(r['improvements'])}")
            print(f"    未变指标数: {r['unchanged_count']}")
            print(f"    回归率: {r['regression_rate']:.4f}")
            if r["regressions"]:
                print(f"\n    回归详情:")
                for reg in r["regressions"][:10]:
                    print(f"      {reg['key']}: {reg['old']} → {reg['new']} "
                          f"({reg['change_pct']}%)")
            if r["improvements"]:
                print(f"\n    改进详情:")
                for imp in r["improvements"][:10]:
                    print(f"      {imp['key']}: {imp['old']} → {imp['new']} "
                          f"({imp['change_pct']}%)")

    if args.history:
        tracker = RegressionTracker()
        hist = tracker.history()
        print("=" * 60)
        print("  回归历史")
        print("=" * 60)
        print(f"\n  总记录数: {hist['total']}")
        for name, count in hist.get("by_name", {}).items():
            print(f"    {name}: {count} 次")
        if hist.get("trends"):
            print(f"\n  趋势分析:")
            for name, t in hist["trends"].items():
                print(f"\n  [{name}] ({t['records']} 次记录)")
                print(f"    {t['first_time']} → {t['last_time']}")
                # 显示变化最大的 5 个指标
                sorted_trends = sorted(
                    t["metric_trends"].items(),
                    key=lambda x: abs(x[1]["delta"]),
                    reverse=True,
                )[:5]
                for k, v in sorted_trends:
                    arrow = "↑" if v["delta"] > 0 else "↓"
                    print(f"      {k}: {v['first']} → {v['last']} ({arrow}{abs(v['delta'])})")


if __name__ == "__main__":
    main()
