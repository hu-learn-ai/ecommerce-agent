"""
记忆系统评估 — 摘要质量 + 长期检索 + 画像迁移 + 遗忘曲线

此前全项目对三层记忆系统(短期摘要/长期 FAISS/跨会话共享)零评估。
本评估补齐:

1. 短期记忆摘要质量
   - 关键事实保留率 (Key Fact Retention)
   - 摘要压缩比 (Compression Ratio)
   - 摘要生成延迟
2. 长期记忆检索质量
   - 写入 → 检索的 Precision/Recall
   - 重要性评估准确性 (重要记忆的 ranking)
3. 跨会话用户画像迁移
   - shared=True 记忆是否被新会话检索到
   - 迁移准确率 (相关记忆命中, 不相关不被迁移)
4. 遗忘曲线 + TTL 过期
   - 遗忘评分随时间的衰减趋势
   - TTL 过期记忆是否被正确清理
   - 高重要性记忆的保护机制

运行方式:
    python tests/memory_eval.py

    # 不调用 LLM (仅测试遗忘曲线/TTL 等纯算法部分)
    python tests/memory_eval.py --no-llm
"""

import json
import os
import shutil
import sys
import tempfile
import time
from typing import List

# Windows GBK 控制台下打印 emoji 会抛 UnicodeEncodeError（评估跑完才崩、报告不落盘）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001 - 流不支持 reconfigure 时忽略
        pass

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from config.settings import settings

# ------------------------------------------------------------------ #
#  测试用例集
# ---------------------------------------------------------------- #

# 摘要测试用例: 模拟一段对话, 标注必须保留的关键事实
SUMMARY_TEST_CASES = [
    {
        "name": "偏好+预算对话",
        "messages": [
            ("user", "我想买一台笔记本电脑"),
            ("assistant", "请问您有什么具体需求?比如预算、用途"),
            ("user", "预算8000左右,主要用来写代码"),
            ("assistant", "写代码建议16GB内存以上,SSD 512GB起"),
            ("user", "最好是14寸的,方便携带"),
        ],
        "must_keep": ["8000", "14", "16", "代码"],  # 关键事实
    },
    {
        "name": "购买决策对话",
        "messages": [
            ("user", "有没有降噪耳机推荐"),
            ("assistant", "降噪耳机推荐 Sony WH-1000XM5 或 Bose QC Ultra"),
            ("user", "Sony那款多少钱"),
            ("assistant", "Sony WH-1000XM5 大概 2299 元"),
            ("user", "太贵了,有没有2000以内的"),
            ("assistant", "可以看看 Sony WH-1000XM4, 约 1599 元"),
        ],
        "must_keep": ["2000", "1599", "Sony", "降噪"],
    },
]

# 长期记忆写入/检索测试用例: 模拟对话 → 应被提取的记忆 → 查询时能否检索到
LONG_TERM_CASES = [
    {
        "conversation": ("我是 Java 开发者,平时用 Spring Boot", "好的,已记录您是 Java/Spring Boot 开发者"),
        "expected_extract": "Java",  # 期望被提取的关键信息
        "category": "preference",
        "probe_query": "我平时用什么编程语言",  # 后续检索查询
        "should_retrieve": True,
    },
    {
        "conversation": ("我对价格敏感,预算一般控制在500以内", "已记录您的预算偏好"),
        "expected_extract": "500",
        "category": "preference",
        "probe_query": "我的预算是多少",
        "should_retrieve": True,
    },
    {
        "conversation": ("上次我买了台华为MateBook", "好的,华为MateBook是不错的选择"),
        "expected_extract": "华为",
        "category": "fact",
        "probe_query": "我之前买过什么电脑",
        "should_retrieve": True,
    },
    {
        "conversation": ("今天天气不错", "是的,天气很好"),
        "expected_extract": None,  # 闲聊不应被提取
        "category": None,
        "probe_query": "天气",
        "should_retrieve": False,
    },
]


# ------------------------------------------------------------------ #
#  摘要质量评估
# ---------------------------------------------------------------- #


class SummaryMetrics:
    """短期记忆摘要质量评估"""

    @staticmethod
    def compute(
        summary: str,
        original_messages: List[tuple],
        must_keep: List[str],
    ) -> dict:
        """
        计算摘要质量指标

        Args:
            summary: 生成的摘要文本
            original_messages: 原始对话 [(role, content), ...]
            must_keep: 必须保留的关键事实列表

        Returns:
            {
                "key_fact_retention": 关键事实保留率 (0-1),
                "missed_facts": 漏掉的关键事实,
                "compression_ratio": 压缩比 (原始长度/摘要长度),
                "summary_length": 摘要字符数,
                "original_length": 原始字符数,
            }
        """
        original_text = "".join(content for _, content in original_messages)

        # 关键事实保留率
        hits = sum(1 for fact in must_keep if fact in summary)
        retention = hits / len(must_keep) if must_keep else 1.0
        missed = [f for f in must_keep if f not in summary]

        # 压缩比
        orig_len = len(original_text)
        sum_len = len(summary)
        ratio = orig_len / sum_len if sum_len > 0 else 0

        return {
            "key_fact_retention": round(retention, 4),
            "missed_facts": missed,
            "compression_ratio": round(ratio, 2),
            "summary_length": sum_len,
            "original_length": orig_len,
        }


# ------------------------------------------------------------------ #
#  记忆系统评估器
# ---------------------------------------------------------------- #


class MemorySystemEvaluator:
    """记忆系统综合评估器"""

    def __init__(self, llm=None, embedder=None):
        """
        Args:
            llm: 用于摘要/长期记忆提取的 LLM (None 时跳过相关测试)
            embedder: SentenceTransformer 实例
        """
        self.llm = llm
        self.embedder = embedder
        # 使用临时目录避免污染生产数据
        self._temp_dir = tempfile.mkdtemp(prefix="memory_eval_")

    def __del__(self):
        # 清理临时目录
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass

    def evaluate_summary_quality(self) -> dict:
        """评估短期记忆摘要质量"""
        if not self.llm:
            return {"available": False, "note": "无 LLM, 跳过摘要质量评估"}

        from orchestration.memory import Message, ShortTermMemory

        results = []
        for case in SUMMARY_TEST_CASES:
            # 直接调用 _generate_summary 传入全部消息, 测试 prompt 本身的效果
            # (而非依赖 window_size 触发, 避免关键消息留在窗口里没被摘要)
            messages = [
                Message(role=role, content=content)
                for role, content in case["messages"]
            ]
            summary = ShortTermMemory(llm=self.llm)._generate_summary(messages)

            metrics = SummaryMetrics.compute(
                summary=summary,
                original_messages=case["messages"],
                must_keep=case["must_keep"],
            )
            metrics["name"] = case["name"]
            metrics["summary"] = summary[:300]
            results.append(metrics)

        retentions = [r["key_fact_retention"] for r in results]
        compressions = [r["compression_ratio"] for r in results]

        return {
            "available": True,
            "total": len(results),
            "results": results,
            "avg_key_fact_retention": round(sum(retentions) / len(retentions), 4) if retentions else 0,
            "avg_compression_ratio": round(sum(compressions) / len(compressions), 2) if compressions else 0,
        }

    def evaluate_long_term_retrieval(self) -> dict:
        """评估长期记忆写入 + 检索"""
        if not self.llm:
            return {"available": False, "note": "无 LLM, 跳过长期记忆评估"}

        from orchestration.memory import LongTermMemory

        ltm = LongTermMemory(
            store_path=self._temp_dir,
            embedder=self.embedder,
            llm=self.llm,
        )

        results = []
        for case in LONG_TERM_CASES:
            user_msg, asst_msg = case["conversation"]

            # 写入 (extract_and_store 是后台线程, 这里直接调)
            try:
                ltm.extract_and_store(user_msg, asst_msg, session_id="test_session")
            except Exception as e:
                print(f"  [长期记忆写入失败] {e}")
                continue

            # 等待线程完成 (extract_and_store 内部是同步的)
            time.sleep(0.5)

            # 检索
            try:
                retrieved = ltm.search(case["probe_query"], k=3)
            except Exception:
                retrieved = []

            retrieved_content = " ".join(r.get("content", "") for r in retrieved)
            hit = case["expected_extract"] in retrieved_content if case["expected_extract"] else False

            results.append({
                "conversation": user_msg[:50],
                "expected_extract": case["expected_extract"],
                "category": case["category"],
                "should_retrieve": case["should_retrieve"],
                "retrieved_count": len(retrieved),
                "hit": hit,
                "retrieved_top": retrieved[0]["content"][:80] if retrieved else "",
            })

        # 计算 Precision/Recall
        # should_retrieve=True 且 hit=True → TP
        # should_retrieve=True 且 hit=False → FN (漏召回)
        # should_retrieve=False 且 retrieved=True → FP (误召回)
        tp = sum(1 for r in results if r["should_retrieve"] and r["hit"])
        fn = sum(1 for r in results if r["should_retrieve"] and not r["hit"])
        fp = sum(1 for r in results if not r["should_retrieve"] and r["retrieved_count"] > 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "available": True,
            "total": len(results),
            "results": results,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    def evaluate_cross_session_migration(self) -> dict:
        """评估跨会话共享记忆迁移"""
        if not self.llm or not self.embedder:
            return {"available": False, "note": "无 LLM/embedder, 跳过跨会话迁移评估"}

        from orchestration.memory import MemoryManager

        # 用**独立的**临时目录建 MemoryManager，避免同一轮评估里别的测试（如摘要质量用了
        # test_session）写进去的记忆串进来——否则"跨会话迁移"会命中别的话题的记忆，
        # 指标看起来是命中、实际上是记忆串味（旧实现就出现过：上下文里混进 Java/Spring Boot，
        # 而本用例期望的是 Python/FastAPI，却仍然判 migration_hit=true）。
        original_path = settings.memory_store_path
        isolated_dir = tempfile.mkdtemp(prefix="memory_eval_cross_session_")
        settings.memory_store_path = isolated_dir
        try:
            mm = MemoryManager(llm=self.llm, embedder=self.embedder)
        finally:
            settings.memory_store_path = original_path

        # session A: 用户提到偏好
        session_a = "session_A"
        mm.add_message(session_a, "user", "我是 Python 开发者,喜欢用 FastAPI")
        mm.add_message(session_a, "assistant",
                       "好的,已记录您偏好 Python + FastAPI")
        time.sleep(3)  # 等待后台提取

        # session B: 新会话,询问用户偏好
        session_b = "session_B"
        # build_context 应能从跨会话共享记忆中召回 "Python/FastAPI"
        ctx = mm.build_context(session_b, current_query="我平时用什么编程语言")
        ctx_text = "".join(m.get("content", "") for m in ctx)

        # 命中 = 期望事实被召回；同时必须检查**没有把无关记忆灌进上下文**，
        # 否则"召回了正确信息 + 顺手带一堆别的"也会被算成成功。
        expected_terms = ("Python", "FastAPI")
        leaked_terms = [t for t in ("Java", "Spring Boot") if t in ctx_text]
        migration_hit = any(t in ctx_text for t in expected_terms)
        strict_hit = migration_hit and not leaked_terms
        shutil.rmtree(isolated_dir, ignore_errors=True)

        return {
            "available": True,
            "session_a_said": "Python 开发者, FastAPI",
            "session_b_query": "我平时用什么编程语言",
            "migration_hit": migration_hit,
            "strict_hit": strict_hit,
            "leaked_terms": leaked_terms,
            "context_built": ctx_text[:300],
            "linked_sessions": [
                getattr(m, "linked_sessions", [])
                for m in mm._long_term.memories
                if getattr(m, "shared", False)
            ][:3] if mm._long_term else [],
        }

    def evaluate_forgetting_curve(self) -> dict:
        """评估遗忘曲线 + TTL 过期机制(纯算法,不需要 LLM)"""
        from orchestration.memory import MemoryItem, auto_ttl

        # 测试不同重要性、不同访问频率下的遗忘曲线
        test_scenarios = [
            {"name": "高重要性", "importance": 0.9, "access_count": 5},
            {"name": "中重要性", "importance": 0.5, "access_count": 2},
            {"name": "低重要性", "importance": 0.2, "access_count": 0},
            {"name": "高重要性未访问", "importance": 0.9, "access_count": 0},
        ]

        # 模拟 0/1/7/30/90/365 天后的遗忘评分
        days_to_check = [0, 1, 7, 30, 90, 365]
        curves = []
        for scenario in test_scenarios:
            mem = MemoryItem(
                content=scenario["name"],
                category="preference",
                session_id="test",
                importance=scenario["importance"],
                access_count=scenario["access_count"],
            )
            mem.last_accessed = time.time()  # 重置访问时间

            curve = []
            for days in days_to_check:
                sim_now = time.time() + days * 86400
                forget_score = mem.compute_forget_score(now=sim_now)
                retention = 1.0 - forget_score
                curve.append({
                    "days": days,
                    "forget_score": round(forget_score, 4),
                    "retention": round(retention, 4),
                })
            curves.append({
                "name": scenario["name"],
                "importance": scenario["importance"],
                "access_count": scenario["access_count"],
                "curve": curve,
            })

        # TTL 过期测试
        ttl_cases = [
            {"category": "preference", "importance": 0.5, "expected_ttl": 365},  # preference 基础 365
            {"category": "fact", "importance": 0.5, "expected_ttl": 90},
            {"category": "context", "importance": 0.5, "expected_ttl": 30},
            {"category": "ephemeral", "importance": 0.5, "expected_ttl": 1},
            {"category": "preference", "importance": 0.95, "expected_ttl": 365 * 3},  # 高重要性 ×3
        ]
        ttl_results = []
        for case in ttl_cases:
            actual_ttl = auto_ttl(case["category"], case["importance"])
            ttl_results.append({
                "category": case["category"],
                "importance": case["importance"],
                "expected_ttl": case["expected_ttl"],
                "actual_ttl": actual_ttl,
                "passed": actual_ttl == case["expected_ttl"],
            })

        # 过期清理测试
        now = time.time()
        expired_mem = MemoryItem(
            content="过期记忆",
            category="ephemeral",
            session_id="test",
            ttl_days=1,
        )
        expired_mem.timestamp = now - 2 * 86400  # 2 天前
        not_expired_mem = MemoryItem(
            content="未过期记忆",
            category="preference",
            session_id="test",
            ttl_days=365,  # 必须显式传 ttl_days, 否则默认 None=不过期
        )
        not_expired_mem.timestamp = now - 1 * 86400  # 1 天前

        return {
            "forgetting_curves": curves,
            "ttl_validation": ttl_results,
            "expired_check": {
                "expired_is_expired": expired_mem.is_expired(now),
                "not_expired_is_expired": not not_expired_mem.is_expired(now),
            },
        }

    def evaluate(self) -> dict:
        """运行完整评估"""
        return {
            "summary_quality": self.evaluate_summary_quality(),
            "long_term_retrieval": self.evaluate_long_term_retrieval(),
            "cross_session_migration": self.evaluate_cross_session_migration(),
            "forgetting_curve": self.evaluate_forgetting_curve(),
        }


# ------------------------------------------------------------------ #
#  报告打印
# ---------------------------------------------------------------- #


def print_report(result: dict):
    """打印可读报告"""

    sq = result["summary_quality"]
    if sq.get("available"):
        print("\n  📝 短期记忆摘要质量:")
        print(f"    用例数: {sq['total']}")
        print(f"    平均关键事实保留率: {sq['avg_key_fact_retention']:.4f}")
        print(f"    平均压缩比: {sq['avg_compression_ratio']}")
        for r in sq["results"]:
            print(f"      [{r['name']}] retention={r['key_fact_retention']:.2f} "
                  f"compress={r['compression_ratio']}x "
                  f"missed={r['missed_facts']}")
    else:
        print(f"\n  📝 摘要质量: {sq.get('note')}")

    lt = result["long_term_retrieval"]
    if lt.get("available"):
        print("\n  🧠 长期记忆检索:")
        print(f"    用例数: {lt['total']}")
        print(f"    Precision: {lt['precision']:.4f}")
        print(f"    Recall: {lt['recall']:.4f}")
        print(f"    F1: {lt['f1']:.4f}")
        for r in lt["results"]:
            status = "OK" if r["hit"] == r["should_retrieve"] else "FAIL"
            print(f"      [{status}] conv='{r['conversation']}' "
                  f"extract={r['expected_extract']} "
                  f"hit={r['hit']} retrieved={r['retrieved_count']}")
    else:
        print(f"\n  🧠 长期记忆: {lt.get('note')}")

    cs = result["cross_session_migration"]
    if cs.get("available"):
        print("\n  🔗 跨会话画像迁移:")
        print(f"    迁移命中(期望事实被召回): {cs['migration_hit']}")
        print(f"    严格命中(无无关记忆串入): {cs['strict_hit']}"
              + (f"   ⚠️ 串入: {cs['leaked_terms']}" if cs.get("leaked_terms") else ""))
        print(f"    context: {cs['context_built'][:120]}")
    else:
        print(f"\n  🔗 跨会话迁移: {cs.get('note')}")

    fc = result["forgetting_curve"]
    print("\n  📉 遗忘曲线:")
    for curve in fc["forgetting_curves"]:
        print(f"    {curve['name']} (imp={curve['importance']}, "
              f"access={curve['access_count']}):")
        for pt in curve["curve"]:
            print(f"      day {pt['days']:>3d}: retention={pt['retention']:.4f} "
                  f"(forget={pt['forget_score']:.4f})")

    ttl_pass = sum(1 for t in fc["ttl_validation"] if t["passed"])
    print(f"\n  ⏰ TTL 过期验证 ({ttl_pass}/{len(fc['ttl_validation'])} 通过):")
    for t in fc["ttl_validation"]:
        mark = "PASS" if t["passed"] else "FAIL"
        print(f"    [{mark}] {t['category']:<12s} imp={t['importance']} "
              f"expected={t['expected_ttl']}d actual={t['actual_ttl']}d")

    ec = fc["expired_check"]
    print("\n  🗑️  过期清理:")
    print(f"    过期记忆被识别为过期: {ec['expired_is_expired']}")
    print(f"    未过期记忆被识别为未过期: {ec['not_expired_is_expired']}")


# ------------------------------------------------------------------ #
#  主入口
# ---------------------------------------------------------------- #


def main():
    import argparse

    parser = argparse.ArgumentParser(description="记忆系统评估")
    parser.add_argument("--no-llm", action="store_true",
                        help="跳过 LLM 相关测试(仅测遗忘曲线/TTL)")
    args = parser.parse_args()

    print("=" * 60)
    print("  记忆系统评估 (摘要+长期+跨会话+遗忘)")
    print("=" * 60)

    llm = None
    embedder = None

    if not args.no_llm:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.0,
            max_tokens=512,
        )

        # 加载共享 embedder
        try:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer(settings.embedding_model)
            print(f"  embedder 已加载: {settings.embedding_model}")
        except Exception as e:
            print(f"  embedder 加载失败 (跨会话迁移将被跳过): {e}")

    evaluator = MemorySystemEvaluator(llm=llm, embedder=embedder)
    result = evaluator.evaluate()
    print_report(result)

    output_path = os.path.join(PROJECT_ROOT, "tests", "memory_eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
