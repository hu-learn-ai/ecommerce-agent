"""
基于 LLM as Judge 模式的自评估系统

使用 LLM 对 Agent 回答质量进行多维度评估：
- relevance: 相关性
- accuracy: 准确性
- completeness: 完整性
- tone: 语气
- helpfulness: 实用性

学习参考: ecommerce-agentic-rag 的 evaluate.py

使用方法:
    python tests/evaluate.py
"""

import json
import os
import sys

# 必须在 import sentence-transformers/huggingface 相关模块前设置
# 否则 SentenceTransformer 加载时仍请求 huggingface.co 超时
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_openai import ChatOpenAI

from config.settings import settings
from orchestration.observability import obs

# 复用 rag_evaluation 的健壮 JSON 解析(支持代码块包裹/夹杂文字/尾随逗号/单引号)
from tests.rag_evaluation import GenerationMetrics, _extract_token_usage


class AgentEvaluator:
    """使用 LLM 对 Agent 回答质量进行评估"""

    EVAL_CRITERIA = {
        "relevance": "回答是否与用户问题相关，是否切中要害",
        "accuracy": "回答中的事实信息是否准确，有无错误或幻觉",
        "completeness": "回答是否完整回应了用户的所有关切",
        "tone": "语气是否专业、友好，是否符合电商客服场景",
        "helpfulness": "回答对用户是否有实际帮助，是否可操作",
    }

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
        # 借用 GenerationMetrics 的静态解析方法,保证与 RAG 评估同款健壮度
        self._parse_json = GenerationMetrics._parse_json_response

    def evaluate(self, question: str, answer: str, context: dict = None) -> dict:
        """
        评估单条回答（批量模式：5 个维度合并为 1 次 LLM 调用）

        Args:
            question: 用户问题
            answer: Agent 回答
            context: 额外上下文（如调用的 Agent 名称）

        Returns:
            {"question": ..., "answer": ..., "scores": {...}, "overall_score": ...}
        """
        context_str = json.dumps(context, ensure_ascii=False) if context else "无"

        # 构建批量评估 prompt — 一次调用评估所有 5 个维度
        criteria_desc = "\n".join(
            f"  - {criterion}: {desc}" for criterion, desc in self.EVAL_CRITERIA.items()
        )

        prompt = f"""作为电商智能体评估专家，请对以下回答进行多维度评分。

评估标准（每个维度给出 0-10 的分数，整数或一位小数，并简要说明理由）：
{criteria_desc}

用户问题: {question}
Agent回答: {answer}
额外上下文: {context_str}

请只回复 JSON 格式，不要其他文字：
{{"relevance": {{"score": 分数, "reason": "理由"}}, "accuracy": {{"score": 分数, "reason": "理由"}}, "completeness": {{"score": 分数, "reason": "理由"}}, "tone": {{"score": 分数, "reason": "理由"}}, "helpfulness": {{"score": 分数, "reason": "理由"}}}}"""

        scores = {}
        parse_error = None
        try:
            msg = self.llm.invoke(prompt)
            # 记录 LLM Judge 的 token 用量到全局 obs
            try:
                model_name = getattr(self.llm, "model_name", "") or getattr(self.llm, "model", "") or "unknown"
                usage = _extract_token_usage(msg, model_name)
                if usage["prompt_tokens"] or usage["completion_tokens"]:
                    obs.record_tokens(
                        model=usage["model"],
                        prompt_tokens=usage["prompt_tokens"],
                        completion_tokens=usage["completion_tokens"],
                    )
            except Exception:
                pass
            result = msg.content.strip()
            try:
                parsed = self._parse_json(result)
            except ValueError as e:
                # 健壮解析仍失败, 显式标记 error, 不再用"提取第一个数字套到所有维度"的虚假降级
                parse_error = f"解析失败: {e}"
                parsed = {}

            # 确保所有评估维度都存在
            for criterion in self.EVAL_CRITERIA:
                if criterion in parsed and isinstance(parsed[criterion], dict):
                    scores[criterion] = parsed[criterion]
                else:
                    scores[criterion] = {
                        "score": 0,
                        "reason": parse_error or "维度缺失",
                    }

        except Exception as e:
            parse_error = f"评估异常: {type(e).__name__}: {e}"
            for criterion in self.EVAL_CRITERIA:
                scores[criterion] = {"score": 0, "reason": parse_error}

        # 计算总分（5个维度各0-10，总分 = 平均分）
        total = 0
        valid_count = 0
        for criterion in self.EVAL_CRITERIA:
            s = scores.get(criterion, {}).get("score", 0)
            if isinstance(s, (int, float)):
                total += float(s)
                valid_count += 1

        overall = total / valid_count if valid_count > 0 else 0

        result = {
            "question": question,
            "answer": answer,
            "scores": scores,
            "overall_score": round(overall, 2),
        }
        if parse_error:
            # 暴露失败原因,便于排查(此前被静默吞掉)
            result["error"] = parse_error
        return result

    def run_benchmark(self, test_cases: list) -> dict:
        """
        运行测试集

        Args:
            test_cases: [{"question": "...", "answer": "...", "context": {...}}]

        Returns:
            {"total": N, "average_score": ..., "details": [...]}
        """
        results = []
        for case in test_cases:
            result = self.evaluate(
                case["question"],
                case["answer"],
                case.get("context"),
            )
            results.append(result)
            print(f"  [{result['overall_score']:.1f}/10] {case['question'][:30]}...")

        avg_score = sum(r["overall_score"] for r in results) / len(results) if results else 0

        return {
            "total": len(results),
            "average_score": round(avg_score, 2),
            "details": results,
        }


# ------------------------------------------------------------------ #
#  测试用例集
# ------------------------------------------------------------------ #

DEFAULT_TEST_CASES = [
    {
        "question": "有没有500元以下的蓝牙耳机？",
        "answer": "🔍 共找到 5 个相关商品：\n1. 小米蓝牙耳机 Air 2 SE | 价格: ¥199\n2. 漫步者 GM4 | 价格: ¥159\n3. QCY T13 | 价格: ¥89\n4. 倍思 Bowie M1 | 价格: ¥129\n5. JBL Tune 215TWS | 价格: ¥299",
        "context": {"agent": "catalog_agent/search_products", "intent": "search"},
    },
    {
        "question": "怎么退货？",
        "answer": "您可以在签收后7天内申请退换货。操作步骤：\n1. 打开APP → 我的订单\n2. 找到对应订单 → 点击「申请退换货」\n3. 选择退货原因并提交\n4. 客服审核通过后，按提示寄回商品\n\n注意：商品需保持全新状态，包含所有配件和原包装。退款将在审核通过后3-7个工作日内原路返回。",
        "context": {"agent": "cs_agent", "intent": "customer_service"},
    },
    {
        "question": "纸尿裤属于什么类别？",
        "answer": "分类结果: 母婴用品 (置信度: 98.76%)",
        "context": {"agent": "catalog_agent/classify_product", "intent": "classify"},
    },
    {
        "question": "你好",
        "answer": "你好！我是电商智能助手 🛒。你可以问我商品搜索、推荐、订单查询、售后政策等问题，随时为你服务！",
        "context": {"agent": "llm_direct", "intent": "chitchat"},
    },
    {
        "question": "最近什么品类卖得最好？",
        "answer": "📊 商品销量 TOP 5\n1. 手机数码 | 销量: 12,580 | 销售额: ¥3,560,000\n2. 服饰内衣 | 销量: 8,920 | 销售额: ¥890,000\n3. 食品生鲜 | 销量: 7,650 | 销售额: ¥420,000\n4. 美妆护肤 | 销量: 6,200 | 销售额: ¥1,240,000\n5. 家用电器 | 销量: 3,100 | 销售额: ¥2,100,000",
        "context": {"agent": "business_data_agent/data_analysis", "intent": "analytics"},
    },
]


def main():
    """运行默认测试集"""
    print("=" * 60)
    print("  电商智能体自评估系统 (LLM as Judge)")
    print("=" * 60)
    print("  ⚠️ 注意: 本脚本评估的是手写静态答案, 不代表真实链路质量;")
    print("         真实链路基线请运行 tests/e2e_eval.py")

    llm = ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.1,
        max_tokens=1024,
    )

    evaluator = AgentEvaluator(llm)

    print(f"\n共 {len(DEFAULT_TEST_CASES)} 个测试用例\n")

    result = evaluator.run_benchmark(DEFAULT_TEST_CASES)

    print(f"\n{'=' * 60}")
    print(f"  总测试数: {result['total']}")
    print(f"  平均得分: {result['average_score']}/10")
    print(f"{'=' * 60}\n")

    # 保存结果
    output_path = os.path.join(PROJECT_ROOT, "tests", "evaluation_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"评估报告已保存: {output_path}")

    return result


if __name__ == "__main__":
    main()
