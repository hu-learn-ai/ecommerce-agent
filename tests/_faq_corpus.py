"""FAQ 语料辅助：稳定 ID、文本映射、人工标注 GT 读取。

`data/processed/faq_data.json` 只有 question/answer/category 三个字段、没有 ID，
而 `scripts/build_faq_index.py` 按 `【类目】问: Q\\n答: A` 的格式把问答拼成一段文本入索引。
本模块按同一格式生成与索引一致的文本，并给出稳定的 `FAQ0001...` 形式的 ID，
供 FAQ 检索评估做 pooling、人工标注与指标计算。
"""

import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAQ_JSON = os.path.join(PROJECT_ROOT, "data", "processed", "faq_data.json")
FAQ_LABELED_PATH = os.path.join(PROJECT_ROOT, "tests", "retrieval_gt_labeled_faq.json")


def faq_text(question: str, answer: str, category: str = "") -> str:
    """与 scripts/build_faq_index.py 完全一致的入库文本。"""
    question = (question or "").strip()
    answer = (answer or "").strip()
    category = (category or "").strip()
    if category:
        return f"【{category}】问: {question}\n答: {answer}"
    return f"问: {question}\n答: {answer}"


def load_faq_items() -> list:
    """返回 [{id, question, answer, category, text}, ...]（顺序即索引顺序，ID 稳定）。"""
    if not os.path.exists(FAQ_JSON):
        return []
    with open(FAQ_JSON, encoding="utf-8") as handle:
        raw = json.load(handle)
    items = []
    for index, item in enumerate(raw, 1):
        question = item.get("question", "")
        answer = item.get("answer", "")
        if not question.strip() or not answer.strip():
            continue
        category = item.get("category", "")
        items.append(
            {
                "id": f"FAQ{index:04d}",
                "question": question.strip(),
                "answer": answer.strip(),
                "category": category.strip(),
                "text": faq_text(question, answer, category),
            }
        )
    return items


def id_by_text(items: list) -> dict:
    """文本 → FAQ ID（用于把 LangChain 检索结果映射回稳定 ID）。"""
    mapping = {}
    for item in items:
        mapping.setdefault(item["text"], item["id"])
        # 兜底：仅用问题文本也能匹配（索引里包含答案，但个别条目可能被截断）
        mapping.setdefault(f"问: {item['question']}", item["id"])
    return mapping


def load_labeled_faq_gt() -> dict:
    """读取人工标注的 FAQ GT：{查询: [FAQ ID, ...]}；不存在时返回空 dict。"""
    if not os.path.exists(FAQ_LABELED_PATH):
        return {}
    try:
        with open(FAQ_LABELED_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        print(f"[FAQ] 已加载人工标注 GT: {len(data)} 个查询（{FAQ_LABELED_PATH}）")
        return data
    except Exception as exc:  # noqa: BLE001
        print(f"[FAQ] 人工 GT 读取失败: {exc}")
        return {}
