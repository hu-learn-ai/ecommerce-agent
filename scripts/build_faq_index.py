"""
构建 FAQ 知识库的 FAISS 索引

将客服 FAQ 文本转为向量存储，供客服 Agent 的 RAG 检索使用

数据来源:
    data/processed/faq_data.json  (由 import_taobao_data.py 生成)
    若文件不存在则使用内置 FAQ 兜底

使用方法:
    python scripts/build_faq_index.py

输出:
    data/faiss_index/faq/index.faiss — LangChain FAISS 索引
    data/faiss_index/faq/index.pkl   — 索引元数据
"""

import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.settings import settings


def load_faq_data():
    """
    加载 FAQ 数据

    优先从 data/processed/faq_data.json 读取 (由 import_taobao_data.py 生成)
    文件不存在时使用内置兜底数据
    """
    faq_json_path = os.path.join(PROJECT_ROOT, "data", "processed", "faq_data.json")

    if os.path.exists(faq_json_path):
        print(f"[FAQ Builder] 从文件加载: {faq_json_path}")
        with open(faq_json_path, "r", encoding="utf-8") as f:
            faq_list = json.load(f)

        # 将 JSON 问答对拼接为 FAQ 文本
        faqs = []
        for item in faq_list:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            cat = item.get("category", "").strip()
            if q and a:
                text = f"【{cat}】问: {q}\n答: {a}" if cat else f"问: {q}\n答: {a}"
                faqs.append(text)
        print(f"[FAQ Builder] 从 JSON 加载 {len(faqs)} 条 FAQ")
        return faqs

    # 兜底：内置 FAQ
    print("[FAQ Builder] faq_data.json 未找到，使用内置兜底 FAQ")
    return [
        "退换货政策：支持7天无理由退换货，需保证商品完好不影响二次销售。"
        "签收后7天内可在订单页申请退换货。",
        "退款时效：审核通过后，退款将在3-7个工作日内原路返回支付账户。",
        "运费规则：非质量问题退换货运费由买家承担，质量问题由卖家承担。满99元包邮。",
        "发货时效：下单后48小时内发货（节假日顺延）。预售商品以页面标注时间为准。",
        "发票开具：支持电子发票，在下单时可选择开具。",
        "价格保护：签收后7天内若商品降价，可在APP申请价保退款。",
        "客服联系方式：在线客服 9:00-21:00 全年无休。电话客服 9:00-18:00（工作日）。",
        "如何修改订单：未发货时可在订单详情页修改收货地址和商品数量。",
        "取消订单：未发货的订单可在订单页直接取消，全额退款。",
        "支付方式：支持微信、支付宝、银行卡、花呗、白条等多种支付方式。",
        "会员权益：注册会员享专属折扣、生日礼、积分翻倍等特权。",
    ]


def build_faq_knowledge_base():
    """
    构建 FAQ 知识库向量索引
    """
    # 设置 HuggingFace 镜像
    os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    # 加载 FAQ 数据
    faqs = load_faq_data()

    print(f"[FAQ Builder] 共 {len(faqs)} 条 FAQ")

    if not faqs:
        print("[FAQ Builder] 无 FAQ 数据，跳过索引构建")
        return

    # 加载嵌入模型
    print("[FAQ Builder] 加载嵌入模型...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS as LCFAISS

    embedding_model = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 分批构建向量库（避免 OOM）
    print("[FAQ Builder] 构建向量索引...")
    batch_size = 500
    vector_store = None

    for i in range(0, len(faqs), batch_size):
        batch = faqs[i : i + batch_size]
        if vector_store is None:
            vector_store = LCFAISS.from_texts(batch, embedding=embedding_model)
        else:
            vector_store.add_texts(batch)
        print(f"  进度: {min(i + batch_size, len(faqs))}/{len(faqs)}")

    # 保存
    faq_dir = os.path.join(settings.faiss_index_path, "faq")
    os.makedirs(faq_dir, exist_ok=True)
    vector_store.save_local(faq_dir)

    print("\n[FAQ Builder] FAQ 知识库构建完成!")
    print(f"  索引目录: {faq_dir}")
    print(f"  FAQ 数量: {len(faqs)}")


if __name__ == "__main__":
    build_faq_knowledge_base()
