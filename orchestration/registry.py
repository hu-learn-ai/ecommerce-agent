"""
Agent 注册中心 — 统一初始化所有组件

职责：
1. 创建统一的 LLM 实例
2. 初始化数据库连接 (Neo4j, MySQL)
3. 实例化并注册所有子 Agent
4. 提供 Agent 实例的统一获取入口

学习参考: Price Pilot 的 agent 初始化模式
"""

import os
from typing import Optional

from langchain_openai import ChatOpenAI

from config.settings import settings


def create_llm() -> ChatOpenAI:
    """
    创建统一的 LLM 实例 (DeepSeek Chat)

    所有 Agent 共享同一个 LLM 实例
    """
    # 设置 HuggingFace 镜像端点
    if settings.hf_endpoint:
        os.environ["HF_ENDPOINT"] = settings.hf_endpoint

    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.request_timeout,  # 30s 超时
        max_retries=settings.max_retries,  # 最多重试 3 次
    )


def create_neo4j_driver():
    """创建 Neo4j 驱动实例"""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    # 验证连接
    try:
        driver.verify_connectivity()
        print("[Registry] Neo4j 连接成功")
    except Exception as e:
        print(f"[Registry] Neo4j 连接失败: {e}")
        print("[Registry] 知识图谱相关功能将不可用")
    return driver


def create_mysql_engine():
    """创建 MySQL 连接池（SQLAlchemy engine）"""
    from sqlalchemy import create_engine

    cfg = settings.get_mysql_config()
    url = f"mysql+pymysql://{cfg['user']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}?charset={cfg.get('charset', 'utf8mb4')}"

    engine = create_engine(
        url,
        pool_size=5,  # 常驻连接数
        max_overflow=10,  # 突发连接数
        pool_pre_ping=True,  # 自动检测断连
        pool_recycle=3600,  # 1小时回收
    )
    print("[Registry] MySQL 连接池已创建")
    return engine


def register_all_agents(
    neo4j_driver=None,
    mysql_config: Optional[dict] = None,
    llm: Optional[ChatOpenAI] = None,
) -> dict:
    """
    注册所有 Agent 并返回字典

    Args:
        neo4j_driver: Neo4j 驱动实例（可选，为 None 时跳过图相关 Agent 的图查询功能）
        mysql_config: MySQL 连接配置（可选，为 None 时使用 settings 默认配置）
        llm: LLM 实例（可选，为 None 时创建新实例）

    Returns:
        {"catalog_agent": CatalogAgent(), "kg_qa_agent": KGQAAgent(), ...}
    """
    from agents.analytics_agent import AnalyticsAgent
    from agents.business_data_agent import BusinessDataAgent
    from agents.catalog_agent import CatalogAgent
    from agents.classify_agent import ClassifyAgent
    from agents.cs_agent import CustomerServiceAgent
    from agents.kg_qa_agent import KGQAAgent
    from agents.ner_agent import NerAgent
    from agents.order_agent import OrderAgent
    from agents.recommend_agent import RecommendAgent
    from agents.search_agent import ProductSearchAgent

    # 创建 LLM 实例
    if llm is None:
        llm = create_llm()

    # MySQL 配置
    if mysql_config is None:
        mysql_config = settings.get_mysql_config()

    # 创建 MySQL 连接池
    mysql_engine = None
    try:
        mysql_engine = create_mysql_engine()
    except Exception as e:
        print(f"[Registry] MySQL 连接池创建失败（降级为每次新建连接）: {e}")

    # 创建共享嵌入模型（避免 SearchAgent 和 CSAgent 各加载一份 400MB 模型）
    shared_embedder = None
    try:
        from sentence_transformers import SentenceTransformer

        shared_embedder = SentenceTransformer(settings.embedding_model)
        # 加载期 int8 动态量化（CPU 下提速约 2-3x、权重降 4 倍）
        from models.persistence import quantize_sentence_transformer

        shared_embedder = quantize_sentence_transformer(shared_embedder)
        print(f"[Registry] 共享嵌入模型已加载: {settings.embedding_model}")
    except Exception as e:
        print(f"[Registry] 嵌入模型加载失败（Agent 将懒加载）: {e}")

    # 分类模型标签文件路径
    import os

    labels_path = os.path.join(os.path.dirname(settings.classify_model_path), "labels.txt")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(settings.classify_model_path, "labels.txt")

    # FAQ 索引路径
    faq_index_path = os.path.join(settings.faiss_index_path, "faq")

    # 实例化底层能力组件（保持原实现不变）
    search_agent = ProductSearchAgent(
        embedding_model_name=settings.embedding_model,
        neo4j_driver=neo4j_driver,
        faiss_index_path=settings.faiss_index_path,
        shared_embedder=shared_embedder,
    )
    # 预热 BM25 关键词索引：构建需 ~10s（7000 条商品名 jieba 分词），
    # 放在启动期完成，避免第一个搜索请求承担冷启动延迟
    search_agent.warmup()
    recommend_agent = RecommendAgent(
        neo4j_driver=neo4j_driver,
        llm=llm,
        db_config=mysql_config,
        search_agent=search_agent,
        engine=mysql_engine,
    )
    classify_agent = ClassifyAgent(
        model_path=settings.classify_model_path,
        labels_path=labels_path,
    )
    order_agent = OrderAgent(
        db_config=mysql_config,
        engine=mysql_engine,
    )
    analytics_agent = AnalyticsAgent(
        db_config=mysql_config,
        llm=llm,
        engine=mysql_engine,
    )

    # 按数据域合并为 4 个 Agent：
    #   商品域 catalog_agent       = search + recommend + classify
    #   图谱域 kg_qa_agent         = 知识图谱问答
    #   业务数据域 business_data_agent = order + analytics
    #   客服域 cs_agent            = FAQ RAG 客服
    catalog_agent = CatalogAgent(
        search_agent=search_agent,
        recommend_agent=recommend_agent,
        classify_agent=classify_agent,
        ner_agent=NerAgent(model_path=settings.ner_model_path),
    )
    business_data_agent = BusinessDataAgent(
        order_agent=order_agent,
        analytics_agent=analytics_agent,
    )

    agents = {
        "catalog_agent": catalog_agent,
        "kg_qa_agent": KGQAAgent(
            neo4j_driver=neo4j_driver,
            llm=llm,
        ),
        "business_data_agent": business_data_agent,
        "cs_agent": CustomerServiceAgent(
            faiss_index_path=faq_index_path,
            embedding_model_name=settings.embedding_model,
            llm=llm,
            shared_embedder=shared_embedder,
        ),
    }
    # 暴露共享嵌入实例，供 API 层复用（避免重复加载 400MB 模型）
    agents["__shared_embedder__"] = shared_embedder

    print(f"[Registry] 已注册 {len(agents)} 个 Agent:")
    for name, agent in agents.items():
        print(f"  - {name}: {agent.__class__.__name__}")

    return agents
