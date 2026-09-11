"""
统一配置中心
所有连接信息、模型路径、运行参数集中管理
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# 加载 .env 环境变量
load_dotenv()

# 强制 HuggingFace 使用镜像 + 离线模式 (模型已本地缓存)
# 必须在任何 sentence_transformers/transformers import 前生效
# 否则运行时会请求 huggingface.co 超时 (每次重试 10s × 5次)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@dataclass
class Settings:
    """全局配置 — 通过环境变量注入，dataclass 默认值在类定义时求值"""

    # === LLM ===
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )

    # === 自托管客服模型（微调后 vLLM 接入，Tier1 切换）===
    cs_model_base_url: str = field(
        default_factory=lambda: os.getenv("CS_MODEL_BASE_URL", "http://localhost:8000/v1")
    )
    cs_model_name: str = field(default_factory=lambda: os.getenv("CS_MODEL_NAME", "qwen-cs-7b"))
    self_hosted_cs: bool = os.getenv("SELF_HOSTED_CS", "false").lower() == "true"

    # === Neo4j ===
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))

    # === MySQL ===
    mysql_host: str = field(default_factory=lambda: os.getenv("MYSQL_HOST", "localhost"))
    mysql_port: int = field(default_factory=lambda: int(os.getenv("MYSQL_PORT", "3306")))
    mysql_user: str = field(default_factory=lambda: os.getenv("MYSQL_USER", "root"))
    mysql_password: str = field(default_factory=lambda: os.getenv("MYSQL_PASSWORD", ""))
    mysql_db: str = field(default_factory=lambda: os.getenv("MYSQL_DB", "gmall"))

    # === Embedding ===
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-zh-v1.5")
    )
    embedding_dim: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "768")))
    hf_endpoint: str = field(
        default_factory=lambda: os.getenv("HF_ENDPOINT", "https://hf-mirror.com")
    )

    # === Classification ===
    classify_model_path: str = field(
        default_factory=lambda: os.getenv("CLASSIFY_MODEL_PATH", "./models/best")
    )
    # 域外拒识参考统计（由 scripts/build_ood_stats.py 生成）
    ood_stats_path: str = field(
        default_factory=lambda: os.getenv("OOD_STATS_PATH", "./models/best/ood_stats.npz")
    )
    # 加载期 int8 动态量化（仅 CPU 生效，减少内存/延迟；可用 MODEL_QUANTIZE=false 关闭）
    model_quantize: bool = os.getenv("MODEL_QUANTIZE", "true").lower() == "true"

    # === NER ===
    ner_model_path: str = field(
        default_factory=lambda: os.getenv("NER_MODEL_PATH", "./models/ner/best_model")
    )

    # === FAISS ===
    faiss_index_path: str = field(
        default_factory=lambda: os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    )
    search_min_cosine_score: float = field(
        default_factory=lambda: float(os.getenv("SEARCH_MIN_COSINE_SCORE", "0.30"))
    )

    # === API Server ===
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8002")))

    # === Agent 运行参数 ===
    max_retries: int = 3
    request_timeout: int = 30
    lite_timeout: int = field(default_factory=lambda: int(os.getenv("LITE_TIMEOUT", "10")))
    heavy_timeout: int = field(default_factory=lambda: int(os.getenv("HEAVY_TIMEOUT", "60")))
    llm_temperature: float = 0.3
    llm_max_tokens: int = 2048

    # === 模型分级 (成本优化) ===
    # 三层模型分级：lite / standard / heavy，使用不同模型实现真正的成本差异化
    # DeepSeek API 定价: deepseek-chat (V3) 缓存命中 ¥0.5/M, 未命中 ¥2/M
    #                    deepseek-reasoner (R1) 缓存命中 ¥1/M, 未命中 ¥8/M (约 4x)
    lite_model: str = field(default_factory=lambda: os.getenv("LITE_MODEL", "deepseek-chat"))
    lite_temperature: float = float(os.getenv("LITE_TEMPERATURE", "0.1"))
    lite_max_tokens: int = int(os.getenv("LITE_MAX_TOKENS", "512"))

    standard_model: str = field(
        default_factory=lambda: os.getenv("STANDARD_MODEL", "deepseek-chat")
    )
    standard_temperature: float = float(os.getenv("STANDARD_TEMPERATURE", "0.3"))
    standard_max_tokens: int = int(os.getenv("STANDARD_MAX_TOKENS", "2048"))

    heavy_model: str = field(default_factory=lambda: os.getenv("HEAVY_MODEL", "deepseek-reasoner"))
    heavy_temperature: float = float(os.getenv("HEAVY_TEMPERATURE", "0.5"))
    heavy_max_tokens: int = int(os.getenv("HEAVY_MAX_TOKENS", "4096"))

    # === 记忆系统 ===
    memory_window: int = int(os.getenv("MEMORY_WINDOW", "10"))  # 短期记忆保留最近 N 轮
    memory_summary_threshold: int = int(
        os.getenv("MEMORY_SUMMARY_THRESHOLD", "6")
    )  # 超过 N 轮触发摘要
    memory_long_term_enabled: bool = os.getenv("MEMORY_LONG_TERM_ENABLED", "true").lower() == "true"
    memory_store_path: str = field(
        default_factory=lambda: os.getenv("MEMORY_STORE_PATH", "./data/memory_store")
    )
    # 跨会话共享记忆（M2 修复：默认关闭，避免不同会话/用户的偏好互相串味；
    # 确认多会话归属同一用户时再开启）
    memory_shared_enabled: bool = os.getenv("MEMORY_SHARED_ENABLED", "false").lower() == "true"

    # === 可观测性 ===
    langsmith_api_key: str = field(default_factory=lambda: os.getenv("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_PROJECT", "ecommerce-agent")
    )
    langsmith_endpoint: str = field(
        default_factory=lambda: os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    )
    tracing_enabled: bool = os.getenv("TRACING_ENABLED", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # === ReAct 循环 ===
    react_max_iterations: int = int(os.getenv("REACT_MAX_ITERATIONS", "5"))
    react_enabled: bool = os.getenv("REACT_ENABLED", "true").lower() == "true"

    # === 缓存分层 ===
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))
    cache_max_size: int = int(os.getenv("CACHE_MAX_SIZE", "500"))
    cache_persist_path: str = field(
        default_factory=lambda: os.getenv("CACHE_PERSIST_PATH", "./data/cache")
    )

    # === 流式输出 ===
    streaming_enabled: bool = os.getenv("STREAMING_ENABLED", "true").lower() == "true"

    # === 推荐 ===
    recommend_rerank_enabled: bool = os.getenv("RECOMMEND_LLM_RERANK", "true").lower() == "true"

    def get_mysql_config(self) -> dict:
        """返回 PyMySQL 连接字典"""
        return {
            "host": self.mysql_host,
            "port": self.mysql_port,
            "user": self.mysql_user,
            "password": self.mysql_password,
            "database": self.mysql_db,
            "charset": "utf8mb4",
        }


# 单例
settings = Settings()
