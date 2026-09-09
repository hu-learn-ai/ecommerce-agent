"""
智能客服 Agent — 退换货政策、常见问题、投诉处理

使用 FAISS + BGE 嵌入构建 FAQ 知识库的 RAG 系统
用户问题 → 向量检索 → 召回相关 FAQ → LLM 生成回答

学习参考: MultiAgent-Ecom 的 Policy Agent (FAISS RAG)
          E-Commerce Shopping Assistant 的 RAG 实现
"""

import os
from typing import List

from cachetools import TTLCache

from agents.tools.base import BaseAgentTool


class _SharedEmbedderAdapter:
    """
    轻量级适配器 — 将共享的 SentenceTransformer 包装为 LangChain Embeddings 接口
    避免重复加载 400MB 的 BGE 模型
    """

    def __init__(self, model):
        self._model = model

    def __call__(self, text: str) -> List[float]:
        """
        兼容 LangChain 新版本 FAISS 的调用方式:
        新版 langchain_community.faiss._embed_query 直接调用 self.embedding_function(text)
        （要求 embedding 对象可调用），旧版则调用 .embed_query(text)。
        同时实现 __call__ 与 embed_* 接口可兼容两种版本。
        """
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self._model.encode([text], normalize_embeddings=True)
        return embedding[0].tolist()


class CustomerServiceAgent(BaseAgentTool):
    """
    智能客服

    能力：
    - 退换货政策问答
    - 常见问题解答 (FAQ)
    - 投诉处理引导
    - 售后流程指引

    数据来源：FAISS 向量库中的 FAQ 知识
    需要先运行 scripts/build_faq_index.py 构建索引
    """

    name: str = "cs_agent"
    description: str = "智能客服：回答退换货政策、售后规则、常见问题等客服相关内容"

    def __init__(
        self, faiss_index_path: str, embedding_model_name: str, llm=None, shared_embedder=None
    ):
        super().__init__()
        self.faiss_index_path = faiss_index_path
        self.embedding_model_name = embedding_model_name
        self.llm = llm

        # 延迟加载（除非传入了共享模型）
        self._embedding_model = None
        self._vector_store = None
        self._shared_embedder = shared_embedder  # SentenceTransformer 实例

        # LLM 响应缓存：TTL=10min, maxsize=100，客服问题重复率高
        self._response_cache = TTLCache(maxsize=100, ttl=600)

    def preload(self):
        """启动时预加载嵌入模型和 FAISS 索引，避免首次请求超时"""
        try:
            self._ensure_loaded()
            print("[CSAgent] 预加载完成")
        except Exception as e:
            print(f"[CSAgent] 预加载失败（不影响启动，首次调用时重试）: {e}")

    # ------------------------------------------------------------------ #
    #  懒加载
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self):
        """懒加载向量库和嵌入模型"""
        if self._vector_store is not None:
            return

        # 如果有共享的 SentenceTransformer，用它创建轻量级 Embeddings 适配器
        if self._shared_embedder is not None:
            self._embedding_model = _SharedEmbedderAdapter(self._shared_embedder)
        else:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            self._embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )

        index_dir = self.faiss_index_path
        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            from langchain_community.vectorstores import FAISS as LCFAISS

            # ⚠️ 安全注意: allow_dangerous_deserialization=True 使用 pickle 反序列化
            # 必须确保 index 目录 (data/faiss_index/faq/) 的写入权限仅限受信任的构建脚本
            # 生产环境应将此目录挂载为只读卷，且使用非 root 用户运行
            self._vector_store = LCFAISS.load_local(
                index_dir,
                self._embedding_model,
                allow_dangerous_deserialization=True,
            )
            print("[CSAgent] FAQ 知识库已加载")
        else:
            print(f"[CSAgent] FAQ 索引未找到: {index_dir}")
            print("[CSAgent] 请先运行 scripts/build_faq_index.py")
            self._vector_store = None

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行客服问答"""
        return self.policy_qa(query)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心问答逻辑
    # ------------------------------------------------------------------ #

    def policy_qa(self, question: str) -> str:
        """
        回答退换货政策、售后规则等常见问题

        RAG 流程：
        1. 向量检索召回 Top-3 相关 FAQ
        2. 构建带 context 的 prompt
        3. LLM 生成自然语言回答

        Args:
            question: 用户问题

        Returns:
            回答字符串
        """
        self._ensure_loaded()

        # 检查缓存
        cache_key = question.strip().lower()
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        # Step 1: 向量检索
        docs = self._retrieve(question, k=3)

        if not docs:
            # 无 FAQ 知识库时降级为 LLM 直接回答
            return self._llm_direct_answer(question)

        # Step 2: 构建 context
        context = "\n\n".join(f"[FAQ {i + 1}] {d.page_content}" for i, d in enumerate(docs))

        # Step 3: LLM 生成回答
        if self.llm:
            answer = self._llm_generate(question, context)
            self._response_cache[cache_key] = answer
            return answer
        else:
            # 无 LLM 时直接返回检索到的 FAQ
            return f"根据我们的政策：\n\n{context}"

    def _retrieve(self, question: str, k: int = 3) -> list:
        """向量检索召回 FAQ"""
        if not self._vector_store:
            return []

        try:
            docs = self._vector_store.similarity_search(question, k=k)
            return docs
        except Exception as e:
            print(f"[CSAgent] 检索失败: {e}")
            return []

    def _llm_generate(self, question: str, context: str) -> str:
        """LLM 基于 context 生成回答"""
        prompt = f"""你是一个专业的电商客服助手。请根据以下 FAQ 知识回答用户问题。

FAQ 知识库:
{context}

用户问题: {question}

要求：
1. 基于以上 FAQ 内容回答，不要编造
2. 如果 FAQ 中没有相关信息，诚实告知并建议联系人工客服
3. 语气友好、专业
4. 用中文回答
"""
        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"[CSAgent] LLM 生成失败: {e}")
            return f"根据我们的政策：\n\n{context}"

    def _llm_direct_answer(self, question: str) -> str:
        """无 FAQ 知识库时的降级方案"""
        # 内置常见 FAQ 回答
        builtin_faqs = {
            "退换货": "支持7天无理由退换货，需保证商品完好不影响二次销售。",
            "退款": "审核通过后，退款将在3-7个工作日内原路返回。",
            "运费": "非质量问题退换货运费由买家承担，质量问题由卖家承担。",
            "发货": "下单后48小时内发货，预售商品以页面标注为准。",
            "发票": "支持电子发票，在下单时可选择开具。",
            "价保": "签收后7天内若商品降价，可在APP申请价保退款。",
            "客服": "在线客服 9:00-21:00，电话客服 9:00-18:00。",
            "会员": "注册会员享专属折扣、生日礼、积分翻倍等特权。",
        }

        # 关键词匹配
        for keyword, answer in builtin_faqs.items():
            if keyword in question:
                return f"关于您的问题：\n\n{answer}\n\n如需进一步帮助，请联系在线客服。"

        # 无匹配
        if self.llm:
            prompt = f"你是电商客服。用户问: {question}\n请基于电商常识友好回答。"
            try:
                return self.llm.invoke(prompt).content.strip()
            except Exception:
                pass

        return (
            "抱歉，我暂时无法回答这个问题。建议您联系在线客服 (9:00-21:00) 或拨打客服电话获取帮助。"
        )
