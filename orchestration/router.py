"""
意图路由器 — Router Agent

分析用户输入，识别意图并路由到对应的子 Agent

学习参考: MultiAgent-Ecom 的 Router Agent
          Price Pilot 的 ChatAgent (任务委托)
"""

import re
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 意图类型定义
IntentType = [
    "kg_qa",  # 知识图谱问答（品牌/属性/关系）
    "search",  # 商品搜索
    "classify",  # 商品分类
    "recommend",  # 商品推荐
    "order",  # 订单查询
    "customer_service",  # 售后服务/FAQ
    "analytics",  # 数据分析
    "chitchat",  # 闲聊/问候
]


class RouterAgent:
    """
    意图识别路由器

    两级路由策略：
    Level 1: 基于关键词的快速规则匹配（低延迟）
    Level 2: LLM 语义理解（高准确率，当 Level 1 无法判断时使用）

    意图 → Agent 映射（8 个意图收敛到 4 个领域 Agent）:
        search / classify / recommend → catalog_agent
        kg_qa                          → kg_qa_agent
        order / analytics              → business_data_agent
        customer_service               → cs_agent
        chitchat                       → 无 Agent（编排器 LLM 直接回复）
    """

    # 意图 → Agent 名称映射
    AGENT_MAP = {
        "kg_qa": "kg_qa_agent",
        "search": "catalog_agent",
        "classify": "catalog_agent",
        "recommend": "catalog_agent",
        "order": "business_data_agent",
        "customer_service": "cs_agent",
        "analytics": "business_data_agent",
        "chitchat": "",  # 无专属 Agent，由编排器 LLM 直接处理
    }

    # 意图 → Agent 方法名（降级路由按 mode 分派）
    AGENT_MODE = {
        "kg_qa": "run",
        "search": "search",
        "classify": "classify",
        "recommend": "recommend",
        "order": "query_order",
        "customer_service": "run",
        "analytics": "analyze",
    }

    # 关键词路由规则（Level 1 快速匹配）
    # 注意：kg_qa 的 "属于什么" / "什么分类" 必须在 classify 的 "属于" / "分类" 之前匹配
    KEYWORD_RULES = {
        "order": ["订单", "物流", "快递", "发货", "签收", "运单", "order", "tracking"],
        "customer_service": [
            "退换货",
            "退货",
            "退款",
            "售后",
            "投诉",
            "运费",
            "发票",
            "价保",
            "客服",
        ],
        "analytics": [
            "销量",
            "排行",
            "趋势",
            "分析",
            "数据",
            "统计",
            "占比",
            "top",
            "卖得最好",
            "最好卖",
            "热销",
            "畅销",
            "销量最高",
        ],
        # kg_qa 放在 classify 之前：问句式 "属于什么分类" 优先路由到知识图谱
        "kg_qa": [
            "品牌",
            "属性",
            "什么关系",
            "属于什么",
            "有哪些产品",
            "什么牌子",
            "什么分类",
            "什么类别",
            "哪个分类",
            "哪个类别",
        ],
        # classify 只处理直接的分类请求（如 "分类这个商品"），不处理问句
        "classify": ["分类", "类别", "什么类"],
        "recommend": [
            "推荐",
            "建议",
            "买什么",
            "选什么",
            "选个",
            "有什么好",
            "预算",
            "求推荐",
        ],
        "search": ["搜索", "查找", "找", "有没有", "卖", "买", "多少钱", "价格"],
        "chitchat": ["你好", "您好", "早上好", "下午好", "晚上好", "hi", "hello", "谢谢", "再见"],
    }

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        self.router_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """你是一个电商智能客服的路由器。
根据用户输入，判断意图并路由到对应模块：

- kg_qa: 关于品牌、商品属性、分类关系的问题
  例: "Apple有哪些产品？" "Nike是什么品牌？" "纸尿裤属于什么分类？"

- search: 搜索具体商品
  例: "有没有500元以下的蓝牙耳机？" "推荐一款洗发水" "搜索手机"

- classify: 询问商品属于哪个分类
  例: "纸尿裤属于什么类别？" "这个商品是什么分类？"

- recommend: 个性化推荐
  例: "我最近喜欢户外运动，有什么推荐？" "帮我选个礼物"

- order: 订单查询/物流
  例: "我的订单到哪了？" "查询订单 ORD123456"

- customer_service: 售后/退换货/政策
  例: "怎么退货？" "你们的退换货政策是什么？" "运费谁出？"

- analytics: 数据分析/趋势
  例: "最近什么品类卖得最好？" "销售趋势如何？"

- chitchat: 问候/闲聊/其他
  例: "你好" "今天天气不错" "谢谢"

只回复意图类型关键词(从上述列表中选择)，不要加其他任何文字。""",
                ),
                ("user", "{input}"),
            ]
        )

    def route(self, user_input: str, history: list = None) -> dict:
        """
        识别意图并返回路由信息

        Args:
            user_input: 用户输入
            history: 对话历史（可选，用于上下文感知路由）

        Returns:
            {"intent": "search", "agent": "catalog_agent", "mode": "search", "input": "..."}
        """
        # Level 1: 关键词快速匹配
        intent = self._keyword_route(user_input)

        # Level 2: LLM 语义路由（Level 1 无结果时）
        if not intent:
            intent = self._llm_route(user_input)

        # Level 3: 上下文感知（追问场景：复用最近的非闲聊意图）
        if not intent and history:
            for item in reversed(history):
                it = item.get("intent") if isinstance(item, dict) else None
                if it and it != "chitchat":
                    intent = it
                    break

        # 清理与验证
        intent = self._normalize_intent(intent)

        return {
            "intent": intent,
            "agent": self.AGENT_MAP.get(intent, ""),
            "mode": self.AGENT_MODE.get(intent, "run"),
            "input": user_input,
        }

    # ------------------------------------------------------------------ #
    #  路由策略
    # ------------------------------------------------------------------ #

    def _keyword_route(self, user_input: str) -> Optional[str]:
        """
        Level 1: 关键词快速匹配

        基于预设关键词规则进行意图匹配
        优势：零延迟，无需调用 LLM
        """
        text = user_input.lower().strip()

        # 按优先级检查（订单和客服优先级较高，kg_qa 在 classify 之前）
        priority_order = [
            "order",
            "customer_service",
            "analytics",
            "kg_qa",  # kg_qa 优先于 classify：问句式分类查询走知识图谱
            "classify",  # classify 仅处理直接分类请求
            "recommend",
            "search",
            "chitchat",
        ]

        for intent in priority_order:
            for keyword in self.KEYWORD_RULES.get(intent, []):
                if keyword in text:
                    return intent

        return None

    def _llm_route(self, user_input: str) -> str:
        """
        Level 2: LLM 语义路由

        当关键词匹配失败时，使用 LLM 进行语义理解
        """
        try:
            chain = self.router_prompt | self.llm
            result = chain.invoke({"input": user_input})
            return result.content.strip().lower()
        except Exception as e:
            print(f"[Router] LLM 路由失败: {e}")
            return "chitchat"

    def _normalize_intent(self, intent: str) -> str:
        """标准化意图类型"""
        intent = intent.strip().lower()

        # 去除可能的标点和多余文字
        intent = re.sub(r"[^\w_]", "", intent)

        # 别名映射
        alias_map = {
            "qa": "kg_qa",
            "kgqa": "kg_qa",
            "knowledge": "kg_qa",
            "cs": "customer_service",
            "service": "customer_service",
            "faq": "customer_service",
            "analytics": "analytics",
            "analysis": "analytics",
            "data": "analytics",
            "chitchat": "chitchat",
            "chat": "chitchat",
            "small_talk": "chitchat",
            "greeting": "chitchat",
        }

        if intent in alias_map:
            return alias_map[intent]

        # 验证是否为有效意图
        if intent in IntentType:
            return intent

        # 默认兜底
        return "chitchat"
