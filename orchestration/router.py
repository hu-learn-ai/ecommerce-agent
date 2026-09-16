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

    # 匹配优先级（前者优先）。抽成类属性，便于调参脚本复用同一套顺序
    KEYWORD_PRIORITY = [
        "order",
        "customer_service",
        "analytics",
        "kg_qa",  # kg_qa 优先于 classify：问句式分类查询走知识图谱
        "classify",  # classify 仅处理直接分类请求
        "recommend",
        "search",
        "chitchat",
    ]

    # 弱关键词：这些词在多个意图里都会出现，**不足以单独定意图**——
    # 命中它们时不直接返回，继续交给 Level-2 的 LLM 语义路由判断。
    # 依据（2026-09-16，209 条用例的误判归因）：单一个"买"字就造成 15 次误判
    # （"想给男朋友买个生日礼物"→被判 search、"我买的东西现在到哪儿了"→被判 search），
    # "卖/找/价格"同样把 analytics/customer_service 请求抢成 search。
    # 准入过程见 tests/tune_router.py（留出集 + 4 折交叉验证）。
    WEAK_KEYWORDS = frozenset({"买", "卖", "找", "价格"})

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

【对话历史】（可能为空）
{history}

判断规则补充：如果用户输入是**简短的追问或省略句**（如"那平板呢""第二个呢""有降噪吗""能改地址吗"），
必须结合【对话历史】判断他在延续哪件事，并按**延续的那个意图**回答——不要只看这几个字本身。
例如上文在推荐耳机，用户说"那平板呢"，意图仍是 recommend。

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

        # Level 2: LLM 语义路由（Level 1 无结果时）——**带上对话历史**，
        # 否则"那平板呢"这类省略式追问只能看到这几个字本身，会被判成别的意图
        if not intent:
            intent = self._llm_route(user_input, history=history)

        # Level 3: LLM 不可用时，退回复用最近一次有效意图（追问场景）
        # 注：此前这段是**死代码**——_llm_route 失败也返回 "chitchat"（非空），
        # 所以 `if not intent and history` 永远不成立，上下文感知从未生效。
        if not intent and history:
            for item in reversed(history):
                it = item.get("intent") if isinstance(item, dict) else None
                if it and it != "chitchat":
                    intent = it
                    break

        # Level 4: 兜底闲聊（避免 None 进入标准化导致异常）
        intent = intent or "chitchat"

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
        has_weak = False
        for intent in self.KEYWORD_PRIORITY:
            for keyword in self.KEYWORD_RULES.get(intent, []):
                if keyword not in text:
                    continue
                if keyword in self.WEAK_KEYWORDS:
                    has_weak = True
                    continue
                # 问候语与购物诉求混说（"你好,我想买耳机"）：只剩 chitchat 一个强命中、
                # 但句子里还有购物弱词时，不要判闲聊，交给 LLM（chitchat 在优先级最后，
                # 走到这里说明前面几类都没有强命中）
                if intent == "chitchat" and has_weak:
                    continue
                return intent

        return None

    @staticmethod
    def _format_history(history: list, turns: int = 3) -> str:
        """把最近几轮对话压成简短文本，供 LLM 判断省略式追问的意图。"""
        if not history:
            return "（无）"
        lines = []
        for item in history[-turns * 2:]:
            if not isinstance(item, dict):
                continue
            role = "用户" if item.get("role") == "user" else "客服"
            content = str(item.get("content", ""))[:80]
            if content:
                lines.append(f"{role}: {content}")
        return "\n".join(lines) or "（无）"

    def _llm_route(self, user_input: str, history: list = None) -> Optional[str]:
        """
        Level 2: LLM 语义路由

        当关键词匹配失败时，使用 LLM 进行语义理解；**带对话历史**，
        以便把"那平板呢"这类省略式追问归到它真正延续的意图上。

        返回 None 表示 LLM 不可用/失败，由调用方决定是否复用历史意图。
        """
        try:
            chain = self.router_prompt | self.llm
            result = chain.invoke(
                {"input": user_input, "history": self._format_history(history)}
            )
            return result.content.strip().lower()
        except Exception as e:
            print(f"[Router] LLM 路由失败: {e}")
            return None

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
