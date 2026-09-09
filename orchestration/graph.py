"""
LangGraph 降级编排器 — 状态机驱动的 ReAct Agent 工具调度 (ReAct 失败时的 fallback)

流程：
    用户输入 → Router(意图识别) → 条件分发
        → Executor(执行对应 Agent)
        → Clarify(需要澄清时追问)
        → Fallback(闲聊/兜底)
        → Finalize(返回结果)

学习参考:
    - Multi-AI-Agent4OnlineShopping 的 LangGraph State Machine
    - JoyAgent-JDGenie 的 DAG 执行引擎
    - llm-based-recommender 的 LangGraph workflow
"""

import asyncio
import operator
from typing import Annotated, Optional, Sequence, Tuple, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    """Agent 状态定义"""

    messages: Annotated[Sequence[str], operator.add]
    intent: str
    agent: str
    mode: str
    current_response: str
    needs_clarification: bool
    tool_results: list
    input: str  # 当前用户输入（独立存储，避免 messages 累积影响取值）


class ECommerceOrchestrator:
    """
    主编排器 — LangGraph 状态机驱动

    管理 4 个领域 Agent 的调度与编排（8 个意图 → 4 个数据域）：
    1. Router Agent 识别用户意图
    2. 根据意图分发到对应子 Agent 执行
    3. 需要澄清时追问用户
    4. 闲聊/兜底由 LLM 直接回复
    5. 汇总结果返回
    """

    def __init__(self, llm: ChatOpenAI, agents: dict, router):
        """
        Args:
            llm: 统一 LLM 实例 (DeepSeek Chat)
            agents: Agent 实例字典 {"search_agent": SearchAgent(), ...}
            router: RouterAgent 路由器实例
        """
        self.llm = llm
        self.agents = agents
        self.router = router
        self._last_state: Optional[AgentState] = None
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ #
    #  图构建
    # ------------------------------------------------------------------ #

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态机"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("router", self._route_intent)
        workflow.add_node("executor", self._execute_agent)
        workflow.add_node("clarify", self._clarify)
        workflow.add_node("fallback", self._fallback)
        workflow.add_node("finalize", self._finalize)

        # 定义边
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self._after_route,
            {
                "execute": "executor",
                "clarify": "clarify",
                "fallback": "fallback",
            },
        )
        workflow.add_edge("executor", "finalize")
        workflow.add_edge("clarify", "finalize")
        workflow.add_edge("fallback", "finalize")
        workflow.add_edge("finalize", END)

        return workflow.compile()

    # ------------------------------------------------------------------ #
    #  节点实现
    # ------------------------------------------------------------------ #

    def _route_intent(self, state: AgentState) -> dict:
        """路由节点：识别用户意图（只返回需要更新的字段）"""
        user_input = state.get("input") or state["messages"][-1]
        route_info = self.router.route(user_input)

        # 简单澄清判断：输入过短且不是问候
        if len(user_input.strip()) < 3 and route_info["intent"] not in ("chitchat",):
            needs_clarify = True
        else:
            needs_clarify = False

        return {
            "intent": route_info["intent"],
            "agent": route_info["agent"],
            "mode": route_info["mode"],
            "input": user_input,
            "needs_clarification": needs_clarify,
        }

    def _execute_agent(self, state: AgentState) -> dict:
        """执行节点：调用对应子 Agent（只返回需要更新的字段）"""
        agent_name = state["agent"]
        user_input = state.get("input") or state["messages"][-1]

        if agent_name in self.agents:
            agent = self.agents[agent_name]
            try:
                # 商品分类需要提取商品名，而不是传入完整问句
                if state.get("mode") == "classify":
                    query = self._extract_product_name(user_input)
                else:
                    query = user_input
                response = self._dispatch(agent, state.get("mode", "run"), query)
            except TypeError:
                # 某些方法可能不接受 query 参数名
                response = self._dispatch(agent, state.get("mode", "run"), user_input)
            except Exception as e:
                response = f"执行 {agent_name} 时出错: {e}。请稍后重试。"
                print(f"[Orchestrator] Agent {agent_name} 执行失败: {e}")
        else:
            # Agent 不存在，降级为 LLM 直接回复
            try:
                response = self.llm.invoke(user_input).content
            except Exception as e:
                response = f"服务暂时不可用: {e}"

        return {"current_response": response}

    def _dispatch(self, agent, mode: str, query: str) -> str:
        """按 mode 分派到领域 Agent 的方法，方法不存在时回退 run()"""
        method = getattr(agent, mode, None)
        if callable(method):
            try:
                return method(query=query)
            except TypeError:
                return method(query)
        return agent.run(query=query)

    def _extract_product_name(self, user_input: str) -> str:
        """
        从用户问句中提取商品名称

        例: "iPhone 15 属于什么分类" → "iPhone 15"
            "小米手机 分类" → "小米手机"
            "蓝牙耳机" → "蓝牙耳机" (已经是商品名)
        """
        import re

        # 去掉常见的问句模板，保留商品名
        patterns = [
            r"(.*?)属于什么分类",
            r"(.*?)属于什么类别",
            r"(.*?)什么分类",
            r"(.*?)什么类别",
            r"(.*?)什么类",
            r"分类[:：\s]*(.*)",
            r"类别[:：\s]*(.*)",
        ]

        for pattern in patterns:
            match = re.match(pattern, user_input.strip(), re.IGNORECASE)
            if match:
                product = match.group(1).strip()
                # 去掉末尾的标点和空格
                product = re.sub(r"[，,。.!！?？\s]+$", "", product)
                if product:
                    return product

        return user_input.strip()

    def _after_route(self, state: AgentState) -> str:
        """条件边：决定走哪个分支"""
        if state.get("needs_clarification"):
            return "clarify"
        if state["intent"] in ("chitchat", "unknown"):
            return "fallback"
        return "execute"

    def _clarify(self, state: AgentState) -> dict:
        """澄清节点：需要更多信息时追问用户"""
        return {
            "current_response": (
                "您能再详细描述一下吗？比如您想了解哪个商品、什么分类，"
                "或者具体的问题？我可以帮您搜索商品、查订单、推荐好物等。"
            )
        }

    def _fallback(self, state: AgentState) -> dict:
        """兜底节点：闲聊或无法分类的意图（LLM 以电商人设直接回复）"""
        user_input = state.get("input") or state["messages"][-1]

        # 零成本问候快速匹配
        greeting = self._match_greeting(user_input)
        if greeting:
            return {"current_response": greeting}

        # LLM 以电商助手人设直接回复
        try:
            prompt = f"{self.CHITCHAT_SYSTEM_PROMPT}\n\n用户: {user_input}\n\n助手:"
            return {"current_response": self.llm.invoke(prompt).content}
        except Exception as e:
            return {"current_response": f"抱歉，我暂时无法回复: {e}"}

    # 闲聊人设（替代原 chitchat_agent）
    CHITCHAT_SYSTEM_PROMPT = """你是一个友好、专业的电商智能助手。

你的能力包括：
- 🔍 商品搜索（支持分类和价格过滤）
- 📚 知识图谱问答（品牌、属性、分类关系）
- 🏷️ 商品分类（自动识别商品类别）
- 💡 智能推荐（个性化商品推荐）
- 📦 订单查询（订单状态、物流追踪）
- ❓ 售后服务（退换货政策、FAQ）
- 📊 数据分析（销售趋势、商品排行）

要求：
1. 保持友好、简洁的语气
2. 如果用户意图不明确，引导用户使用上述功能
3. 用中文回答
4. 如果用户问到商品、推荐、价格等购物相关问题，引导他们直接描述需求
   （如"搜索蓝牙耳机"或"推荐户外装备"）。"""

    @staticmethod
    def _match_greeting(text: str) -> str:
        """匹配常见问候语并返回预设回复（零成本，替代 chitchat_agent）"""
        text_lower = text.lower().strip()

        greeting_map = {
            "你好": "你好！我是电商智能助手 🛒。你可以问我商品搜索、推荐、订单查询、售后政策等问题，随时为你服务！",
            "您好": "您好！我是电商智能助手 🛒。有什么可以帮你的吗？可以搜索商品、查订单、问售后政策等。",
            "早上好": "早上好！☀️ 今天想了解什么商品？我可以帮你搜索、推荐或查询订单。",
            "下午好": "下午好！☀️ 有什么购物需求吗？搜索商品、查推荐、问售后都可以找我。",
            "晚上好": "晚上好！🌙 有什么可以帮你的？商品搜索、智能推荐、订单查询随时在线。",
            "哈喽": "哈喽！👋 我是电商智能助手，帮你搜索商品、查订单、问售后政策都可以！",
            "hello": "Hello! 我是电商智能助手 🛒，可以帮你搜索商品、推荐好物、查询订单和售后政策。",
            "hi": "Hi! 👋 有什么购物问题可以帮你？",
            "谢谢": "不客气！😊 如果还有其他问题，随时问我。祝你购物愉快！",
            "感谢": "不客气！😊 很高兴能帮到你。",
            "再见": "再见！👋 下次有购物问题随时找我。祝生活愉快！",
            "拜拜": "拜拜！👋 随时欢迎回来找我。",
        }

        for keyword, reply in greeting_map.items():
            if keyword in text_lower:
                return reply

        return ""

    def _finalize(self, state: AgentState) -> dict:
        """终节点：保存最终状态用于元数据返回（注意：并发场景下会被覆盖）"""
        self._last_state = state
        return {}

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def invoke(self, user_input: str) -> tuple:
        """
        同步调用入口

        Args:
            user_input: 用户输入文本

        Returns:
            (回复字符串, 状态信息 dict)
        """
        result = self.graph.invoke(
            {
                "messages": [user_input],
                "input": user_input,
                "intent": "",
                "agent": "",
                "mode": "",
                "current_response": "",
                "needs_clarification": False,
                "tool_results": [],
            }
        )
        return result["current_response"], {
            "intent": result.get("intent", "unknown"),
            "agent": result.get("agent", "unknown"),
        }

    async def ainvoke(self, user_input: str) -> Tuple[str, dict]:
        """
        异步调用入口 — 使用 asyncio.to_thread 包装同步 graph.invoke，
        避免阻塞 asyncio 事件循环

        Args:
            user_input: 用户输入文本

        Returns:
            (回复字符串, 状态信息 dict)
        """
        initial_state = {
            "messages": [user_input],
            "input": user_input,
            "intent": "",
            "agent": "",
            "mode": "",
            "current_response": "",
            "needs_clarification": False,
            "tool_results": [],
        }
        # 在线程池中执行同步的 graph.invoke，不阻塞事件循环
        result = await asyncio.to_thread(self.graph.invoke, initial_state)
        return result["current_response"], {
            "intent": result.get("intent", "unknown"),
            "agent": result.get("agent", "unknown"),
        }

    def get_last_state(self) -> Optional[dict]:
        """获取最近一次执行的状态信息（已废弃，推荐使用 ainvoke 返回值）"""
        if not self._last_state:
            return {}
        return {
            "intent": self._last_state.get("intent", "unknown"),
            "agent": self._last_state.get("agent", "unknown"),
        }
