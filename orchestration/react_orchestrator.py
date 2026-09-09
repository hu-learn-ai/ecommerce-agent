"""
ReAct 工具循环编排器 — LLM 自主决策的工具调用循环

核心改进 (对比原固定路由):
1. LLM 自主决定调用哪个工具 (function-calling)
2. 支持多步推理: 思考 → 调用工具 → 观察结果 → 再思考 → 再调用 → ... → 最终回答
3. 工具间可以链式调用 (如: 先搜索 → 再分类 → 再推荐)
4. 保留原路由作为 fallback (ReAct 失败时降级为固定路由)

工作流程:
    用户输入 + 记忆上下文
        → LLM 思考: 需要调用哪些工具?
        → 调用工具 1 → 观察结果
        → LLM 再思考: 信息够了吗? 需要更多工具?
        → 调用工具 2 → 观察结果 (可选)
        → LLM 生成最终回答
        → 更新记忆

使用 LangChain 的 create_tool_calling_agent + AgentExecutor 实现。
"""

import asyncio
import json
import threading
import time
from collections import defaultdict
from typing import AsyncGenerator, List, Optional, Tuple

try:
    # langchain 1.x: 旧版 agents API 移入 langchain-classic 包
    from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
except ImportError:  # pragma: no cover - langchain 0.x 兼容路径
    from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

from config.settings import settings
from orchestration.memory import MemoryManager
from orchestration.model_router import cost_optimizer
from orchestration.observability import obs
from orchestration.router import RouterAgent


class RepeatDetectionCallback(BaseCallbackHandler):
    """
    ReAct 防死循环回调 — 检测重复工具调用

    当同一工具被以相同参数连续调用超过阈值时，抛出异常中断循环。
    这防止 LLM 陷入 "调用同一工具 → 获得相同结果 → 再次调用" 的死循环。
    """

    def __init__(self, max_repeats: int = 2):
        self.max_repeats = max_repeats
        self._call_history: List[tuple] = []  # [(tool_name, tool_input), ...]
        self._repeat_counts: dict = defaultdict(int)

    def on_tool_start(self, serialized_tool: dict, input_str: str, **kwargs):
        """工具调用开始时检测重复"""
        tool_name = serialized_tool.get("name", "unknown")
        call_key = (tool_name, input_str.strip()[:200])  # 截断防止内存膨胀

        self._call_history.append(call_key)
        self._repeat_counts[call_key] += 1

        if self._repeat_counts[call_key] > self.max_repeats:
            obs.logger.warning(
                "ReAct 重复调用检测: 中断循环",
                tool=tool_name,
                input=input_str[:100],
                count=self._repeat_counts[call_key],
            )
            raise ValueError(
                f"检测到重复调用: 工具 '{tool_name}' 已被相同参数调用 "
                f"{self._repeat_counts[call_key]} 次，中断以防死循环"
            )

    def reset(self):
        """重置调用历史（每次新对话调用）"""
        self._call_history.clear()
        self._repeat_counts.clear()

    @property
    def call_history(self) -> List[tuple]:
        return list(self._call_history)


class ReactOrchestrator:
    """
    ReAct 工具循环编排器

    将 4 个领域 Agent 的 7 个能力包装为 LangChain Tool，
    让 LLM 自主决定调用顺序和次数。支持多步推理、工具链式调用、观察-决策循环。
    """

    SYSTEM_PROMPT = """你是一个电商智能助手，拥有以下工具能力：

1. **search_products** — 搜索商品 (支持分类/价格过滤)
2. **kg_qa** — 知识图谱问答 (品牌/属性/分类关系)
3. **classify_product** — 商品分类识别
4. **recommend_products** — 个性化商品推荐
5. **query_order** — 订单查询/物流追踪
6. **customer_service** — 售后/退换货政策
7. **data_analysis** — 数据分析/销售趋势/排行

工作原则：
- 先理解用户意图，选择最合适的工具
- 如果一个工具的结果不够，可以继续调用其他工具
- 基于工具返回的结果生成最终回答
- 用中文回答，保持友好专业

你可以多步推理：调用工具 → 观察结果 → 判断是否需要更多信息 → 继续调用或给出最终回答。

【关键约束 — 必须严格遵守】
1. 涉及商品搜索、推荐、分类、品牌关系等问题时，必须先调用对应工具获取数据，严禁基于自身知识编造商品信息。
2. 如果工具返回的结果为空或不相关，如实告诉用户"没有找到相关商品"，不要自行补充推荐。
3. 你只能推荐工具返回的商品，不得添加工具结果之外的商品名称、价格、规格等信息。
4. 对于问候、感谢、告别等纯闲聊场景，直接友好回复即可，不需要调用任何工具。
5. 如果用户询问的商品在工具返回结果中不存在，直接告知用户未找到，不要编造。"""

    def __init__(
        self,
        llm: ChatOpenAI,
        agents: dict,
        router: RouterAgent,
        memory_manager: Optional[MemoryManager] = None,
    ):
        self.llm = llm
        self.agents = agents
        self.router = router
        self.memory = memory_manager or MemoryManager(llm=llm)
        self._tools = self._build_tools()
        # 按模型层级构建 executor（接线三级模型分级：lite/standard/heavy）
        self._executors: dict = {}
        try:
            for tier, tier_llm in cost_optimizer.model_router._models.items():
                self._executors[tier] = self._build_agent_executor(tier_llm)
        except Exception as exc:  # noqa: BLE001
            print(f"[ReactOrchestrator] 分层 executor 构建失败，使用默认: {exc}")
        self._agent_executor = self._executors.get(2) or self._build_agent_executor(self.llm)
        self._fallback_orchestrator = None  # 延迟初始化
        # 线程局部变量：当前请求的用户画像（推荐工具读取，并发安全）
        self._profile_store = threading.local()

    def _build_tools(self) -> list:
        """将所有 Agent 包装为 LangChain Tool"""

        tools = []

        # 1. 商品域（catalog_agent）：搜索 / 推荐 / 分类
        catalog = self.agents.get("catalog_agent")
        if catalog:
            tools.append(
                Tool(
                    name="search_products",
                    description=(
                        "搜索具体商品。仅当用户给出了明确的商品关键词/名称时使用，"
                        "如 '蓝牙耳机'、'iPhone 15'、'跑步鞋'、'纸尿裤'。输入: 商品关键词。"
                        "注意: 如果用户表达的是推荐诉求（'推荐…'、'求推荐…'、'预算X'、'适合…'、'想买…'），"
                        "必须使用 recommend_products 工具，而不是本工具。"
                    ),
                    func=lambda query: catalog.search(query=query),
                )
            )
            tools.append(
                Tool(
                    name="recommend_products",
                    description=(
                        "个性化商品推荐。当用户表达推荐诉求时必须调用此工具："
                        "'推荐…'、'求推荐…'、'预算X'、'适合…'、'想买…'、'有什么…推荐'。"
                        "输入: 用户的完整需求描述（含预算、用途、品牌偏好等）。"
                        "示例: '预算8000，主要拍视频和拍照，内存512G' 或 '推荐一些适合户外的装备'。"
                        "严禁自行编造推荐，必须通过此工具获取真实商品数据。"
                    ),
                    func=lambda query: catalog.recommend(
                        query=query,
                        user_profile=self._get_current_profile(),
                    ),
                )
            )
            tools.append(
                Tool(
                    name="classify_product",
                    description="对商品标题进行自动分类。输入: 商品标题文本。例如: '小米 Redmi Note 12 5G 手机'",
                    func=lambda query: catalog.classify(query=query),
                )
            )

        # 2. 图谱域（kg_qa_agent）
        kg_agent = self.agents.get("kg_qa_agent")
        if kg_agent:
            tools.append(
                Tool(
                    name="kg_qa",
                    description="基于知识图谱回答商品相关问题，如品牌关系、属性查询、分类导航。输入: 自然语言问题。例如: 'Apple有哪些产品?' 或 '纸尿裤属于什么分类?'",
                    func=lambda query: kg_agent.run(query=query),
                )
            )

        # 3. 业务数据域（business_data_agent）：订单 / 数据分析
        business = self.agents.get("business_data_agent")
        if business:
            tools.append(
                Tool(
                    name="query_order",
                    description="查询订单状态和物流信息。输入: 订单号或包含订单号的查询。例如: '查询订单 ORD123456' 或 '123456789'",
                    func=lambda query: business.query_order(
                        query=query,
                        user_id=(self._get_current_profile() or {}).get("user_id"),
                    ),
                )
            )
            tools.append(
                Tool(
                    name="data_analysis",
                    description="查询销售趋势、商品排行、品类占比等数据分析。输入: 分析需求。例如: '最近什么品类卖得最好?' 或 '销售趋势'",
                    func=lambda query: business.analyze(query=query),
                )
            )

        # 4. 客服域（cs_agent）
        cs_agent = self.agents.get("cs_agent")
        if cs_agent:
            tools.append(
                Tool(
                    name="customer_service",
                    description="回答售后政策、退换货规则、常见问题。输入: 用户问题。例如: '怎么退货?' 或 '运费谁出?'",
                    func=lambda query: cs_agent.run(query=query),
                )
            )

        return tools

    def _build_agent_executor(self, llm: ChatOpenAI) -> AgentExecutor:
        """构建 ReAct Agent Executor（含防死循环 + 超时控制）"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm, self._tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=self._tools,
            max_iterations=settings.react_max_iterations,  # 最大迭代次数 (默认 5)
            max_execution_time=30,  # 总执行超时 30 秒
            verbose=False,
            handle_parsing_errors=True,
            return_intermediate_steps=True,
            early_stopping_method="generate",
        )

    def _get_fallback(self):
        """获取降级编排器 (原固定路由)"""
        if self._fallback_orchestrator is None:
            from orchestration.graph import ECommerceOrchestrator

            self._fallback_orchestrator = ECommerceOrchestrator(self.llm, self.agents, self.router)
        return self._fallback_orchestrator

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def invoke(
        self,
        user_input: str,
        session_id: str = "default",
        user_profile: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        """
        同步调用入口

        Args:
            user_input: 用户输入
            session_id: 会话 ID (用于记忆隔离)
            user_profile: 用户画像 (可选)

        Returns:
            (回复字符串, 状态信息 dict)
        """
        try:
            return self._invoke_react(user_input, session_id, user_profile)
        except Exception as e:
            obs.logger.error("ReAct 执行失败，降级到固定路由", error=str(e))
            # 降级到原固定路由
            fallback = self._get_fallback()
            response, state = fallback.invoke(user_input)
            return response, {
                **state,
                "mode": "fallback",
                "error": str(e),
            }

    def _invoke_react(
        self,
        user_input: str,
        session_id: str,
        user_profile: Optional[dict],
    ) -> Tuple[str, dict]:
        """执行 ReAct 循环（含防死循环 + 超时保护）"""
        with obs.trace_context("react_orchestrator") as span:
            span.set_attribute("session_id", session_id)
            span.set_attribute("input_length", len(user_input))

            # 1. 先用关键词路由快速判断意图 (用于模型分级 + 缓存 key)
            route_info = self.router.route(
                user_input,
                history=self.memory.get_history(session_id),
            )
            intent = route_info["intent"]
            span.set_attribute("intent", intent)

            # 2. 检查缓存（个性化意图不缓存，避免画像串味）
            cached = (
                None
                if intent == "recommend"
                else cost_optimizer.cache_get(intent, user_input)
            )
            if cached:
                span.set_attribute("cache_hit", True)
                # 仍然记录到记忆
                self.memory.add_message(session_id, "user", user_input, intent=intent)
                self.memory.add_message(session_id, "assistant", cached, intent=intent)
                return cached, {"intent": intent, "mode": "react_cached", "agent": "cache"}

            # 3. 构建记忆上下文
            chat_history = self._build_chat_history(session_id, user_input)

            # 3.1 注入意图提示：关键词路由已确定业务意图时，
            #     明确引导 LLM 调用对应的业务工具，纠正误路由（如推荐→搜索）
            intent_hint = self._build_intent_hint(intent)
            if intent_hint:
                chat_history.insert(0, SystemMessage(content=intent_hint))

            # 4. 执行 ReAct 循环 (max_iterations + max_execution_time + repeat_detection 三重保护)
            # 3.2 构建并绑定当前请求的用户画像（显式传入 + 长期记忆偏好），
            #     供 recommend_products 工具在线程局部读取（并发安全）
            request_profile = self._build_user_profile(session_id, user_input, user_profile)
            self._profile_store.user_profile = request_profile
            start_time = time.time()
            try:
                # 按意图选择模型层级 executor（接线三级模型分级）
                tier = cost_optimizer.model_router.get_tier(intent, len(user_input))
                executor = self._executors.get(tier) or self._agent_executor
                # 请求级防死循环回调（避免并发下共享单例互相 reset）
                repeat_cb = RepeatDetectionCallback(max_repeats=2)
                from langchain_community.callbacks import get_openai_callback

                with get_openai_callback() as cb:
                    result = executor.invoke(
                        {
                            "input": user_input,
                            "chat_history": chat_history,
                        },
                        config={"callbacks": [repeat_cb]},
                    )
                if cb.total_tokens:
                    obs.record_tokens(
                        model=self.llm.model_name,
                        prompt_tokens=cb.prompt_tokens,
                        completion_tokens=cb.completion_tokens,
                    )
            except ValueError as e:
                # 重复调用检测触发的中断
                obs.logger.warning("ReAct 重复调用中断，降级处理", error=str(e))
                span.set_attribute("anti_loop_triggered", True)
                span.set_attribute("repeat_history", repeat_cb.call_history)
                # 降级: 直接用路由的 agent 回答
                agent_name = route_info["agent"]
                mode = route_info.get("mode", "run")
                if agent_name in self.agents:
                    response = self._run_agent_method(
                        self.agents[agent_name],
                        mode,
                        user_input,
                        user_profile=request_profile,
                    ) or "抱歉，处理您的请求时遇到问题，请稍后重试。"
                else:
                    response = "抱歉，处理您的请求时遇到问题，请稍后重试。"
                return response, {
                    "intent": intent,
                    "mode": "react_anti_loop",
                    "agent": agent_name,
                    "tools_called": [],
                    "error": str(e),
                }
            finally:
                # 请求结束（或异常）后清理线程局部画像，避免串号
                self._profile_store.user_profile = None

            elapsed = time.time() - start_time

            response = result.get("output", "")
            intermediate_steps = result.get("intermediate_steps", [])

            # 结构化事件记忆：把工具调用 JSON 结果显式存入短期记忆（供后续轮次引用）
            try:
                for step in intermediate_steps[:5]:
                    action, observation = step
                    self.memory.add_event(
                        session_id,
                        "tool_call",
                        {
                            "tool": getattr(action, "tool", ""),
                            "tool_input": str(getattr(action, "tool_input", ""))[:300],
                            "result": str(observation)[:600],
                        },
                    )
            except Exception as e:  # noqa: BLE001
                print(f"[ReactOrchestrator] 工具事件记忆写入失败: {e}")

            # 提取调用的工具链
            tools_called = (
                [step[0].tool for step in intermediate_steps] if intermediate_steps else []
            )

            # 关键兜底：关键词路由已命中业务意图（recommend/search/order/cs/kg_qa...），
            # 但 LLM 未调用任何工具直接凭知识回答（幻觉）。
            # 此时强制调用路由对应的 Agent，确保回答基于真实商品数据。
            if not tools_called and intent not in ("chitchat",):
                agent_name = route_info["agent"]
                if agent_name in self.agents:
                    obs.logger.info(
                        "ReAct 未调用工具，强制走路由 Agent",
                        intent=intent,
                        agent=agent_name,
                    )
                    span.set_attribute("forced_agent_route", agent_name)
                    try:
                        response = self._run_agent_method(
                            self.agents[agent_name],
                            route_info.get("mode", "run"),
                            user_input,
                            user_profile=request_profile,
                        )
                        tools_called = [agent_name]
                    except Exception as e:
                        obs.logger.warning("强制路由 Agent 调用失败", error=str(e))

            span.set_attribute("tools_called", tools_called)
            span.set_attribute("response_length", len(response))
            span.set_attribute("elapsed_seconds", round(elapsed, 2))
            span.set_attribute("tool_call_count", len(tools_called))

            # 5. 记录到记忆
            self.memory.add_message(session_id, "user", user_input, intent=intent)
            self.memory.add_message(
                session_id, "assistant", response, intent=intent, tools=tools_called
            )

            # 6. 写入缓存 (非闲聊才缓存；个性化推荐不缓存)
            if intent not in ("chitchat", "recommend"):
                cost_optimizer.cache_set(intent, user_input, response)

            return response, {
                "intent": intent,
                "mode": "react",
                "agent": tools_called[0] if tools_called else route_info["agent"],
                "tools_called": tools_called,
                "steps": len(intermediate_steps),
                "elapsed_seconds": round(elapsed, 2),
            }

    def _run_agent_method(self, agent, mode: str, query: str, **kwargs) -> Optional[str]:
        """
        按 mode 分派到领域 Agent 的方法

        mode 由 Router 的 AGENT_MODE 定义（search/recommend/classify/
        query_order/analyze/run）。方法不存在时回退到 run()。
        """
        method = getattr(agent, mode, None)
        if callable(method):
            try:
                return method(query=query, **kwargs)
            except TypeError:
                # 某些方法不接受关键字参数（回退时丢弃额外参数）
                return method(query)
        # 回退：统一入口
        try:
            return agent.run(query=query, **kwargs)
        except TypeError:
            return agent.run(query)

    def _get_current_profile(self) -> dict:
        """读取当前请求绑定的用户画像（线程局部），无则返回空 dict"""
        return getattr(self._profile_store, "user_profile", None) or {}

    def _build_user_profile(self, session_id: str, query: str, explicit: Optional[dict]) -> dict:
        """
        构建当前请求的用户画像：
        显式画像（API 传入） + 长期记忆中的偏好/决策事实
        """
        return self.memory.build_user_profile(
            session_id=session_id,
            current_query=query,
            explicit=explicit,
        )

    def _build_chat_history(self, session_id: str, current_input: str) -> list:
        """
        从记忆系统构建聊天历史

        返回 LangChain Message 列表
        """
        context_messages = self.memory.build_context(session_id, current_input)

        history = []
        for msg in context_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                history.append(HumanMessage(content=content))
            elif role == "assistant":
                history.append(AIMessage(content=content))
            elif role == "system":
                history.append(SystemMessage(content=content))

        return history

    def _build_intent_hint(self, intent: str) -> Optional[str]:
        """
        根据关键词路由意图生成工具选择提示

        目的：关键词路由是确定性的（如"推荐"→recommend），而 LLM 工具选择
        是概率性的，可能误路由（如把推荐请求导去 search_products）。
        在 chat_history 中注入提示，引导 LLM 调用与路由一致的工具。
        """
        hints = {
            "recommend": (
                "【意图提示】用户请求被判定为「商品推荐」。"
                "必须优先调用 recommend_products 工具获取真实推荐结果，"
                "不要用 search_products 代替，也不要凭自身知识直接回答。"
            ),
            "search": (
                "【意图提示】用户请求被判定为「商品搜索」。"
                "请调用 search_products 工具搜索商品库获取真实商品。"
            ),
            "kg_qa": "【意图提示】用户请求与品牌/属性/分类关系相关，请调用 kg_qa 工具查询知识图谱。",
            "classify": "【意图提示】用户请求是商品分类，请调用 classify_product 工具。",
            "order": "【意图提示】用户请求涉及订单/物流，请调用 query_order 工具。",
            "customer_service": "【意图提示】用户请求涉及售后/退换货，请调用 customer_service 工具。",
            "analytics": "【意图提示】用户请求涉及数据分析/排行，请调用 data_analysis 工具。",
        }
        return hints.get(intent)

    async def ainvoke(
        self,
        user_input: str,
        session_id: str = "default",
        user_profile: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        """异步调用入口"""
        try:
            # 在线程池中执行同步的 agent.invoke
            result = await asyncio.to_thread(
                self._invoke_react, user_input, session_id, user_profile
            )
            return result
        except Exception as e:
            obs.logger.error("ReAct 异步执行失败，降级", error=str(e))
            fallback = self._get_fallback()
            response, state = await fallback.ainvoke(user_input)
            return response, {**state, "mode": "fallback", "error": str(e)}

    async def ainvoke_stream(
        self,
        user_input: str,
        session_id: str = "default",
        user_profile: Optional[dict] = None,
    ) -> AsyncGenerator[str, None]:
        """
        真 token 级流式调用入口

        实现：手动 function-calling 循环（llm.bind_tools + astream）：
        1. 模型调用时逐 token 输出内容；若返回工具调用则执行工具后继续循环
        2. 无工具调用时，该轮内容即最终回答，已逐 token 输出
        3. 最终回答只生成一次，且基于真实工具结果

        Yields:
            逐 token 的回答文本
        """
        try:
            with obs.trace_context("react_orchestrator_stream") as span:
                span.set_attribute("session_id", session_id)
                route_info = self.router.route(
                    user_input,
                    history=self.memory.get_history(session_id),
                )
                intent = route_info["intent"]
                span.set_attribute("intent", intent)

                # 缓存命中（个性化推荐不缓存）
                if intent != "recommend":
                    cached = cost_optimizer.cache_get(intent, user_input)
                    if cached:
                        span.set_attribute("cache_hit", True)
                        self.memory.add_message(session_id, "user", user_input, intent=intent)
                        self.memory.add_message(session_id, "assistant", cached, intent=intent)
                        for i in range(0, len(cached), 20):
                            yield cached[i : i + 20]
                        return

                request_profile = self._build_user_profile(
                    session_id, user_input, user_profile
                )
                self._profile_store.user_profile = request_profile

                chat_history = self._build_chat_history(session_id, user_input)
                intent_hint = self._build_intent_hint(intent)
                if intent_hint:
                    chat_history.insert(0, SystemMessage(content=intent_hint))
                messages = [
                    SystemMessage(content=self.SYSTEM_PROMPT),
                    *chat_history,
                    HumanMessage(content=user_input),
                ]

                # 按意图选择模型层级（接线三级模型分级）
                tier = cost_optimizer.model_router.get_tier(intent, len(user_input))
                stream_llm = cost_optimizer.get_llm(intent, len(user_input))
                model = stream_llm.bind_tools(self._tools)
                tools_called: list[str] = []
                call_history: list[tuple] = []
                final_text_parts: list[str] = []
                last_usage = None
                start_time = time.time()

                for step in range(settings.react_max_iterations):
                    if time.time() - start_time > 30:
                        break

                    content_chunks: list[str] = []
                    tool_call_chunks = []
                    # 流式调用模型：工具轮次内容为空（不输出），最终回答逐 token 输出
                    async for chunk in model.astream(messages):
                        if getattr(chunk, "content", None):
                            content_chunks.append(chunk.content)
                            yield chunk.content
                        usage = getattr(chunk, "response_metadata", {}).get("token_usage")
                        if usage:
                            last_usage = usage
                        tool_call_chunks.extend(
                            getattr(chunk, "tool_call_chunks", None) or []
                        )
                    content = "".join(content_chunks)
                    final_text_parts.append(content)

                    tool_calls = self._aggregate_tool_calls(tool_call_chunks)
                    if not tool_calls:
                        break  # 无工具调用：当前内容即最终回答（已逐 token 输出）

                    # 防死循环：同工具同参数连续超过 2 次即中断
                    valid_calls = []
                    for tc in tool_calls:
                        key = (tc["name"], json.dumps(tc.get("args", {}), ensure_ascii=False))
                        if call_history.count(key) >= 2:
                            yield "\n（检测到重复工具调用，已停止）"
                            valid_calls = []
                            break
                        call_history.append(key)
                        valid_calls.append(tc)
                    if not valid_calls:
                        break

                    messages.append(
                        AIMessage(content=content, tool_calls=valid_calls)
                    )
                    for tc in valid_calls:
                        tool = next(
                            (t for t in self._tools if t.name == tc["name"]), None
                        )
                        if tool is None:
                            messages.append(
                                ToolMessage(
                                    content=f"未知工具: {tc['name']}",
                                    tool_call_id=tc["id"],
                                )
                            )
                            continue
                        try:
                            result = tool.func(**tc.get("args", {}))
                        except TypeError:
                            result = tool.invoke(tc.get("args", {}))
                        except Exception as e:  # noqa: BLE001
                            result = f"工具执行失败: {e}"
                        tools_called.append(tc["name"])
                        messages.append(
                            ToolMessage(
                                content=str(result)[:2000],
                                tool_call_id=tc["id"],
                            )
                        )
                        self.memory.add_event(
                            session_id,
                            "tool_call",
                            {
                                "tool": tc["name"],
                                "tool_input": str(tc.get("args", {}))[:300],
                                "result": str(result)[:600],
                            },
                        )

                response = "".join(final_text_parts)
                if last_usage:
                    obs.record_tokens(
                        model=stream_llm.model_name,
                        prompt_tokens=int(last_usage.get("prompt_tokens", 0)),
                        completion_tokens=int(last_usage.get("completion_tokens", 0)),
                    )
                span.set_attribute("tools_called", tools_called)
                span.set_attribute("response_length", len(response))
                span.set_attribute("elapsed_seconds", round(time.time() - start_time, 2))

                # 记忆 + 缓存
                self.memory.add_message(session_id, "user", user_input, intent=intent)
                self.memory.add_message(
                    session_id,
                    "assistant",
                    response,
                    intent=intent,
                    tools=tools_called,
                )
                if intent not in ("chitchat", "recommend"):
                    cost_optimizer.cache_set(intent, user_input, response)

        except Exception as e:
            obs.logger.error("流式输出失败", error=str(e))
            yield f"抱歉，处理您的请求时出错: {e}"

    @staticmethod
    def _aggregate_tool_calls(tool_call_chunks: list) -> list:
        """把流式 tool_call_chunks 聚合为完整工具调用列表。"""
        calls: dict = {}
        for c in tool_call_chunks:
            idx = c.get("index", 0)
            entry = calls.setdefault(idx, {"name": "", "args": "", "id": ""})
            entry["name"] += c.get("name", "") or ""
            entry["args"] += c.get("args", "") or ""
            entry["id"] += c.get("id", "") or ""
        result = []
        for idx in sorted(calls):
            e = calls[idx]
            try:
                args = json.loads(e["args"]) if e["args"].strip() else {}
            except Exception:  # noqa: BLE001
                args = {}
            if not isinstance(args, dict):
                args = {"query": str(args)}
            result.append({"name": e["name"], "args": args, "id": e["id"]})
        return result

    def _invoke_react_tools_only(
        self,
        user_input: str,
        session_id: str,
        user_profile: Optional[dict],
    ) -> dict:
        """
        执行 ReAct 工具循环, 但不生成最终回答
        返回工具调用结果, 供流式生成最终回答使用

        如果 ReAct 失败或无需工具调用, 返回预生成的完整回答
        """
        try:
            # 先走正常的 ReAct 循环
            response, state = self._invoke_react(user_input, session_id, user_profile)
            return {
                "response_prefix": response,
                "state": state,
                "use_direct": True,  # 标记: 直接返回预生成回答
            }
        except Exception as e:
            return {
                "response_prefix": f"抱歉，处理您的请求时出错: {e}",
                "state": {"mode": "error"},
                "use_direct": True,
            }

    def _build_final_answer_prompt(self, user_input: str, tool_results: dict) -> str:
        """
        基于工具调用结果构建最终回答的 prompt
        用于 LLM 原生 stream 生成
        """
        if tool_results.get("use_direct"):
            # 如果已经有完整回答 (如缓存命中或 ReAct 已生成), 直接返回
            # 这种情况下流式没有意义, 但至少保证逻辑正确
            return tool_results["response_prefix"]

        # 正常情况下, 基于 ReAct 的 intermediate steps 构建回答 prompt
        response = tool_results.get("response_prefix", "")

        return response

    def get_memory_stats(self, session_id: str = None) -> dict:
        """获取记忆系统统计（可按会话过滤）"""
        stats = self.memory.get_stats()
        if session_id and isinstance(stats, dict):
            sessions = stats.get("sessions", {})
            if isinstance(sessions, dict):
                stats["sessions"] = {
                    k: v for k, v in sessions.items() if k == session_id
                }
        return stats

    def clear_session(self, session_id: str):
        """清除会话记忆"""
        self.memory.clear_session(session_id)
