"""
业务数据域 Agent — 统一封装订单查询与数据分析两个能力

精简背景：order/analytics 两个原 Agent 共享 MySQL 业务数据源，
合并为 1 个「业务数据域」Agent，对外暴露 2 个职责清晰的工具方法。
"""

from agents.tools.base import BaseAgentTool


class BusinessDataAgent(BaseAgentTool):
    """
    业务数据域 Agent

    能力：
    - query_order:  订单状态 / 物流查询（MySQL）
    - analyze:      销售趋势 / 商品排行 / 品类占比分析（MySQL）
    """

    name: str = "business_data_agent"
    description: str = (
        "业务数据域 Agent：统一提供订单/物流查询与销售数据分析两个能力"
    )

    def __init__(self, order_agent, analytics_agent):
        super().__init__()
        self.order_agent = order_agent
        self.analytics_agent = analytics_agent

    # ------------------------------------------------------------------ #
    #  工具方法（ReAct Tool 直接绑定）
    # ------------------------------------------------------------------ #

    def query_order(self, query: str, **kwargs) -> str:
        """订单状态与物流追踪（MySQL 订单/物流表）"""
        # 透传 user_id（画像中确有 user_id 时才校验归属，避免空值干扰）
        profile_uid = (kwargs.get("user_profile") or {}).get("user_id")
        if profile_uid and not kwargs.get("user_id"):
            kwargs["user_id"] = profile_uid
        return self.order_agent.run(query=query, **kwargs)

    def analyze(self, query: str, **kwargs) -> str:
        """销售数据分析（趋势 / 排行 / 品类占比 / 用户行为）"""
        return self.analytics_agent.run(query=query, **kwargs)

    # ------------------------------------------------------------------ #
    #  统一入口（降级路由按 mode 分派）
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        mode = kwargs.get("mode", "query_order")
        if mode == "analyze":
            return self.analyze(query, **kwargs)
        return self.query_order(query, **kwargs)

    async def arun(self, **kwargs) -> str:
        """异步执行（当前复用同步逻辑）"""
        return self.run(**kwargs)
