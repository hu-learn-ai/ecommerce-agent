"""
订单查询 Agent — 订单状态与物流追踪

查询 MySQL (gmall) 中的订单数据
支持订单状态查询、物流追踪、订单详情

学习参考: Price Pilot 的 OrderAgent
          MultiAgent-Ecom 的 order-agent-service
"""

from typing import Optional

import pymysql

from agents.tools.base import BaseAgentTool


class OrderAgent(BaseAgentTool):
    """
    订单查询与物流追踪

    数据来源: MySQL (gmall) 的订单相关表
    支持按订单号查询订单状态和物流信息
    """

    name: str = "order_agent"
    description: str = "订单查询：查询订单状态、物流追踪、订单详情信息"

    def __init__(self, db_config: dict, engine=None):
        super().__init__()
        self.db_config = db_config
        self.engine = engine  # SQLAlchemy 连接池（可选）

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行订单查询"""
        order_id = kwargs.get("order_id")
        user_id = kwargs.get("user_id") or (kwargs.get("user_profile") or {}).get("user_id")
        if not order_id:
            # 从用户输入中提取订单号
            order_id = self._extract_order_id(query)
        if not order_id:
            return "请提供您的订单号，格式如: ORD123456 或 纯数字订单号"
        return self.query_order(order_id, user_id=user_id)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心查询逻辑
    # ------------------------------------------------------------------ #

    def query_order(self, order_id: str, user_id: str = None) -> str:
        """
        查询订单状态

        Args:
            order_id: 订单号
            user_id: 当前用户 ID（提供时校验订单归属，防越权）

        Returns:
            格式化的订单信息字符串
        """
        conn = None
        try:
            # 优先使用连接池
            if self.engine:
                conn = self.engine.connect()
                # 使用 SQLAlchemy 执行（需要用 text() 包装 SQL）
                order = self._fetch_order_sa(conn, order_id)
                if not order:
                    return f"未找到订单 {order_id}，请确认订单号是否正确。"
                if user_id and str(order.get("user_id", "")) != str(user_id):
                    # 与"未找到订单"返回相同措辞，避免泄露"订单存在但非本人"（防订单号枚举）
                    return f"未找到订单 {order_id}，请确认订单号是否正确。"
                logistics = self._fetch_logistics_sa(conn, order_id)
                return self._format_order(
                    order, logistics, show_pii=bool(user_id and str(order.get("user_id", "")) == str(user_id))
                )
            else:
                # 降级：每次新建连接
                conn = pymysql.connect(**self.db_config)
                with conn.cursor(pymysql.cursors.DictCursor) as cursor:
                    order = self._fetch_order(cursor, order_id)
                    if not order:
                        return f"未找到订单 {order_id}，请确认订单号是否正确。"
                    if user_id and str(order.get("user_id", "")) != str(user_id):
                        # 与"未找到订单"返回相同措辞，避免泄露"订单存在但非本人"（防订单号枚举）
                        return f"未找到订单 {order_id}，请确认订单号是否正确。"
                    logistics = self._fetch_logistics(cursor, order_id)
                    return self._format_order(
                        order,
                        logistics,
                        show_pii=bool(user_id and str(order.get("user_id", "")) == str(user_id)),
                    )

        except pymysql.Error as e:
            print(f"[OrderAgent] 数据库查询失败: {e}")
            return "订单查询服务暂时不可用，请稍后重试。"
        except Exception as e:
            print(f"[OrderAgent] 查询异常: {e}")
            return f"查询失败: {e}"
        finally:
            if conn:
                conn.close()

    # ------------------------------------------------------------------ #
    #  SQLAlchemy 连接池版本（优先使用）
    # ------------------------------------------------------------------ #

    def _fetch_order_sa(self, conn, order_id: str) -> Optional[dict]:
        """使用 SQLAlchemy 连接池查询订单"""
        from sqlalchemy import text as sa_text

        table_candidates = ["order_info", "orders", "order", "ods_order_info"]
        for table in table_candidates:
            try:
                result = conn.execute(
                    sa_text(f"SELECT * FROM {table} WHERE order_id = :oid OR id = :oid LIMIT 1"),
                    {"oid": order_id},
                )
                row = result.mappings().first()
                if row:
                    return dict(row)
            except Exception:
                continue
        return None

    def _fetch_logistics_sa(self, conn, order_id: str) -> Optional[dict]:
        """使用 SQLAlchemy 连接池查询物流"""
        from sqlalchemy import text as sa_text

        table_candidates = ["logistics_info", "shipment", "express_info"]
        for table in table_candidates:
            try:
                result = conn.execute(
                    sa_text(f"SELECT * FROM {table} WHERE order_id = :oid LIMIT 1"),
                    {"oid": order_id},
                )
                row = result.mappings().first()
                if row:
                    return dict(row)
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------ #
    #  原始 PyMySQL 版本（降级时使用）
    # ------------------------------------------------------------------ #

    def _fetch_order(self, cursor, order_id: str) -> Optional[dict]:
        """
        查询订单主表

        兼容多种表结构：order_info / orders / order
        """
        # 尝试常见的订单表名
        table_candidates = ["order_info", "orders", "order", "ods_order_info"]

        for table in table_candidates:
            try:
                # 先检查表是否存在
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_name = %s",
                    (self.db_config.get("database", ""), table),
                )
                if not cursor.fetchone():
                    continue

                # 安全: 表名来自上方硬编码白名单 table_candidates，无注入风险
                # order_id 使用参数化查询 (%s) 防止 SQL 注入
                cursor.execute(
                    f"SELECT * FROM {table} WHERE order_id = %s OR id = %s LIMIT 1",
                    (order_id, order_id),
                )
                order = cursor.fetchone()
                if order:
                    return order
            except pymysql.Error:
                continue

        return None

    def _fetch_logistics(self, cursor, order_id: str) -> Optional[dict]:
        """查询物流信息"""
        table_candidates = ["logistics_info", "shipment", "express_info"]

        for table in table_candidates:
            try:
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = %s AND table_name = %s",
                    (self.db_config.get("database", ""), table),
                )
                if not cursor.fetchone():
                    continue

                # 安全: 表名来自上方硬编码白名单 table_candidates，无注入风险
                cursor.execute(
                    f"SELECT * FROM {table} WHERE order_id = %s LIMIT 1",
                    (order_id,),
                )
                logistics = cursor.fetchone()
                if logistics:
                    return logistics
            except pymysql.Error:
                continue

        return None

    def _format_order(self, order: dict, logistics: Optional[dict], show_pii: bool = False) -> str:
        """格式化订单信息为可读字符串（默认隐藏收货人/地址等 PII，防越权泄露）"""
        lines = ["📦 订单信息\n" + "=" * 40]

        # 订单基本信息（字段名可能不同，兼容处理）
        order_id = order.get("order_id") or order.get("id") or "未知"
        status = self._map_order_status(
            order.get("order_status") or order.get("status") or order.get("state")
        )
        total_amount = (
            order.get("total_amount") or order.get("amount") or order.get("price") or "未知"
        )
        create_time = (
            order.get("create_time") or order.get("created_at") or order.get("order_time") or "未知"
        )
        user_id = order.get("user_id") or order.get("uid") or "未知"
        # 用户 ID 同属 PII：未获授权（show_pii=False）时同样脱敏，防止横向越权枚举
        user_id_display = (
            user_id if show_pii or user_id == "未知" else self._mask_pii(str(user_id))
        )

        lines.append(f"订单号: {order_id}")
        lines.append(f"状态: {status}")
        lines.append(f"金额: ¥{total_amount}")
        lines.append(f"下单时间: {create_time}")
        lines.append(f"用户ID: {user_id_display}")

        # 收货人/地址（PII，默认打码；show_pii=True 时才明文）
        address = order.get("consignee") or order.get("receiver_address") or order.get("address")
        if address:
            lines.append(f"收货地址: {address if show_pii else self._mask_pii(address)}")
        consignee = order.get("consignee_name") or order.get("receiver") or ""
        if consignee and not show_pii:
            lines.append(f"收货人: {self._mask_pii(str(consignee))}")

        # 物流信息
        if logistics:
            lines.append("\n🚚 物流信息")
            lines.append("-" * 40)
            tracking_no = logistics.get("tracking_no") or logistics.get("express_no") or "未知"
            company = logistics.get("company") or logistics.get("express_company") or "未知"
            logistics_status = (
                logistics.get("status") or logistics.get("logistics_status") or "未知"
            )

            lines.append(f"快递公司: {company}")
            lines.append(f"运单号: {tracking_no}")
            lines.append(f"物流状态: {logistics_status}")
        else:
            lines.append("\n🚚 物流信息暂未更新")

        lines.append("=" * 40)
        return "\n".join(lines)

    @staticmethod
    def _mask_pii(text: str) -> str:
        """PII 脱敏：保留前 2 个字符，其余打星。"""
        text = str(text or "")
        if len(text) <= 2:
            return "*" * len(text)
        return text[:2] + "*" * min(len(text) - 2, 6)

    def _map_order_status(self, status_code) -> str:
        """将订单状态码映射为可读文字

        支持两种格式:
        - 数字状态码 (1-7): import_taobao_data.py STATUS_MAP 定义的编码
        - 中文文本: 如 "待付款"、"已付款" (直接存储在 order_info.order_status 中)
        """
        if status_code is None:
            return "未知"

        # 数字状态码映射
        status_map = {
            "1": "待付款",
            "2": "已付款",
            "3": "已发货",
            "4": "已完成",
            "5": "已取消",
            "6": "退款中",
            "7": "已退款",
            1: "待付款",
            2: "已付款",
            3: "已发货",
            4: "已完成",
            5: "已取消",
            6: "退款中",
            7: "已退款",
        }

        # 中文文本直接返回 (DB 中 order_status 存储的就是中文)
        if isinstance(status_code, str):
            # 先尝试数字字符串映射
            if status_code.lower() in status_map:
                return status_map[status_code.lower()]
            # 已经是中文文本，直接返回
            if status_code in status_map.values():
                return status_code
            # 其他文本原样返回
            return status_code

        if status_code in status_map:
            return status_map[status_code]

        return str(status_code)

    def _extract_order_id(self, text: str) -> Optional[str]:
        """
        从用户输入文本中提取订单号

        支持：
        - "查询订单 ORD123456"
        - "订单号 123456789"
        - 纯数字订单号
        """
        import re

        # 匹配 "订单" + 字母数字组合
        patterns = [
            r"(?:订单号?|order[_ ]?id|查(?:询)?)[\s:：]*([A-Za-z0-9]{4,20})",
            r"\b([A-Z]{1,4}\d{4,16})\b",  # ORD123456 / O00000001 格式
            r"\b(\d{8,20})\b",  # 纯数字订单号
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)

        return None
