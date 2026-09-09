"""
数据分析 Agent — 销售趋势、商品排行、用户行为分析

查询 MySQL (gmall) 业务数据，进行分析并生成报告
支持：销售趋势分析、商品销量排行、用户行为统计、品类分析

学习参考: Price Pilot 的 AnalyticsAgent
"""

import pymysql

from agents.tools.base import BaseAgentTool


class AnalyticsAgent(BaseAgentTool):
    """
    数据分析 Agent

    能力：
    - 销售趋势分析（日/周/月维度）
    - 商品销量 TOP N 排行
    - 品类销售占比分析
    - 用户行为统计

    数据来源：MySQL (gmall) 业务数据表
    """

    name: str = "analytics_agent"
    description: str = "数据分析：查询销售趋势、商品排行、品类占比等电商数据分析"

    def __init__(self, db_config: dict, llm=None, engine=None):
        super().__init__()
        self.db_config = db_config
        self.llm = llm
        self.engine = engine  # SQLAlchemy 连接池（可选）

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行数据分析"""
        analysis_type = kwargs.get("analysis_type")
        if not analysis_type:
            analysis_type = self._detect_analysis_type(query)
        return self.analyze(query, analysis_type, **kwargs)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心分析逻辑
    # ------------------------------------------------------------------ #

    def analyze(self, query: str, analysis_type: str, **kwargs) -> str:
        """
        执行数据分析

        Args:
            query: 用户原始查询
            analysis_type: 分析类型 (sales_trend / top_products / category_share / user_behavior)
        """
        conn = None
        try:
            # 优先使用连接池（raw_connection 返回 DBAPI 兼容连接）
            if self.engine:
                conn = self.engine.raw_connection()
            else:
                conn = pymysql.connect(**self.db_config)

            if analysis_type == "sales_trend":
                data = self._sales_trend(conn, kwargs.get("days", 30))
                return self._format_trend(data, query)
            elif analysis_type == "top_products":
                data = self._top_products(conn, kwargs.get("n", 10))
                return self._format_top(data, query)
            elif analysis_type == "category_share":
                data = self._category_share(conn)
                return self._format_category(data, query)
            elif analysis_type == "user_behavior":
                data = self._user_behavior(conn)
                return self._format_behavior(data, query)
            else:
                # 让 LLM 判断
                return self._llm_analyze(query, conn)

        except pymysql.Error as e:
            print(f"[AnalyticsAgent] 数据库错误: {e}")
            return f"数据分析服务暂时不可用: {e}"
        finally:
            if conn:
                conn.close()

    def _sales_trend(self, conn, days: int = 30) -> list:
        """销售趋势分析"""
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            # 尝试多种表结构
            sql_candidates = [
                # gmall 标准表结构
                """
                SELECT dt AS date, SUM(payment_amount) AS sales, COUNT(*) AS order_count
                FROM dwd_fact_payment_info
                WHERE dt >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                GROUP BY dt ORDER BY dt
                """,
                # 通用订单表
                """
                SELECT DATE(create_time) AS date, SUM(total_amount) AS sales, COUNT(*) AS order_count
                FROM order_info
                WHERE create_time >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
                GROUP BY DATE(create_time) ORDER BY date
                """,
            ]

            for sql in sql_candidates:
                try:
                    cursor.execute(sql, (days,))
                    return cursor.fetchall()
                except pymysql.Error:
                    continue

            return []

    def _top_products(self, conn, n: int = 10) -> list:
        """商品销量排行"""
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql_candidates = [
                """
                SELECT sku_name AS product, SUM(sku_num) AS quantity, SUM(order_amount) AS revenue
                FROM dwd_fact_order_detail
                GROUP BY sku_name ORDER BY quantity DESC LIMIT %s
                """,
                """
                SELECT product_name AS product, SUM(quantity) AS quantity, SUM(amount) AS revenue
                FROM order_detail
                GROUP BY product_name ORDER BY quantity DESC LIMIT %s
                """,
            ]

            for sql in sql_candidates:
                try:
                    cursor.execute(sql, (n,))
                    return cursor.fetchall()
                except pymysql.Error:
                    continue

            return []

    def _category_share(self, conn) -> list:
        """品类销售占比"""
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            sql_candidates = [
                """
                SELECT category3_name AS category, SUM(order_amount) AS revenue
                FROM dwd_fact_order_detail
                GROUP BY category3_name
                ORDER BY revenue DESC
                """,
                """
                SELECT c3.name AS category, SUM(od.quantity * od.price) AS revenue
                FROM order_detail od
                JOIN spu s ON od.spu_id = s.id
                JOIN category3 c3 ON s.category3_id = c3.id
                GROUP BY c3.name ORDER BY revenue DESC
                """,
            ]

            for sql in sql_candidates:
                try:
                    cursor.execute(sql)
                    return cursor.fetchall()
                except pymysql.Error:
                    continue

            return []

    def _user_behavior(self, conn) -> dict:
        """用户行为统计"""
        with conn.cursor(pymysql.cursors.DictCursor) as cursor:
            result = {}

            # 用户总数
            for sql in [
                "SELECT COUNT(*) AS total FROM user_info",
                "SELECT COUNT(*) AS total FROM users",
            ]:
                try:
                    cursor.execute(sql)
                    result["total_users"] = cursor.fetchone().get("total", 0)
                    break
                except pymysql.Error:
                    continue

            # 订单总数
            for sql in [
                "SELECT COUNT(*) AS total FROM order_info",
                "SELECT COUNT(*) AS total FROM orders",
            ]:
                try:
                    cursor.execute(sql)
                    result["total_orders"] = cursor.fetchone().get("total", 0)
                    break
                except pymysql.Error:
                    continue

            # 平均客单价
            for sql in [
                "SELECT ROUND(AVG(total_amount), 2) AS avg_amount FROM order_info",
                "SELECT ROUND(AVG(amount), 2) AS avg_amount FROM orders",
            ]:
                try:
                    cursor.execute(sql)
                    result["avg_order_value"] = cursor.fetchone().get("avg_amount", 0)
                    break
                except pymysql.Error:
                    continue

            return result

    # ------------------------------------------------------------------ #
    #  辅助方法
    # ------------------------------------------------------------------ #

    def _detect_analysis_type(self, query: str) -> str:
        """从用户查询中检测分析类型"""
        if any(kw in query for kw in ["趋势", "走势", "趋势图", "增长"]):
            return "sales_trend"
        elif any(kw in query for kw in ["排行", "排名", "top", "销量", "热销", "best"]):
            return "top_products"
        elif any(kw in query for kw in ["品类", "分类占比", "结构", "占比"]):
            return "category_share"
        elif any(kw in query for kw in ["用户", "行为", "客单", "活跃"]):
            return "user_behavior"
        return "auto"

    def _llm_analyze(self, query: str, conn) -> str:
        """LLM 辅助分析"""
        if not self.llm:
            return "暂时无法分析该类型的数据，请尝试查询销售趋势、商品排行或品类占比。"

        prompt = f"""你是一个电商数据分析师。用户想了解: {query}

请基于电商常见数据分析场景，给出分析建议和可能的结论。
注意：当前数据库查询可能不可用，请基于常识回答。"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except Exception:
            return "数据分析服务暂时不可用。"

    def _format_trend(self, data: list, query: str) -> str:
        """格式化销售趋势"""
        if not data:
            return "暂无销售趋势数据。数据库中可能不存在相关表。"

        lines = [f"📊 近 {len(data)} 天销售趋势\n" + "=" * 40]
        total_sales = 0
        total_orders = 0
        for row in data:
            date = row.get("date", "")
            sales = float(row.get("sales", 0))
            orders = int(row.get("order_count", 0))
            total_sales += sales
            total_orders += orders
            lines.append(f"  {date}: 销售额 ¥{sales:.2f} | 订单 {orders}")

        lines.append("=" * 40)
        lines.append(f"总销售额: ¥{total_sales:.2f}")
        lines.append(f"总订单数: {total_orders}")
        lines.append(f"日均销售额: ¥{total_sales / len(data):.2f}")
        return "\n".join(lines)

    def _format_top(self, data: list, query: str) -> str:
        """格式化商品排行"""
        if not data:
            return "暂无商品排行数据。"

        lines = [f"🏆 商品销量 TOP {len(data)}\n" + "=" * 40]
        for i, row in enumerate(data, 1):
            product = row.get("product", "未知商品")
            quantity = row.get("quantity", 0)
            revenue = float(row.get("revenue", 0))
            lines.append(f"  {i}. {product}")
            lines.append(f"     销量: {quantity} | 销售额: ¥{revenue:.2f}")
        lines.append("=" * 40)
        return "\n".join(lines)

    def _format_category(self, data: list, query: str) -> str:
        """格式化品类占比"""
        if not data:
            return "暂无品类数据。"

        total = sum(float(row.get("revenue", 0)) for row in data)
        if total == 0:
            return "暂无品类销售数据。"

        lines = ["📊 品类销售占比\n" + "=" * 40]
        for i, row in enumerate(data, 1):
            category = row.get("category", "未知")
            revenue = float(row.get("revenue", 0))
            share = revenue / total * 100
            bar = "█" * int(share / 5)  # 简易柱状图
            lines.append(f"  {i}. {category}")
            lines.append(f"     销售额: ¥{revenue:.2f} ({share:.1f}%) {bar}")
        lines.append("=" * 40)
        lines.append(f"总销售额: ¥{total:.2f}")
        return "\n".join(lines)

    def _format_behavior(self, data: dict, query: str) -> str:
        """格式化用户行为"""
        if not data:
            return "暂无用户行为数据。"

        lines = ["📊 用户行为统计\n" + "=" * 40]
        lines.append(f"  总用户数: {data.get('total_users', 'N/A')}")
        lines.append(f"  总订单数: {data.get('total_orders', 'N/A')}")
        lines.append(f"  平均客单价: ¥{data.get('avg_order_value', 'N/A')}")
        lines.append("=" * 40)
        return "\n".join(lines)
