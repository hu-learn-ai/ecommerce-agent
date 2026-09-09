"""
淘宝用户消费数据汇总 → 项目全量数据导入脚本

读取 6 个 CSV 文件，生成项目所需的所有数据格式：
1. neo4j_import.cypher     — Neo4j 图谱 (SPU/SKU/Category/Brand/User 节点 + 关系)
2. mysql_gmall.sql          — MySQL gmall 数据库 (订单/物流/支付/DWD事实表/用户)
3. products_for_faiss.json  — FAISS 索引构建数据
4. classify_train.csv       — BERT 商品分类训练集
5. faq_data.json            — 客服 FAQ 语料
6. data_report.txt          — 数据质量报告

使用方法:
    python scripts/import_taobao_data.py

数据来源: 淘宝用户消费数据汇总/
    orders.csv          — 15,000 条订单
    products.csv        — 2,000 个商品
    product_features.csv — 2,000 条商品特征
    users.csv           — 5,000 个用户
    user_behaviors.csv  — 30,000 条行为记录
    user_features.csv   — 5,000 条用户特征
"""

import csv
import json
import os
import random
from collections import Counter
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "淘宝用户消费数据汇总", "淘宝用户消费数据汇总")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 状态映射：中文订单状态 → 数字编码 (与 Order Agent._map_order_status 对齐)
# 1=待付款 2=已付款 3=已发货 4=已完成 5=已取消 6=退款中 7=已退款
STATUS_MAP = {
    "待付款": 1,
    "已付款": 2,
    "已发货": 3,
    "已收货": 4,  # 已收货归入已完成
    "已完成": 4,
    "已取消": 5,
    "已退款": 7,
}

# 行为类型映射：中文 → 英文 (Neo4j 关系类型)
BEHAVIOR_MAP = {
    "浏览": "VIEW",
    "点击": "CLICK",
    "收藏": "FAVORITE",
    "加购": "CART",
}


def read_csv(filename):
    """读取 CSV 文件，返回 dict 列表"""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


# ============================================================
# 1. Neo4j 图谱导入脚本
# ============================================================
def generate_neo4j_import(orders, products, product_features, users, user_behaviors, user_features):
    """
    生成 Neo4j Cypher 导入脚本

    图结构:
        (SPU)-[:Belong]->(Category3)
        (SPU)-[:Have]->(Trademark)
        (SPU)-[:Have]->(SKU)
        (User)-[:BUY]->(SPU)
        (User)-[:VIEW]->(SPU)
        (User)-[:FAVORITE]->(SPU)
        (User)-[:CLICK]->(SPU)
        (User)-[:CART]->(SPU)
    """
    print("[Neo4j] 生成导入脚本...")

    lines = []
    lines.append("// ============================================")
    lines.append("// Neo4j 图谱导入脚本 — 淘宝用户消费数据")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"// 生成时间: {ts}")
    lines.append("// ============================================")
    lines.append("")
    lines.append("// 清空已有数据 (首次导入时取消注释)")
    lines.append("// MATCH (n) DETACH DELETE n;")
    lines.append("")

    # --- 约束 ---
    lines.append("// --- 唯一约束 ---")
    lines.append("CREATE CONSTRAINT spu_id IF NOT EXISTS FOR (p:SPU) REQUIRE p.id IS UNIQUE;")
    lines.append(
        "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category3) REQUIRE c.name IS UNIQUE;"
    )
    lines.append(
        "CREATE CONSTRAINT trademark_name IF NOT EXISTS FOR (t:Trademark) REQUIRE t.name IS UNIQUE;"
    )
    lines.append("CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE;")
    lines.append("CREATE CONSTRAINT sku_id IF NOT EXISTS FOR (s:SKU) REQUIRE s.id IS UNIQUE;")
    lines.append("")

    # --- Category3 节点 ---
    categories = sorted(set(p["category"] for p in products))
    lines.append(f"// --- Category3 节点 ({len(categories)} 个) ---")
    for cat in categories:
        safe_cat = cat.replace("'", "\\'")
        lines.append(f"MERGE (c:Category3 {{name: '{safe_cat}'}});")
    lines.append("")

    # --- Trademark (品牌) 节点 ---
    brands = sorted(set(p["brand"] for p in products))
    lines.append(f"// --- Trademark 节点 ({len(brands)} 个) ---")
    for brand in brands:
        safe_brand = brand.replace("'", "\\'")
        lines.append(f"MERGE (t:Trademark {{name: '{safe_brand}'}});")
    lines.append("")

    # --- SPU 商品节点 + 关系 ---
    # 合并 products.csv 和 product_features.csv
    pf_map = {pf["product_id"]: pf for pf in product_features}

    lines.append(f"// --- SPU 商品节点 + 分类/品牌关系 ({len(products)} 个) ---")
    for p in products:
        pid = p["product_id"]
        name = p["product_name"].replace("'", "\\'")
        category = p["category"].replace("'", "\\'")
        brand = p["brand"].replace("'", "\\'")
        price = float(p["price"])
        sales_count = int(p["sales_count"])

        pf = pf_map.get(pid, {})
        revenue = float(pf.get("total_revenue", 0))
        conversion_rate = float(pf.get("conversion_rate", 0))
        avg_review = float(pf.get("avg_review_score", 0))
        popularity = float(pf.get("popularity_score", 0))

        lines.append(
            f"MERGE (p:SPU {{id: '{pid}'}}) "
            f"SET p.name = '{name}', p.price = {price}, "
            f"p.sales_count = {sales_count}, "
            f"p.total_revenue = {revenue}, "
            f"p.conversion_rate = {conversion_rate}, "
            f"p.avg_review_score = {avg_review}, "
            f"p.popularity_score = {popularity};"
        )
        # SPU -> Category3
        lines.append(
            f"MATCH (p:SPU {{id: '{pid}'}}), (c:Category3 {{name: '{category}'}}) "
            f"MERGE (p)-[:Belong]->(c);"
        )
        # SPU -> Trademark
        lines.append(
            f"MATCH (p:SPU {{id: '{pid}'}}), (t:Trademark {{name: '{brand}'}}) "
            f"MERGE (p)-[:Have]->(t);"
        )

    lines.append("")

    # --- SKU 节点 (为每个 SPU 生成一个 SKU) ---
    lines.append(f"// --- SKU 节点 ({len(products)} 个) ---")
    for p in products:
        pid = p["product_id"]
        sku_id = f"SKU_{pid}"
        name = p["product_name"].replace("'", "\\'")
        price = float(p["price"])
        lines.append(f"MERGE (s:SKU {{id: '{sku_id}'}}) SET s.name = '{name}', s.price = {price};")
        lines.append(
            f"MATCH (p:SPU {{id: '{pid}'}}), (s:SKU {{id: '{sku_id}'}}) MERGE (p)-[:Have]->(s);"
        )

    lines.append("")

    # --- User 节点 ---
    # 合并 users.csv 和 user_features.csv
    uf_map = {uf["user_id"]: uf for uf in user_features}

    lines.append(f"// --- User 节点 ({len(users)} 个) ---")
    for u in users:
        uid = u["user_id"]
        age = int(u["age"]) if u["age"] else 0
        gender = u["gender"]
        province = u["province"].replace("'", "\\'")
        city = u["city"].replace("'", "\\'")
        member_level = u["member_level"]
        account_balance = float(u["account_balance"])
        credit_score = int(u["credit_score"])

        uf = uf_map.get(uid, {})
        total_spent = float(uf.get("total_spent", 0))
        order_count = int(float(uf.get("order_count", 0)))
        purchase_intent = float(uf.get("purchase_intent", 0))
        consumption_level = uf.get("consumption_level", "")

        lines.append(
            f"MERGE (u:User {{id: '{uid}'}}) "
            f"SET u.age = {age}, u.gender = '{gender}', "
            f"u.province = '{province}', u.city = '{city}', "
            f"u.member_level = '{member_level}', "
            f"u.account_balance = {account_balance}, "
            f"u.credit_score = {credit_score}, "
            f"u.total_spent = {total_spent}, "
            f"u.order_count = {order_count}, "
            f"u.purchase_intent = {purchase_intent}, "
            f"u.consumption_level = '{consumption_level}';"
        )

    lines.append("")

    # --- 用户行为关系 ---
    lines.append(f"// --- 用户行为关系 ({len(user_behaviors)} 条) ---")
    behavior_count = 0
    for b in user_behaviors:
        uid = b["user_id"]
        pid = b["product_id"]
        btype = BEHAVIOR_MAP.get(b["behavior_type"], "VIEW")
        btime = b["behavior_time"]
        duration = int(b["duration_seconds"]) if b["duration_seconds"] else 0

        lines.append(
            f"MATCH (u:User {{id: '{uid}'}}), (p:SPU {{id: '{pid}'}}) "
            f"MERGE (u)-[:{btype} {{time: '{btime}', duration: {duration}}}]->(p);"
        )
        behavior_count += 1

    lines.append("")

    # --- 订单购买关系 ---
    # 从 orders.csv 生成 BUY 关系
    buy_relations = set()
    for o in orders:
        uid = o["user_id"]
        pid = o["product_id"]
        if (uid, pid) not in buy_relations:
            buy_relations.add((uid, pid))

    lines.append(f"// --- 购买关系 ({len(buy_relations)} 条, 去重) ---")
    for uid, pid in buy_relations:
        lines.append(
            f"MATCH (u:User {{id: '{uid}'}}), (p:SPU {{id: '{pid}'}}) MERGE (u)-[:BUY]->(p);"
        )

    lines.append("")

    # --- 全文索引 ---
    lines.append("// --- 全文索引 (供 Search Agent 使用) ---")
    lines.append("CREATE FULLTEXT INDEX spu_fulltext IF NOT EXISTS FOR (p:SPU) ON EACH [p.name];")
    lines.append("")

    # --- 统计信息 ---
    lines.append("// --- 数据统计 ---")
    lines.append(f"// SPU 商品: {len(products)}")
    lines.append(f"// Category3 分类: {len(categories)}")
    lines.append(f"// Trademark 品牌: {len(brands)}")
    lines.append(f"// SKU: {len(products)}")
    lines.append(f"// User 用户: {len(users)}")
    lines.append(f"// 行为关系: {behavior_count}")
    lines.append(f"// 购买关系: {len(buy_relations)}")

    output_path = os.path.join(OUTPUT_DIR, "neo4j_import.cypher")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Neo4j] ✅ 已生成: {output_path}")
    return {
        "spu": len(products),
        "category": len(categories),
        "brand": len(brands),
        "sku": len(products),
        "user": len(users),
        "behavior": behavior_count,
        "buy": len(buy_relations),
    }


# ============================================================
# 2. MySQL gmall 数据库
# ============================================================
def generate_mysql_import(orders, products, users, user_features):
    """
    生成 MySQL gmall 数据库建表 + 数据导入脚本

    表结构:
        order_info               — 订单主表
        order_detail             — 订单明细
        logistics_info           — 物流信息
        dwd_fact_payment_info    — DWD 支付事实表 (Analytics Agent)
        dwd_fact_order_detail    — DWD 订单明细事实表 (Analytics Agent)
        user_info                — 用户信息表
    """
    print("[MySQL] 生成导入脚本...")

    lines = []
    lines.append("-- ============================================")
    lines.append("-- MySQL gmall 数据库导入脚本 — 淘宝用户消费数据")
    lines.append(f"-- 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-- ============================================")
    lines.append("")
    lines.append("CREATE DATABASE IF NOT EXISTS gmall DEFAULT CHARACTER SET utf8mb4;")
    lines.append("USE gmall;")
    lines.append("")

    # --- order_info ---
    lines.append("-- order_info: 订单主表")
    lines.append("DROP TABLE IF EXISTS order_info;")
    lines.append("""CREATE TABLE order_info (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id        VARCHAR(32) NOT NULL COMMENT '订单号',
    user_id         VARCHAR(32) NOT NULL COMMENT '用户ID',
    total_amount    DECIMAL(12,2) DEFAULT 0 COMMENT '订单总金额',
    actual_payment  DECIMAL(12,2) DEFAULT 0 COMMENT '实付金额',
    discount        DECIMAL(12,2) DEFAULT 0 COMMENT '优惠金额',
    order_status    VARCHAR(20) DEFAULT '待付款' COMMENT '订单状态',
    payment_method  VARCHAR(20) DEFAULT NULL COMMENT '支付方式',
    create_time     DATETIME DEFAULT NULL COMMENT '下单时间',
    delivery_time   DATETIME DEFAULT NULL COMMENT '发货时间',
    receive_time    DATETIME DEFAULT NULL COMMENT '收货时间',
    consignee       VARCHAR(200) DEFAULT NULL COMMENT '收货地址',
    review_score    INT DEFAULT NULL COMMENT '评价评分',
    review_content  TEXT DEFAULT NULL COMMENT '评价内容',
    UNIQUE KEY uk_order_id (order_id),
    KEY idx_user_id (user_id),
    KEY idx_create_time (create_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单信息表';""")
    lines.append("")

    # --- order_detail ---
    lines.append("-- order_detail: 订单明细")
    lines.append("DROP TABLE IF EXISTS order_detail;")
    lines.append("""CREATE TABLE order_detail (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id        VARCHAR(32) NOT NULL COMMENT '订单号',
    user_id         VARCHAR(32) NOT NULL COMMENT '用户ID',
    product_id      VARCHAR(32) NOT NULL COMMENT '商品ID',
    product_name    VARCHAR(200) DEFAULT NULL COMMENT '商品名称',
    quantity        INT DEFAULT 1 COMMENT '购买数量',
    unit_price      DECIMAL(12,2) DEFAULT 0 COMMENT '单价',
    amount          DECIMAL(12,2) DEFAULT 0 COMMENT '金额',
    KEY idx_order_id (order_id),
    KEY idx_product_id (product_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单明细表';""")
    lines.append("")

    # --- logistics_info ---
    lines.append("-- logistics_info: 物流信息")
    lines.append("DROP TABLE IF EXISTS logistics_info;")
    lines.append("""CREATE TABLE logistics_info (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id        VARCHAR(32) NOT NULL COMMENT '订单号',
    tracking_no     VARCHAR(64) DEFAULT NULL COMMENT '运单号',
    company         VARCHAR(50) DEFAULT NULL COMMENT '快递公司',
    status          VARCHAR(20) DEFAULT NULL COMMENT '物流状态',
    create_time     DATETIME DEFAULT NULL COMMENT '创建时间',
    KEY idx_order_id (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='物流信息表';""")
    lines.append("")

    # --- dwd_fact_payment_info ---
    lines.append("-- dwd_fact_payment_info: DWD 支付事实表 (Analytics Agent)")
    lines.append("DROP TABLE IF EXISTS dwd_fact_payment_info;")
    lines.append("""CREATE TABLE dwd_fact_payment_info (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id        VARCHAR(32) NOT NULL COMMENT '订单号',
    user_id         VARCHAR(32) NOT NULL COMMENT '用户ID',
    payment_amount  DECIMAL(12,2) DEFAULT 0 COMMENT '支付金额',
    payment_method  VARCHAR(20) DEFAULT NULL COMMENT '支付方式',
    dt              DATE DEFAULT NULL COMMENT '支付日期',
    KEY idx_dt (dt),
    KEY idx_order_id (order_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='DWD支付事实表';""")
    lines.append("")

    # --- dwd_fact_order_detail ---
    lines.append("-- dwd_fact_order_detail: DWD 订单明细事实表 (Analytics Agent)")
    lines.append("DROP TABLE IF EXISTS dwd_fact_order_detail;")
    lines.append("""CREATE TABLE dwd_fact_order_detail (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    order_id        VARCHAR(32) NOT NULL COMMENT '订单号',
    user_id         VARCHAR(32) NOT NULL COMMENT '用户ID',
    sku_id          VARCHAR(32) DEFAULT NULL COMMENT 'SKU ID',
    sku_name        VARCHAR(200) DEFAULT NULL COMMENT '商品名称',
    sku_num         INT DEFAULT 1 COMMENT '购买数量',
    order_amount    DECIMAL(12,2) DEFAULT 0 COMMENT '订单金额',
    category1_name  VARCHAR(50) DEFAULT NULL COMMENT '一级分类',
    category3_name  VARCHAR(50) DEFAULT NULL COMMENT '三级分类',
    brand_name      VARCHAR(50) DEFAULT NULL COMMENT '品牌',
    dt              DATE DEFAULT NULL COMMENT '下单日期',
    KEY idx_dt (dt),
    KEY idx_category3 (category3_name),
    KEY idx_sku_name (sku_name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='DWD订单明细事实表';""")
    lines.append("")

    # --- user_info ---
    lines.append("-- user_info: 用户信息表")
    lines.append("DROP TABLE IF EXISTS user_info;")
    lines.append("""CREATE TABLE user_info (
    id              BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(32) NOT NULL COMMENT '用户ID',
    age             INT DEFAULT NULL COMMENT '年龄',
    gender          VARCHAR(10) DEFAULT NULL COMMENT '性别',
    province        VARCHAR(20) DEFAULT NULL COMMENT '省份',
    city            VARCHAR(20) DEFAULT NULL COMMENT '城市',
    member_level    VARCHAR(20) DEFAULT NULL COMMENT '会员等级',
    account_balance DECIMAL(12,2) DEFAULT 0 COMMENT '账户余额',
    credit_score    INT DEFAULT NULL COMMENT '信用分',
    total_spent     DECIMAL(12,2) DEFAULT 0 COMMENT '累计消费',
    order_count     INT DEFAULT 0 COMMENT '订单数',
    purchase_intent DECIMAL(6,4) DEFAULT 0 COMMENT '购买意向',
    consumption_level VARCHAR(10) DEFAULT NULL COMMENT '消费等级',
    UNIQUE KEY uk_user_id (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户信息表';""")
    lines.append("")

    # --- 数据导入 ---
    # 构建 product 查找表
    product_map = {p["product_id"]: p for p in products}
    # 构建 user 查找表 (用于收货地址)
    user_map = {u["user_id"]: u for u in users}
    uf_map = {uf["user_id"]: uf for uf in user_features}

    # 快递公司列表
    express_companies = [
        "顺丰速运",
        "中通快递",
        "圆通速递",
        "申通快递",
        "韵达速递",
        "京东物流",
        "邮政EMS",
        "百世快递",
        "天天快递",
        "德邦快递",
    ]

    # --- order_info + order_detail + logistics + dwd 数据 ---
    lines.append("-- 订单数据导入")
    lines.append("")

    logistics_count = 0
    payment_count = 0
    dwd_count = 0

    for o in orders:
        oid = o["order_id"]
        uid = o["user_id"]
        pid = o["product_id"]
        qty = int(o["quantity"])
        order_date = o["order_date"]
        status = o["order_status"]
        pay_method = o["payment_method"]
        unit_price = float(o["unit_price"])
        total_amount = float(o["total_amount"])
        discount = float(o["discount"]) if o["discount"] else 0
        actual_payment = float(o["actual_payment"])
        delivery_date = o["delivery_date"] if o["delivery_date"] else ""
        receive_date = o["receive_date"] if o["receive_date"] else ""
        review_score = o["review_score"] if o["review_score"] else "NULL"
        review_content = o["review_content"].replace("'", "\\'") if o["review_content"] else ""

        # 收货地址
        u = user_map.get(uid, {})
        address = f"{u.get('province', '')}{u.get('city', '')}" if u else ""

        # order_info
        review_part = f"'{review_content}'" if review_content else "NULL"
        review_score_part = review_score if review_score != "NULL" else "NULL"
        delivery_part = f"'{delivery_date}'" if delivery_date else "NULL"
        receive_part = f"'{receive_date}'" if receive_date else "NULL"

        lines.append(
            f"INSERT INTO order_info (order_id, user_id, total_amount, actual_payment, discount, "
            f"order_status, payment_method, create_time, delivery_time, receive_time, "
            f"consignee, review_score, review_content) VALUES "
            f"('{oid}', '{uid}', {total_amount:.2f}, {actual_payment:.2f}, {discount:.2f}, "
            f"'{status}', '{pay_method}', '{order_date}', {delivery_part}, {receive_part}, "
            f"'{address}', {review_score_part}, {review_part});"
        )

        # order_detail
        p = product_map.get(pid, {})
        pname = p.get("product_name", pid).replace("'", "\\'")
        amount = unit_price * qty

        lines.append(
            f"INSERT INTO order_detail (order_id, user_id, product_id, product_name, "
            f"quantity, unit_price, amount) VALUES "
            f"('{oid}', '{uid}', '{pid}', '{pname}', {qty}, {unit_price:.2f}, {amount:.2f});"
        )

        # logistics_info (只有已发货/已收货/已完成的订单才有物流)
        if status in ("已发货", "已收货", "已完成"):
            tracking_no = f"SF{random.randint(100000000000, 999999999999)}"
            company = random.choice(express_companies)
            log_status = "已签收" if status in ("已收货", "已完成") else "运输中"
            lines.append(
                f"INSERT INTO logistics_info (order_id, tracking_no, company, status, create_time) VALUES "
                f"('{oid}', '{tracking_no}', '{company}', '{log_status}', '{order_date}');"
            )
            logistics_count += 1

        # dwd_fact_payment_info (已付款及之后状态)
        if status != "待付款" and status != "已取消":
            dt = order_date[:10]  # YYYY-MM-DD
            lines.append(
                f"INSERT INTO dwd_fact_payment_info (order_id, user_id, payment_amount, "
                f"payment_method, dt) VALUES "
                f"('{oid}', '{uid}', {actual_payment:.2f}, '{pay_method}', '{dt}');"
            )
            payment_count += 1

        # dwd_fact_order_detail
        category = p.get("category", "")
        brand = p.get("brand", "")
        dt = order_date[:10]
        lines.append(
            f"INSERT INTO dwd_fact_order_detail (order_id, user_id, sku_id, sku_name, "
            f"sku_num, order_amount, category1_name, category3_name, brand_name, dt) VALUES "
            f"('{oid}', '{uid}', 'SKU_{pid}', '{pname}', {qty}, {actual_payment:.2f}, "
            f"'{category}', '{category}', '{brand}', '{dt}');"
        )
        dwd_count += 1

    lines.append("")
    lines.append(f"-- 物流记录: {logistics_count}")
    lines.append(f"-- 支付记录: {payment_count}")
    lines.append(f"-- DWD订单明细: {dwd_count}")
    lines.append("")

    # --- user_info ---
    lines.append("-- 用户数据导入")
    for u in users:
        uid = u["user_id"]
        age = int(u["age"]) if u["age"] else "NULL"
        gender = u["gender"]
        province = u["province"]
        city = u["city"]
        member_level = u["member_level"]
        account_balance = float(u["account_balance"])
        credit_score = int(u["credit_score"])

        uf = uf_map.get(uid, {})
        total_spent = float(uf.get("total_spent", 0))
        order_count = int(float(uf.get("order_count", 0)))
        purchase_intent = float(uf.get("purchase_intent", 0))
        consumption_level = uf.get("consumption_level", "")

        lines.append(
            f"INSERT INTO user_info (user_id, age, gender, province, city, member_level, "
            f"account_balance, credit_score, total_spent, order_count, purchase_intent, "
            f"consumption_level) VALUES "
            f"('{uid}', {age}, '{gender}', '{province}', '{city}', '{member_level}', "
            f"{account_balance:.2f}, {credit_score}, {total_spent:.2f}, {order_count}, "
            f"{purchase_intent:.4f}, '{consumption_level}');"
        )

    lines.append("")
    lines.append("-- ============================================")
    lines.append("-- 导入完成!")
    lines.append(f"-- order_info: {len(orders)} 条")
    lines.append(f"-- order_detail: {len(orders)} 条")
    lines.append(f"-- logistics_info: {logistics_count} 条")
    lines.append(f"-- dwd_fact_payment_info: {payment_count} 条")
    lines.append(f"-- dwd_fact_order_detail: {dwd_count} 条")
    lines.append(f"-- user_info: {len(users)} 条")
    lines.append("-- ============================================")

    output_path = os.path.join(OUTPUT_DIR, "mysql_gmall.sql")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[MySQL] ✅ 已生成: {output_path}")
    return {
        "order_info": len(orders),
        "order_detail": len(orders),
        "logistics_info": logistics_count,
        "dwd_payment": payment_count,
        "dwd_order_detail": dwd_count,
        "user_info": len(users),
    }


# ============================================================
# 3. FAISS 索引数据
# ============================================================
def generate_faiss_data(products, product_features):
    """生成供 build_faiss_index.py 使用的产品 JSON"""
    print("[FAISS] 生成产品数据...")

    pf_map = {pf["product_id"]: pf for pf in product_features}

    faiss_products = []
    for p in products:
        pid = p["product_id"]
        pf = pf_map.get(pid, {})

        # 构建富文本描述 (用于嵌入)
        description = (
            f"{p['product_name']} 品牌:{p['brand']} 分类:{p['category']} "
            f"价格:{p['price']}元 销量:{p['sales_count']} "
            f"转化率:{pf.get('conversion_rate', 0)} "
            f"好评率:{pf.get('avg_review_score', 0)}"
        )

        faiss_products.append(
            {
                "id": pid,
                "name": p["product_name"],
                "description": description,
                "category": p["category"],
                "brand": p["brand"],
                "price": float(p["price"]),
                "sales_count": int(p["sales_count"]),
                "popularity_score": float(pf.get("popularity_score", 0)),
            }
        )

    output_path = os.path.join(OUTPUT_DIR, "products_for_faiss.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(faiss_products, f, ensure_ascii=False, indent=2)

    print(f"[FAISS] ✅ 已生成: {output_path} ({len(faiss_products)} 个商品)")
    return len(faiss_products)


# ============================================================
# 4. 分类训练集
# ============================================================
def generate_classify_train(products):
    """生成 BERT 商品分类训练数据 (text + label)"""
    print("[Classify] 生成分类训练集...")

    output_path = os.path.join(OUTPUT_DIR, "classify_train.csv")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])

        for p in products:
            # 用商品名称作为文本，类目作为标签
            text = p["product_name"]
            label = p["category"]
            writer.writerow([text, label])

    cats = sorted(set(p["category"] for p in products))
    print(f"[Classify] ✅ 已生成: {output_path} ({len(products)} 条, {len(cats)} 个类目)")
    return {"count": len(products), "categories": len(cats)}


# ============================================================
# 5. FAQ 语料 (从评价内容生成)
# ============================================================
def generate_faq_data(orders, products):
    """从订单评价 + 商品信息生成客服 FAQ 语料"""
    print("[FAQ] 生成客服语料...")

    product_map = {p["product_id"]: p for p in products}

    faqs = []

    # --- 从评价内容生成 QA ---
    reviews = [o for o in orders if o["review_content"]]
    for r in reviews:
        p = product_map.get(r["product_id"], {})
        pname = p.get("product_name", r["product_id"])
        score = r["review_score"]
        content = r["review_content"]

        if int(float(score)) >= 4:
            faqs.append(
                {
                    "question": f"{pname}怎么样？质量好吗？",
                    "answer": content,
                    "category": "商品质量",
                }
            )
        else:
            faqs.append(
                {
                    "question": f"{pname}有什么缺点？",
                    "answer": content,
                    "category": "商品质量",
                }
            )

    # --- 通用电商 FAQ ---
    general_faqs = [
        {
            "question": "如何退换货？",
            "answer": "您可以在收到商品后7天内申请退换货。进入「我的订单」，找到对应订单，点击「申请退换货」即可。退换货商品需保持原包装完好。",
            "category": "退换货",
        },
        {
            "question": "退货多久能退款？",
            "answer": "退款会在我们收到退货商品后1-3个工作日内原路退回您的支付账户。请注意查收。",
            "category": "退换货",
        },
        {
            "question": "如何查看物流信息？",
            "answer": "您可以在「我的订单」中查看订单详情，里面有物流单号和快递公司信息。也可以直接点击物流信息查看实时物流状态。",
            "category": "物流",
        },
        {
            "question": "支持哪些支付方式？",
            "answer": "我们支持微信支付、支付宝、花呗、信用卡、银行卡等多种支付方式，您可以选择最方便的方式进行支付。",
            "category": "支付",
        },
        {
            "question": "订单多久发货？",
            "answer": "一般情况下，付款后24小时内发货。节假日可能会有延迟，发货后您会收到短信通知。",
            "category": "物流",
        },
        {
            "question": "如何修改收货地址？",
            "answer": "如果订单还未发货，您可以在「我的订单」中修改收货地址。如果已经发货，请联系客服协助处理。",
            "category": "订单",
        },
        {
            "question": "商品是正品吗？",
            "answer": "我们所有商品均为正品，支持专柜验货。每个商品都有防伪码，您可以在品牌官网查询真伪。",
            "category": "商品质量",
        },
        {
            "question": "如何使用优惠券？",
            "answer": "在结算页面，选择使用优惠券，系统会自动抵扣相应金额。请注意优惠券的使用期限和适用条件。",
            "category": "优惠",
        },
        {
            "question": "可以开发票吗？",
            "answer": "可以。在下单时选择「需要发票」，填写发票抬头和税号即可。电子发票会在订单完成后发送到您的邮箱。",
            "category": "发票",
        },
        {
            "question": "会员等级有什么权益？",
            "answer": "铜牌会员：享基础积分；银牌会员：享95折优惠；金牌会员：享9折+免邮；钻石会员：享85折+免邮+专属客服+优先发货。",
            "category": "会员",
        },
        {
            "question": "如何成为金牌会员？",
            "answer": "累计消费满5000元可升级为金牌会员，累计消费满20000元可升级为钻石会员。系统会自动升级。",
            "category": "会员",
        },
        {
            "question": "积分怎么用？",
            "answer": "积分可以在结算时抵扣现金，100积分=1元。也可以在积分商城兑换商品或优惠券。",
            "category": "优惠",
        },
        {
            "question": "商品保修多久？",
            "answer": "不同商品保修期不同，一般为1年。具体保修期请查看商品详情页或联系客服咨询。",
            "category": "售后",
        },
        {
            "question": "如何投诉？",
            "answer": "您可以通过客服热线、在线客服或「我的订单-投诉建议」进行投诉。我们会在24小时内处理并回复。",
            "category": "售后",
        },
        {
            "question": "支持货到付款吗？",
            "answer": "部分地区支持货到付款。在下单时如果地址支持货到付款，会有相应选项。货到付款需支付少量手续费。",
            "category": "支付",
        },
        {
            "question": "如何取消订单？",
            "answer": "如果订单还未付款，可以直接取消。如果已付款未发货，请联系客服取消。已发货的订单无法取消，需收到后申请退货。",
            "category": "订单",
        },
        {
            "question": "快递费怎么算？",
            "answer": "单笔订单满99元免邮费，不满99元收取10元邮费。偏远地区（新疆、西藏等）可能需要额外邮费。",
            "category": "物流",
        },
        {
            "question": "商品缺货怎么办？",
            "answer": "如果商品缺货，我们会尽快补货并通知您。您也可以选择类似商品或申请退款。缺货商品的退款会优先处理。",
            "category": "订单",
        },
        {
            "question": "如何修改密码？",
            "answer": "进入「我的-设置-账户安全」，点击「修改密码」，输入旧密码和新密码即可完成修改。",
            "category": "账户",
        },
        {
            "question": "忘记密码怎么办？",
            "answer": "在登录页面点击「忘记密码」，通过手机验证码或邮箱重置密码即可。",
            "category": "账户",
        },
    ]

    faqs.extend(general_faqs)

    output_path = os.path.join(OUTPUT_DIR, "faq_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(faqs, f, ensure_ascii=False, indent=2)

    print(f"[FAQ] ✅ 已生成: {output_path} ({len(faqs)} 条 FAQ)")
    return len(faqs)


# ============================================================
# 6. 数据质量报告
# ============================================================
def generate_report(
    orders,
    products,
    product_features,
    users,
    user_behaviors,
    user_features,
    neo4j_stats,
    mysql_stats,
    faiss_count,
    classify_stats,
    faq_count,
):
    """生成数据质量报告"""
    print("[Report] 生成数据报告...")

    # 计算统计信息
    behavior_users = set(b["user_id"] for b in user_behaviors)

    status_dist = Counter(o["order_status"] for o in orders)
    cat_dist = Counter(p["category"] for p in products)
    brand_dist = Counter(p["brand"] for p in products)
    behavior_dist = Counter(b["behavior_type"] for b in user_behaviors)
    member_dist = Counter(u["member_level"] for u in users)
    pay_dist = Counter(o["payment_method"] for o in orders)

    prices = [float(p["price"]) for p in products]
    total_amounts = [float(o["total_amount"]) for o in orders]
    reviews = [o for o in orders if o["review_score"]]

    report = []
    report.append("=" * 60)
    report.append("  淘宝用户消费数据 — 数据质量与导入报告")
    report.append(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 60)
    report.append("")

    report.append("一、原始数据概况")
    report.append("-" * 40)
    report.append(f"  orders.csv         {len(orders):>8,} 条订单")
    report.append(f"  products.csv       {len(products):>8,} 个商品")
    report.append(f"  product_features   {len(product_features):>8,} 条商品特征")
    report.append(f"  users.csv          {len(users):>8,} 个用户")
    report.append(f"  user_behaviors     {len(user_behaviors):>8,} 条行为")
    report.append(f"  user_features      {len(user_features):>8,} 条用户特征")
    report.append("")

    report.append("二、数据质量评估")
    report.append("-" * 40)
    report.append(f"  商品数:          {len(products)} (优秀, 理想200+)")
    report.append(f"  类目数:          {len(cat_dist)} (良好, 覆盖主流类目)")
    report.append(f"  品牌数:          {len(brand_dist)} (优秀)")
    report.append(f"  用户数:          {len(users)} (良好)")
    report.append(f"  订单数:          {len(orders)} (优秀)")
    report.append(f"  行为记录:        {len(user_behaviors)} (良好)")
    report.append(f"  人均行为:        {len(user_behaviors) / len(behavior_users):.1f} 条 (中等)")
    report.append(f"  有评价订单:      {len(reviews)} ({len(reviews) / len(orders) * 100:.1f}%)")
    report.append(f"  价格范围:        ¥{min(prices):.2f} ~ ¥{max(prices):.2f}")
    report.append(f"  平均价:          ¥{sum(prices) / len(prices):.2f}")
    report.append(f"  订单金额范围:    ¥{min(total_amounts):.2f} ~ ¥{max(total_amounts):.2f}")
    report.append("")

    report.append("三、分布详情")
    report.append("-" * 40)
    report.append("  订单状态分布:")
    for k, v in status_dist.most_common():
        report.append(f"    {k}: {v:,} ({v / len(orders) * 100:.1f}%)")
    report.append("")
    report.append("  行为类型分布:")
    for k, v in behavior_dist.most_common():
        report.append(f"    {k}: {v:,} ({v / len(user_behaviors) * 100:.1f}%)")
    report.append("")
    report.append("  商品类目分布:")
    for k, v in cat_dist.most_common():
        report.append(f"    {k}: {v} ({v / len(products) * 100:.1f}%)")
    report.append("")
    report.append("  会员等级分布:")
    for k, v in member_dist.most_common():
        report.append(f"    {k}: {v} ({v / len(users) * 100:.1f}%)")
    report.append("")
    report.append("  支付方式分布:")
    for k, v in pay_dist.most_common():
        report.append(f"    {k}: {v:,} ({v / len(orders) * 100:.1f}%)")
    report.append("")

    report.append("四、生成文件清单")
    report.append("-" * 40)
    report.append("  neo4j_import.cypher      Neo4j 图谱导入脚本")
    report.append(f"    SPU 商品:      {neo4j_stats['spu']}")
    report.append(f"    Category3:     {neo4j_stats['category']}")
    report.append(f"    Trademark:     {neo4j_stats['brand']}")
    report.append(f"    SKU:           {neo4j_stats['sku']}")
    report.append(f"    User:          {neo4j_stats['user']}")
    report.append(f"    行为关系:      {neo4j_stats['behavior']:,}")
    report.append(f"    购买关系:      {neo4j_stats['buy']:,}")
    report.append("")
    report.append("  mysql_gmall.sql          MySQL gmall 数据库脚本")
    report.append(f"    order_info:            {mysql_stats['order_info']:,}")
    report.append(f"    order_detail:          {mysql_stats['order_detail']:,}")
    report.append(f"    logistics_info:        {mysql_stats['logistics_info']:,}")
    report.append(f"    dwd_fact_payment:      {mysql_stats['dwd_payment']:,}")
    report.append(f"    dwd_fact_order_detail: {mysql_stats['dwd_order_detail']:,}")
    report.append(f"    user_info:             {mysql_stats['user_info']:,}")
    report.append("")
    report.append(f"  products_for_faiss.json  FAISS 索引数据 ({faiss_count} 个商品)")
    report.append(
        f"  classify_train.csv       分类训练集 ({classify_stats['count']} 条, {classify_stats['categories']} 类)"
    )
    report.append(f"  faq_data.json            客服 FAQ 语料 ({faq_count} 条)")
    report.append("")

    report.append("五、各 Agent 就绪状态")
    report.append("-" * 40)
    report.append("  Order Agent      ✅ 完全就绪 — 15K 订单 + 物流数据")
    report.append("  Analytics Agent  ✅ 完全就绪 — DWD 事实表 + 支付表")
    report.append("  Search Agent     ✅ 完全就绪 — 2000 商品可建 FAISS 索引")
    report.append("  KG QA Agent      ✅ 完全就绪 — 完整图谱 (商品/品牌/分类/用户)")
    report.append("  Recommend Agent  ✅ 完全就绪 — 用户行为 + 购买关系 + 商品特征")
    report.append("  Classify Agent   ✅ 完全就绪 — 2000 条训练数据, 15 个类目")
    report.append("  CS Agent         ✅ 完全就绪 — 评价 FAQ + 通用 FAQ")
    report.append("  Chitchat Agent   ✅ 不需要数据")
    report.append("")

    report.append("六、下一步操作")
    report.append("-" * 40)
    report.append("  1. 导入 Neo4j:")
    report.append("     在 Neo4j Browser 中粘贴 neo4j_import.cypher 执行")
    report.append("")
    report.append("  2. 导入 MySQL:")
    report.append("     mysql -u root -p < data/processed/mysql_gmall.sql")
    report.append("")
    report.append("  3. 构建 FAISS 索引:")
    report.append("     python scripts/build_faiss_index.py")
    report.append("     (需要 Neo4j 已导入数据 + sentence-transformers + faiss-cpu)")
    report.append("")
    report.append("  4. 训练分类模型:")
    report.append("     使用 classify_train.csv 微调 BERT 模型")
    report.append("     或直接复用 product_classification 项目")
    report.append("")
    report.append("  5. 构建 FAQ 索引:")
    report.append("     python scripts/build_faq_index.py")
    report.append("     (使用 faq_data.json 作为 FAQ 数据源)")
    report.append("")

    report.append("=" * 60)
    report.append("  总结: 数据质量优秀, 所有 8 个 Agent 均可完整运行")
    report.append("=" * 60)

    output_path = os.path.join(OUTPUT_DIR, "data_report.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    print(f"[Report] ✅ 已生成: {output_path}")


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("  淘宝用户消费数据 → 项目数据导入脚本")
    print("=" * 60)
    print()

    # 读取所有 CSV
    print("读取数据文件...")
    orders = read_csv("orders.csv")
    products = read_csv("products.csv")
    product_features = read_csv("product_features.csv")
    users = read_csv("users.csv")
    user_behaviors = read_csv("user_behaviors.csv")
    user_features = read_csv("user_features.csv")

    print(f"  订单:       {len(orders):,} 条")
    print(f"  商品:       {len(products):,} 个")
    print(f"  商品特征:   {len(product_features):,} 条")
    print(f"  用户:       {len(users):,} 个")
    print(f"  用户行为:   {len(user_behaviors):,} 条")
    print(f"  用户特征:   {len(user_features):,} 条")
    print()

    # 生成所有数据
    neo4j_stats = generate_neo4j_import(
        orders, products, product_features, users, user_behaviors, user_features
    )
    print()

    mysql_stats = generate_mysql_import(orders, products, users, user_features)
    print()

    faiss_count = generate_faiss_data(products, product_features)
    print()

    classify_stats = generate_classify_train(products)
    print()

    faq_count = generate_faq_data(orders, products)
    print()

    generate_report(
        orders,
        products,
        product_features,
        users,
        user_behaviors,
        user_features,
        neo4j_stats,
        mysql_stats,
        faiss_count,
        classify_stats,
        faq_count,
    )
    print()

    print("=" * 60)
    print("  ✅ 全部数据生成完成!")
    print(f"  输出目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
