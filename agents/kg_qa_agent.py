"""
知识图谱问答 Agent — 复用 ec_graph 核心流程

基于 Neo4j 电商知识图谱的 RAG 智能问答
流程：用户问题 → LLM 生成 Cypher → 实体对齐 → 执行查询 → LLM 生成回答

学习参考: ec_graph/src/web/service.py 的 RAG 流程
"""

import json
import re

from neo4j import READ_ACCESS

from agents.tools.base import BaseAgentTool


class KGQAAgent(BaseAgentTool):
    """
    基于知识图谱的智能问答

    知识图谱 Schema (来自 ec_graph):
        节点: Category1/2/3(三级分类), SPU(产品), SKU(库存单位),
              Trademark(品牌), BaseAttrName/Value(属性), Tag(标签)
        关系: Belong(归属), Have(拥有)

    复用 ec_graph 的：
    - Neo4j 图数据库及索引
    - KG Schema 描述
    - 三级 fallback 实体对齐策略
    """

    name: str = "kg_qa_agent"
    description: str = "知识图谱问答：基于电商知识图谱回答商品属性、品牌关系、分类导航等问题"

    # KG Schema 描述 — 与实际导入的图谱对齐(运行时 _schema_desc() 会用真实结构覆盖)
    # 实机探测(2026-08): 只有 SPU/Category3/Trademark/SKU/User(空);
    # 无 Category1/2、无 BaseAttrName/Value、Trademark 无 tm_name、SKU 无价格字段
    KG_SCHEMA = """
    知识图谱 Schema:
    - 节点(Node Labels):
      * Category3    三级分类 (字段: name, id)
      * SPU          标准产品单元 (字段: name, id)
      * SKU          库存计量单位 (字段: id)
      * Trademark    品牌 (字段: name)
      * User         用户

    - 关系(Relationship Types):
      * Belong  归属关系 (SPU -> Category3)
      * Have    拥有关系 (SPU -> SKU, SPU -> Trademark)

    - 能力边界:
      * 只有一级分类 Category3(无 Category1/Category2 多级导航)
      * 无商品属性节点(BaseAttrName/BaseAttrValue), 属性类问题(如"内存是多少")回复 NONE
      * SKU 无价格字段, 价格类问题回复 NONE
    """

    # 安全限制常量
    MAX_CYPHER_LENGTH = 800  # Cypher 语句最大长度（字符）
    MAX_MATCH_CLAUSES = 5  # 最大 MATCH/OPTIONAL MATCH 子句数
    MAX_RESULT_ROWS = 100  # 最大返回行数（强制 LIMIT）
    QUERY_TIMEOUT_SECONDS = 10  # 查询超时（秒）

    def __init__(self, neo4j_driver, llm):
        super().__init__()
        self.neo4j_driver = neo4j_driver
        self.llm = llm
        self._schema_cache = None

    def _read_session(self):
        """返回只读会话：LLM 生成的 Cypher 只能在只读事务中执行。

        这是对 _entity_alignment 黑名单校验的纵深防御——即使校验被绕过，
        只读事务在服务端也无法执行写操作，Neo4j 会直接拒绝。
        """
        return self.neo4j_driver.session(default_access_mode=READ_ACCESS)

    def _schema_desc(self) -> str:
        """返回给 LLM 的 schema 描述: 优先用运行时探测的真实结构, 失败用静态默认。"""
        if self.neo4j_driver is None:
            return self.KG_SCHEMA
        if self._schema_cache:
            return self._schema_cache
        try:
            with self._read_session() as session:
                labels = [
                    r[0]
                    for r in session.run(
                        "CALL db.labels() YIELD label RETURN label ORDER BY label"
                    )
                ]
                rels = [
                    r[0]
                    for r in session.run(
                        "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
                    )
                ]
                props = {}
                for lab in labels:
                    row = session.run(
                        f"MATCH (n:`{lab}`) RETURN keys(n) LIMIT 1"
                    ).single()
                    props[lab] = sorted(row[0]) if row else []
            lines = ["知识图谱 Schema (运行时探测):", "- 节点(Node Labels):"]
            for lab in labels:
                fields = ", ".join(props[lab]) if props[lab] else "无"
                lines.append(f"  * {lab} (字段: {fields})")
            lines.append("- 关系(Relationship Types):")
            for r in rels:
                lines.append(f"  * {r}")
            self._schema_cache = "\n".join(lines)
            return self._schema_cache
        except Exception:
            return self.KG_SCHEMA

    # ------------------------------------------------------------------ #
    #  对外接口
    # ------------------------------------------------------------------ #

    def run(self, query: str, **kwargs) -> str:
        """同步执行知识图谱问答"""
        return self.kg_query(query)

    async def arun(self, **kwargs) -> str:
        """异步执行"""
        return self.run(**kwargs)

    # ------------------------------------------------------------------ #
    #  核心问答流程
    # ------------------------------------------------------------------ #

    def kg_query(self, question: str) -> str:
        """
        基于电商知识图谱回答商品相关问题

        适用场景：品牌查询、分类导航、属性对比、商品关系

        Args:
            question: 用户自然语言问题
        """
        # Step 1: LLM 生成 Cypher 查询
        cypher = self._generate_cypher(question)

        # 如果 LLM 判断无法生成 Cypher，直接回答
        if not cypher or cypher.strip().upper() in ("NONE", "NULL", ""):
            return self._llm_direct_answer(question)

        # Step 2: 实体对齐 (复用 ec_graph 三级 fallback 策略)
        aligned_cypher = self._entity_alignment(cypher)

        # Step 3: 执行 Cypher 查询
        records = self._execute_cypher(aligned_cypher)

        # Step 4: 如果查询无结果，尝试降级
        if not records:
            records = self._fallback_query(question)

        # Step 5: LLM 基于查询结果生成回答
        return self._generate_answer(question, records)

    # ------------------------------------------------------------------ #
    #  内部方法
    # ------------------------------------------------------------------ #

    def _generate_cypher(self, question: str) -> str:
        """LLM 生成 Cypher 查询语句"""
        # 输入消毒：截断 + 转义引号/反斜杠，降低提示注入面
        question = (question or "").strip()[:200]
        question = question.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
        prompt = f"""你是一个 Cypher 查询生成专家。

{self._schema_desc()}

请根据用户问题生成可在 Neo4j 中执行的参数化 Cypher 查询。

规则：
1. 只能使用上述定义的节点标签和关系类型
2. 查询必须以 RETURN 结尾
3. 如果问题无法用 Cypher 表达，回复 "NONE"
4. 不要使用 CALL 子句
5. 使用 CONTAINS 进行模糊匹配
6. 限制结果数量，使用 LIMIT
7. schema 中不存在的维度(属性值/多级分类/价格等)回复 "NONE"

示例：
Q: "华为有哪些产品？"
A: MATCH (t:Trademark) WHERE t.name CONTAINS '华为' MATCH (t)<-[:Have]-(p:SPU) RETURN p.name AS product, t.name AS brand LIMIT 10

Q: "吸尘器属于什么分类？"
A: MATCH (p:SPU) WHERE p.name CONTAINS '吸尘器' MATCH (p)-[:Belong]->(c:Category3) RETURN p.name AS product, c.name AS category LIMIT 5

Q: "和iPhone类似的产品有哪些？"
A: MATCH (p:SPU) WHERE p.name CONTAINS 'iPhone' MATCH (p)-[:Belong]->(c:Category3) MATCH (similar:SPU)-[:Belong]->(c) WHERE similar <> p RETURN DISTINCT similar.name AS product, c.name AS category LIMIT 10

用户问题: {question}

Cypher查询:"""

        try:
            response = self.llm.invoke(prompt)
            cypher = response.content.strip()

            # 清理：移除 markdown 代码块标记
            cypher = re.sub(r"^```(?:cypher)?\s*", "", cypher)
            cypher = re.sub(r"\s*```$", "", cypher)
            cypher = cypher.strip()

            return cypher
        except Exception as e:
            print(f"[KGQAAgent] Cypher 生成失败: {e}")
            return ""

    def _entity_alignment(self, cypher: str) -> str:
        """
        Cypher 安全校验 — 多层防御

        Level 0: 注释剥离 + 多语句拒绝
        Level 1: 白名单 — 只允许 MATCH/OPTIONAL MATCH 开头
        Level 2: 黑名单 — 禁止一切写操作和危险关键词
        Level 3: 复杂度限制 — 长度/MATCH 子句数/RETURN 检查
        Level 4: 结果集上限 — 无 LIMIT 时自动注入 LIMIT
        """
        if not cypher:
            return ""

        # Level 0: 剥离 // 与 /* */ 注释，拒绝分号多语句注入
        # 块注释同样剥离，防止用 /* ... */ 把禁止关键词拆开绕过黑名单
        cypher = re.sub(r"//.*", "", cypher)
        cypher = re.sub(r"/\*.*?\*/", "", cypher, flags=re.DOTALL)
        stripped = cypher.strip().rstrip(";")
        if ";" in stripped:
            print("[KGQAAgent] L0: 检测到多语句注入（;），拒绝执行")
            return ""
        cypher = stripped

        cypher_upper = cypher.upper().strip()

        # === Level 1: 白名单 — 必须以 MATCH 或 OPTIONAL MATCH 开头 ===
        # L5 修复：移除 CYPHER 前缀放行（可被用于切换 runtime/planner 等执行参数）
        allowed_starts = ("MATCH", "OPTIONAL MATCH")
        if not any(cypher_upper.startswith(prefix) for prefix in allowed_starts):
            print(f"[KGQAAgent] L1: Cypher 不以 MATCH 开头，拒绝执行: {cypher[:80]}")
            return ""

        # === Level 2: 黑名单 — 绝对禁止的关键词 ===
        forbidden_keywords = [
            "DELETE",
            "DROP",
            "REMOVE",
            "SET",
            "CREATE",
            "MERGE",
            "CALL",
            "LOAD CSV",
            "FOREACH",
            "PERIODIC",
            "SHORTEST",
            "DETACH",
            "DEPRECATE",
            "INSTALL",
            "UNINSTALL",
            "STOP",
            "START",
            "EXPLAIN",
            "PROFILE",
            "-schema",
            "SCHEMA",
            # 补充：UNWIND 可做数据展开（DoS 面），UNION 可拼接第二条子查询绕过白名单
            "UNWIND",
            "UNION",
        ]
        for kw in forbidden_keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", cypher_upper):
                print(f"[KGQAAgent] L2: 检测到禁止操作 {kw}，拒绝执行")
                return ""

        # === Level 3: 复杂度限制 ===
        # 3a: 长度限制
        if len(cypher) > self.MAX_CYPHER_LENGTH:
            print(
                f"[KGQAAgent] L3: Cypher 过长 ({len(cypher)} > {self.MAX_CYPHER_LENGTH})，拒绝执行"
            )
            return ""

        # 3b: MATCH 子句数量限制（防止笛卡尔积爆炸）
        match_count = len(re.findall(r"\bMATCH\b", cypher_upper))
        if match_count > self.MAX_MATCH_CLAUSES:
            print(
                f"[KGQAAgent] L3: MATCH 子句过多 ({match_count} > {self.MAX_MATCH_CLAUSES})，拒绝执行"
            )
            return ""

        # 3c: 必须包含 RETURN
        if "RETURN" not in cypher_upper:
            print("[KGQAAgent] L3: Cypher 缺少 RETURN 子句，拒绝执行")
            return ""

        # 3d: 检测无 WHERE 的多 MATCH（潜在笛卡尔积）
        if match_count >= 2 and "WHERE" not in cypher_upper:
            print("[KGQAAgent] L3: 多 MATCH 无 WHERE，潜在笛卡尔积风险，拒绝执行")
            return ""

        # 3e: 检测单 MATCH 内多节点笛卡尔积
        # MATCH (a), (b), (c) 是笛卡尔积(节点间仅逗号, 无关系连接)
        # MATCH (a)-[:R]->(b) 是合法多节点(有关系连接), 不应误判
        match_clauses = re.findall(
            r"\bMATCH\b(.*?)(?=\b(?:MATCH|WHERE|RETURN|WITH|CALL|LIMIT|OPTIONAL|UNWIND|SET|REMOVE|MERGE|CREATE|DELETE)\b|$)",
            cypher_upper,
            re.DOTALL,
        )
        for clause in match_clauses:
            # 检测 ")," 模式: 多个独立节点用逗号分隔
            if re.search(r"\)\s*,\s*\(", clause):
                # 排除有关系连接的合法查询: (a)-[:R]->(b) / (a)--(b) / (a)<-[:R]-(b)
                if "-[" not in clause and "<-[" not in clause and "--" not in clause:
                    print(
                        "[KGQAAgent] L3: 单 MATCH 内多节点笛卡尔积(无关系连接)，拒绝执行"
                    )
                    return ""

        # === Level 4: 结果集上限 — 无 LIMIT 时自动注入 ===
        if "LIMIT" not in cypher_upper:
            cypher = cypher.rstrip().rstrip(";")
            cypher = f"{cypher} LIMIT {self.MAX_RESULT_ROWS}"
            print(f"[KGQAAgent] L4: 自动注入 LIMIT {self.MAX_RESULT_ROWS}")

        return cypher

    def _execute_cypher(self, cypher: str) -> list:
        """执行 Cypher 查询并返回记录列表（只读事务 + 超时 + 结果集上限保护）"""
        if not cypher or not self.neo4j_driver:
            return []

        try:
            with self._read_session() as session:
                # neo4j 6.x 中 session.run(..., timeout=) 会被当作查询参数而非
                # 事务超时，因此用 begin_transaction 显式传 timeout（DoS 防护）。
                with session.begin_transaction(timeout=self.QUERY_TIMEOUT_SECONDS) as tx:
                    result = tx.run(cypher)
                    records = []
                    row_count = 0
                    for r in result:
                        row_count += 1
                        # 二次防御：即使 LIMIT 被绕过，也在此截断
                        if row_count > self.MAX_RESULT_ROWS:
                            print(f"[KGQAAgent] 结果集截断: 超过 {self.MAX_RESULT_ROWS} 行上限")
                            break
                        record_dict = {}
                        for key in r.keys():
                            val = r[key]
                            if hasattr(val, "items"):
                                record_dict[key] = dict(val)
                            else:
                                record_dict[key] = str(val)
                        records.append(record_dict)
                    return records
        except Exception as e:
            print(f"[KGQAAgent] Cypher 执行失败: {e}")
            print(f"  Cypher: {cypher[:200]}")
            return []

    def _execute_cypher_with_params(self, cypher: str, params: dict) -> list:
        """执行带参数的 Cypher 查询（只读事务 + 参数化 + 超时 + 结果集上限）"""
        if not cypher or not self.neo4j_driver:
            return []

        try:
            with self._read_session() as session:
                with session.begin_transaction(timeout=self.QUERY_TIMEOUT_SECONDS) as tx:
                    result = tx.run(cypher, params)
                    records = []
                    row_count = 0
                    for r in result:
                        row_count += 1
                        if row_count > self.MAX_RESULT_ROWS:
                            print(f"[KGQAAgent] 结果集截断: 超过 {self.MAX_RESULT_ROWS} 行上限")
                            break
                        record_dict = {}
                        for key in r.keys():
                            val = r[key]
                            if hasattr(val, "items"):
                                record_dict[key] = dict(val)
                            else:
                                record_dict[key] = str(val)
                        records.append(record_dict)
                    return records
        except Exception as e:
            print(f"[KGQAAgent] Cypher 执行失败: {e}")
            print(f"  Cypher: {cypher[:200]}")
            return []

    def _fallback_query(self, question: str) -> list:
        """
        降级查询：当 LLM 生成的 Cypher 无结果时使用
        使用简单的关键词匹配搜索 SPU 节点
        """
        # 提取关键词（简单分词）
        keywords = [w.strip() for w in question.split() if len(w.strip()) > 1]
        if not keywords:
            keywords = [question.strip()]

        cypher = """
        MATCH (p:SPU)
        WHERE p.name CONTAINS $keyword
        OPTIONAL MATCH (p)-[:Belong]->(c:Category3)
        OPTIONAL MATCH (p)-[:Have]->(t:Trademark)
        RETURN p.name AS name, p.id AS id, c.name AS category, t.name AS brand
        LIMIT 5
        """

        for kw in keywords:
            records = self._execute_cypher_with_params(cypher, {"keyword": kw})
            if records:
                return records

        return []

    def _generate_answer(self, question: str, records: list) -> str:
        """LLM 基于查询结果生成自然语言回答"""
        if not records:
            return (
                "抱歉，知识图谱中暂时未找到与您问题相关的数据。"
                "您可以换个问法，或试试商品搜索/推荐功能。"
            )

        # 格式化查询结果作为 context
        context = json.dumps(records, ensure_ascii=False, indent=2)

        prompt = f"""你是一个电商知识图谱问答助手。

知识图谱 Schema:
{self._schema_desc()}

用户问题: {question}

知识图谱查询结果(JSON):
{context}

请根据以上查询结果，用自然语言回答用户的问题。
要求：
1. 回答要准确、简洁、友好
2. 如果结果为空或与问题不相关，诚实告知
3. 可以适当补充对结果的解释
4. 用中文回答
"""

        try:
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            print(f"[KGQAAgent] 回答生成失败: {e}")
            # 降级：直接格式化返回查询结果
            return self._format_records(records)

    def _llm_direct_answer(self, question: str) -> str:
        """LLM 直接回答（无图查询结果时）"""
        prompt = f"""你是一个电商知识助手。用户问: {question}

请基于你的电商知识给出回答。如果你不确定，请诚实告知。"""
        try:
            return self.llm.invoke(prompt).content.strip()
        except Exception:
            return "抱歉，我暂时无法回答这个问题。"

    def _format_records(self, records: list) -> str:
        """格式化查询结果为可读字符串（降级方案）"""
        if not records:
            return "未找到相关信息。"
        lines = []
        for i, r in enumerate(records, 1):
            parts = [f"{k}: {v}" for k, v in r.items() if v]
            lines.append(f"{i}. {' | '.join(parts)}")
        return "\n".join(lines)
