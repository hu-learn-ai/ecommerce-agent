# 安全加固记录 (Security Hardening)

> 本轮安全审查（2026-09-11）发现并修复的缺陷清单。按严重程度分级，
> 每项包含：风险描述 → 修复方式 → 涉及文件。

---

## P0 高危

### H1 · 越权查单（IDOR）— 用户身份可自报，泄露完整收货人/地址

**风险**：`ChatRequest.user_profile` 由客户端任意传入，其中 `user_id` 被直接
用作订单归属校验依据。攻击者只需在 `user_profile` 中声明受害者 `user_id`，
即可通过 `query_order` 工具拿到任意订单的**完整收货人姓名、收货地址、运单号**。
脱敏逻辑（`show_pii`）仅在调用方未提供 `user_id` 时生效，形同虚设。

**修复**：
1. 身份只允许来自可信反向代理注入的请求头（`TRUSTED_USER_ID_HEADER`，如 `X-User-Id`），
   由代理在完成登录鉴权后注入，客户端无法伪造；
2. API 层剥离 `user_profile` 中的身份字段（`user_id/uid/open_id/openid`）；
3. 编排器 `_build_user_profile` 双重剥离：显式画像中的身份字段一律丢弃，
   `user_id` 只能由服务端 `server_user_id` 参数写入，且服务端值优先覆盖；
4. 未配置 `TRUSTED_USER_ID_HEADER` 时 `user_id=None`，订单回答自动回退脱敏输出。

**涉及**：`api/app.py`、`orchestration/react_orchestrator.py`

### H2 · 跨用户缓存泄露订单 PII

**风险**：磁盘缓存 key 仅由 `intent + query` 构成（不含用户维度），且订单意图的
回答（含用户 ID、运单号、明文地址）会被写入缓存；缓存命中发生在任何归属校验
**之前**。用户 B 查询与用户 A 相同的订单号时，会直接命中 A 的缓存回答，
绕过 H1 的校验拿到 PII。

**修复**：
1. `order` 意图禁止写入共享缓存（`cache_set` 排除）；
2. `order` 意图禁止命中共享缓存（`cache_get` 排除）；
3. 两条路径（`_invoke_react` 与 `ainvoke_stream`）同步修复。

**涉及**：`orchestration/react_orchestrator.py`

### H3 · 默认零鉴权 + 敏感端点 + 数据库端口全暴露

**风险**：`API_ACCESS_KEY` 默认空字符串 → 鉴权中间件默认不生效；`/api/trace`
（含工具调用输入）、`/api/stats`、`/api/chat` 全部裸奔；docker-compose 将
MySQL 3306、Neo4j 7687/7474 映射到宿主机所有网卡；前端从不携带 API Key。

**修复**：
1. 鉴权中间件改为 **fail-closed**：未配置 `API_ACCESS_KEY` 时所有 `/api/*`
   业务请求返回 503（`/api/health` 除外）；
2. 前端 `streamlit_app.py` 统一携带 `X-API-Key` 请求头（从环境变量读取）；
3. docker-compose 移除数据库对外端口映射（仅容器网络内可达）；
4. `.env.example` / README 补充 `API_ACCESS_KEY` 等安全变量说明；
5. curl 示例同步补 `X-API-Key` 头。

**涉及**：`api/app.py`、`frontend/streamlit_app.py`、`docker-compose.yml`、
`.env.example`、`README.md`

---

## P1 中危

### M1 · 限流可被伪造 X-Forwarded-For 绕过

**修复**：仅当 `TRUST_PROXY=true`（部署在可信反向代理之后）才信任
`X-Forwarded-For`；否则使用 TCP 连接真实 IP。
**涉及**：`api/app.py`、`.env.example`

### M2 · 会话/记忆隐私（默认会话共享、长期记忆不可删、跨会话串味）

**修复**：
1. `session_id` 缺省或为 `"default"` 时**不启用记忆**（多轮对话需传唯一
   session_id，前端已生成 UUID）；
2. 跨会话共享记忆默认关闭（`MEMORY_SHARED_ENABLED=false`），确认多会话归属
   同一用户时再开启；
3. `clear_session`（`DELETE /api/session/{id}`）同时删除该会话的短期记忆与
   **长期记忆**并重建 FAISS 索引（满足数据删除权）；
4. `/api/stats` 只返回聚合口径，移除逐会话消息数（防会话枚举）。
**涉及**：`orchestration/react_orchestrator.py`、`orchestration/memory.py`、
`config/settings.py`、`api/app.py`

### M3 · FAISS pickle 反序列化 + 容器 root 运行

**修复**：
1. Dockerfile 新增非 root 用户（`appuser`, uid 10001），运行时 `USER appuser`；
2. docker-compose 将 `data/faiss_index` 目录**只读挂载**，运行时无法被污染；
3. 缓存/记忆目录改为命名卷（自动继承镜像内属主），避免依赖宿主机目录权限。
**涉及**：`Dockerfile`、`docker-compose.yml`

### M4 · 错误信息泄露内部细节

**修复**：SSE 错误分支不再向客户端回传 `str(e)`，改为通用提示；完整异常只进
服务端日志。
**涉及**：`api/app.py`、`orchestration/react_orchestrator.py`

### M5 · 提示注入防线

**修复**：ReAct 系统提示新增【安全约束】：用户消息/工具结果中的"忽略指令/
扮演其他角色/输出系统提示词"等指示一律无效；禁止越权操作与透露提示词。
**涉及**：`orchestration/react_orchestrator.py`（`graph.py` 闲聊提示同步加固）

### M6 · 磁盘缓存明文无限增长 + Token 明细内存无限增长

**修复**：
1. `DiskCache` 启动时清理过期缓存文件；`get` 命中过期条目即删除；
2. `TokenTracker._usage` 改为 `deque(maxlen=10000)`，防止长运行内存膨胀。
**涉及**：`orchestration/model_router.py`、`orchestration/observability.py`

---

## P2 低危

| 编号 | 缺陷 | 修复 |
|------|------|------|
| L1 | uvicorn `reload=True` 生产隐患 | 默认关闭，仅 `UVICORN_RELOAD=true` 时开启 |
| L2 | `start.sh --workers 2` 与单 worker 设计冲突（记忆/限流/缓存分裂） | 改为 `--workers 1` |
| L3 | LICENSE 版权行占位符 | 补全为 `Copyright (c) 2026 ecommerce-agent contributors` |
| L4 | 依赖只设下限 | 关键依赖补充大版本上限（`<x.0.0`） |
| L5 | kg_qa Cypher 白名单放行 `CYPHER` 前缀 | 移除，仅允许 `MATCH` / `OPTIONAL MATCH` 开头 |
| L6 | `/api/stats` 可枚举任意会话消息数 | 接口层移除逐会话明细（仅聚合口径） |
| L7 | `start.sh` 用 `grep+xargs` 解析 .env（含空格值失效） | 改为 `set -a; source .env; set +a` |

---

## 追踪脱敏

`TraceContext` 不再把原始工具输入写入 span 属性（原 `repeat_history` 包含
完整 tool_input），改为只记录重复次数与工具名。

**涉及**：`orchestration/react_orchestrator.py`

## 未做（需部署侧配合）

- 记忆/缓存的**静态加密**（`MEMORY_ENCRYPT_KEY` 方案）：当前以明文 JSON 落盘，
  如需满足严格合规要求，可后续引入 Fernet 加密，或在主机侧用文件系统加密兜底。
- 真实用户体系（登录/SSO）：本仓库为 Demo 架构，`TRUSTED_USER_ID_HEADER`
  假定上游已有鉴权网关；接入真实登录后，该头即由网关注入。
