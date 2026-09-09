"""
Streamlit 前端 — 电商智能助手聊天 UI

功能：
- 多轮对话（基于后端记忆系统，通过 session_id 隔离）
- 流式输出（SSE 逐 token 打字机效果）
- 显示意图路由信息（Intent + Agent + 工具调用链）
- 侧边栏快捷功能入口
- 分类/搜索独立功能面板
- 系统统计面板（Token 用量、缓存命中率）
"""

import json
import os
import uuid

import requests
import streamlit as st

# API 地址
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8002")

# ------------------------------------------------------------------ #
#  页面配置
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="电商智能助手",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛒 电商领域智能助手")
st.caption("ReAct 工具循环 | 记忆系统 | 流式输出 | LangGraph + DeepSeek")

# ------------------------------------------------------------------ #
#  会话状态初始化
# ------------------------------------------------------------------ #

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True

# ------------------------------------------------------------------ #
#  流式输出支持
# ------------------------------------------------------------------ #


def stream_chat(message: str, session_id: str):
    """
    通过 SSE 流式获取回答

    返回生成器, 逐 chunk yield 文本
    """
    try:
        response = requests.post(
            f"{API_BASE}/api/chat/stream",
            json={"message": message, "session_id": session_id},
            stream=True,
            timeout=120,
        )

        if response.status_code != 200:
            yield f"❌ 请求失败: {response.text}"
            return

        full_response = ""
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                try:
                    data = json.loads(data_str)
                    if data.get("type") == "token":
                        chunk = data.get("content", "")
                        full_response += chunk
                        yield chunk
                    elif data.get("type") == "done":
                        # 可以在这里获取完整回答的元信息
                        break
                    elif data.get("type") == "error":
                        yield f"\n❌ 错误: {data.get('message', '未知错误')}"
                        break
                except json.JSONDecodeError:
                    continue

    except requests.exceptions.ConnectionError:
        yield f"❌ 无法连接到 API 服务 ({API_BASE})"
    except Exception as e:
        yield f"❌ 请求异常: {e}"


def non_stream_chat(message: str, session_id: str) -> dict:
    """非流式调用（降级方案）"""
    try:
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={"message": message, "session_id": session_id},
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "message": f"❌ 请求失败: {response.text}",
                "intent": "error",
                "agent_used": "none",
            }
    except Exception as e:
        return {"message": f"❌ 请求异常: {e}", "intent": "error", "agent_used": "none"}


# ------------------------------------------------------------------ #
#  侧边栏
# ------------------------------------------------------------------ #

with st.sidebar:
    st.header("📋 功能导航")
    st.markdown("""
    - 🔍 **商品搜索**：支持分类/价格过滤
    - 📚 **知识问答**：品牌/属性/分类关系
    - 🏷️ **商品分类**：自动识别类别
    - 💡 **智能推荐**：个性化推荐
    - 📦 **订单查询**：状态/物流追踪
    - ❓ **售后服务**：退换货政策
    - 📊 **数据分析**：销售趋势/排行
    """)

    st.divider()

    # 健康检查
    try:
        resp = requests.get(f"{API_BASE}/api/health", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"✅ 服务在线 | {data.get('agents_registered', 0)} 个 Agent")
            if data.get("react_enabled"):
                st.caption("🧠 ReAct 模式已启用")
            if data.get("streaming_enabled"):
                st.caption("⚡ 流式输出已启用")
        else:
            st.warning("⚠️ 服务异常")
    except Exception:
        st.error("❌ 无法连接到 API 服务")
        st.caption(f"请确保 API 服务已启动: {API_BASE}")

    st.divider()

    # 流式输出开关
    st.session_state.streaming_enabled = st.checkbox(
        "⚡ 流式输出",
        value=st.session_state.streaming_enabled,
        help="启用后回答将逐字显示 (打字机效果)",
    )

    # 系统统计
    with st.expander("📊 系统统计"):
        try:
            resp = requests.get(f"{API_BASE}/api/stats", timeout=5)
            if resp.status_code == 200:
                stats = resp.json()

                # Token 用量
                token_stats = stats.get("token_usage", {})
                st.metric("LLM 调用次数", token_stats.get("total_calls", 0))
                st.metric("总 Token 数", token_stats.get("total_tokens", 0))
                st.metric("预估成本 (USD)", f"${token_stats.get('total_cost_usd', 0):.4f}")

                st.divider()

                # 缓存统计
                cache_stats = stats.get("cache_stats", {}).get("cache", {})
                st.metric("缓存命中率", cache_stats.get("hit_rate", "N/A"))
                st.metric("缓存命中数", cache_stats.get("hits", 0))

                st.divider()

                # 记忆统计
                mem_stats = stats.get("memory", {})
                st.metric("活跃会话", mem_stats.get("active_sessions", 0))
                st.metric("长期记忆", mem_stats.get("long_term_memories", 0))
            else:
                st.caption("统计信息暂不可用")
        except Exception:
            st.caption("统计信息暂不可用")

    st.divider()

    # 独立功能面板
    st.header("🛠️ 独立工具")
    tab_choice = st.radio(
        "选择工具",
        ["商品分类", "商品搜索"],
        label_visibility="collapsed",
    )

    if tab_choice == "商品分类":
        classify_input = st.text_input(
            "商品标题",
            placeholder="例如：小米 Redmi Note 12 5G 手机",
            key="classify_input",
        )
        top_k = st.slider("返回数量", 1, 10, 3)
        if st.button("分类", use_container_width=True):
            if classify_input.strip():
                with st.spinner("分类中..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/classify",
                            json={"title": classify_input, "top_k": top_k},
                            timeout=10,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.write(f"**输入**: {data['title']}")
                            for item in data["top_k"]:
                                st.write(f"- {item['category']} ({item['confidence']})")
                        else:
                            st.error(f"分类失败: {resp.text}")
                    except Exception as e:
                        st.error(f"请求失败: {e}")
            else:
                st.warning("请输入商品标题")

    elif tab_choice == "商品搜索":
        search_input = st.text_input(
            "搜索关键词",
            placeholder="例如：蓝牙耳机",
            key="search_input",
        )
        col1, col2 = st.columns(2)
        with col1:
            min_p = st.number_input("最低价", 0, 100000, 0, step=50)
        with col2:
            max_p = st.number_input("最高价", 0, 100000, 500, step=50)
        cat = st.text_input("分类（可选）", placeholder="如：手机数码")
        if st.button("搜索", use_container_width=True):
            if search_input.strip():
                with st.spinner("搜索中..."):
                    try:
                        resp = requests.post(
                            f"{API_BASE}/api/search",
                            json={
                                "query": search_input,
                                "top_k": 10,
                                "category": cat or None,
                                "min_price": min_p if min_p > 0 else None,
                                "max_price": max_p if max_p > 0 else None,
                            },
                            timeout=15,
                        )
                        if resp.status_code == 200:
                            data = resp.json()
                            st.text(data["result"])
                        else:
                            st.error(f"搜索失败: {resp.text}")
                    except Exception as e:
                        st.error(f"请求失败: {e}")
            else:
                st.warning("请输入搜索关键词")

    st.divider()

    # 清空对话
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        # 清除后端记忆
        try:
            requests.delete(f"{API_BASE}/api/session/{st.session_state.session_id}", timeout=5)
        except Exception:
            pass
        st.rerun()

    # 显示 Session ID
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

# ------------------------------------------------------------------ #
#  聊天历史
# ------------------------------------------------------------------ #

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("agent"):
            st.caption(f"🤖 {msg['agent']}")

# ------------------------------------------------------------------ #
#  用户输入
# ------------------------------------------------------------------ #

if prompt := st.chat_input("输入你的问题...（例如：搜索蓝牙耳机 / 查询订单 / 怎么退货）"):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 API
    with st.chat_message("assistant"):
        if st.session_state.streaming_enabled:
            # 流式输出
            with st.spinner("🤔 思考中..."):
                pass  # spinner 只显示一瞬间

            # 使用 st.write_stream 实现打字机效果
            full_response = st.write_stream(stream_chat(prompt, st.session_state.session_id))

            # 获取元信息 (非流式接口获取)
            meta_info = ""
            try:
                # 从流式结果中无法获取元信息, 尝试从 stats 获取
                pass
            except Exception:
                pass

            agent_info = "ReAct 模式 | 流式输出"
            st.caption(f"🤖 {agent_info}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "agent": agent_info,
                }
            )
        else:
            # 非流式输出 (降级)
            with st.spinner("🤔 思考中..."):
                data = non_stream_chat(prompt, st.session_state.session_id)

            st.markdown(data["message"])

            mode = data.get("mode", "fixed_route")
            tools = data.get("tools_called", [])
            steps = data.get("steps", 0)

            agent_info = f"意图: {data['intent']} → Agent: {data['agent_used']}"
            if mode == "react":
                agent_info += f" | ReAct ({steps}步)"
            if tools:
                agent_info += f" | 工具: {', '.join(tools)}"

            st.caption(f"🤖 {agent_info}")

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": data["message"],
                    "agent": agent_info,
                }
            )
