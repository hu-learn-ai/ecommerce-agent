"""
冒烟测试一键脚本 - 复制以下命令到 PowerShell 运行：
python scripts/smoke_test.py
"""

import sys

import httpx

BASE = "http://localhost:8002"
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        FAIL += 1


def main():
    global PASS, FAIL
    client = httpx.Client(timeout=120)

    print("=" * 60)
    print("  电商智能体 - 冒烟测试")
    print("=" * 60)

    # === 1. 健康检查 ===
    print("\n--- 1. 健康检查 ---")
    r = httpx.get(f"{BASE}/api/health")
    check("状态码 200", r.status_code == 200, f"实际: {r.status_code}")
    data = r.json()
    check(
        "8 个 Agent 已注册",
        data.get("agents_registered") == 8,
        f"实际: {data.get('agents_registered')}",
    )

    # === 2. Agent 列表 ===
    print("\n--- 2. Agent 列表 ---")
    r = httpx.get(f"{BASE}/api/agents")
    agents = r.json().get("agents", [])
    check("返回 8 个 Agent", len(agents) == 8, f"实际: {len(agents)}")
    for a in agents:
        print(f"       - {a['name']}: {a['class']}")

    # === 3. 商品分类 ===
    print("\n--- 3. 商品分类 (BERT) ---")
    print("       (首次加载模型约需 30 秒，请等待...)")
    r = client.post(f"{BASE}/api/classify", json={"title": "小米手机 红米 Note 13", "top_k": 3})
    check("状态码 200", r.status_code == 200, f"实际: {r.status_code}")
    top_k = r.json().get("top_k", [])
    check("返回 Top-3", len(top_k) == 3, f"实际: {len(top_k)}")
    if top_k:
        print(f"       分类结果: {top_k[0]['category']} ({top_k[0]['confidence']})")
        check(
            "首选分类=手机数码", top_k[0]["category"] == "手机数码", f"实际: {top_k[0]['category']}"
        )

    # === 4. 商品搜索 ===
    print("\n--- 4. 商品搜索 (FAISS) ---")
    r = client.post(f"{BASE}/api/search", json={"query": "蓝牙耳机", "top_k": 5})
    check("状态码 200", r.status_code == 200, f"实际: {r.status_code}")
    result = r.json().get("result", "")
    check("返回搜索结果", len(result) > 0, "结果为空")
    print(f"       {result[:120]}...")

    # === 5. 意图路由 (8 种意图 → 4 个领域 Agent) ===
    print("\n--- 5. 意图路由测试 ---")
    tests = [
        # (消息, 期望意图, 可接受的 agent_used)
        ("搜索蓝牙耳机", "search", {"search_products", "catalog_agent"}),
        ("纸尿裤属于什么类别", "classify", {"classify_product", "catalog_agent"}),
        ("有什么推荐的商品", "recommend", {"recommend_products", "catalog_agent"}),
        ("查询订单 ORD123456", "order", {"query_order", "business_data_agent"}),
        ("怎么退货", "customer_service", {"customer_service", "cs_agent"}),
        ("最近销量排行", "analytics", {"data_analysis", "business_data_agent"}),
        ("你好", "chitchat", {""}),
    ]
    for msg, expected_intent, acceptable_agents in tests:
        r = client.post(f"{BASE}/api/chat", json={"message": msg}, timeout=60)
        d = r.json()
        ok = d["intent"] == expected_intent and d["agent_used"] in acceptable_agents
        check(
            f"[{msg}] -> {expected_intent}/{sorted(acceptable_agents)}",
            ok,
            f"实际: {d['intent']}/{d['agent_used']}",
        )
        reply = d.get("message", "")[:80]
        print(f"       回复: {reply}...")

    # === 汇总 ===
    print("\n" + "=" * 60)
    print(f"  通过: {PASS} | 失败: {FAIL}")
    print("=" * 60)
    if FAIL == 0:
        print("  ALL PASSED - 可以进入部署阶段")
    else:
        print("  有失败项，请检查后再部署")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
