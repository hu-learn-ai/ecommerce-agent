#!/usr/bin/env bash
# ============================================================
# 推荐评估 + 完整 RAG 评估一键脚本（在部署环境执行）
#
# 前置条件（部署环境已满足）:
#   1. Neo4j 运行中（图谱: 5000 用户 + 2000 商品）
#   2. MySQL 运行中（gmall 库）
#   3. FAISS 索引与 BGE 模型就绪（data/faiss_index/、models/）
#   4. .env 中 DEEPSEEK_API_KEY 有效
#
# 用法:
#   bash scripts/eval_recommendation.sh          # 推荐评估
#   bash scripts/eval_recommendation.sh --rag    # 完整 RAG 评估（含 Neo4j 混合检索）
#   bash scripts/eval_recommendation.sh --all    # 两者都跑
# ============================================================
set -e
cd "$(dirname "$0")/.."

# 环境检查
echo "=== 前置检查 ==="
if ! curl -s -o /dev/null --max-time 3 http://localhost:7474; then
    echo "[WARN] Neo4j (7474) 未响应 — 推荐评估需要图谱数据"
fi
if [ -z "$DEEPSEEK_API_KEY" ] && ! grep -q "DEEPSEEK_API_KEY=.\+" .env 2>/dev/null; then
    echo "[WARN] .env 中未检测到 DEEPSEEK_API_KEY"
fi

run_recommendation() {
    echo
    echo "============================================================"
    echo "  推荐 Agent 离线评估（graph_only vs graph+LLM_rerank）"
    echo "============================================================"
    python tests/recommendation_eval.py
    echo "→ 结果: tests/recommendation_eval_report.json"
}

run_rag() {
    echo
    echo "============================================================"
    echo "  完整 RAG 评估（FAISS 向量 + Neo4j 全文检索 + LLM Judge）"
    echo "============================================================"
    python tests/rag_evaluation.py
    echo "→ 结果: tests/rag_evaluation_report.json"
}

case "${1:-}" in
    --rag)  run_rag ;;
    --all)  run_recommendation; run_rag ;;
    *)      run_recommendation ;;
esac

echo
echo "✅ 评估完成。回填 README「评估系统」表格即可。"
