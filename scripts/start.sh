#!/bin/bash
# ============================================
# 电商智能体 — API 服务启动脚本
# ============================================

set -e

# 项目目录
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=========================================="
echo "  电商智能体系统启动"
echo "  项目目录: $PROJECT_DIR"
echo "=========================================="

# 激活虚拟环境（如果存在）
if [ -f "venv/bin/activate" ]; then
    echo "激活虚拟环境..."
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    echo "激活虚拟环境..."
    source .venv/bin/activate
fi

# 加载环境变量
if [ -f ".env" ]; then
    echo "加载 .env 配置..."
    export $(grep -v '^#' .env | xargs)
else
    echo "⚠️  .env 文件不存在，请先配置: cp .env.example .env"
    exit 1
fi

# 检查关键配置
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "⚠️  DEEPSEEK_API_KEY 未配置，LLM 功能将不可用"
fi

# 启动 API 服务
echo ""
echo "启动 API 服务..."
echo "  地址: http://${API_HOST:-0.0.0.0}:${API_PORT:-8002}"
echo "  文档: http://${API_HOST:-0.0.0.0}:${API_PORT:-8002}/docs"
echo ""

exec uvicorn api.app:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8002}" \
    --workers 2 \
    --log-level info
