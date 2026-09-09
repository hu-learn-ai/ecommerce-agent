#!/usr/bin/env bash
# vLLM 启动脚本（对应 docs/微调项目执行计划.md 5.1）
#
# 用法：
#   MODEL_PATH=/root/autodl-tmp/models/qwen-cs-7b-merged ./finetune/scripts/serve_vllm.sh
# 或直接指定全部参数：
#   MODEL_PATH=... SERVED_MODEL_NAME=qwen-cs-7b MAX_MODEL_LEN=4096 GPU_UTIL=0.90 PORT=8000 ./finetune/scripts/serve_vllm.sh
#
# 验证：curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
#   -d '{"model":"qwen-cs-7b","messages":[{"role":"user","content":"你好"}]}'
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/qwen-cs-7b-merged}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen-cs-7b}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT="${PORT:-8000}"

exec vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT"
