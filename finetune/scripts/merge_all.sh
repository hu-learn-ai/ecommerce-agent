#!/usr/bin/env bash
# ----------------------------------------------------------------- #
#  合并两个 LoRA adapter → 完整可推理模型
#
#  用途：把 finetune/checkpoints/ 下的 LoRA adapter 合并到 base 模型，
#        输出完整模型目录供 vLLM / transformers 加载
#
#  对应补齐路径：
#    Run A (qwen-cs-7b)    — 15k 全量，loss 2.28→0.36（过拟合倾向，作对照）
#    Run B (qwen-cs-7b-5k) — 5k 消融，loss 0.38（更平滑，推荐主线）
#
#  显存需求：7B bf16 ≈ 14GB；24G 单卡可完成
#  预计时间：每个 merge 约 10-15 分钟（含 base 加载 + 合并 + 落盘）
#
#  用法（AutoDL 容器内）：
#    1. 把脚本上传到容器（或直接复制命令）
#    2. 编辑下面的 BASE_MODEL 路径匹配你的实例
#    3. bash finetune/scripts/merge_all.sh
#
#  产出：
#    /root/autodl-tmp/models/qwen-cs-7b-merged/      （Run A 合并）
#    /root/autodl-tmp/models/qwen-cs-7b-5k-merged/   （Run B 合并，推荐）
# ----------------------------------------------------------------- #

set -euo pipefail

# ====== 路径配置（按你的实例改） ======
REPO_ROOT="${REPO_ROOT:-/root/autodl-tmp/ecommerce-agent}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"

# adapter 路径（两个 run 都合并，便于三方对比）
ADAPTER_RUN_A="${ADAPTER_RUN_A:-${REPO_ROOT}/finetune/checkpoints/qwen-cs-7b}"
ADAPTER_RUN_B="${ADAPTER_RUN_B:-${REPO_ROOT}/finetune/checkpoints/qwen-cs-7b-5k}"

# 输出路径（建议放 /root/autodl-tmp/models/，与 base 同级便于 vLLM 启动）
OUT_RUN_A="${OUT_RUN_A:-/root/autodl-tmp/models/qwen-cs-7b-merged}"
OUT_RUN_B="${OUT_RUN_B:-/root/autodl-tmp/models/qwen-cs-7b-5k-merged}"

# ====== 环境 ======
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$OUT_RUN_A")" "$(dirname "$OUT_RUN_B")"

run_merge() {
    local adapter="$1"
    local output="$2"
    local tag="$3"

    if [[ -d "$output" && -f "$output/config.json" ]]; then
        echo "[merge_all] ${tag}: 已存在，跳过 → ${output}"
        return 0
    fi

    echo ""
    echo "=========================================================="
    echo "[merge_all] ${tag} 开始合并"
    echo "  base:    ${BASE_MODEL}"
    echo "  adapter: ${adapter}"
    echo "  output:  ${output}"
    echo "=========================================================="

    python "${REPO_ROOT}/finetune/scripts/merge_adapter.py" \
        --base-model "${BASE_MODEL}" \
        --adapter   "${adapter}" \
        --output    "${output}" \
        --dtype     bf16

    # 完整性自检：config.json + tokenizer + 至少一个 safetensors
    # （transformers 5.x 可能把 7B 存成单文件 model.safetensors，不分片，故用 model*.safetensors）
    if [[ ! -f "${output}/config.json" ]] || [[ ! -f "${output}/tokenizer.json" ]]; then
        echo "[merge_all] ${tag}: 合并产物不完整，请检查" >&2
        return 1
    fi
    if ! ls "${output}"/model*.safetensors >/dev/null 2>&1; then
        echo "[merge_all] ${tag}: 未生成 model*.safetensors，请检查" >&2
        return 1
    fi

    local size_gb
    size_gb=$(du -sh "$output" | awk '{print $1}')
    echo "[merge_all] ${tag}: 合并完成，体积 ${size_gb}"
}

# ====== 主流程 ======
echo "[merge_all] REPO_ROOT=${REPO_ROOT}"
echo "[merge_all] BASE_MODEL=${BASE_MODEL}"

run_merge "${ADAPTER_RUN_A}" "${OUT_RUN_A}" "Run A (15k, 作对照)"
run_merge "${ADAPTER_RUN_B}" "${OUT_RUN_B}" "Run B (5k, 主线)"

echo ""
echo "=========================================================="
echo "[merge_all] 全部合并完成"
echo "  Run A merged → ${OUT_RUN_A}"
echo "  Run B merged → ${OUT_RUN_B}（推荐）"
echo ""
echo "下一步：跑三方评估（Run B 主线 + base + DeepSeek）"
echo "  python ${REPO_ROOT}/finetune/scripts/evaluate_llm.py \\"
echo "      --base-model ${BASE_MODEL} \\"
echo "      --finetuned-model ${OUT_RUN_B} \\"
echo "      --judge-repeat 2"
echo "=========================================================="
