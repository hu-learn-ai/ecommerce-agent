#!/bin/bash
# ============================================================
# AutoDL GPU 训练脚本 — 4 类真实数据 (京东商品分类)
# 使用方法:
#   1. 上传本项目到服务器 /root/ecommerce-agent
#   2. cd /root/ecommerce-agent && bash scripts/gpu_train.sh
# ============================================================
set -e

echo "============================================"
echo "  4 类真实数据 — GPU 训练"
echo "============================================"

# Step 0: GPU 检测
if ! command -v nvidia-smi &> /dev/null; then
    echo "[ERROR] 未检测到 NVIDIA GPU"; exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PYTHON_BIN=${PYTHON_BIN:-python3}
echo "  Python: $($PYTHON_BIN --version 2>&1)"

# Step 1: 依赖 (AutoDL PyTorch 镜像自带 CUDA 版 torch)
echo "[Step 1] 安装依赖..."
$PYTHON_BIN -m pip install -q "transformers>=4.40.0,<5.0.0" "accelerate>=0.26.0" scikit-learn numpy

# Step 2: HuggingFace 国内镜像
echo "[Step 2] 配置 HF 镜像..."
export HF_ENDPOINT="https://hf-mirror.com"

# Step 3: 数据检查
echo "[Step 3] 检查数据..."
PROJECT_ROOT="/root/ecommerce-agent"
cd "$PROJECT_ROOT"
DATA_FILE="${DATA_FILE:-$PROJECT_ROOT/data/processed/classify_train_real_aug_train.csv}"
VAL_FILE="${VAL_FILE:-$PROJECT_ROOT/data/processed/classify_train_real_val.csv}"

for f in "$DATA_FILE" "$VAL_FILE"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] 缺少数据文件: $f"; exit 1
    fi
done
echo "  训练数据: $DATA_FILE ($(( $(wc -l < "$DATA_FILE") - 1 )) 条)"
echo "  验证数据: $VAL_FILE ($(( $(wc -l < "$VAL_FILE") - 1 )) 条)"

# Step 4: 训练 (首次运行会自动从 hf-mirror 下载 bert-base-chinese)
echo "[Step 4] 开始训练..."
echo "  参数: epochs=5, batch_size=32, lr=2e-5, max_length=128"

$PYTHON_BIN scripts/train_classify_model.py \
    --data "$DATA_FILE" \
    --val_data "$VAL_FILE" \
    --epochs 5 \
    --batch_size 32 \
    --lr 2e-5 \
    --max_length 128 \
    --output_dir "$PROJECT_ROOT/models/product_classification/checkpoint/best"

# Step 5: 验证并打包
echo ""
echo "[Step 5] 训练结果:"
BEST_DIR="$PROJECT_ROOT/models/product_classification/checkpoint/best"
if [ -f "$BEST_DIR/training_config.json" ]; then
    $PYTHON_BIN -c "
import json
with open('$BEST_DIR/training_config.json') as f:
    cfg = json.load(f)
print(f\"  类别数: {cfg['num_labels']}  标签: {cfg['labels']}\")
print(f\"  训练样本: {cfg['train_samples']}  验证样本: {cfg['val_samples']}\")
print(f\"  准确率: {cfg['eval_result'].get('eval_accuracy', 0):.4f}  F1: {cfg['eval_result'].get('eval_f1', 0):.4f}\")
"
    cd "$PROJECT_ROOT/models/product_classification/checkpoint"
    tar -czf /root/best_model_4class.tar.gz best/
    echo "  模型包: /root/best_model_4class.tar.gz ($(du -sh /root/best_model_4class.tar.gz | cut -f1))"
else
    echo "[ERROR] 未找到训练结果!"; exit 1
fi

echo ""
echo "============================================"
echo "  训练完成! 下载: scp -P <端口> root@<地址>:/root/best_model_4class.tar.gz ./"
echo "============================================"
