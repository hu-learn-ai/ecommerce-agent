#!/bin/bash
# ============================================================
# AutoDL GPU 训练 — 电商 NER 实体抽取模型
# 数据: data/processed/ner/{train,dev,test}.txt
# 输出: models/ner/best_model/
# ============================================================
set -e

echo "============================================"
echo "  电商 NER 实体抽取 — GPU 训练"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHON_BIN=${PYTHON_BIN:-python3}

$PYTHON_BIN -m pip install -q "transformers>=4.40.0,<5.0.0" "accelerate>=0.26.0" scikit-learn numpy
export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="/root/ecommerce-agent"
cd "$PROJECT_ROOT"

for f in data/processed/ner/train.txt data/processed/ner/dev.txt data/processed/ner/test.txt; do
    if [ ! -f "$f" ]; then echo "[ERROR] 缺少数据: $f"; exit 1; fi
done

echo "[Train] 数据检查通过，开始训练..."
$PYTHON_BIN scripts/train_ner.py \
    --epochs 5 \
    --batch_size 32 \
    --lr 2e-5 \
    --max_length 128 \
    --output_dir "$PROJECT_ROOT/models/ner/best_model"

echo ""
echo "============================================"
echo "  训练完成! 下载: scp -P <端口> root@<地址>:/root/ecommerce-agent/models/ner/best_model ./"
echo "============================================"
