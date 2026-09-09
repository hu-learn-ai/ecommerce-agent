#!/bin/bash
# ============================================================
# AutoDL 一键训练 — 分类模型（增强数据）+ NER 模型
# 一次 GPU 会话完成所有训练任务
# ============================================================
set -e

echo "============================================"
echo "  一键训练: 4 类分类(增强) + NER 实体抽取"
echo "============================================"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
PYTHON_BIN=${PYTHON_BIN:-python3}

echo "[Step 1] 安装依赖..."
$PYTHON_BIN -m pip install -q "transformers>=4.40.0,<5.0.0" "accelerate>=0.26.0" scikit-learn numpy
export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="/root/ecommerce-agent"
cd "$PROJECT_ROOT"

# 数据检查
for f in \
    data/processed/classify_train_real_aug_train.csv \
    data/processed/classify_train_real_val.csv \
    data/processed/ner/train.txt \
    data/processed/ner/dev.txt \
    data/processed/ner/test.txt; do
    if [ ! -f "$f" ]; then echo "[ERROR] 缺少数据: $f"; exit 1; fi
done

echo "[Step 2] 训练分类模型（增强数据，修复短查询误判）..."
if [ -f "$PROJECT_ROOT/models/product_classification/checkpoint/best/training_config.json" ]; then
    echo "[Step 2] 检测到分类模型已存在，跳过分类训练"
else
    $PYTHON_BIN scripts/train_classify_model.py \
        --data "$PROJECT_ROOT/data/processed/classify_train_real_aug_train.csv" \
        --val_data "$PROJECT_ROOT/data/processed/classify_train_real_val.csv" \
        --epochs 5 \
        --batch_size 32 \
        --lr 2e-5 \
        --max_length 128 \
        --output_dir "$PROJECT_ROOT/models/product_classification/checkpoint/best"
fi

echo "[Step 3] 训练 NER 实体抽取模型（修复评估记录）..."
if [ -f "$PROJECT_ROOT/models/ner/best_model/training_config.json" ]; then
    echo "[Step 3] 检测到 NER 模型已存在，跳过 NER 训练"
else
    $PYTHON_BIN scripts/train_ner.py \
        --epochs 5 \
        --batch_size 32 \
        --lr 2e-5 \
        --max_length 128 \
        --output_dir "$PROJECT_ROOT/models/ner/best_model"
fi

echo "[Step 4] 打包模型..."
cd "$PROJECT_ROOT"
tar -czf /root/classify_model.tar.gz -C models/product_classification/checkpoint best
tar -czf /root/ner_model.tar.gz -C models/ner best_model
ls -lh /root/*.tar.gz

echo ""
echo "============================================"
echo "  全部训练完成!"
echo "  下载:"
echo "    scp -P <端口> root@<地址>:/root/classify_model.tar.gz ./"
echo "    scp -P <端口> root@<地址>:/root/ner_model.tar.gz ./"
echo "============================================"
