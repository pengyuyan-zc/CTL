#!/bin/bash

# Federated CTN-LT Quick Start Script
# 快速开始脚本

echo "=================================="
echo "Federated CTN-LT Quick Start"
echo "=================================="

# 1. 生成示例数据
echo ""
echo "Step 1: Generating sample data..."
python -m federated_ctn_lt.scripts.generate_sample_data \
    --num_documents 1000 \
    --num_labels 50 \
    --long_tail \
    --output_path data/sample_data.json

# 2. 训练模型（基础版本）
echo ""
echo "Step 2: Training basic federated model..."
python -m federated_ctn_lt.train \
    --data_path data/sample_data.json \
    --output_dir outputs/basic \
    --num_clients 5 \
    --num_rounds 10 \
    --batch_size 16 \
    --save_model

# 3. 训练模型（个性化版本）
echo ""
echo "Step 3: Training personalized federated model..."
python -m federated_ctn_lt.train \
    --data_path data/sample_data.json \
    --output_dir outputs/personalized \
    --num_clients 5 \
    --num_rounds 10 \
    --batch_size 16 \
    --personalized \
    --use_global_negatives \
    --save_model

echo ""
echo "=================================="
echo "Training completed!"
echo "Check outputs/ directory for results"
echo "=================================="
