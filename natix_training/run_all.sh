#!/bin/bash
# Complete training pipeline

set -e

VENV_PATH=~/natix-mining-project/venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "NATIX MINING - COMPLETE TRAINING PIPELINE"
echo "======================================================================"

# Activate venv
source $VENV_PATH/bin/activate

# Step 1: Download datasets
echo ""
echo "[STEP 1/4] Downloading datasets..."
python $SCRIPT_DIR/download_datasets.py

# Step 2: Prepare data
echo ""
echo "[STEP 2/4] Preparing data..."
python $SCRIPT_DIR/prepare_data.py

# Step 3: Train model
echo ""
echo "[STEP 3/4] Training DINOv2-Large..."
echo "This will take 20-25 hours on RTX 3090"
python $SCRIPT_DIR/train_dinov2.py

# Step 4: Upload to HuggingFace
echo ""
echo "[STEP 4/4] Upload to HuggingFace..."
python $SCRIPT_DIR/upload_to_hf.py

echo ""
echo "======================================================================"
echo "✓ COMPLETE PIPELINE FINISHED!"
echo "======================================================================"
