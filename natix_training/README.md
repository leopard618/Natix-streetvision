# Natix Mining Training Scripts

Complete training pipeline for DINOv2-Large on Natix subnet.

## Quick Start

```bash
# Activate environment
source ~/natix-mining-project/venv/bin/activate

# Run complete pipeline (automatic)
cd natix_training
./run_all.sh
```

## Manual Steps

```bash
# 1. Download datasets (2-4 hours)
python download_datasets.py

# 2. Prepare data (1-2 hours)
python prepare_data.py

# 3. Train model (20-25 hours)
python train_dinov2.py

# 4. Test inference
python inference.py path/to/test/image.jpg

# 5. Upload to HuggingFace
python upload_to_hf.py
```

## Files

- `download_datasets.py` - Download CSDS + Natix datasets
- `prepare_data.py` - Preprocess and split data
- `train_dinov2.py` - Train DINOv2-Large model
- `inference.py` - Test trained model
- `upload_to_hf.py` - Upload to HuggingFace
- `run_all.sh` - Run complete pipeline

## Expected Results

- Accuracy: 93-96%
- F1 Score: 0.90-0.95
- Training time: 20-25 hours (RTX 3090)
