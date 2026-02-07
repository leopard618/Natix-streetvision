# 🚀 START HERE - Natix Mining Training

## Your System is Ready!

✅ **GPU:** NVIDIA RTX 3090 (24GB VRAM) - Perfect!
✅ **Environment:** All dependencies installed
✅ **Training scripts:** Ready to use
✅ **Expected time:** 20-25 hours training

---

## Option 1: Automatic (Recommended)

Run everything automatically:

```bash
cd ~/Romeo/streetvision-subnet/natix_training
source ~/natix-mining-project/venv/bin/activate
./run_all.sh
```

This will:
1. Download datasets (2-4 hours)
2. Prepare data (1-2 hours)  
3. Train DINOv2-Large (20-25 hours)
4. Upload to HuggingFace

---

## Option 2: Step by Step

```bash
# Activate environment
source ~/natix-mining-project/venv/bin/activate
cd ~/Romeo/streetvision-subnet/natix_training

# Step 1: Download datasets
python download_datasets.py

# Step 2: Prepare data
python prepare_data.py

# Step 3: Train model (20-25 hours)
python train_dinov2.py

# Step 4: Test model
python inference.py path/to/test/image.jpg

# Step 5: Upload to HuggingFace
python upload_to_hf.py
```

---

## Monitor Training

**Terminal 1 - Training:**
```bash
python train_dinov2.py
```

**Terminal 2 - GPU Monitor:**
```bash
watch -n 1 nvidia-smi
```

**Terminal 3 - Disk Space:**
```bash
watch -n 60 df -h
```

---

## What You'll Get

After training completes:

- **Model:** DINOv2-Large fine-tuned for roadwork detection
- **Accuracy:** 93-96% expected
- **F1 Score:** 0.90-0.95 expected
- **Location:** `~/natix-mining-project/models/dinov2_roadwork_final/`

---

## Next Steps After Training

1. **Test your model:**
   ```bash
   python inference.py test_image.jpg
   ```

2. **Upload to HuggingFace:**
   ```bash
   python upload_to_hf.py
   ```

3. **Update miner config:**
   Edit `miner.env` in the natix-subnet repo:
   ```
   MODEL_URL=https://huggingface.co/your-username/dinov2-roadwork-detector
   ```

4. **Start mining:**
   ```bash
   ./start_miner.sh
   ```

---

## Files Created

```
natix_training/
├── download_datasets.py    # Download CSDS + Natix datasets
├── prepare_data.py          # Preprocess and split data
├── train_dinov2.py          # Train DINOv2-Large
├── inference.py             # Test trained model
├── upload_to_hf.py          # Upload to HuggingFace
├── run_all.sh               # Run complete pipeline
└── README.md                # Documentation
```

---

## Troubleshooting

**Out of Memory:**
```python
# Edit train_dinov2.py, line ~120
per_device_train_batch_size=12  # Reduce from 16
```

**Disk Space Full:**
```bash
# Delete old checkpoints
rm -rf ~/natix-mining-project/models/dinov2_roadwork/checkpoint-*
```

**Dataset Download Fails:**
- CSDS requires accepting terms: https://huggingface.co/datasets/issai/CSDS_dataset
- Login: `huggingface-cli login`

---

## Ready to Start!

Choose your option and run the commands above.

**Estimated total time:** 25-33 hours
**Your RTX 3090 is perfect for this task!**

Good luck! 🎉
