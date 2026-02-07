#!/usr/bin/env python3
"""Upload trained model to HuggingFace."""

import json
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi, create_repo

BASE_DIR = Path.home() / "natix-mining-project"
MODEL_PATH = BASE_DIR / "models" / "dinov2_roadwork_final"

print("=" * 70)
print("UPLOAD MODEL TO HUGGINGFACE")
print("=" * 70)

if not MODEL_PATH.exists():
    print("✗ Model not found. Train first!")
    exit(1)

# Get your info
print("\nEnter your details:")
hf_username = input("HuggingFace username: ").strip()
hotkey_address = input("Your wallet hotkey address: ").strip()

# Create model card
model_card = {
    "model_name": "DINOv2-Large-Roadwork-Detector",
    "description": "Fine-tuned DINOv2-large for roadwork detection on Natix subnet",
    "version": "1.0.0",
    "submitted_by": hotkey_address,
    "submission_time": int(datetime.now().timestamp())
}

with open(MODEL_PATH / "model_card.json", "w") as f:
    json.dump(model_card, f, indent=2)

print(f"\n✓ Model card created")

# Upload
repo_name = f"{hf_username}/dinov2-roadwork-detector"
print(f"\nUploading to: {repo_name}")

try:
    api = HfApi()
    create_repo(repo_name, exist_ok=True)
    
    api.upload_folder(
        folder_path=str(MODEL_PATH),
        repo_id=repo_name,
        repo_type="model"
    )
    
    print(f"\n✓ Model uploaded!")
    print(f"URL: https://huggingface.co/{repo_name}")
    print(f"\nUpdate your miner.env:")
    print(f"MODEL_URL=https://huggingface.co/{repo_name}")
    
except Exception as e:
    print(f"✗ Upload failed: {e}")
