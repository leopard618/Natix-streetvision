#!/usr/bin/env python3
"""
Download CSDS and Natix Roadwork datasets for training.
"""

import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download
import shutil

# Setup directories
BASE_DIR = Path.home() / "natix-mining-project"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("DOWNLOADING DATASETS FOR NATIX MINING")
print("=" * 70)

# Download CSDS Dataset
print("\n[1/2] Downloading CSDS Dataset (Construction Site Detection)...")
print("This may take 30-60 minutes depending on your connection...")

try:
    csds_path = DATA_DIR / "csds"
    csds_path.mkdir(exist_ok=True)
    
    # Download CSDS dataset
    snapshot_download(
        repo_id="issai/CSDS_dataset",
        repo_type="dataset",
        local_dir=str(csds_path),
        ignore_patterns=["*.md", "*.txt"]
    )
    print("✓ CSDS Dataset downloaded successfully!")
    
    # Check size
    size = sum(f.stat().st_size for f in csds_path.rglob('*') if f.is_file())
    print(f"  Size: {size / 1024**3:.2f} GB")
    
except Exception as e:
    print(f"✗ Error downloading CSDS: {e}")
    print("  You may need to accept the dataset terms on HuggingFace first:")
    print("  https://huggingface.co/datasets/issai/CSDS_dataset")

# Download Natix Roadwork Dataset
print("\n[2/2] Downloading Natix Roadwork Dataset...")
print("This is the subnet-specific dataset...")

try:
    natix_path = DATA_DIR / "natix"
    natix_path.mkdir(exist_ok=True)
    
    # Download Natix dataset
    natix_dataset = load_dataset("natix-network-org/roadwork", split="train")
    
    # Save to disk
    natix_dataset.save_to_disk(str(natix_path))
    print("✓ Natix Roadwork Dataset downloaded successfully!")
    
    # Check size
    size = sum(f.stat().st_size for f in natix_path.rglob('*') if f.is_file())
    print(f"  Size: {size / 1024**3:.2f} GB")
    print(f"  Images: {len(natix_dataset)}")
    
except Exception as e:
    print(f"✗ Error downloading Natix dataset: {e}")

# Summary
print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)

total_size = 0
datasets_info = []

if (DATA_DIR / "csds").exists():
    csds_size = sum(f.stat().st_size for f in (DATA_DIR / "csds").rglob('*') if f.is_file())
    total_size += csds_size
    datasets_info.append(f"✓ CSDS: {csds_size / 1024**3:.2f} GB")
else:
    datasets_info.append("✗ CSDS: Not downloaded")

if (DATA_DIR / "natix").exists():
    natix_size = sum(f.stat().st_size for f in (DATA_DIR / "natix").rglob('*') if f.is_file())
    total_size += natix_size
    datasets_info.append(f"✓ Natix: {natix_size / 1024**3:.2f} GB")
else:
    datasets_info.append("✗ Natix: Not downloaded")

for info in datasets_info:
    print(info)

print(f"\nTotal downloaded: {total_size / 1024**3:.2f} GB")
print(f"Location: {DATA_DIR}")

# Check disk space
total, used, free = shutil.disk_usage("/")
free_gb = free / 1024**3
print(f"Remaining disk space: {free_gb:.1f} GB")

if free_gb < 30:
    print("\n⚠️  WARNING: Low disk space! Consider freeing up space before training.")

print("\n✓ Dataset download complete!")
print("Next step: python prepare_data.py")
print("=" * 70)
