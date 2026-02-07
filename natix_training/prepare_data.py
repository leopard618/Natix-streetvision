#!/usr/bin/env python3
"""
Prepare and preprocess datasets for training.
Combines CSDS and Natix datasets, creates train/val/test splits.
"""

import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil

BASE_DIR = Path.home() / "natix-mining-project"
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("PREPARING TRAINING DATA")
print("=" * 70)

# Step 1: Process CSDS Dataset
print("\n[1/3] Processing CSDS Dataset...")

csds_images = []
csds_labels = []

csds_path = DATA_DIR / "csds"
if csds_path.exists():
    # CSDS contains construction sites (label = 1 for roadwork)
    # Look for image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for img_path in tqdm(list(csds_path.rglob('*')), desc="Scanning CSDS"):
        if img_path.suffix.lower() in image_extensions:
            try:
                # Verify image can be opened
                with Image.open(img_path) as img:
                    if img.mode in ['RGB', 'L']:
                        csds_images.append(str(img_path))
                        csds_labels.append(1)  # Construction site = roadwork
            except:
                continue
    
    print(f"✓ Found {len(csds_images)} valid images in CSDS")
else:
    print("✗ CSDS dataset not found")

# Step 2: Process Natix Dataset
print("\n[2/3] Processing Natix Roadwork Dataset...")

natix_images = []
natix_labels = []

natix_path = DATA_DIR / "natix"
if natix_path.exists():
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(str(natix_path))
        
        # Save images and collect paths
        natix_img_dir = PROCESSED_DIR / "natix_images"
        natix_img_dir.mkdir(exist_ok=True)
        
        for idx, item in enumerate(tqdm(dataset, desc="Processing Natix")):
            # Assuming dataset has 'image' and 'label' fields
            # Adjust based on actual dataset structure
            img_path = natix_img_dir / f"image_{idx:06d}.jpg"
            
            if 'image' in item:
                img = item['image']
                if isinstance(img, Image.Image):
                    img.save(img_path)
                    natix_images.append(str(img_path))
                    
                    # Get label (1 = roadwork, 0 = no roadwork)
                    label = item.get('label', 1)
                    natix_labels.append(label)
        
        print(f"✓ Processed {len(natix_images)} images from Natix dataset")
    except Exception as e:
        print(f"✗ Error processing Natix dataset: {e}")
else:
    print("✗ Natix dataset not found")

# Step 3: Combine and split datasets
print("\n[3/3] Creating train/val/test splits...")

all_images = csds_images + natix_images
all_labels = csds_labels + natix_labels

print(f"\nTotal images: {len(all_images)}")
print(f"  - Roadwork (label=1): {sum(all_labels)}")
print(f"  - No roadwork (label=0): {len(all_labels) - sum(all_labels)}")

if len(all_images) == 0:
    print("\n✗ No images found! Please check dataset downloads.")
    exit(1)

# Create balanced dataset if needed
# For now, use all data

# Split: 70% train, 15% val, 15% test
train_imgs, temp_imgs, train_labels, temp_labels = train_test_split(
    all_images, all_labels, test_size=0.3, random_state=42, stratify=all_labels
)

val_imgs, test_imgs, val_labels, test_labels = train_test_split(
    temp_imgs, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"\nSplit sizes:")
print(f"  Train: {len(train_imgs)} images")
print(f"  Val:   {len(val_imgs)} images")
print(f"  Test:  {len(test_imgs)} images")

# Save splits
splits = {
    'train': {'images': train_imgs, 'labels': train_labels},
    'val': {'images': val_imgs, 'labels': val_labels},
    'test': {'images': test_imgs, 'labels': test_labels}
}

splits_file = PROCESSED_DIR / "splits.json"
with open(splits_file, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"\n✓ Splits saved to: {splits_file}")

# Create dataset info
dataset_info = {
    'total_images': len(all_images),
    'num_classes': 2,
    'class_names': ['no_roadwork', 'roadwork'],
    'train_size': len(train_imgs),
    'val_size': len(val_imgs),
    'test_size': len(test_imgs),
    'train_distribution': {
        'no_roadwork': len([l for l in train_labels if l == 0]),
        'roadwork': len([l for l in train_labels if l == 1])
    }
}

info_file = PROCESSED_DIR / "dataset_info.json"
with open(info_file, 'w') as f:
    json.dump(dataset_info, f, indent=2)

print(f"✓ Dataset info saved to: {info_file}")

# Check disk space
total, used, free = shutil.disk_usage("/")
free_gb = free / 1024**3
print(f"\nRemaining disk space: {free_gb:.1f} GB")

print("\n" + "=" * 70)
print("✓ Data preparation complete!")
print("Next step: python train_dinov2.py")
print("=" * 70)
