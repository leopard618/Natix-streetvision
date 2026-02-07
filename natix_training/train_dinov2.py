#!/usr/bin/env python3
"""
Train DINOv2-Large for roadwork detection on Natix subnet.
Optimized for RTX 3090 (24GB VRAM).
"""

import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import (
    Dinov2ForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Setup
BASE_DIR = Path.home() / "natix-mining-project"
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("TRAINING DINOV2-LARGE FOR NATIX MINING")
print("=" * 70)

# Check GPU
if not torch.cuda.is_available():
    print("✗ No GPU detected! Training will be very slow.")
    exit(1)

print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Load splits
print("\nLoading dataset splits...")
with open(DATA_DIR / "splits.json", 'r') as f:
    splits = json.load(f)

with open(DATA_DIR / "dataset_info.json", 'r') as f:
    dataset_info = json.load(f)

print(f"✓ Train: {dataset_info['train_size']} images")
print(f"✓ Val: {dataset_info['val_size']} images")
print(f"✓ Test: {dataset_info['test_size']} images")

# Custom Dataset
class RoadworkDataset(Dataset):
    def __init__(self, image_paths, labels, processor, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        
        if augment:
            self.transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.3),
                A.RandomRain(rain_type='drizzle', p=0.2),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.MotionBlur(blur_limit=7, p=0.3),
                A.Rotate(limit=15, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            # Return a blank image
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply augmentation
        if self.augment:
            image_np = np.array(image)
            augmented = self.transform(image=image_np)
            image = Image.fromarray(augmented['image'])
        
        # Process for DINOv2
        inputs = self.processor(images=image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load processor and model
print("\nLoading DINOv2-Large model...")
print("This will download ~1.2 GB on first run...")

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")

model = Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-large",
    num_labels=2,
    ignore_mismatched_sizes=True
)

print("✓ Model loaded successfully!")

# Create datasets
print("\nCreating datasets...")

train_dataset = RoadworkDataset(
    splits['train']['images'],
    splits['train']['labels'],
    processor,
    augment=True
)

val_dataset = RoadworkDataset(
    splits['val']['images'],
    splits['val']['labels'],
    processor,
    augment=False
)

print(f"✓ Train dataset: {len(train_dataset)} samples")
print(f"✓ Val dataset: {len(val_dataset)} samples")

# Metrics
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Training configuration
print("\nConfiguring training...")
print("Optimized for RTX 3090 (24GB VRAM)")

training_args = TrainingArguments(
    output_dir=str(MODEL_DIR / "dinov2_roadwork"),
    
    # Training duration
    num_train_epochs=100,
    
    # Batch sizes (optimized for 24GB VRAM)
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,  # Effective batch = 64
    
    # Learning rate
    learning_rate=5e-6,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    
    # Regularization
    weight_decay=0.01,
    
    # Optimization
    fp16=True,  # Mixed precision for speed
    optim="adamw_torch",
    
    # Evaluation & saving
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    save_total_limit=1,  # Only keep 1 best checkpoint to save disk space!
    
    # Logging
    logging_dir=str(MODEL_DIR / "logs"),
    logging_steps=50,
    report_to="none",  # Change to "wandb" if you want W&B tracking
    
    # Reproducibility
    seed=42,
    data_seed=42,
)

print(f"✓ Batch size: {training_args.per_device_train_batch_size}")
print(f"✓ Effective batch: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"✓ Learning rate: {training_args.learning_rate}")
print(f"✓ Epochs: {training_args.num_train_epochs}")
print(f"✓ Mixed precision: {training_args.fp16}")

# Initialize trainer
print("\nInitializing trainer...")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("✓ Trainer initialized!")

# Start training
print("\n" + "=" * 70)
print("STARTING TRAINING")
print("=" * 70)
print(f"Expected time: 20-25 hours on RTX 3090")
print(f"Monitor GPU: watch -n 1 nvidia-smi")
print("=" * 70 + "\n")

try:
    trainer.train()
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE!")
    print("=" * 70)
    
    # Save final model
    final_model_path = MODEL_DIR / "dinov2_roadwork_final"
    trainer.save_model(str(final_model_path))
    processor.save_pretrained(str(final_model_path))
    
    print(f"\n✓ Model saved to: {final_model_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_dataset = RoadworkDataset(
        splits['test']['images'],
        splits['test']['labels'],
        processor,
        augment=False
    )
    
    test_results = trainer.evaluate(test_dataset)
    print("\nTest Results:")
    for key, value in test_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save test results
    with open(final_model_path / "test_results.json", 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\n" + "=" * 70)
    print("✓ ALL DONE!")
    print("=" * 70)
    print(f"Model location: {final_model_path}")
    print("Next step: python upload_to_hf.py")
    print("=" * 70)
    
except KeyboardInterrupt:
    print("\n\n✗ Training interrupted by user")
    print("Partial model saved in checkpoints")
except Exception as e:
    print(f"\n\n✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
