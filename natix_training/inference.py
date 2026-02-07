#!/usr/bin/env python3
"""
Inference script for roadwork detection.
Use this to test your trained model.
"""

import torch
from transformers import Dinov2ForImageClassification, AutoImageProcessor
from PIL import Image
from pathlib import Path
import sys

class RoadworkDetector:
    def __init__(self, model_path):
        """Initialize detector with trained model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and processor
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = Dinov2ForImageClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model loaded from: {model_path}")
    
    def predict(self, image_path):
        """
        Predict if image contains roadwork.
        
        Args:
            image_path: Path to image file
            
        Returns:
            float: Probability of roadwork (0.0 to 1.0)
                   >0.5 = roadwork present
                   <=0.5 = no roadwork
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            roadwork_prob = probs[0][1].item()
        
        return roadwork_prob
    
    def predict_batch(self, image_paths):
        """Predict for multiple images."""
        results = []
        for img_path in image_paths:
            prob = self.predict(img_path)
            results.append({
                'image': str(img_path),
                'probability': prob,
                'prediction': 'roadwork' if prob > 0.5 else 'no_roadwork'
            })
        return results

def main():
    """Test the detector."""
    BASE_DIR = Path.home() / "natix-mining-project"
    MODEL_PATH = BASE_DIR / "models" / "dinov2_roadwork_final"
    
    if not MODEL_PATH.exists():
        print(f"✗ Model not found at: {MODEL_PATH}")
        print("Please train the model first: python train_dinov2.py")
        return
    
    # Initialize detector
    detector = RoadworkDetector(str(MODEL_PATH))
    
    # Test with command line argument
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"✗ Image not found: {image_path}")
            return
        
        print(f"\nAnalyzing: {image_path}")
        prob = detector.predict(image_path)
        
        print(f"\nResults:")
        print(f"  Roadwork probability: {prob:.4f}")
        print(f"  Prediction: {'ROADWORK' if prob > 0.5 else 'NO ROADWORK'}")
    else:
        print("\nUsage: python inference.py <image_path>")
        print("Example: python inference.py test_image.jpg")

if __name__ == "__main__":
    main()
