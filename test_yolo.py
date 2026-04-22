#!/usr/bin/env python
"""
Test script to verify YOLOv8 integration
"""
import os
import sys
from pathlib import Path

print("="*60)
print("YOLOv8 INTEGRATION TEST")
print("="*60)

# Check if models can be loaded
try:
    print("\n1. Testing YOLOv8 model loading...")
    from predict_person import load_detection_model, load_classification_model
    
    print("   Loading YOLOv8 model...")
    yolo_model = load_detection_model()
    print("   ✓ YOLOv8 model loaded successfully!")
    
    print("   Loading MobileNetV2 classification model...")
    classification_model = load_classification_model()
    print("   ✓ Classification model loaded successfully!")
    
except Exception as e:
    print(f"   ✗ Error loading models: {e}")
    sys.exit(1)

# Test with a sample image if available
print("\n2. Testing with sample images...")
dataset_dir = Path("dataset")
person_dir = dataset_dir / "person"
no_person_dir = dataset_dir / "no_person"

# Find first image
test_image = None
if person_dir.exists():
    images = list(person_dir.glob("*.jpg")) + list(person_dir.glob("*.png"))
    if images:
        test_image = images[0]
        print(f"   Found test image: {test_image.name}")

if test_image:
    try:
        print(f"   Running prediction on: {test_image}")
        from predict_person import predict_image, count_people
        
        result = predict_image(str(test_image))
        detection = count_people(str(test_image))
        
        print(f"\n   Prediction Results:")
        print(f"   - Label: {result['label']}")
        print(f"   - Confidence: {result['confidence']:.2%}")
        print(f"   - People Count: {detection['people_count']}")
        if detection['other_detections']:
            print(f"   - Other Detections: {len(detection['other_detections'])} items")
        
        print("\n   ✓ Prediction successful!")
        
    except Exception as e:
        print(f"   ✗ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
else:
    print("   ⚠ No test images found in dataset")

print("\n" + "="*60)
print("✓ YOLOv8 INTEGRATION SUCCESSFUL!")
print("="*60)
print("\nYou can now run the Flask app:")
print("  python app_person.py")
print("\nThen visit: http://127.0.0.1:5002")
