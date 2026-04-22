import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import sys
from ultralytics import YOLO


# CONFIGURATION
class Config:
    MODEL_PATH = Path("output/person_classifier.pth")
    IMG_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    YOLO_MODEL = "yolov8m.pt"  # Small model: yolov8n, Medium: yolov8m, Large: yolov8l


# GLOBAL MODEL VARIABLES
_classification_model = None
_detection_model = None


# MODEL LOADING
def load_classification_model():
    global _classification_model
    if _classification_model is None:
        print("Loading classification model...")
        model = models.mobilenet_v2()
        model.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=Config.DEVICE))
        model = model.to(Config.DEVICE)
        model.eval()
        _classification_model = model
        print(f"Classification model loaded from: {Config.MODEL_PATH}")
    return _classification_model


def load_detection_model():
    global _detection_model
    if _detection_model is None:
        print("Loading YOLOv8 detection model...")
        _detection_model = YOLO(Config.YOLO_MODEL)
        _detection_model.to(Config.DEVICE)
        print(f"YOLOv8 model ({Config.YOLO_MODEL}) loaded successfully!")
    return _detection_model


# PRE-LOAD MODELS ON IMPORT
print("Pre-loading models...")
load_classification_model()
load_detection_model()
print("All models loaded and ready!")


# PREDICTION FUNCTION
def predict_image(image_path):

    # Load models
    classification_model = load_classification_model()
    
    # Transform for classification
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and transform image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {
            'label': 'error',
            'confidence': 0.0,
            'probability_person': 0.0,
            'error': str(e)
        }
    
    image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
    
    # Get classification prediction
    with torch.no_grad():
        output = classification_model(image_tensor).squeeze()
        probability = output.item()
    
    # Get detection results to verify if there's at least 1 person
    detection_result = count_people(image_path)
    people_count = detection_result['people_count']
    
    # Determine label: if at least 1 person is detected, classify as 'person'
    if people_count > 0:
        # Person detected by object detection model
        label = 'person'
        confidence = 0.99  # High confidence since we have object detection confirmation
    elif probability > 0.5:
        # Classification model says person
        label = 'person'
        confidence = probability
    else:
        # Classification model says no person and no detection
        label = 'no_person'
        confidence = 1 - probability
    
    return {
        'label': label,
        'confidence': confidence,
        'probability_person': probability,
        'people_count': people_count
    }


# COUNT PEOPLE FUNCTION USING YOLOv8
def count_people(image_path):
    # Load YOLOv8 model
    model = load_detection_model()
    
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {'people_count': 0, 'other_detections': []}
    
    # Run YOLOv8 inference
    results = model(image_path, conf=0.5, verbose=False)
    
    people_count = 0
    other_detections = []
    
    # Process detections
    for result in results:
        for detection in result.boxes.data:
            cls_id = int(detection[5])  # Class ID
            confidence = float(detection[4])  # Confidence score
            class_name = result.names[cls_id]  # Class name from model
            
            if class_name.lower() == 'person':
                people_count += 1
            else:
                other_detections.append({
                    'class': class_name,
                    'confidence': confidence
                })
    
    return {
        'people_count': people_count,
        'other_detections': other_detections
    }


def predict_batch(image_paths):
    
    model = load_classification_model()
    
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(Config.DEVICE)
            
            with torch.no_grad():
                output = model(image_tensor).squeeze()
                probability = output.item()
            
            # Get detection results to verify if there's at least 1 person
            detection_result = count_people(image_path)
            people_count = detection_result['people_count']
            
            # Determine label: if at least 1 person is detected, classify as 'person'
            if people_count > 0:
                label = 'person'
                confidence = 0.99
            elif probability > 0.5:
                label = 'person'
                confidence = probability
            else:
                label = 'no_person'
                confidence = 1 - probability
            
            results.append({
                'image': str(image_path),
                'label': label,
                'confidence': confidence,
                'probability_person': probability,
                'people_count': people_count
            })
            
        except Exception as e:
            results.append({
                'image': str(image_path),
                'label': 'error',
                'confidence': 0.0,
                'probability_person': 0.0,
                'error': str(e)
            })
    
    return results


# MAIN

def main():
    """Main function for command-line usage."""
    
    if len(sys.argv) < 2:
        print("Usage: python predict_person.py <image_path>")
        print("Or: python predict_person.py <image1> <image2> ...")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    if len(image_paths) == 1:
        # Single image
        result = predict_image(image_paths[0])
        
        print("\n" + "=" * 40)
        print("PREDICTION RESULT")
        print("=" * 40)
        print(f"Image: {image_paths[0]}")
        print(f"Label: {result['label']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print(f"Probability (Person): {result['probability_person']:.4f}")
        
    else:
        # Multiple images
        results = predict_batch(image_paths)
        
        print("\n" + "=" * 40)
        print("BATCH PREDICTION RESULTS")
        print("=" * 40)
        
        for result in results:
            print(f"\nImage: {result['image']}")
            print(f"  Label: {result['label']}")
            print(f"  Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()