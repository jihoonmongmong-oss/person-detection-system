import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset


# CONFIGURATION

class Config:
    # Dataset paths
    DATA_DIR = Path("dataset")
    PERSON_DIR = DATA_DIR / "person"
    NO_PERSON_DIR = DATA_DIR / "no_person"
    
    # Model path
    MODEL_PATH = Path("output/person_classifier.pth")
    
    # Output paths
    OUTPUT_DIR = Path("output")
    
    # Image settings
    IMG_SIZE = 224
    BATCH_SIZE = 16
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# CUSTOM DATASET
class PersonDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)

from torch.utils.data import Dataset


# LOAD MODEL & DATA

def load_model_and_data():
    
    # Load model
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
    
    print(f"Model loaded from: {Config.MODEL_PATH}")
    
    # Load all data
    image_paths = []
    labels = []
    
    # Person images (label = 1)
    for img_path in Config.PERSON_DIR.glob("*.jpg"):
        image_paths.append(img_path)
        labels.append(1)
    for img_path in Config.PERSON_DIR.glob("*.png"):
        image_paths.append(img_path)
        labels.append(1)
    
    # No person images (label = 0)
    for img_path in Config.NO_PERSON_DIR.glob("*.jpg"):
        image_paths.append(img_path)
        labels.append(0)
    for img_path in Config.NO_PERSON_DIR.glob("*.png"):
        image_paths.append(img_path)
        labels.append(0)
    
    print(f"\nTotal samples: {len(image_paths)}")
    print(f"  Person: {sum(labels)}")
    print(f"  No Person: {len(labels) - sum(labels)}")
    
    return model, image_paths, labels


# EVALUATION

def evaluate_model(model, image_paths, labels):
    
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and loader
    dataset = PersonDataset(image_paths, labels, transform)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # Get predictions
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, label_batch in loader:
            images = images.to(Config.DEVICE)
            outputs = model(images).squeeze()
            
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).cpu().numpy().astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(label_batch.numpy().astype(int))
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Person', 'Person']))
    
    return all_labels, all_preds, all_probs


# CONFUSION MATRIX

def plot_confusion_matrix(y_true, y_pred):
   
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Person Detection')
    plt.colorbar()
    
    classes = ['No Person', 'Person']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    cm_path = Config.OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    
    print(f"\nConfusion matrix saved to: {cm_path}")
    
    return cm


# SAMPLE PREDICTIONS

def visualize_predictions(image_paths, labels, predictions, probabilities, num_samples=16):
    
    # Select random samples
    indices = random.sample(range(len(image_paths)), min(num_samples, len(image_paths)))
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for idx, i in enumerate(indices):
        # Load and transform image
        img = Image.open(image_paths[i]).convert('RGB')
        img_display = img.resize((224, 224))
        
        true_label = 'Person' if labels[i] == 1 else 'No Person'
        pred_label = 'Person' if predictions[i] == 1 else 'No Person'
        confidence = probabilities[i] if predictions[i] == 1 else 1 - probabilities[i]
        
        color = 'green' if labels[i] == predictions[i] else 'red'
        
        axes[idx].imshow(img_display)
        axes[idx].axis('off')
        axes[idx].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}", 
                           color=color, fontsize=10)
    
    plt.suptitle('Sample Predictions\n(Green=Correct, Red=Wrong)', fontsize=14)
    plt.tight_layout()
    
    samples_path = Config.OUTPUT_DIR / "sample_predictions.png"
    plt.savefig(samples_path)
    plt.close()
    
    print(f"Sample predictions saved to: {samples_path}")


# MISCLASSIFIED EXAMPLES

def analyze_misclassified(image_paths, labels, predictions, probabilities):
  
    misclassified_idx = np.where(predictions != labels)[0]
    
    print(f"\n" + "=" * 60)
    print("MISCLASSIFIED EXAMPLES ANALYSIS")
    print("=" * 60)
    print(f"Total misclassified: {len(misclassified_idx)} out of {len(labels)}")
    
    if len(misclassified_idx) == 0:
        print("No misclassified examples!")
        return
    
    # Separate by type of error
    false_positives = [i for i in misclassified_idx if predictions[i] == 1 and labels[i] == 0]
    false_negatives = [i for i in misclassified_idx if predictions[i] == 0 and labels[i] == 1]
    
    print(f"\nFalse Positives (Predicted Person, Actually No Person): {len(false_positives)}")
    print(f"False Negatives (Predicted No Person, Actually Person): {len(false_negatives)}")
    
    # Visualize misclassified
    num_each = min(4, len(false_positives), len(false_negatives))
    if num_each == 0:
        return
    
    fig, axes = plt.subplots(2, num_each, figsize=(12, 6))
    
    # Plot false positives
    for i, idx in enumerate(false_positives[:num_each]):
        img = Image.open(image_paths[idx]).convert('RGB')
        img = img.resize((224, 224))
        confidence = probabilities[idx]
        
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Pred: Person\nConf: {confidence:.2f}", color='red', fontsize=10)
    
    # Plot false negatives
    for i, idx in enumerate(false_negatives[:num_each]):
        img = Image.open(image_paths[idx]).convert('RGB')
        img = img.resize((224, 224))
        confidence = 1 - probabilities[idx]
        
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Pred: No Person\nConf: {confidence:.2f}", color='red', fontsize=10)
    
    # Add row labels
    if num_each > 0:
        axes[0, 0].set_ylabel('False Positives\n(Pred: Person, True: No Person)', fontsize=10)
        axes[1, 0].set_ylabel('False Negatives\n(Pred: No Person, True: Person)', fontsize=10)
    
    plt.suptitle('Misclassified Examples', fontsize=14)
    plt.tight_layout()
    
    misclassified_path = Config.OUTPUT_DIR / "misclassified_examples.png"
    plt.savefig(misclassified_path)
    plt.close()
    
    print(f"Misclassified examples saved to: {misclassified_path}")


# VISUALIZE RANDOM SAMPLES

def visualize_random_samples(image_paths, labels):
    
    # Select random samples from each class
    person_indices = [i for i, l in enumerate(labels) if l == 1]
    no_person_indices = [i for i, l in enumerate(labels) if l == 0]
    
    person_sample = random.sample(person_indices, min(8, len(person_indices)))
    no_person_sample = random.sample(no_person_indices, min(8, len(no_person_indices)))
    
    fig, axes = plt.subplots(2, 8, figsize=(16, 6))
    
    # Plot person samples
    for i, idx in enumerate(person_sample):
        img = Image.open(image_paths[idx]).convert('RGB')
        img = img.resize((224, 224))
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        axes[0, i].set_title('Person', fontsize=10)
    
    # Plot no_person samples
    for i, idx in enumerate(no_person_sample):
        img = Image.open(image_paths[idx]).convert('RGB')
        img = img.resize((224, 224))
        axes[1, i].imshow(img)
        axes[1, i].axis('off')
        axes[1, i].set_title('No Person', fontsize=10)
    
    axes[0, 0].set_ylabel('Person', fontsize=12)
    axes[1, 0].set_ylabel('No Person', fontsize=12)
    
    plt.suptitle('Random Samples from Dataset', fontsize=14)
    plt.tight_layout()
    
    samples_path = Config.OUTPUT_DIR / "dataset_samples.png"
    plt.savefig(samples_path)
    plt.close()
    
    print(f"Dataset samples saved to: {samples_path}")


# MAIN

def main():
    print("=" * 60)
    print("PERSON DETECTION EVALUATION")
    print("=" * 60)
    
    # Load model and data
    model, image_paths, labels = load_model_and_data()
    
    # Evaluate model
    y_true, y_pred, y_prob = evaluate_model(model, image_paths, labels)
    
    # Confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred)
    
    # Sample predictions
    visualize_predictions(image_paths, y_true, y_pred, y_prob)
    
    # Misclassified examples
    analyze_misclassified(image_paths, y_true, y_pred, y_prob)
    
    # Random samples
    visualize_random_samples(image_paths, labels)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {Config.OUTPUT_DIR}")

if __name__ == "__main__":
    main()