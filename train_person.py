import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

class Config:
    DATA_DIR = Path("dataset")
    PERSON_DIR = DATA_DIR / "person"
    NO_PERSON_DIR = DATA_DIR / "no_person"
    
    OUTPUT_DIR = Path("output")
    MODEL_PATH = OUTPUT_DIR / "person_classifier.pth"
    
    IMG_SIZE = 224
    BATCH_SIZE = 8
    
    EPOCHS = 5
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.3
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Config.OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Using device: {Config.DEVICE}")


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
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (Config.IMG_SIZE, Config.IMG_SIZE), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def load_dataset():
    image_paths = []
    labels = []
    
    person_images = list(Config.PERSON_DIR.glob("*.jpg")) + list(Config.PERSON_DIR.glob("*.png"))
    for img_path in person_images:
        image_paths.append(img_path)
        labels.append(1)
    
    no_person_images = list(Config.NO_PERSON_DIR.glob("*.jpg")) + list(Config.NO_PERSON_DIR.glob("*.png"))
    for img_path in no_person_images:
        image_paths.append(img_path)
        labels.append(0)
    
    print(f"\nDataset loaded:")
    print(f"  Person images: {len(person_images)}")
    print(f"  No person images: {len(no_person_images)}")
    print(f"  Total: {len(image_paths)}")
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=Config.VALIDATION_SPLIT,
        random_state=SEED,
        stratify=labels
    )
    
    print(f"\nData split (70/30):")
    print(f"  Training: {len(train_paths)} images")
    print(f"  Validation: {len(val_paths)} images")
    
    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    
    print(f"\n  Training - Person: {train_pos}, No Person: {train_neg}")
    print(f"  Validation - Person: {val_pos}, No Person: {val_neg}")
    
    return train_paths, train_labels, val_paths, val_labels


def build_model():
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.Linear(1280, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    
    model = model.to(Config.DEVICE)
    
    return model

def fine_tune_model(model):
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=Config.LEARNING_RATE / 10)
    
    return model, optimizer


def train_model():
    print("=" * 60)
    print("PERSON DETECTION TRAINING PIPELINE")
    print("=" * 60)
    
    train_paths, train_labels, val_paths, val_labels = load_dataset()
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = PersonDataset(train_paths, train_labels, train_transform)
    val_dataset = PersonDataset(val_paths, val_labels, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    model = build_model()
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    print("\n" + "=" * 40)
    print("PHASE 1: Training with frozen base")
    print("=" * 40)
    
    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {Config.MODEL_PATH}")
    
    print("\n" + "=" * 40)
    print("PHASE 2: Fine-tuning")
    print("=" * 40)
    
    model, optimizer = fine_tune_model(model)
    
    for epoch in range(Config.EPOCHS // 2):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = train_correct / train_total
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.DEVICE), labels.to(Config.DEVICE)
                outputs = model(images).squeeze()
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{Config.EPOCHS//2} - "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.MODEL_PATH)
    
    print(f"\nFinal best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {Config.MODEL_PATH}")
    
    return model


if __name__ == "__main__":
    model = train_model()
    print("\nTraining complete!")