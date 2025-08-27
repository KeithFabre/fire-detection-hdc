# Modified for VGG16 feature extraction + BinHD HDC classification with memory optimization

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import time
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

# Import for HDC
import torchhd
from torchhd import embeddings
from torchhd.models import Centroid

# vgg16 expects 224x224 images
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# HDC parameters
DIMENSIONS = 1000  # Hypervector dimension
NUM_LEVELS = 100   # Number of levels for encoding

def get_dataloaders(data_dir, batch_size=BATCH_SIZE, val_split=0.2, num_workers=4):
    """
    Creates training and validation dataloaders optimized for VGG16.
    """
    # VGG16 preprocessing
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the entire dataset using ImageFolder
    full_dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms)
    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")

    # Split dataset into training and validation sets
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Total images: {total_size}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names

class RecordEncoder(nn.Module):
    """
    Encoder for converting features to hypervectors using random projection and scatter coding.
    Using consistent MAP tensor type for both position and value.
    """
    def __init__(self, out_features, size, levels, low, high, device=None):
        super(RecordEncoder, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Random projection for position (MAP)
        self.position = embeddings.Random(size, out_features, vsa="MAP")
        # Level encoding for value (also MAP for consistency)
        self.value = embeddings.Level(levels, out_features, low=low, high=high, vsa="MAP")

    def forward(self, x):
        # Process in batches to avoid memory issues
        x = x.to(self.device)
        
        # Get position and value hypervectors (both MAP)
        pos_hv = self.position.weight
        val_hv = self.value(x)
        
        # Bind position and value hypervectors
        sample_hv = torchhd.bind(pos_hv, val_hv)
        # Create multiset of hypervectors
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv

def extract_features(model, x):
    """
    Extract features from VGG16 up to the last convolutional layer.
    """
    with torch.no_grad():
        x = model.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        features = torch.flatten(x, 1)
    return features

# Configuration
DATA_DIR = '../Training'
NUM_CLASSES = 2  # Binary classification for fire detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Get data loaders
train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
print(f"Classes: {class_names}")

# Load pre-trained VGG16
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()

# Get min and max values for encoding by processing a subset
print("Calculating feature range...")
sample_features = []
sample_size = min(100, len(train_loader.dataset))  # Use 100 samples or all if less

# Create a subset of the training data for range calculation
subset_indices = torch.randperm(len(train_loader.dataset))[:sample_size]
subset_dataset = Subset(train_loader.dataset, subset_indices)
subset_loader = DataLoader(subset_dataset, batch_size=BATCH_SIZE, shuffle=False)

for images, _ in tqdm(subset_loader, desc="Sampling features"):
    images = images.to(device)
    features = extract_features(vgg16, images)
    sample_features.append(features.cpu())

sample_features = torch.cat(sample_features, dim=0)
min_val = sample_features.min().item()
max_val = sample_features.max().item()
print(f"Feature min: {min_val}, max: {max_val}")

# Create record encoder
feature_size = sample_features.shape[1]  # Get the feature dimension
record_encode = RecordEncoder(DIMENSIONS, feature_size, NUM_LEVELS, min_val, max_val, device=device)
record_encode = record_encode.to(device)

# Create and train Centroid model (no vsa parameter needed)
model = Centroid(DIMENSIONS, NUM_CLASSES)
model.to(device)

# Train the model by processing batches
print("Training Centroid model...")
for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
    images = images.to(device)
    labels = labels.to(device)
    
    # Extract features
    with torch.no_grad():
        features = extract_features(vgg16, images)
    
    # Encode features to hypervectors
    encoded_features = record_encode(features)
    
    # Add to centroid model
    for i in range(len(encoded_features)):
        model.add(encoded_features[i].unsqueeze(0), labels[i].unsqueeze(0))
    
    # Clear memory periodically
    if batch_idx % 10 == 0:
        torch.cuda.empty_cache()

model.normalize()

# Validate the model
print("Validating Centroid model...")
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Extract features
        features = extract_features(vgg16, images)
        
        # Encode features to hypervectors
        encoded_features = record_encode(features)
        
        # Predict
        outputs = model(encoded_features, dot=True)
        predictions = outputs.argmax(dim=1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Clear memory
        torch.cuda.empty_cache()

accuracy = correct / total
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Additional metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))