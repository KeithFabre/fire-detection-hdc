# Modified for VGG16 feature extraction + Random Projection HDC classification with memory optimization

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

def get_dataloaders(data_dir, batch_size=BATCH_SIZE, val_split=0.2, num_workers=4):
    """
    Creates training and validation dataloaders optimized for VGG16.
    """
    # VGG16 preprocessing: resize to 224x224 and normalize with ImageNet stats
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

    # Ensure consistent splitting
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"Total images: {total_size}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names

class RandomProjectionEncoder(nn.Module):
    """
    Encoder for converting features to hypervectors using random projection.
    Based on the random_projection.py implementation.
    """
    def __init__(self, out_features, size, device=None):
        super(RandomProjectionEncoder, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Nonlinear projection for feature encoding
        self.nonlinear_projection = embeddings.Sinusoid(size, out_features, vsa="MAP")
        self.nonlinear_projection = self.nonlinear_projection.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        sample_hv = self.nonlinear_projection(x)
        return torchhd.hard_quantize(sample_hv)

def extract_features(model, x):
    """
    Extract features from VGG16 up to the last convolutional layer.
    """
    with torch.no_grad():
        # Use only the features part of VGG16 (convolutional layers)
        x = model.features(x)
        # Global average pooling to reduce to 1x1 spatial dimensions
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # Flatten to 1D feature vector
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

# Load pre-trained VGG16 and use only features
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()  # Set to evaluation mode since we're only extracting features

# Get feature dimension by processing a small batch
print("Getting feature dimension...")
sample_batch, _ = next(iter(train_loader))
sample_batch = sample_batch.to(device)
sample_features = extract_features(vgg16, sample_batch)
feature_dim = sample_features.shape[1]
print(f"Feature dimension: {feature_dim}")

# Create random projection encoder
encode = RandomProjectionEncoder(DIMENSIONS, feature_dim, device=device)
encode = encode.to(device)

# Create and train Centroid model
model = Centroid(DIMENSIONS, NUM_CLASSES)
model.to(device)

# Train the model by processing batches
print("Training Centroid model with random projection...")
for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
    images = images.to(device)
    labels = labels.to(device)
    
    # Extract features
    with torch.no_grad():
        features = extract_features(vgg16, images)
    
    # Encode features to hypervectors
    encoded_features = encode(features)
    
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
        encoded_features = encode(features)
        
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