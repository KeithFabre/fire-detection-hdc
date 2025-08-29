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
TRAIN_DATA_DIR = '../Training'
TEST_DATA_DIR = '../Test'
NUM_CLASSES = 2  # Binary classification for fire detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# Get data loaders
# VGG16 preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the entire dataset using ImageFolder
train_dataset = datasets.ImageFolder(root=TRAIN_DATA_DIR, transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
class_names = train_dataset.classes
test_dataset = datasets.ImageFolder(root=TEST_DATA_DIR, transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

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
    for images, labels in tqdm(test_loader, desc="Validation"):
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