# Modified for VGG16 feature extraction + BinHD HDC classification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from tqdm import tqdm
import time
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

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
    
    Args:
        data_dir (str): Path to the root data directory (e.g., '../Training').
        batch_size (int): The number of samples per batch.
        val_split (float): The proportion of the dataset to use for validation.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, class_names).
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
    print(f"Class to index mapping: {full_dataset.class_to_idx}")

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
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        #shuffle=True,
        #num_workers=num_workers,
        #pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        #shuffle=False,
        #num_workers=num_workers,
        #pin_memory=True
    )

    return train_loader, val_loader, class_names

class RecordEncoder(nn.Module):
    """
    Encoder for converting features to hypervectors using random projection and scatter coding.
    Based on the notebook implementation.
    """
    def __init__(self, out_features, size, levels, low, high, device=None):
        super(RecordEncoder, self).__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Random projection for position
        self.position = embeddings.Random(size, out_features, vsa="BSC", dtype=torch.uint8)
        # Level encoding for value
        self.value = embeddings.Level(levels, out_features, low=low, high=high)

    def forward(self, x):
        # Move input to device
        x = x.to(self.device)
        
        # Bind position and value hypervectors
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        # Create multiset of hypervectors
        sample_hv = torchhd.multiset(sample_hv)
        return sample_hv

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
DATA_DIR = '../data/Training'
NUM_CLASSES = 2  # Binary classification for fire detection

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Get data loaders
train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
print(f"Classes: {class_names}")

# Load pre-trained VGG16 and use only features
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()  # Set to evaluation mode since we're only extracting features

print("Extracting features from training set...")
# Extract features from training set
all_train_features = []
all_train_labels = []

for images, labels in tqdm(train_loader, desc="Training feature extraction"):
    images = images.to(device)
    features = extract_features(vgg16, images)
    all_train_features.append(features.cpu())
    all_train_labels.append(labels.cpu())

X_train = torch.cat(all_train_features, dim=0)
y_train = torch.cat(all_train_labels, dim=0)

print("Extracting features from validation set...")
# Extract features from validation set
all_val_features = []
all_val_labels = []

for images, labels in tqdm(val_loader, desc="Validation feature extraction"):
    images = images.to(device)
    features = extract_features(vgg16, images)
    all_val_features.append(features.cpu())
    all_val_labels.append(labels.cpu())

X_val = torch.cat(all_val_features, dim=0)
y_val = torch.cat(all_val_labels, dim=0)

print(f"Training features shape: {X_train.shape}")
print(f"Training labels shape: {y_train.shape}")
print(f"Validation features shape: {X_val.shape}")
print(f"Validation labels shape: {y_val.shape}")

# Get min and max values for encoding
min_val = X_train.min().item()
max_val = X_train.max().item()
print(f"Feature min: {min_val}, max: {max_val}")

# Create record encoder
record_encode = RecordEncoder(DIMENSIONS, X_train.shape[1], NUM_LEVELS, min_val, max_val, device=device)
record_encode = record_encode.to(device)

# Encode training features
print("Encoding training features...")
X_train_encoded = record_encode(X_train.to(device))
y_train = y_train.to(device)

# Encode validation features
print("Encoding validation features...")
X_val_encoded = record_encode(X_val.to(device))
y_val = y_val.to(device)

# Create and train Centroid model
print("Training Centroid model...")
model = Centroid(DIMENSIONS, NUM_CLASSES)
model.to(device)

# Train the model by adding encoded samples
for i in tqdm(range(len(X_train_encoded)), desc="Adding training samples"):
    model.add(X_train_encoded[i].unsqueeze(0), y_train[i].unsqueeze(0))

# Validate the model
print("Validating Centroid model...")
model.normalize()

predictions = []
for i in tqdm(range(len(X_val_encoded)), desc="Predicting validation samples"):
    output = model(X_val_encoded[i].unsqueeze(0), dot=True)
    predictions.append(output.argmax(dim=1).item())

predictions = torch.tensor(predictions, device=device)
accuracy = (predictions == y_val).float().mean().item()

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Additional metrics can be added here for binary classification
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(y_val.cpu().numpy(), predictions.cpu().numpy(), target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val.cpu().numpy(), predictions.cpu().numpy()))
