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
from torchhd import embeddings
from torchhd.models import Centroid
import torch.nn.functional as F
from torch import Tensor
from torchhd.embeddings import  Random, Level, Sinusoid
import math 
from tqdm import trange
import torchhd.functional as functional

# vgg16 expects 224x224 images
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# HDC parameters
DIMENSIONS = 1000  # Hypervector dimension
NUM_LEVELS = 100   # Number of levels for encoding


class AdaptHD(nn.Module):
    r"""Implements `AdaptHD: Adaptive Efficient Training for Brain-Inspired Hyperdimensional Computing <https://ieeexplore.ieee.org/document/8918974>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        n_levels (int, optional): The number of discretized levels for the level-hypervectors.
        min_level (int, optional): The lower-bound of the range represented by the level-hypervectors.
        max_level (int, optional): The upper-bound of the range represented by the level-hypervectors.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        n_levels: int = 100,
        min_level: int = -1,
        max_level: int = 1,
        epochs: int = 120,
        lr: float = 0.035,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()  # Remove the parameters from super().__init__()
        
        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        self.keys = Random(n_features, n_dimensions, device=self.device, dtype=self.dtype)
        self.levels = Level(
            n_levels,
            n_dimensions,
            low=min_level,
            high=max_level,
            device=self.device,
            dtype=self.dtype,
        )
        self.model = Centroid(n_dimensions, n_classes, device=self.device, dtype=self.dtype)

    def encoder(self, samples: Tensor) -> Tensor:
        return functional.hash_table(self.keys.weight, self.levels(samples)).sign()

    def fit(self, input: Tensor, target: Tensor):
        for _ in trange(self.epochs, desc="fit", disable=True):        
            samples = input.to(self.device)
            labels = target.to(self.device)

            encoded = self.encoder(samples)
            self.model.add_adapt(encoded, labels, lr=self.lr)

        return self


    def fit_data_loader(self, data_loader: DataLoader):

        for _ in trange(self.epochs, desc="fit", disable=True):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                self.model.add_adapt(encoded, labels, lr=self.lr)

        return self


# Adapted from: https://gitlab.com/biaslab/onlinehd/
class OnlineHD(nn.Module):
    r"""Implements `OnlineHD: Robust, Efficient, and Single-Pass Online Learning Using Hyperdimensional System <https://ieeexplore.ieee.org/abstract/document/9474107>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    encoder: Sinusoid
    model: Centroid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        epochs: int = 120,
        lr: float = 0.035,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()  # Remove parameters from super()
        
        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.epochs = epochs
        self.lr = lr
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        self.encoder = Sinusoid(n_features, n_dimensions, device=self.device, dtype=self.dtype)
        self.model = Centroid(n_dimensions, n_classes, device=self.device, dtype=self.dtype)

    def fit(self, input: Tensor, target: Tensor):

        for _ in trange(self.epochs, desc="fit", disable=True):            
            samples = input.to(self.device)
            labels = target.to(self.device)

            encoded = self.encoder(samples)
            self.model.add_online(encoded, labels, lr=self.lr)

        return self

    def fit_data_loader(self, data_loader: DataLoader):

        for _ in trange(self.epochs, desc="fit", disable=True):
            for samples, labels in data_loader:
                samples = samples.to(self.device)
                labels = labels.to(self.device)

                encoded = self.encoder(samples)
                self.model.add_online(encoded, labels, lr=self.lr)

        return self


class NeuralHD(nn.Module):
    r"""Implements `Scalable edge-based hyperdimensional learning system with brain-like neural adaptation <https://dl.acm.org/doi/abs/10.1145/3458817.3480958>`_.

    Args:
        n_features (int): Size of each input sample.
        n_dimensions (int): The number of hidden dimensions to use.
        n_classes (int): The number of classes.
        regen_freq (int, optional): The frequency in epochs at which to regenerate hidden dimensions.
        regen_rate (float, optional): The fraction of hidden dimensions to regenerate.
        epochs (int, optional): The number of iteration over the training data.
        lr (float, optional): The learning rate.
        device (``torch.device``, optional):  the desired device of the weights. Default: if ``None``, uses the current device for the default tensor type (see ``torch.set_default_tensor_type()``). ``device`` will be the CPU for CPU tensor types and the current CUDA device for CUDA tensor types.
        dtype (``torch.dtype``, optional): the desired data type of the weights. Default: if ``None``, uses ``torch.get_default_dtype()``.

    """

    model: Centroid
    encoder: Sinusoid

    def __init__(
        self,
        n_features: int,
        n_dimensions: int,
        n_classes: int,
        *,
        regen_freq: int = 20,
        regen_rate: float = 0.04,
        epochs: int = 120,
        lr: float = 0.37,
        device: torch.device = None,
        dtype: torch.dtype = None
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.regen_freq = regen_freq
        self.regen_rate = regen_rate
        self.epochs = epochs
        self.lr = lr

        self.encoder = Sinusoid(n_features, n_dimensions, device=device, dtype=dtype)
        self.model = Centroid(n_dimensions, n_classes, device=device, dtype=dtype)

    def fit(self, input: Tensor, target: Tensor):
        encoded = self.encoder(input)
        n_regen_dims = math.ceil(self.regen_rate * self.n_dimensions)
        self.model.add(encoded, target)

        for epoch_idx in trange(1, self.epochs, desc="fit", disable=True):  # Disable progress bar
            encoded = self.encoder(input)
            self.model.add_adapt(encoded, target, lr=self.lr)

            # Regenerate feature dimensions
            if (epoch_idx % self.regen_freq) == (self.regen_freq - 1):
                weight = F.normalize(self.model.weight, dim=1)
                scores = torch.var(weight, dim=0)

                regen_dims = torch.topk(scores, n_regen_dims, largest=False).indices
                self.model.weight.data[:, regen_dims].zero_()
                self.encoder.weight.data[regen_dims, :].normal_()
                self.encoder.bias.data[:, regen_dims].uniform_(0, 2 * math.pi)

        return self

    def forward(self, samples: Tensor) -> Tensor:
        return self.model(self.encoder(samples))

    def predict(self, samples: Tensor) -> Tensor:
        return torch.argmax(self(samples), dim=-1)
    
    def normalize(self):
        """Normalize the model weights"""
        self.model.weight.data = F.normalize(self.model.weight.data, dim=1)




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


# Create and train model (no vsa parameter needed)
epochs = 10
regen_freq = 5
sample_image, _ = train_dataset[0] 

model = AdaptHD(feature_size, DIMENSIONS, NUM_CLASSES, device=device, epochs=epochs)
model.to(device)

# Train the model by processing batches
print("Training model...")
for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc="Training")):
    images = images.to(device)
    labels = labels.to(device)
    
    # Extract features
    with torch.no_grad():
        features = extract_features(vgg16, images)
    
    model.fit(features, labels)
    
    # Clear memory periodically
    if batch_idx % 10 == 0:
        torch.cuda.empty_cache()

model.normalize()

# Validate the model
print("Validating model...")
correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Test"):
        images = images.to(device)
        labels = labels.to(device)
        
        # Extract features
        features = extract_features(vgg16, images)
        
        # Predict
        predictions = model.predict(features) 

        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Clear memory
        torch.cuda.empty_cache()

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Additional metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:")
print(classification_report(all_labels, all_predictions, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(all_labels, all_predictions))