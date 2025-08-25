# random-projection encoding with HSV color space

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Note: this example requires the torchmetrics library: https://torchmetrics.readthedocs.io
import torchmetrics
from tqdm import tqdm

import torchhd
from torchhd.models import Centroid
from torchhd import embeddings


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 1000
IMG_SIZE = 256
BATCH_SIZE = 1  # for GPUs with enough memory we can process multiple images at ones

# Image dimensions and batch size
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_BATCH_SIZE = 32
NUM_WORKERS = 1

# training directory
training_directory = '../data/Training'
# test directory
test_directory = '../data/Test'

# Manual RGB to HSV conversion
def rgb_to_hsv(img):
    # img is a tensor of shape [3, H, W] in range [0, 1]
    r = img[0, :, :]
    g = img[1, :, :]
    b = img[2, :, :]
    
    # Calculate Value (V)
    v, max_idx = torch.max(img, dim=0)
    min_val, _ = torch.min(img, dim=0)
    
    # Calculate Saturation (S)
    s = torch.where(v > 0, (v - min_val) / v, torch.zeros_like(v))
    
    # Calculate Hue (H)
    h = torch.zeros_like(v)
    
    # Add small epsilon to avoid division by zero
    eps = 1e-6
    
    # Red is max
    red_mask = (max_idx == 0)
    denominator_red = (v[red_mask] - min_val[red_mask] + eps)
    h[red_mask] = (60 * ((g[red_mask] - b[red_mask]) / denominator_red)) % 360
    
    # Green is max
    green_mask = (max_idx == 1)
    denominator_green = (v[green_mask] - min_val[green_mask] + eps)
    h[green_mask] = (60 * (2 + (b[green_mask] - r[green_mask]) / denominator_green)) % 360
    
    # Blue is max
    blue_mask = (max_idx == 2)
    denominator_blue = (v[blue_mask] - min_val[blue_mask] + eps)
    h[blue_mask] = (60 * (4 + (r[blue_mask] - g[blue_mask]) / denominator_blue)) % 360
    
    # Handle undefined hue (when S=0)
    h[s == 0] = 0
    
    # Normalize H to [0, 1] by dividing by 360
    h_normalized = h / 360.0
    
    # Stack channels: H, S, V
    hsv_img = torch.stack([h_normalized, s, v], dim=0)
    return hsv_img

transform = transforms.Compose([
    transforms.Resize(IMG_HEIGHT),
    transforms.ToTensor(),
    transforms.Lambda(rgb_to_hsv),
    # No normalization for HSV as values are already in [0,1] range
])

training_ds = datasets.ImageFolder(root=training_directory, transform=transform)
training_classes_names = training_ds.classes
print(f"Training Classes found: {training_classes_names}")
print(f"Training Class to index mapping: {training_ds.class_to_idx}")

test_ds = datasets.ImageFolder(root=test_directory, transform=transform)
test_classes_names = test_ds.classes
print(f"Test Classes found: {test_classes_names}")
print(f"Test Class to index mapping: {test_ds.class_to_idx}")

generator = torch.Generator().manual_seed(42)

train_ld = DataLoader(
    training_ds,
    batch_size=IMG_BATCH_SIZE
)

test_ld = DataLoader(
    test_ds,
    batch_size=IMG_BATCH_SIZE
)

class Encoder(nn.Module):
    def __init__(self, out_features, size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        # Create a nonlinear projection for each channel (H, S, V)
        self.nonlinear_projection_h = embeddings.Sinusoid(size * size, out_features)
        self.nonlinear_projection_s = embeddings.Sinusoid(size * size, out_features)
        self.nonlinear_projection_v = embeddings.Sinusoid(size * size, out_features)

    def forward(self, x):
        # x shape: [batch_size, 3, height, width] - HSV channels
        # Split into H, S, V channels
        x_h = x[:, 0, :, :]  # Hue channel
        x_s = x[:, 1, :, :]  # Saturation channel
        x_v = x[:, 2, :, :]  # Value channel
        
        # Flatten each channel
        x_h_flat = self.flatten(x_h)
        x_s_flat = self.flatten(x_s)
        x_v_flat = self.flatten(x_v)
        
        # Apply nonlinear projection to each channel
        h_h = self.nonlinear_projection_h(x_h_flat)
        h_s = self.nonlinear_projection_s(x_s_flat)
        h_v = self.nonlinear_projection_v(x_v_flat)
        
        # Bundle the hypervectors: HX = HH + HS + HV
        hx = h_h + h_s + h_v
        
        return torchhd.hard_quantize(hx)

encode = Encoder(DIMENSIONS, IMG_SIZE)
encode = encode.to(device)

num_classes = len(training_ds.classes)
model = Centroid(DIMENSIONS, num_classes)
model = model.to(device)

print('-'*50)
print('Starting Training')
print('-'*50)

with torch.no_grad():
    for samples, labels in tqdm(train_ld, desc="Training"):
        samples = samples.to(device)
        labels = labels.to(device)

        samples_hv = encode(samples)
        model.add(samples_hv, labels)

accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
print('-'*50)
print('Starting Test')
print('-'*50)

with torch.no_grad():
    model.normalize()

    for samples, labels in tqdm(test_ld, desc="Testing"):
        samples = samples.to(device)

        samples_hv = encode(samples)
        outputs = model(samples_hv, dot=True)
        accuracy.update(outputs.cpu(), labels)

print(f"Testing accuracy of {(accuracy.compute().item() * 100):.3f}%")
