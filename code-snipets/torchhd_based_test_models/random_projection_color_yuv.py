# random-projection encoding with YUV color space

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

# Custom transform to convert RGB to YUV
def rgb_to_yuv(img):
    # img is a tensor of shape [3, H, W] in range [0, 1]
    r = img[0, :, :]
    g = img[1, :, :]
    b = img[2, :, :]
    
    # Convert to YUV using standard formulas
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = 0.492 * (b - y) + 0.5  # Center and scale U to [0, 1]
    v = 0.877 * (r - y) + 0.5  # Center and scale V to [0, 1]
    
    # Stack channels
    yuv_img = torch.stack([y, u, v], dim=0)
    return yuv_img

transform = transforms.Compose([
    transforms.Resize(IMG_HEIGHT),
    transforms.ToTensor(),
    transforms.Lambda(rgb_to_yuv),
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
        # Create a nonlinear projection for each channel (Y, U, V)
        self.nonlinear_projection_y = embeddings.Sinusoid(size * size, out_features)
        self.nonlinear_projection_u = embeddings.Sinusoid(size * size, out_features)
        self.nonlinear_projection_v = embeddings.Sinusoid(size * size, out_features)

    def forward(self, x):
        # x shape: [batch_size, 3, height, width] - YUV channels
        # Split into Y, U, V channels
        x_y = x[:, 0, :, :]  # Luminance channel
        x_u = x[:, 1, :, :]  # Chrominance U channel
        x_v = x[:, 2, :, :]  # Chrominance V channel
        
        # Flatten each channel
        x_y_flat = self.flatten(x_y)
        x_u_flat = self.flatten(x_u)
        x_v_flat = self.flatten(x_v)
        
        # Apply nonlinear projection to each channel
        h_y = self.nonlinear_projection_y(x_y_flat)
        h_u = self.nonlinear_projection_u(x_u_flat)
        h_v = self.nonlinear_projection_v(x_v_flat)
        
        # Bundle the hypervectors: HX = HY + HU + HV
        hx = h_y + h_u + h_v
        
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
