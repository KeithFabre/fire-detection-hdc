# random-projection encoding with RGB channels

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

#DIMENSIONS = 10000
DIMENSIONS = 1000
IMG_SIZE = 256
# qt de HV pra representar um valor numerico
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


#transform = torchvision.transforms.ToTensor()

transform = transforms.Compose([
        transforms.Resize(IMG_HEIGHT),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Removed grayscale transformation to keep RGB channels
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
        #shuffle=True
        #num_workers=NUM_WORKERS,
        #pin_memory=True
    )

test_ld = DataLoader(
        test_ds,
        batch_size=IMG_BATCH_SIZE
        #shuffle=True,
        #num_workers=NUM_WORKERS,
        #pin_memory=True
    )


#exit()  


class Encoder(nn.Module):
    def __init__(self, out_features, size):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        # Create a nonlinear projection for each channel (R, G, B)
        self.nonlinear_projection_r = embeddings.Sinusoid(size * size, out_features)
        self.nonlinear_projection_g = embeddings.Sinusoid(size * size, out_features)
        self.nonlinear_projection_b = embeddings.Sinusoid(size * size, out_features)

    def forward(self, x):
        # x shape: [batch_size, 3, height, width]
        # Split into R, G, B channels
        x_r = x[:, 0, :, :]  # Red channel
        x_g = x[:, 1, :, :]  # Green channel
        x_b = x[:, 2, :, :]  # Blue channel
        
        # Flatten each channel
        x_r_flat = self.flatten(x_r)
        x_g_flat = self.flatten(x_g)
        x_b_flat = self.flatten(x_b)
        
        # Apply nonlinear projection to each channel
        hr = self.nonlinear_projection_r(x_r_flat)
        hg = self.nonlinear_projection_g(x_g_flat)
        hb = self.nonlinear_projection_b(x_b_flat)
        
        # Bundle the hypervectors: HX = HR + HG + HB
        hx = hr + hg + hb
        
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
        #print('Encode Done')
        model.add(samples_hv, labels)
        #exit()

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
