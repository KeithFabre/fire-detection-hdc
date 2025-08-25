# record-based encoding

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
NUM_LEVELS = 100
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
        transforms.Grayscale(num_output_channels=1)
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
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features)

    def forward(self, x):

        x = self.flatten(x)
        '''
            print(x.shape, self.position.weight.shape) gives:
            torch.Size([32, 196608]) torch.Size([65536, 1000])
            and it kills the process
        '''

        #torchhd.bind(self.position.weight, self.value(x[0]))
        #print('fiz um bind')    
        #exit()

        sample_hv = torchhd.bind(self.position.weight, self.value(x)) # binds position and value
        sample_hv = torchhd.multiset(sample_hv) 

        return torchhd.hard_quantize(sample_hv)


encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
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