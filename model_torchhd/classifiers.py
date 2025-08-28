import torch
import torchhd
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

classifiers = [
    "NeuralHD",
    "AdaptHD",
    "OnlineHD"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

DIMENSIONS = 1024  # number of hypervector dimensions
BATCH_SIZE = 1  # set batch size to 1 for minimal memory usage
IMG_SIZE = 64  # reduced image size for faster processing (from 256x256 to 64x64)

# Define the same transform as in random_projection.py with added flattening
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(lambda x: x.flatten())  # Flatten the image to a vector
])

# Load training and test datasets using ImageFolder
train_ds = datasets.ImageFolder(root='../data/Training', transform=transform)
test_ds = datasets.ImageFolder(root='../data/Test', transform=transform)

train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Get number of classes from the dataset
num_classes = len(train_ds.classes)
print(f"Training Classes: {train_ds.classes}")
print(f"Test Classes: {test_ds.classes}")

# Calculate number of features (flattened image size)
# Since we're using grayscale and resizing to 256x256, the flattened size is 256*256 = 65536
sample_image, _ = train_ds[0]
num_features = sample_image.numel()  # This gives 256*256 = 65536 for grayscale 256x256 images
print(f"Number of features: {num_features}")
print(f"Number of classes: {num_classes}")

params = {
    "NeuralHD": {
        "epochs": 10,
        "regen_freq": 5,
    },
    "AdaptHD": {
        "epochs": 10,
    },
    "OnlineHD": {
        "epochs": 10,
    }
}

for classifier in classifiers:
    print()
    print(classifier)

    model_cls = getattr(torchhd.classifiers, classifier)
    model: torchhd.classifiers.Classifier = model_cls(
        num_features, DIMENSIONS, num_classes, device=device, **params[classifier]
    )

    # Custom training with progress bar
    print("Training...")
    for epoch in range(params[classifier]["epochs"]):
        model.train()
        progress_bar = tqdm(train_ld, desc=f"Epoch {epoch+1}/{params[classifier]['epochs']}")
        for samples, labels in progress_bar:
            samples = samples.to(device)
            labels = labels.to(device)
            model.encoder.train()
            model.encoder(samples)  # This updates the model internally
        progress_bar.close()

    # Custom testing with progress bar
    print("Testing...")
    model.eval()
    correct = 0
    total = 0
    progress_bar = tqdm(test_ld, desc="Testing")
    with torch.no_grad():
        for samples, labels in progress_bar:
            samples = samples.to(device)
            labels = labels.to(device)
            outputs = model(samples)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            progress_bar.set_postfix(accuracy=f"{100 * correct / total:.2f}%")
    progress_bar.close()

    accuracy = 100 * correct / total
    print(f"Testing accuracy of {accuracy:.3f}%")
