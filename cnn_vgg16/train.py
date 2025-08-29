import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time
from torchvision import datasets, transforms

# Import our custom modules
#from dataset import get_dataloaders
from model import get_vgg16_model

# --- Configuration ---
DATA_DIR = '../Training'
MODEL_SAVE_DIR = 'checkpoints'
MODEL_NAME = 'vgg16_fire_detection_best.pth'
NUM_EPOCHS = 10  # Fewer epochs for transfer learning
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning
BATCH_SIZE = 16  # Smaller batch size for VGG16 (memory constraints)

# VGG16 expects 224x224 images
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs a single training epoch.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Reshape labels for BCEWithLogitsLoss
        labels = labels.float().unsqueeze(1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        preds = torch.sigmoid(outputs) > 0.5
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(correct_predictions/total_samples):.2f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """
    Runs validation on the model.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(correct_predictions/total_samples):.2f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def main():
    """
    Main training loop for VGG16 transfer learning.
    """
    # Create directory for saving models
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----------- GETS DATA --------------
    # Configuration
    TRAIN_DATA_DIR = '../Training'
    TEST_DATA_DIR = '../Test'
    #NUM_CLASSES = 2  # Binary classification for fire detection

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

    # Get DataLoaders
    #train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    #print(f"Classes: {class_names}")

    # Initialize model, criterion, and optimizer
    model = get_vgg16_model(pretrained=True).to(device)
    
    # Use different learning rates for different parts of the model
    # Lower learning rate for pre-trained features, higher for new classifier
    feature_params = list(map(id, model.vgg16.features.parameters()))
    classifier_params = filter(lambda p: id(p) not in feature_params, model.parameters())
    
    optimizer = optim.Adam([
        {'params': model.vgg16.features.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ])
    
    criterion = nn.BCEWithLogitsLoss()

    best_test_acc = 0.0
    start_time = time.time()

    print("--- Starting VGG16 Transfer Learning ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        
        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Test Accuracy: {test_acc:.4f}")

        # Save the model if it has the best validation accuracy so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with accuracy: {best_test_acc:.4f}")

    total_time = time.time() - start_time
    print(f"\n--- Finished Training ---")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_test_acc:.4f}")

if __name__ == '__main__':
    main() 