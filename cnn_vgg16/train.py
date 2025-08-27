import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time

# Import our custom modules
from dataset import get_dataloaders
from model import get_vgg16_model

# --- Configuration ---
DATA_DIR = '../Training'
MODEL_SAVE_DIR = 'checkpoints'
MODEL_NAME = 'vgg16_fire_detection_best.pth'
NUM_EPOCHS = 20  # Fewer epochs for transfer learning
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning
BATCH_SIZE = 16  # Smaller batch size for VGG16 (memory constraints)

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

    # Get DataLoaders
    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    print(f"Classes: {class_names}")

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

    best_val_acc = 0.0
    start_time = time.time()

    print("--- Starting VGG16 Transfer Learning ---")
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Save the model if it has the best validation accuracy so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with accuracy: {best_val_acc:.4f}")

    total_time = time.time() - start_time
    print(f"\n--- Finished Training ---")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main() 