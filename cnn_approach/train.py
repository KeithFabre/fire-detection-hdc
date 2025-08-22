import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm

# Import our custom modules
from dataset import get_dataloaders
from model import FireDetectionModel

# --- Configuration ---
DATA_DIR = '../Training'
MODEL_SAVE_DIR = 'checkpoints'
MODEL_NAME = 'fire_detection_best.pth'
NUM_EPOCHS = 40  # Keras model was trained for 40 epochs
LEARNING_RATE = 1e-3
BATCH_SIZE = 32

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Runs a single training epoch.
    """
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    # Using tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Reshape labels for BCEWithLogitsLoss
        labels = labels.float().unsqueeze(1)
        
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
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
    model.eval()  # Set model to evaluation mode
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
    Main training loop.
    """
    # Create directory for saving models
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # Initialize model, criterion, and optimizer
    model = FireDetectionModel().to(device)
    
    # Binary Cross-Entropy with Logits, combines Sigmoid layer and BCELoss
    criterion = nn.BCEWithLogitsLoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = 0.0

    print("--- Starting Training ---")
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

    print("\n--- Finished Training ---")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    main() 