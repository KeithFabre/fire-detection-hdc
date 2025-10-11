import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time
from torchvision import datasets, transforms
import psutil
import json
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix

# Try to import codecarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
    print("CodeCarbon available - will track carbon emissions")
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not available - install with: pip install codecarbon")

# Import our custom modules
#from dataset import get_dataloaders
from model_vgg16 import get_vgg16_model

# --- Configuration ---
DATA_DIR = '../Training'
MODEL_SAVE_DIR = 'checkpoints'
MODEL_NAME = 'vgg16_fire_detection_best.pth'
NUM_EPOCHS = 20  
LEARNING_RATE = 1e-4  # Lower learning rate for fine-tuning
BATCH_SIZE = 16  # Smaller batch size for VGG16 (memory constraints)

# VGG16 expects 224x224 images
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# GPU monitoring functions using PyTorch
class GPUMonitor:
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.initial_memory = 0
        self.peak_memory = 0
        
    def start_monitoring(self):
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
            self.initial_memory = torch.cuda.memory_allocated()
            
    def get_memory_usage(self):
        if self.cuda_available:
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            return {
                'current_mb': current_memory / 1024**2,
                'peak_mb': peak_memory / 1024**2,
                'allocated_mb': (current_memory - self.initial_memory) / 1024**2
            }
        return {'current_mb': 0, 'peak_mb': 0, 'allocated_mb': 0}
    
    def get_gpu_utilization(self):
        if self.cuda_available:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            current_memory = torch.cuda.memory_allocated()
            return (current_memory / total_memory) * 100
        return 0

# RAM monitoring functions
def get_memory_usage():
    """Get current RAM usage in MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024**2,  # Resident Set Size
        'vms_mb': memory_info.vms / 1024**2,   # Virtual Memory Size
        'percent': psutil.virtual_memory().percent
    }

def get_system_memory():
    """Get system-wide memory information"""
    vm = psutil.virtual_memory()
    return {
        'total_mb': vm.total / 1024**2,
        'available_mb': vm.available / 1024**2,
        'used_mb': vm.used / 1024**2,
        'percent': vm.percent
    }

# Energy estimation functions
def get_cpu_power():
    """Estimate CPU power consumption"""
    cpu_percent = psutil.cpu_percent(interval=0.1)
    estimated_power = (cpu_percent / 100) * 65  # Watts
    return estimated_power

def estimate_gpu_power(gpu_utilization_percent):
    """Estimate GPU power based on utilization"""
    max_gpu_power = 250  # Watts
    estimated_power = (gpu_utilization_percent / 100) * max_gpu_power
    return estimated_power

# CodeCarbon helper functions - simplified like the docs
def start_carbon_tracker():
    """Start carbon tracker - simple approach like TensorFlow example"""
    if not CODECARBON_AVAILABLE:
        return None, False
    
    try:
        tracker = EmissionsTracker()
        tracker.start()
        return tracker, True
    except Exception as e:
        print(f"Warning: Could not initialize carbon tracker: {e}")
        return None, False

def stop_carbon_tracker(tracker, available):
    """Stop carbon tracker - simple approach"""
    if not available or tracker is None:
        return 0.0
    
    try:
        emissions = tracker.stop()
        return emissions if emissions is not None else 0.0
    except Exception as e:
        print(f"Warning: Could not stop carbon tracker: {e}")
        return 0.0

def train_one_epoch(model, dataloader, criterion, optimizer, device, gpu_monitor, epoch_metrics):
    """
    Runs a single training epoch with comprehensive metrics.
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    epoch_cpu_energy = 0
    epoch_gpu_energy = 0
    peak_ram_usage = epoch_metrics['ram_usage']['training']['start_rss_mb']

    progress_bar = tqdm(dataloader, desc="Training", unit="batch")
    for inputs, labels in progress_bar:
        batch_start = time.time()
        
        inputs, labels = inputs.to(device), labels.to(device)

        # Monitor resources at the beginning of each batch
        cpu_power = get_cpu_power()
        gpu_util = gpu_monitor.get_gpu_utilization()
        gpu_power = estimate_gpu_power(gpu_util)
        
        # Monitor RAM
        current_ram = get_memory_usage()
        peak_ram_usage = max(peak_ram_usage, current_ram['rss_mb'])

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
        
        # Calculate energy for this batch
        batch_duration = time.time() - batch_start
        epoch_cpu_energy += (cpu_power * batch_duration) / 3600
        epoch_gpu_energy += (gpu_power * batch_duration) / 3600
        
        progress_bar.set_postfix(loss=loss.item(), acc=f"{(correct_predictions/total_samples):.2f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    # Update epoch metrics
    epoch_metrics['training_loss'] = epoch_loss
    epoch_metrics['training_accuracy'] = epoch_acc
    epoch_metrics['energy_consumption']['training']['cpu_kwh'] = epoch_cpu_energy
    epoch_metrics['energy_consumption']['training']['gpu_kwh'] = epoch_gpu_energy
    epoch_metrics['energy_consumption']['training']['total_kwh'] = epoch_cpu_energy + epoch_gpu_energy
    epoch_metrics['ram_usage']['training']['peak_rss_mb'] = peak_ram_usage
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, gpu_monitor, epoch_metrics):
    """
    Runs validation on the model with comprehensive metrics.
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    epoch_cpu_energy = 0
    epoch_gpu_energy = 0
    peak_ram_usage = epoch_metrics['ram_usage']['testing']['start_rss_mb']

    progress_bar = tqdm(dataloader, desc="Validation", unit="batch")
    with torch.no_grad():
        for inputs, labels in progress_bar:
            batch_start = time.time()
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Monitor resources at the beginning of each batch
            cpu_power = get_cpu_power()
            gpu_util = gpu_monitor.get_gpu_utilization()
            gpu_power = estimate_gpu_power(gpu_util)
            
            # Monitor RAM
            current_ram = get_memory_usage()
            peak_ram_usage = max(peak_ram_usage, current_ram['rss_mb'])
            
            outputs = model(inputs)
            
            labels = labels.float().unsqueeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
            preds = torch.sigmoid(outputs) > 0.5
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Calculate energy for this batch
            batch_duration = time.time() - batch_start
            epoch_cpu_energy += (cpu_power * batch_duration) / 3600
            epoch_gpu_energy += (gpu_power * batch_duration) / 3600
            
            progress_bar.set_postfix(loss=loss.item(), acc=f"{(correct_predictions/total_samples):.2f}")

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    
    # Update epoch metrics
    epoch_metrics['testing_loss'] = epoch_loss
    epoch_metrics['testing_accuracy'] = epoch_acc
    epoch_metrics['energy_consumption']['testing']['cpu_kwh'] = epoch_cpu_energy
    epoch_metrics['energy_consumption']['testing']['gpu_kwh'] = epoch_gpu_energy
    epoch_metrics['energy_consumption']['testing']['total_kwh'] = epoch_cpu_energy + epoch_gpu_energy
    epoch_metrics['ram_usage']['testing']['peak_rss_mb'] = peak_ram_usage
    
    return epoch_loss, epoch_acc

def main():
    """
    Main training loop for VGG16 transfer learning with comprehensive metrics.
    """
    # Create directory for saving models
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check if CUDA is available for detailed GPU monitoring
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ----------- GETS DATA --------------
    TRAIN_DATA_DIR = '../Training'
    TEST_DATA_DIR = '../Test'

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

    # Initialize model, criterion, and optimizer
    model = get_vgg16_model(pretrained=True).to(device)
    
    # Use different learning rates for different parts of the model
    feature_params = list(map(id, model.vgg16.features.parameters()))
    classifier_params = filter(lambda p: id(p) not in feature_params, model.parameters())
    
    optimizer = optim.Adam([
        {'params': model.vgg16.features.parameters(), 'lr': LEARNING_RATE * 0.1},
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ])
    
    criterion = nn.BCEWithLogitsLoss()

    best_test_acc = 0.0
    start_time = time.time()
    
    # Initialize monitoring
    gpu_monitor = GPUMonitor()
    
    # Initialize results array for all epochs
    all_epochs_metrics = []

    print("--- Starting VGG16 Transfer Learning ---")
    print(f"{'='*60}")
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'-'*50}")
        print(f"EPOCH {epoch + 1}/{NUM_EPOCHS}")
        print(f"{'-'*50}")
        
        # Initialize logging for this epoch
        epoch_metrics = {
            'epoch_number': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'training_loss': 0,
            'training_accuracy': 0,
            'testing_loss': 0,
            'testing_accuracy': 0,
            'carbon_emissions': {
                'training': 0,
                'testing': 0,
                'total': 0
            },
            'energy_consumption': {
                'training': {'cpu_kwh': 0, 'gpu_kwh': 0, 'total_kwh': 0},
                'testing': {'cpu_kwh': 0, 'gpu_kwh': 0, 'total_kwh': 0}
            },
            'gpu_memory': {
                'training_peak_mb': 0,
                'testing_peak_mb': 0
            },
            'ram_usage': {
                'training': {
                    'start_rss_mb': 0,
                    'end_rss_mb': 0,
                    'peak_rss_mb': 0,
                    'start_vms_mb': 0,
                    'end_vms_mb': 0
                },
                'testing': {
                    'start_rss_mb': 0,
                    'end_rss_mb': 0,
                    'peak_rss_mb': 0,
                    'start_vms_mb': 0,
                    'end_vms_mb': 0
                }
            }
        }
        
        # Record initial RAM usage for training
        initial_ram = get_memory_usage()
        epoch_metrics['ram_usage']['training']['start_rss_mb'] = initial_ram['rss_mb']
        epoch_metrics['ram_usage']['training']['start_vms_mb'] = initial_ram['vms_mb']
        
        # TRAINING PHASE
        training_tracker, training_carbon_available = start_carbon_tracker()
        gpu_monitor.start_monitoring()
        training_start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, gpu_monitor, epoch_metrics)
        
        training_end_time = time.time()
        training_time = training_end_time - training_start_time
        training_memory = gpu_monitor.get_memory_usage()
        
        # Record final RAM usage for training
        final_ram = get_memory_usage()
        epoch_metrics['ram_usage']['training']['end_rss_mb'] = final_ram['rss_mb']
        epoch_metrics['ram_usage']['training']['end_vms_mb'] = final_ram['vms_mb']
        epoch_metrics['gpu_memory']['training_peak_mb'] = training_memory['peak_mb']
        
        # Stop carbon tracking for training
        training_emissions = stop_carbon_tracker(training_tracker, training_carbon_available)
        epoch_metrics['carbon_emissions']['training'] = training_emissions
        
        print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}")
        print(f"Training Time: {training_time:.2f}s")
        
        # TESTING PHASE
        testing_tracker, testing_carbon_available = start_carbon_tracker()
        gpu_monitor.start_monitoring()
        testing_start_time = time.time()
        
        # Record initial RAM usage for testing
        initial_ram_test = get_memory_usage()
        epoch_metrics['ram_usage']['testing']['start_rss_mb'] = initial_ram_test['rss_mb']
        epoch_metrics['ram_usage']['testing']['start_vms_mb'] = initial_ram_test['vms_mb']
        
        test_loss, test_acc = validate(model, test_loader, criterion, device, gpu_monitor, epoch_metrics)
        
        testing_end_time = time.time()
        testing_time = testing_end_time - testing_start_time
        testing_memory = gpu_monitor.get_memory_usage()
        
        # Record final RAM usage for testing
        final_ram_test = get_memory_usage()
        epoch_metrics['ram_usage']['testing']['end_rss_mb'] = final_ram_test['rss_mb']
        epoch_metrics['ram_usage']['testing']['end_vms_mb'] = final_ram_test['vms_mb']
        epoch_metrics['gpu_memory']['testing_peak_mb'] = testing_memory['peak_mb']
        
        # Store testing time in metrics (coherent with HDC model implementation)
        epoch_metrics['testing_time'] = testing_time
        
        # Stop carbon tracking for testing
        testing_emissions = stop_carbon_tracker(testing_tracker, testing_carbon_available)
        epoch_metrics['carbon_emissions']['testing'] = testing_emissions
        epoch_metrics['carbon_emissions']['total'] = training_emissions + testing_emissions
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
        print(f"Testing Time: {testing_time:.2f}s")
        print(f"Carbon Emissions: {epoch_metrics['carbon_emissions']['total']:.6f} kg CO2")
        print(f"Total Energy: {epoch_metrics['energy_consumption']['training']['total_kwh'] + epoch_metrics['energy_consumption']['testing']['total_kwh']:.6f} kWh")

        # Save the model if it has the best validation accuracy so far
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path} with accuracy: {best_test_acc:.4f}")

        # Add to results array
        all_epochs_metrics.append(epoch_metrics)

    total_time = time.time() - start_time
    print(f"\n{'-'*60}")
    print("--- Finished Training ---")
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_test_acc:.4f}")

    # Load best model for final evaluation
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    # Compute final classification metrics on test set
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            all_predictions.extend(preds.cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # Compute classification report and confusion matrix
    final_classification_report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    final_confusion_matrix = confusion_matrix(all_labels, all_predictions).tolist()

    print("\n--- Final Classification Report ---")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    print("\n--- Confusion Matrix ---")
    print(confusion_matrix(all_labels, all_predictions))

    # Save all results to JSON file
    output_file = f'vgg16_training_metrics_{NUM_EPOCHS}_epochs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    final_output = {
        'experiment_info': {
            'total_epochs': NUM_EPOCHS,
            'experiment_date': datetime.now().isoformat(),
            'configuration': {
                'learning_rate': LEARNING_RATE,
                'batch_size': BATCH_SIZE,
                'img_size': IMG_WIDTH,
                'num_classes': len(class_names),
                'model_type': 'VGG16_TransferLearning'
            },
            'best_accuracy': best_test_acc,
            'total_time_minutes': total_time / 60
        },
        'classification_metrics': {
            'classification_report': final_classification_report,
            'confusion_matrix': final_confusion_matrix
        },
        'individual_epochs': all_epochs_metrics
    }

    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("="*60)

if __name__ == '__main__':
    main()
