# Modified for VGG16 feature extraction + NeuralHD HDC classification with comprehensive metrics

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

# Import for metrics
import psutil
import json
from datetime import datetime

# Try to import codecarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
    print("CodeCarbon available - will track carbon emissions")
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not available - install with: pip install codecarbon")

# vgg16 expects 224x224 images
IMG_WIDTH = 224
IMG_HEIGHT = 224
BATCH_SIZE = 32

# HDC parameters
DIMENSIONS = 1000  # Hypervector dimension
NUM_LEVELS = 100   # Number of levels for encoding

# =============================================================================
# CONFIGURATION - Modify these parameters
# =============================================================================
NUM_RUNS = 3  # Number of experimental runs
# =============================================================================

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
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()

        self.encoder = Sinusoid(n_features, n_dimensions, device=self.device, dtype=self.dtype)
        self.model = Centroid(n_dimensions, n_classes, device=self.device, dtype=self.dtype)

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

# Check if CUDA is available for detailed GPU monitoring
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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

# Get feature size
feature_size = sample_features.shape[1]

# Initialize results array for all runs
all_results = []

print(f"\n{'='*60}")
print(f"STARTING {NUM_RUNS} EXPERIMENTAL RUNS")
print(f"{'='*60}")

# Run the experiment multiple times
for run_num in range(1, NUM_RUNS + 1):
    print(f"\n{'-'*50}")
    print(f"RUN {run_num}/{NUM_RUNS}")
    print(f"{'-'*50}")
    
    # Initialize monitoring for this run
    gpu_monitor = GPUMonitor()
    
    # Initialize logging for this run
    run_metrics = {
        'run_number': run_num,
        'timestamp': datetime.now().isoformat(),
        'training_time': 0,
        'prediction_time': 0,
        'accuracy': 0,
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
            },
            'system_memory': {
                'total_mb': 0,
                'training_start_available_mb': 0,
                'testing_start_available_mb': 0
            }
        },
        'system_info': {
            'device': str(device),
            'cuda_available': cuda_available,
            'gpu_name': torch.cuda.get_device_name() if cuda_available else 'N/A',
            'dimensions': DIMENSIONS,
            'batch_size': BATCH_SIZE,
            'img_size': IMG_WIDTH,
            'feature_size': feature_size,
            'num_classes': NUM_CLASSES,
            'model_type': 'VGG16_NeuralHD'
        }
    }
    
    # Create and train model
    epochs = 10
    regen_freq = 5
    model = NeuralHD(feature_size, DIMENSIONS, NUM_CLASSES, device=device, epochs=epochs, regen_freq=regen_freq)
    model.to(device)
    
    # TRAINING PHASE
    print("Training model...")
    
    # Start carbon tracking for training
    training_tracker, training_carbon_available = start_carbon_tracker()
    
    # Record initial RAM usage for training
    initial_ram = get_memory_usage()
    system_memory = get_system_memory()
    run_metrics['ram_usage']['training']['start_rss_mb'] = initial_ram['rss_mb']
    run_metrics['ram_usage']['training']['start_vms_mb'] = initial_ram['vms_mb']
    run_metrics['ram_usage']['system_memory']['total_mb'] = system_memory['total_mb']
    run_metrics['ram_usage']['system_memory']['training_start_available_mb'] = system_memory['available_mb']
    
    gpu_monitor.start_monitoring()
    training_start_time = time.time()
    training_cpu_energy = 0
    training_gpu_energy = 0
    peak_ram_usage = initial_ram['rss_mb']
    
    # Train the model by processing batches
    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Run {run_num}")):
        batch_start = time.time()
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Monitor resources
        cpu_power = get_cpu_power()
        gpu_util = gpu_monitor.get_gpu_utilization()
        gpu_power = estimate_gpu_power(gpu_util)
        
        # Monitor RAM
        current_ram = get_memory_usage()
        peak_ram_usage = max(peak_ram_usage, current_ram['rss_mb'])
        
        # Extract features
        with torch.no_grad():
            features = extract_features(vgg16, images)
        
        model.fit(features, labels)
        
        # Calculate energy for this batch
        batch_duration = time.time() - batch_start
        training_cpu_energy += (cpu_power * batch_duration) / 3600
        training_gpu_energy += (gpu_power * batch_duration) / 3600
        
        # Clear memory periodically
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    training_memory = gpu_monitor.get_memory_usage()
    
    # Record final RAM usage for training
    final_ram = get_memory_usage()
    run_metrics['ram_usage']['training']['end_rss_mb'] = final_ram['rss_mb']
    run_metrics['ram_usage']['training']['end_vms_mb'] = final_ram['vms_mb']
    run_metrics['ram_usage']['training']['peak_rss_mb'] = peak_ram_usage
    
    # Stop carbon tracking for training
    training_emissions = stop_carbon_tracker(training_tracker, training_carbon_available)
    
    model.normalize()
    
    # TESTING PHASE
    print("Testing model...")
    
    # Start carbon tracking for testing
    testing_tracker, testing_carbon_available = start_carbon_tracker()
    
    # Record initial RAM usage for testing
    initial_ram_test = get_memory_usage()
    system_memory_test = get_system_memory()
    run_metrics['ram_usage']['testing']['start_rss_mb'] = initial_ram_test['rss_mb']
    run_metrics['ram_usage']['testing']['start_vms_mb'] = initial_ram_test['vms_mb']
    run_metrics['ram_usage']['system_memory']['testing_start_available_mb'] = system_memory_test['available_mb']
    
    gpu_monitor.start_monitoring()
    testing_start_time = time.time()
    testing_cpu_energy = 0
    testing_gpu_energy = 0
    peak_ram_usage_test = initial_ram_test['rss_mb']
    
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing Run {run_num}"):
            batch_start = time.time()
            
            images = images.to(device)
            labels = labels.to(device)
            
            # Monitor resources
            cpu_power = get_cpu_power()
            gpu_util = gpu_monitor.get_gpu_utilization()
            gpu_power = estimate_gpu_power(gpu_util)
            
            # Monitor RAM
            current_ram = get_memory_usage()
            peak_ram_usage_test = max(peak_ram_usage_test, current_ram['rss_mb'])
            
            # Extract features
            features = extract_features(vgg16, images)
            
            # Predict
            predictions = model.predict(features)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Calculate energy for this batch
            batch_duration = time.time() - batch_start
            testing_cpu_energy += (cpu_power * batch_duration) / 3600
            testing_gpu_energy += (gpu_power * batch_duration) / 3600
            
            # Clear memory
            torch.cuda.empty_cache()
    
    testing_end_time = time.time()
    testing_time = testing_end_time - testing_start_time
    testing_memory = gpu_monitor.get_memory_usage()
    final_accuracy = correct / total
    
    # Record final RAM usage for testing
    final_ram_test = get_memory_usage()
    run_metrics['ram_usage']['testing']['end_rss_mb'] = final_ram_test['rss_mb']
    run_metrics['ram_usage']['testing']['end_vms_mb'] = final_ram_test['vms_mb']
    run_metrics['ram_usage']['testing']['peak_rss_mb'] = peak_ram_usage_test
    
    # Stop carbon tracking for testing
    testing_emissions = stop_carbon_tracker(testing_tracker, testing_carbon_available)
    
    # Calculate totals for this run
    total_training_energy = training_cpu_energy + training_gpu_energy
    total_testing_energy = testing_cpu_energy + testing_gpu_energy
    total_emissions = training_emissions + testing_emissions
    
    # Update run metrics
    run_metrics['training_time'] = training_time
    run_metrics['prediction_time'] = testing_time
    run_metrics['accuracy'] = final_accuracy * 100
    run_metrics['carbon_emissions']['training'] = training_emissions
    run_metrics['carbon_emissions']['testing'] = testing_emissions
    run_metrics['carbon_emissions']['total'] = total_emissions
    run_metrics['energy_consumption']['training']['cpu_kwh'] = training_cpu_energy
    run_metrics['energy_consumption']['training']['gpu_kwh'] = training_gpu_energy
    run_metrics['energy_consumption']['training']['total_kwh'] = total_training_energy
    run_metrics['energy_consumption']['testing']['cpu_kwh'] = testing_cpu_energy
    run_metrics['energy_consumption']['testing']['gpu_kwh'] = testing_gpu_energy
    run_metrics['energy_consumption']['testing']['total_kwh'] = total_testing_energy
    run_metrics['gpu_memory']['training_peak_mb'] = training_memory['peak_mb']
    run_metrics['gpu_memory']['testing_peak_mb'] = testing_memory['peak_mb']
    
    # Add to results array
    all_results.append(run_metrics)
    
    # Print results for this run
    print(f"\nRUN {run_num} RESULTS:")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Testing Time: {testing_time:.2f}s")
    print(f"Accuracy: {final_accuracy * 100:.3f}%")
    print(f"Carbon Emissions: {total_emissions:.6f} kg CO2")
    print(f"Total Energy: {total_training_energy + total_testing_energy:.6f} kWh")
    print(f"RAM Usage - Training: {run_metrics['ram_usage']['training']['peak_rss_mb']:.1f} MB (peak)")
    print(f"RAM Usage - Testing: {run_metrics['ram_usage']['testing']['peak_rss_mb']:.1f} MB (peak)")
    
    # Clear GPU memory between runs
    if cuda_available:
        torch.cuda.empty_cache()

# FINAL SUMMARY
print(f"\n{'='*60}")
print("SUMMARY OF ALL RUNS")
print(f"{'='*60}")

# Calculate statistics across all runs
accuracies = [run['accuracy'] for run in all_results]
training_times = [run['training_time'] for run in all_results]
testing_times = [run['prediction_time'] for run in all_results]
total_energies = [run['energy_consumption']['training']['total_kwh'] + 
                 run['energy_consumption']['testing']['total_kwh'] for run in all_results]
carbon_emissions = [run['carbon_emissions']['total'] for run in all_results]
ram_peaks_training = [run['ram_usage']['training']['peak_rss_mb'] for run in all_results]
ram_peaks_testing = [run['ram_usage']['testing']['peak_rss_mb'] for run in all_results]

print(f"Runs completed: {NUM_RUNS}")
print(f"Accuracy - Mean: {sum(accuracies)/len(accuracies):.3f}%, Std: {(sum([(x-sum(accuracies)/len(accuracies))**2 for x in accuracies])/len(accuracies))**0.5:.3f}%")
print(f"Training Time - Mean: {sum(training_times)/len(training_times):.2f}s")
print(f"Testing Time - Mean: {sum(testing_times)/len(testing_times):.2f}s")
print(f"Total Energy - Mean: {sum(total_energies)/len(total_energies):.6f} kWh")
print(f"Carbon Emissions - Mean: {sum(carbon_emissions)/len(carbon_emissions):.6f} kg CO2")
print(f"RAM Peak (Training) - Mean: {sum(ram_peaks_training)/len(ram_peaks_training):.1f} MB")
print(f"RAM Peak (Testing) - Mean: {sum(ram_peaks_testing)/len(ram_peaks_testing):.1f} MB")

# Save all results to JSON file
output_file = f'vgg16_neuralhd_metrics_{NUM_RUNS}_runs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
final_output = {
    'experiment_info': {
        'total_runs': NUM_RUNS,
        'experiment_date': datetime.now().isoformat(),
        'configuration': {
            'dimensions': DIMENSIONS,
            'img_size': IMG_WIDTH,
            'batch_size': BATCH_SIZE,
            'feature_size': feature_size,
            'num_classes': NUM_CLASSES,
            'model_type': 'VGG16_NeuralHD'
        }
    },
    'summary_statistics': {
        'accuracy': {
            'mean': sum(accuracies)/len(accuracies),
            'std': (sum([(x-sum(accuracies)/len(accuracies))**2 for x in accuracies])/len(accuracies))**0.5,
            'min': min(accuracies),
            'max': max(accuracies)
        },
        'training_time': {
            'mean': sum(training_times)/len(training_times),
            'std': (sum([(x-sum(training_times)/len(training_times))**2 for x in training_times])/len(training_times))**0.5
        },
        'total_energy': {
            'mean': sum(total_energies)/len(total_energies),
            'std': (sum([(x-sum(total_energies)/len(total_energies))**2 for x in total_energies])/len(total_energies))**0.5
        },
        'ram_usage': {
            'training_peak_mean': sum(ram_peaks_training)/len(ram_peaks_training),
            'training_peak_std': (sum([(x-sum(ram_peaks_training)/len(ram_peaks_training))**2 for x in ram_peaks_training])/len(ram_peaks_training))**0.5,
            'testing_peak_mean': sum(ram_peaks_testing)/len(ram_peaks_testing),
            'testing_peak_std': (sum([(x-sum(ram_peaks_testing)/len(ram_peaks_testing))**2 for x in ram_peaks_testing])/len(ram_peaks_testing))**0.5
        }
    },
    'individual_runs': all_results
}

with open(output_file, 'w') as f:
    json.dump(final_output, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("="*60)

# Print final classification report from the last run
from sklearn.metrics import classification_report, confusion_matrix
print("\nFinal Classification Report (from last run):")
print(classification_report(all_labels, all_predictions, target_names=class_names))

print("\nFinal Confusion Matrix (from last run):")
print(confusion_matrix(all_labels, all_predictions))