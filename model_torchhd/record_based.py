# record-based encoding with multiple runs and comprehensive logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchmetrics
from tqdm import tqdm
import torchhd
from torchhd.models import Centroid
from torchhd import embeddings
import time
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

torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# Check if CUDA is available for detailed GPU monitoring
cuda_available = torch.cuda.is_available()
if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# =============================================================================
# CONFIGURATION - Modify these parameters
# =============================================================================
NUM_RUNS = 1  
DIMENSIONS = 1000
IMG_SIZE = 256
NUM_LEVELS = 100
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_BATCH_SIZE = 8
NUM_WORKERS = 1

# Directories
training_directory = './Training'
test_directory = './Test'
# =============================================================================

# Data preprocessing
transform = transforms.Compose([
    transforms.Resize(IMG_HEIGHT),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=1)
])

# Load datasets (only once)
print("Loading datasets...")
training_ds = datasets.ImageFolder(root=training_directory, transform=transform)
training_classes_names = training_ds.classes
print(f"Training Classes found: {training_classes_names}")
print(f"Training Class to index mapping: {training_ds.class_to_idx}")

test_ds = datasets.ImageFolder(root=test_directory, transform=transform)
test_classes_names = test_ds.classes
print(f"Test Classes found: {test_classes_names}")
print(f"Test Class to index mapping: {test_ds.class_to_idx}")

# Data loaders
train_ld = DataLoader(training_ds, batch_size=IMG_BATCH_SIZE)
test_ld = DataLoader(test_ds, batch_size=IMG_BATCH_SIZE)

# Encoder class
class Encoder(nn.Module):
    def __init__(self, out_features, size, levels):
        super(Encoder, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.position = embeddings.Random(size * size, out_features)
        self.value = embeddings.Level(levels, out_features)
    
    def forward(self, x):
        x = self.flatten(x)
        sample_hv = torchhd.bind(self.position.weight, self.value(x))
        sample_hv = torchhd.multiset(sample_hv)
        return torchhd.hard_quantize(sample_hv)

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
    
    # Initialize models for this run
    encode = Encoder(DIMENSIONS, IMG_SIZE, NUM_LEVELS)
    encode = encode.to(device)
    
    num_classes = len(training_ds.classes)
    model = Centroid(DIMENSIONS, num_classes)
    model = model.to(device)
    
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
            'batch_size': IMG_BATCH_SIZE
        }
    }
    
    # TRAINING PHASE
    print('Starting Training...')
    
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
    
    with torch.no_grad():
        for batch_idx, (samples, labels) in enumerate(tqdm(train_ld, desc=f"Training Run {run_num}")):
            batch_start = time.time()
            
            samples = samples.to(device)
            labels = labels.to(device)
            
            # Monitor resources
            cpu_power = get_cpu_power()
            gpu_util = gpu_monitor.get_gpu_utilization()
            gpu_power = estimate_gpu_power(gpu_util)
            
            # Monitor RAM
            current_ram = get_memory_usage()
            peak_ram_usage = max(peak_ram_usage, current_ram['rss_mb'])
            
            samples_hv = encode(samples)
            model.add(samples_hv, labels)
            
            # Calculate energy for this batch
            batch_duration = time.time() - batch_start
            training_cpu_energy += (cpu_power * batch_duration) / 3600
            training_gpu_energy += (gpu_power * batch_duration) / 3600
    
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
    
    # TESTING PHASE
    print('Starting Testing...')
    
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
    
    accuracy = torchmetrics.Accuracy("multiclass", num_classes=num_classes)
    
    with torch.no_grad():
        model.normalize()
        for samples, labels in tqdm(test_ld, desc=f"Testing Run {run_num}"):
            batch_start = time.time()
            
            samples = samples.to(device)
            
            # Monitor resources
            cpu_power = get_cpu_power()
            gpu_util = gpu_monitor.get_gpu_utilization()
            gpu_power = estimate_gpu_power(gpu_util)
            
            # Monitor RAM
            current_ram = get_memory_usage()
            peak_ram_usage_test = max(peak_ram_usage_test, current_ram['rss_mb'])
            
            samples_hv = encode(samples)
            outputs = model(samples_hv, dot=True)
            
            # Calculate energy for this batch
            batch_duration = time.time() - batch_start
            testing_cpu_energy += (cpu_power * batch_duration) / 3600
            testing_gpu_energy += (gpu_power * batch_duration) / 3600
            
            accuracy.update(outputs.cpu(), labels)
    
    testing_end_time = time.time()
    testing_time = testing_end_time - testing_start_time
    testing_memory = gpu_monitor.get_memory_usage()
    final_accuracy = accuracy.compute().item()
    
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
output_file = f'hdc_metrics_{NUM_RUNS}_runs_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
final_output = {
    'experiment_info': {
        'total_runs': NUM_RUNS,
        'experiment_date': datetime.now().isoformat(),
        'configuration': {
            'dimensions': DIMENSIONS,
            'img_size': IMG_SIZE,
            'num_levels': NUM_LEVELS,
            'batch_size': IMG_BATCH_SIZE
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