import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
from tqdm import tqdm, trange
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import psutil
import json
from datetime import datetime
import torchhd

# Try to import codecarbon
try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
    print("CodeCarbon available - will track carbon emissions")
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not available - install with: pip install codecarbon")


# =============================================================================
# CONFIGURATION - Modify these parameters
# =============================================================================
IMG_SIZE = 64
DIMENSIONS = 10000  # Increased for better accuracy as per paper
NUM_EPOCHS = 10     # For iterative training
NUM_RUNS = 1
BATCH_SIZE = 8
# StrideHD specific parameters from the paper
W_REC = 5  # Receptive Field Length (width and height)
T_REC = 3  # Stride of the window
L_B = 4    # Binary Levels for thermometer encoding
L_E = 2    # Number of training subsets (from paper)
# =============================================================================


# Define the transform to keep the image in RGB (3 channels)
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load training and test datasets
train_ds = datasets.ImageFolder(root='./Training', transform=transform)
test_ds = datasets.ImageFolder(root='./Test', transform=transform)

train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_ld = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Get number of classes and input image size
num_classes = len(train_ds.classes)
sample_image, _ = train_ds[0]
num_channels = sample_image.shape[0]

print(f"Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"Number of channels: {num_channels}")
print(f"Number of classes: {num_classes}")


# =============================================================================
# Corrected StrideHD Encoder
# =============================================================================
class StrideHDEncoder(nn.Module):
    def __init__(self, in_channels, out_dims, window_size, stride, binary_levels, num_training_subsets, device):
        super().__init__()
        self.in_channels = in_channels
        self.out_dims = out_dims
        self.window_size = window_size
        self.stride = stride
        self.binary_levels = binary_levels
        self.num_training_subsets = num_training_subsets
        self.device = device
        
        # Paper parameters - M is number of binary vectors, B is range
        self.M = 64  # Number of binary vectors per distributed hypervector
        self.B = 8   # Range for binary number generation
        self.vector_size = out_dims // self.M  # Dimensions per binary vector
        
        # Precompute random selection matrix R (paper Eq. 4)
        self.register_buffer('R', torch.randint(0, 1000, (self.M, self.B)))
        
    def window_striding_maxpool(self, x):
        """Apply window striding and max-pooling"""
        batch_size, num_channels, height, width = x.shape
        
        # Calculate number of patches
        h_patches = (height - self.window_size) // self.stride + 1
        w_patches = (width - self.window_size) // self.stride + 1
        
        # Extract patches for each channel
        all_patches = []
        for c in range(num_channels):
            patches = x[:, c:c+1].unfold(2, self.window_size, self.stride).unfold(3, self.window_size, self.stride)
            # patches shape: [batch_size, 1, h_patches, w_patches, window_size, window_size]
            patches = patches.contiguous().view(batch_size, h_patches * w_patches, self.window_size, self.window_size)
            all_patches.append(patches)
        
        # Combine channels
        all_patches = torch.cat(all_patches, dim=1)  # [batch_size, total_patches, window_size, window_size]
        
        # Apply max-pooling within each window
        max_pooled = F.adaptive_max_pool2d(all_patches, (1, 1)).squeeze(-1).squeeze(-1)
        # max_pooled shape: [batch_size, total_patches]
        
        return max_pooled, h_patches * w_patches * num_channels

    def thermometer_binarize(self, x):
        """Proper thermometer encoding per paper Eq. 3"""
        batch_size, num_features = x.shape
        
        # Normalize each feature independently to [0, 1] range
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)
        
        # Create thresholds for each binary level
        thresholds = torch.linspace(0, 1, self.binary_levels + 1, device=self.device)[1:]
        
        # Apply thermometer encoding
        binarized = (x_normalized.unsqueeze(-1) > thresholds.view(1, 1, -1)).float()
        # binarized shape: [batch_size, num_features, binary_levels]
        
        return binarized

    def pseudo_random_encoding(self, binary_patterns):
        """Paper's pseudo-random encoding from Section IV-B"""
        batch_size, num_features, binary_levels = binary_patterns.shape
        n_elements = num_features * binary_levels
        
        # Reshape to get all binary features
        all_features = binary_patterns.reshape(batch_size, -1)  # [batch_size, n_elements]
        
        # Generate hypervectors using pseudo-random mechanism
        encoded_hypervectors = torch.zeros(batch_size, self.out_dims, device=self.device)
        
        for i in range(batch_size):
            features = all_features[i]  # [n_elements]
            
            # Generate addresses p_i (paper Eq. 5)
            for m in range(self.M):
                # Select random features using precomputed R
                selected_indices = self.R[m] % n_elements
                selected_features = features[selected_indices]
                
                # Convert to binary number
                p_i = torch.sum(selected_features * (2 ** torch.arange(self.B-1, -1, -1, device=self.device)))
                p_i = p_i.long() % self.vector_size
                
                # Set the bit at position p_i in the m-th segment
                bit_position = m * self.vector_size + p_i
                if bit_position < self.out_dims:
                    encoded_hypervectors[i, bit_position] = 1
        
        return encoded_hypervectors

    def encode(self, x):
        """Main encoding function"""
        # 1. Window striding and max-pooling
        max_pooled, num_windows = self.window_striding_maxpool(x)
        
        # 2. Thermometer binarization
        binarized_features = self.thermometer_binarize(max_pooled)
        
        # 3. Pseudo-random encoding to hypervectors
        encoded_hvs = self.pseudo_random_encoding(binarized_features)
        
        return encoded_hvs


# A custom transform to integrate the StrideHD encoding into the DataLoader
class StrideHDTransform(nn.Module):
    def __init__(self, encoder, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.device = device

    def forward(self, img_tensor):
        # Ensure the tensor has a batch dimension for the encoder
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Move tensor to the correct device
        img_tensor = img_tensor.to(self.device)
        
        # Encode the tensor
        encoded_hv = self.encoder.encode(img_tensor).squeeze(0)
        
        return encoded_hv


# =============================================================================
# Corrected StrideHD Classifier
# =============================================================================
class StrideHDClassifier(nn.Module):
    def __init__(
        self,
        n_dimensions: int,
        n_classes: int,
        *,
        epochs: int = 10,
        device: torch.device = None,
    ) -> None:
        super().__init__()

        self.n_dimensions = n_dimensions
        self.n_classes = n_classes
        self.epochs = epochs
        self.device = device
        
        # Integer counters for training (paper uses integer accumulation)
        self.classes_counter = torch.zeros((n_classes, n_dimensions), device=device, dtype=torch.int32)
        # Binary hypervectors for inference
        self.classes_hv = torch.zeros((n_classes, n_dimensions), device=device, dtype=torch.bool)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.classes_counter)
        nn.init.zeros_(self.classes_hv)

    def single_pass_fit(self, input: Tensor, target: Tensor):
        """Single-pass training from paper Section IV-C"""
        # Convert binary hypervectors to integer addition (1 = +1, 0 = 0)
        input_int = input.to(torch.int32)
        
        # Accumulate hypervectors for each class
        for class_idx in range(self.n_classes):
            class_mask = (target == class_idx)
            if class_mask.sum() > 0:
                class_vectors = input_int[class_mask]
                self.classes_counter[class_idx] += class_vectors.sum(dim=0)
        
        # Binarize for inference (simple threshold)
        self.classes_hv = (self.classes_counter > 0)

    def iterative_fit(self, input: Tensor, target: Tensor):
        """Iterative training from paper Section IV-F"""
        print("Starting iterative training...")
        
        for epoch in range(self.epochs):
            wrong_predictions = 0
            
            for i in range(len(input)):
                sample = input[i:i+1]
                true_label = target[i]
                
                # Get prediction
                pred = self.predict(sample)[0]
                
                if pred != true_label:
                    wrong_predictions += 1
                    # Update: add to correct class, subtract from wrong class
                    sample_int = sample.to(torch.int32).squeeze(0)
                    self.classes_counter[true_label] += sample_int
                    self.classes_counter[pred] -= sample_int
            
            # Re-binarize after updates
            self.classes_hv = (self.classes_counter > 0)
            
            print(f"Epoch {epoch+1}/{self.epochs}: {wrong_predictions} wrong predictions")
            
            if wrong_predictions == 0:
                print("Early stopping - all predictions correct")
                break

    def forward(self, samples: Tensor) -> Tensor:
        """Compute similarity scores"""
        # Ensure samples are boolean for bitwise operations
        samples_bool = samples.to(self.device, dtype=torch.bool)
        
        response = torch.zeros((samples.shape[0], self.n_classes), device=self.device)
        
        for i in range(self.n_classes):
            # Paper's similarity: bitwise AND and counting (simplified)
            # This implements the essence of paper Eq. 7
            matches = (samples_bool & self.classes_hv[i].unsqueeze(0)).sum(dim=1)
            response[:, i] = matches.float()
        
        return response

    def predict(self, samples: Tensor) -> Tensor:
        """Predict class labels"""
        similarities = self(samples)
        return torch.argmax(similarities, dim=1)

    def accuracy(self, dataloader):
        """Calculate accuracy on a dataloader"""
        correct = 0
        total = 0
        with torch.no_grad():
            for samples, labels in tqdm(dataloader, desc="Calculating Accuracy"):
                samples = samples.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.predict(samples)
                total += labels.size(0)
                correct += (outputs == labels).sum().item()
        return correct / total


# =============================================================================
# EXPERIMENT EXECUTION
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))

# Initialize encoder
encoder = StrideHDEncoder(
    in_channels=num_channels, 
    out_dims=DIMENSIONS, 
    window_size=W_REC, 
    stride=T_REC, 
    binary_levels=L_B,
    num_training_subsets=L_E,
    device=device
)

# Define the custom transform pipeline with our encoder
custom_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    StrideHDTransform(encoder=encoder, device=device)
])

# Re-load the datasets with the custom transform
print("Encoding datasets...")
train_ds_encoded = datasets.ImageFolder(root='./Training', transform=custom_transform)
test_ds_encoded = datasets.ImageFolder(root='./Test', transform=custom_transform)

train_ld_encoded = DataLoader(train_ds_encoded, batch_size=BATCH_SIZE, shuffle=True)
test_ld_encoded = DataLoader(test_ds_encoded, batch_size=BATCH_SIZE, shuffle=False)

print(f"Encoded dataset size - Training: {len(train_ds_encoded)}, Test: {len(test_ds_encoded)}")

# Monitoring classes (from your original code)
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

def get_memory_usage():
    """Get current RAM usage in MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss_mb': memory_info.rss / 1024**2,
        'vms_mb': memory_info.vms / 1024**2,
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

# Placeholder for CodeCarbon
def start_carbon_tracker(): 
    if CODECARBON_AVAILABLE:
        tracker = EmissionsTracker()
        tracker.start()
        return tracker, True
    return None, False

def stop_carbon_tracker(tracker, available):
    if available and tracker:
        emissions = tracker.stop()
        return emissions
    return 0.0

# Run experiments
all_results = []
params = {
    "StrideHD": {
        "epochs": NUM_EPOCHS,
        "dimensions": DIMENSIONS,
        "window_size": W_REC,
        "stride": T_REC,
        "binary_levels": L_B
    }
}

print(f"\n{'='*60}")
print(f"STARTING EXPERIMENTS FOR STRIDEHD")
print(f"{'='*60}")

classifier_results = []

for run_num in range(1, NUM_RUNS + 1):
    print(f"\n{'-'*30}")
    print(f"RUN {run_num}/{NUM_RUNS}")
    print(f"{'-'*30}")

    gpu_monitor = GPUMonitor()
    
    run_metrics = {
        'classifier': 'StrideHD',
        'run_number': run_num,
        'timestamp': datetime.now().isoformat(),
        'training_time': 0,
        'prediction_time': 0,
        'accuracy': 0,
        'carbon_emissions': {'training': 0, 'testing': 0, 'total': 0},
        'energy_consumption': {'training': {'cpu_kwh': 0, 'gpu_kwh': 0, 'total_kwh': 0},
                              'testing': {'cpu_kwh': 0, 'gpu_kwh': 0, 'total_kwh': 0}},
        'gpu_memory': {'training_peak_mb': 0, 'testing_peak_mb': 0},
        'ram_usage': {'training': {'start_rss_mb': 0, 'end_rss_mb': 0, 'peak_rss_mb': 0, 'start_vms_mb': 0, 'end_vms_mb': 0},
                     'testing': {'start_rss_mb': 0, 'end_rss_mb': 0, 'peak_rss_mb': 0, 'start_vms_mb': 0, 'end_vms_mb': 0},
                     'system_memory': {'total_mb': 0, 'training_start_available_mb': 0, 'testing_start_available_mb': 0}},
        'system_info': {'device': str(device), 'cuda_available': torch.cuda.is_available(),
                       'gpu_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A',
                       'dimensions': DIMENSIONS, 'batch_size': BATCH_SIZE, 'img_size': IMG_SIZE,
                       'num_classes': num_classes, 'classifier_params': params["StrideHD"]}
    }
    
    model = StrideHDClassifier(DIMENSIONS, num_classes, device=device, epochs=NUM_EPOCHS)

    # SINGLE-PASS TRAINING PHASE
    print("Single-pass training...")
    training_tracker, training_carbon_available = start_carbon_tracker()
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
    
    # Single-pass training
    all_samples = []
    all_labels = []
    
    for samples, labels in tqdm(train_ld_encoded, desc="Single-pass training"):
        batch_start = time.time()
        samples = samples.to(device)
        labels = labels.to(device)
        
        # Store for potential iterative training
        all_samples.append(samples)
        all_labels.append(labels)
        
        model.single_pass_fit(samples, labels)

        # Monitor resources
        cpu_power = get_cpu_power()
        gpu_util = gpu_monitor.get_gpu_utilization()
        gpu_power = estimate_gpu_power(gpu_util)
        current_ram = get_memory_usage()
        peak_ram_usage = max(peak_ram_usage, current_ram['rss_mb'])
        batch_duration = time.time() - batch_start
        training_cpu_energy += (cpu_power * batch_duration) / 3600
        training_gpu_energy += (gpu_power * batch_duration) / 3600

    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    training_memory = gpu_monitor.get_memory_usage()

    # ITERATIVE TRAINING PHASE (optional - uncomment to use)
    print("Iterative training...")
    all_samples_tensor = torch.cat(all_samples)
    all_labels_tensor = torch.cat(all_labels)
    
    with torch.no_grad():
        model.iterative_fit(all_samples_tensor, all_labels_tensor)

    final_ram = get_memory_usage()
    run_metrics['ram_usage']['training']['end_rss_mb'] = final_ram['rss_mb']
    run_metrics['ram_usage']['training']['end_vms_mb'] = final_ram['vms_mb']
    run_metrics['ram_usage']['training']['peak_rss_mb'] = peak_ram_usage
    training_emissions = stop_carbon_tracker(training_tracker, training_carbon_available)
    
    # TESTING PHASE
    print("Testing...")
    testing_tracker, testing_carbon_available = start_carbon_tracker()
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
    
    # Calculate accuracy
    final_accuracy = model.accuracy(test_ld_encoded)
    
    testing_end_time = time.time()
    testing_time = testing_end_time - testing_start_time
    testing_memory = gpu_monitor.get_memory_usage()
    
    final_ram_test = get_memory_usage()
    run_metrics['ram_usage']['testing']['end_rss_mb'] = final_ram_test['rss_mb']
    run_metrics['ram_usage']['testing']['end_vms_mb'] = final_ram_test['vms_mb']
    run_metrics['ram_usage']['testing']['peak_rss_mb'] = peak_ram_usage_test
    testing_emissions = stop_carbon_tracker(testing_tracker, testing_carbon_available)
    
    total_training_energy = training_cpu_energy + training_gpu_energy
    total_testing_energy = testing_cpu_energy + testing_gpu_energy
    total_emissions = training_emissions + testing_emissions
    
    run_metrics['training_time'] = training_time
    run_metrics['prediction_time'] = testing_time
    run_metrics['accuracy'] = final_accuracy * 100
    run_metrics['carbon_emissions']['training'] = training_emissions
    run_metrics['carbon_emissions']['testing'] = testing_emissions
    run_metrics['carbon_emissions']['total'] = total_emissions
    run_metrics['energy_consumption']['training']['cpu_kwh'] = training_cpu_energy / 1000
    run_metrics['energy_consumption']['training']['gpu_kwh'] = training_gpu_energy / 1000
    run_metrics['energy_consumption']['training']['total_kwh'] = total_training_energy / 1000
    run_metrics['energy_consumption']['testing']['cpu_kwh'] = testing_cpu_energy / 1000
    run_metrics['energy_consumption']['testing']['gpu_kwh'] = testing_gpu_energy / 1000
    run_metrics['energy_consumption']['testing']['total_kwh'] = total_testing_energy / 1000
    run_metrics['gpu_memory']['training_peak_mb'] = training_memory['peak_mb']
    run_metrics['gpu_memory']['testing_peak_mb'] = testing_memory['peak_mb']
    
    classifier_results.append(run_metrics)
    
    print(f"\nStrideHD - RUN {run_num} RESULTS:")
    print(f"Training Time: {training_time:.2f}s")
    print(f"Testing Time: {testing_time:.2f}s")
    print(f"Accuracy: {final_accuracy * 100:.3f}%")
    print(f"Carbon Emissions: {total_emissions:.6f} kg CO2")
    print(f"Total Energy: {(total_training_energy + total_testing_energy) / 1000:.6f} kWh")
    print(f"RAM Usage - Training: {run_metrics['ram_usage']['training']['peak_rss_mb']:.1f} MB (peak)")
    print(f"RAM Usage - Testing: {run_metrics['ram_usage']['testing']['peak_rss_mb']:.1f} MB (peak)")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

all_results.extend(classifier_results)

# Save results
output_file = f'stridehd_corrected_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
final_output = {
    'experiment_info': {
        'total_classifiers': 1, 
        'runs_per_classifier': NUM_RUNS, 
        'experiment_date': datetime.now().isoformat(), 
        'configuration': {
            'dimensions': DIMENSIONS, 
            'img_size': IMG_SIZE, 
            'batch_size': BATCH_SIZE, 
            'num_classes': num_classes
        }
    }, 
    'classifiers': {}, 
    'individual_runs': all_results
}

with open(output_file, 'w') as f:
    json.dump(final_output, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("="*60)