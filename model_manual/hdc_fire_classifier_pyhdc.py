import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2

# Configuration
IMAGE_SIZE = (64, 64)  # All images will be resized to this
DIMENSIONS = 10000  # Hypervector dimension

# Feature extractor for static images
class ImageFeatureExtractor:
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape
        
        # Compute image gradients
        self.compute_gradients()
        
        # Compute features
        self.compute_features()
    
    def compute_gradients(self):
        # Convert to float32 for gradient calculation
        img_float = self.image.astype(np.float32)
        
        # Compute gradients using Sobel operators
        grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute gradient magnitude and orientation
        self.grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        self.grad_orientation = np.arctan2(grad_y, grad_x)
    
    def compute_features(self):
        # Basic statistics
        self.mean_val = np.mean(self.image)
        self.std_val = np.std(self.image)
        self.min_val = np.min(self.image)
        self.max_val = np.max(self.image)
        
        # Gradient statistics
        self.mean_grad = np.mean(self.grad_magnitude)
        self.std_grad = np.std(self.grad_magnitude)
        
        # Non-zero pixel count
        self.non_zero_count = np.count_nonzero(self.image)
        self.non_zero_ratio = self.non_zero_count / (self.height * self.width)
        
        # Edge density (using Canny edge detection)
        edges = cv2.Canny(self.image, 100, 200)
        self.edge_density = np.count_nonzero(edges) / (self.height * self.width)
        
        # Feature vector
        self.features = np.array([
            self.mean_val,
            self.std_val,
            self.min_val,
            self.max_val,
            self.mean_grad,
            self.std_grad,
            self.non_zero_count,
            self.non_zero_ratio,
            self.edge_density
        ])

# Create random hypervectors for feature encoding
def create_feature_vectors(feature_count):
    return [np.random.randint(0, 2, DIMENSIONS) for _ in range(feature_count)]

# Encode image using features
def encode_image(image):
    # Create feature extractor
    extractor = ImageFeatureExtractor(image)
    features = extractor.features
    
    # Normalize features to 0-1 range
    normalized_features = (features - np.min(features)) / (np.max(features) - np.min(features) + 1e-8)
    
    # Create hypervectors for each feature
    feature_vectors = create_feature_vectors(len(features))
    
    # Initialize accumulator
    accumulator = np.zeros(DIMENSIONS)
    
    # Combine feature vectors
    for i, feature_value in enumerate(normalized_features):
        # Scale feature value to [0, 1] and use as weight
        weight = feature_value
        accumulator += weight * feature_vectors[i]
    
    # Binarize the accumulator
    median = np.median(accumulator)
    hypervector = (accumulator > median).astype(int)
    
    return hypervector

# Calculate Hamming distance
def hamming_distance(hv1, hv2):
    return np.sum(hv1 != hv2)

# Classify hypervector using Hamming distance
def classify_hv(hv, fire_prototype, no_fire_prototype):
    fire_dist = hamming_distance(hv, fire_prototype)
    no_fire_dist = hamming_distance(hv, no_fire_prototype)
    # Lower distance means more similar
    return "Fire" if fire_dist < no_fire_dist else "No_Fire"

def load_image(path):
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = img.resize(IMAGE_SIZE)
    return np.array(img)

def main():
    print("Feature-Based Fire Classifier")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Hypervector Dimension: {DIMENSIONS}")
    
    # Prepare datasets
    datasets = {
        "Training/Fire": [],
        "Training/No_Fire": [],
        "Test/Fire": [],
        "Test/No_Fire": []
    }
    
    # Load and process all images
    print("\nProcessing images...")
    for category, hvs in datasets.items():
        if not os.path.exists(category):
            print(f"Warning: Directory '{category}' not found!")
            continue
            
        files = [f for f in os.listdir(category) if f.endswith('.jpg')]
        for file in tqdm(files, desc=category):
            img_path = os.path.join(category, file)
            try:
                img = load_image(img_path)
                hv = encode_image(img)  # No vmap needed
                hvs.append(hv)
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
    
    # Train classifier - create prototypes
    print("\nTraining classifier...")
    # Create prototype by averaging all vectors
    fire_prototype = np.mean(datasets["Training/Fire"], axis=0)
    no_fire_prototype = np.mean(datasets["Training/No_Fire"], axis=0)
    
    # Binarize prototypes
    fire_prototype = (fire_prototype > 0.5).astype(int)
    no_fire_prototype = (no_fire_prototype > 0.5).astype(int)
    
    # Test classifier
    print("\nTesting classifier...")
    correct = 0
    total = 0
    
    # Test Fire images
    for hv in tqdm(datasets["Test/Fire"], desc="Testing Fire"):
        pred = classify_hv(hv, fire_prototype, no_fire_prototype)
        if pred == "Fire":
            correct += 1
        total += 1
    
    # Test No_Fire images
    for hv in tqdm(datasets["Test/No_Fire"], desc="Testing No_Fire"):
        pred = classify_hv(hv, fire_prototype, no_fire_prototype)
        if pred == "No_Fire":
            correct += 1
        total += 1
    
    # Results
    accuracy = correct / total
    print(f"\nResults:")
    print(f"Tested {total} images")
    print(f"Correct classifications: {correct}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
