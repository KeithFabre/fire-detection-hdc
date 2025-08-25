import os
import numpy as np
import time
import tracemalloc
from tqdm import tqdm
from PIL import Image

# Hyperparameters
D = 10000  # Hypervector dimension
IMG_SIZE = (100, 100)  # All images resized to 100x100
PATCH_SIZE = 4  # For patch-based encoding

def load_images_from_directory(directory, label):
    """Load and preprocess images from directory"""
    images = []
    for filename in tqdm(os.listdir(directory), desc=f"Loading {os.path.basename(directory)}"):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            img_path = os.path.join(directory, filename)
            try:
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize(IMG_SIZE)
                img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
                images.append((img_array, label))
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
    return images

def generate_random_vectors(n, D):
    """Generate n random float vectors of dimension D"""
    return np.random.normal(0, 1, size=(n, D))

def pixel_based_encoding(image, random_vectors):
    """Pixel-based hypervector encoding with float values"""
    hypervector = np.zeros(D)
    h, w = image.shape
    for i in range(h):
        for j in range(w):
            # Weight by pixel intensity
            hypervector += random_vectors[i * w + j] * image[i, j]
    return hypervector

def patch_based_encoding(image, random_vectors, patch_size=4):
    """Patch-based hypervector encoding with float values"""
    hypervector = np.zeros(D)
    h, w = image.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            patch_mean = np.mean(patch)
            if patch_mean > 0:  # Only encode non-zero patches
                patch_id = (i // patch_size) * (w // patch_size) + (j // patch_size)
                hypervector += random_vectors[patch_id] * patch_mean
    return hypervector

def train(images, encoding_fn, random_vectors):
    """Train HDC classifier with float vectors"""
    class_vectors = {}
    for image, label in tqdm(images, desc="Training"):
        hv = encoding_fn(image, random_vectors)
        if label not in class_vectors:
            class_vectors[label] = hv
        else:
            class_vectors[label] += hv
    
    # Normalize class vectors
    for label in class_vectors:
        norm = np.linalg.norm(class_vectors[label])
        if norm > 0:
            class_vectors[label] /= norm
    return class_vectors

def classify(image, encoding_fn, random_vectors, class_vectors):
    """Classify image using cosine similarity"""
    hv = encoding_fn(image, random_vectors)
    best_class = None
    max_similarity = -float('inf')
    for label, class_hv in class_vectors.items():
        # Cosine similarity
        norm_hv = np.linalg.norm(hv)
        if norm_hv == 0:
            similarity = 0
        else:
            similarity = np.dot(hv, class_hv) / (norm_hv * np.linalg.norm(class_hv))
        if similarity > max_similarity:
            max_similarity = similarity
            best_class = label
    return best_class

def measure_time_and_memory(func, *args, **kwargs):
    """Measure execution time and memory usage"""
    tracemalloc.start()
    start_time = time.time()
    
    result = func(*args, **kwargs)
    
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return result, end_time - start_time, peak / 10**6  # MB

if __name__ == "__main__":
    # Load dataset
    print("Loading training data...")
    fire_train = load_images_from_directory("Training/Fire", 1)
    nofire_train = load_images_from_directory("Training/No_Fire", 0)
    train_data = fire_train + nofire_train
    
    print("\nLoading test data...")
    fire_test = load_images_from_directory("Test/Fire", 1)
    nofire_test = load_images_from_directory("Test/No_Fire", 0)
    test_data = fire_test + nofire_test
    np.random.shuffle(test_data)
    
    # Prepare test images and labels
    test_images, test_labels = zip(*test_data)
    
    print(f"\nDataset stats:")
    print(f"- Training: {len(train_data)} images ({len(fire_train)} fire, {len(nofire_train)} no-fire)")
    print(f"- Test: {len(test_data)} images ({len(fire_test)} fire, {len(nofire_test)} no-fire)")
    
    # Initialize HDC
    h, w = IMG_SIZE
    n_pixels = h * w
    n_patches = (h // PATCH_SIZE) * (w // PATCH_SIZE)
    
    # Test both encoding methods
    for encoding_name, encoding_fn, n_vectors in [
        ("pixel", pixel_based_encoding, n_pixels),
        ("patch", lambda img, rv: patch_based_encoding(img, rv, PATCH_SIZE), n_patches)
    ]:
        print(f"\n{'='*50}")
        print(f"Training with {encoding_name}-based encoding")
        print(f"Generating {n_vectors} random vectors...")
        random_vectors = generate_random_vectors(n_vectors, D)
        
        # Train
        class_vectors, train_time, train_mem = measure_time_and_memory(
            train, train_data, encoding_fn, random_vectors)
        print(f"Training completed in {train_time:.2f}s, peak memory: {train_mem:.2f}MB")
        
        # Test
        correct = 0
        test_images, test_labels = zip(*test_data)  # Regenerate after possible shuffling
        start_time = time.time()
        for i, (image, true_label) in tqdm(enumerate(zip(test_images, test_labels)), 
                                          total=len(test_data), desc="Testing"):
            pred_label = classify(image, encoding_fn, random_vectors, class_vectors)
            if pred_label == true_label:
                correct += 1
        test_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = correct / len(test_data)
        print(f"\nResults ({encoding_name}-based encoding):")
        print(f"- Accuracy: {accuracy:.4f} ({correct}/{len(test_data)})")
        print(f"- Test time: {test_time:.2f}s ({test_time/len(test_data):.4f}s per image)")
        print(f"- Total time: {train_time + test_time:.2f}s")
        print(f"{'='*50}")