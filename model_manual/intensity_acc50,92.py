# intensity encoding
# 50.92% accuracy

import os
import numpy as np
from PIL import Image
from tqdm import tqdm  # For progress bars

# Configuration
DIMENSION = 10000  # Hypervector dimension
IMAGE_SIZE = (64, 64)  # All images will be resized to this
BINARY_THRESHOLD = 128  # For fire/non-fire pixel differentiation

# Generate random hypervectors
def generate_hypervectors(num_vectors):
    return np.random.choice([-1, 1], size=(num_vectors, DIMENSION))

# Encode image to hypervector
def image_to_hypervector(image, position_hvs, intensity_hvs):
    img_array = np.array(image.convert('L').resize(IMAGE_SIZE))
    hv = np.zeros(DIMENSION)
    
    height, width = img_array.shape
    for y in range(height):
        for x in range(width):
            # get pixel intensity (0-255)
            intensity = img_array[y, x]
            # binarize based on fire-like intensity
            intensity_bin = 0 if intensity < BINARY_THRESHOLD else 1
            
            # get position index (unique per pixel)
            pos_idx = y * width + x
            # get hypervectors for this position and intensity
            pos_hv = position_hvs[pos_idx]
            int_hv = intensity_hvs[intensity_bin]
            
            # binding and bundling
            hv += pos_hv * int_hv  # Binding is multiplication
            
    # normalize and binarize
    return np.sign(hv)

# classify hypervector
def classify_hv(hv, fire_prototype, no_fire_prototype):
    fire_sim = np.dot(hv, fire_prototype)
    no_fire_sim = np.dot(hv, no_fire_prototype)
    return "Fire" if fire_sim > no_fire_sim else "Not_Fire"

def main():
    print("HDC Fire Classifier")
    print(f"Hypervector Dimension: {DIMENSION}")
    print(f"Image Size: {IMAGE_SIZE}")
    
    # prepare hypervectors
    num_positions = IMAGE_SIZE[0] * IMAGE_SIZE[1]
    position_hvs = generate_hypervectors(num_positions)
    intensity_hvs = generate_hypervectors(2)  # 2 intensity levels
    
    # prepare datasets
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
                img = Image.open(img_path)
                hv = image_to_hypervector(img, position_hvs, intensity_hvs)
                hvs.append(hv)
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
    
    # Train classifier - create prototypes
    print("\nTraining classifier...")
    fire_prototype = np.mean(datasets["Training/Fire"], axis=0)
    no_fire_prototype = np.mean(datasets["Training/No_Fire"], axis=0)
    
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
        if pred == "Not_Fire":
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
