import numpy as np
from pyhdc import HDC
from PIL import Image
import matplotlib.pyplot as plt

# Initialize HDC
hdc = HDC()

# Load and preprocess image
def load_image(path, size=(64, 64)):
    img = Image.open(path).convert('L')  # Convert to grayscale
    img = img.resize(size)
    return np.array(img)

# Encode image using PyHDC
def encode_image(image):
    # Flatten image to 1D array
    flat_img = image.flatten().astype(float)
    # Normalize to [0, 1]
    flat_img /= 255.0
    # Create hypervector
    return hdc.vector(flat_img)

# Example usage
if __name__ == "__main__":
    # Load sample image
    img_path = "Test/Fire/resized_test_fire_frame0.jpg"
    image = load_image(img_path)
    
    # Encode to hypervector
    hv = encode_image(image)
    
    print(f"Created hypervector with dimension: {len(hv)}")
    print(f"First 10 values: {hv[:10]}")
    
    # Visualize original and encoded information
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.hist(hv, bins=50)
    plt.title("Hypervector Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig("pyhdc_encoding_example.png")
    print("Saved visualization: pyhdc_encoding_example.png")
