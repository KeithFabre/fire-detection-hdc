
import numpy as np
from PIL import Image

def create_random_hypervector(dimensions):
    return np.random.randint(0, 2, dimensions) * 2 - 1  # Bipolar hypervector (-1, 1)

def hamming_distance(hv1, hv2):
    return np.sum(hv1 != hv2)

def pixel_based_encoding(image_path, dimensions=10000):
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img_array = np.array(img)

    # Create a base hypervector for each possible pixel intensity (0-255)
    # For simplicity, let's use a fixed set of random hypervectors for intensities
    # In a real scenario, these would be learned or generated systematically
    intensity_hypervectors = {i: create_random_hypervector(dimensions) for i in range(256)}

    image_hypervector = np.zeros(dimensions) # Initialize with zeros for bundling

    for pixel_value in img_array.flatten():
        image_hypervector += intensity_hypervectors[pixel_value]

    # Binarize the bundled hypervector
    image_hypervector = np.where(image_hypervector > 0, 1, -1)

    return image_hypervector

# Example Usage (will be replaced by actual dataset processing)
# fire_image_path = '/home/ubuntu/upload/resized_test_fire_frame0.jpg'
# fire_hv = pixel_based_encoding(fire_image_path)
# print(f'Hypervector for fire image created with shape: {fire_hv.shape}')

# Placeholder for classification logic
# def classify(query_hv, class_prototypes):
#     min_distance = float('inf')
#     predicted_class = None
#     for class_name, prototype_hv in class_prototypes.items():
#         dist = hamming_distance(query_hv, prototype_hv)
#         if dist < min_distance:
#             min_distance = dist
#             predicted_class = class_name
#     return predicted_class

# Placeholder for training (creating class prototypes)
# def train(encoded_images_with_labels):
#     class_prototypes = {}
#     for hv, label in encoded_images_with_labels:
#         if label not in class_prototypes:
#             class_prototypes[label] = np.zeros(hv.shape)
#         class_prototypes[label] += hv
#     # Binarize prototypes
#     for label in class_prototypes:
#         class_prototypes[label] = np.where(class_prototypes[label] > 0, 1, -1)
#     return class_prototypes





def classify(query_hv, class_prototypes):
    min_distance = float("inf")
    predicted_class = None
    for class_name, prototype_hv in class_prototypes.items():
        dist = hamming_distance(query_hv, prototype_hv)
        if dist < min_distance:
            min_distance = dist
            predicted_class = class_name
    return predicted_class

def train(encoded_images_with_labels, dimensions=10000):
    class_prototypes = {}
    for hv, label in encoded_images_with_labels:
        if label not in class_prototypes:
            class_prototypes[label] = np.zeros(dimensions)
        class_prototypes[label] += hv
    # Binarize prototypes
    for label in class_prototypes:
        class_prototypes[label] = np.where(class_prototypes[label] > 0, 1, -1)
    return class_prototypes




def patch_based_encoding(image_path, patch_size=(16, 16), dimensions=10000):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)

    img_height, img_width = img_array.shape
    patch_h, patch_w = patch_size

    # Create random hypervectors for each unique patch position
    # This is a simplified positional encoding. More complex schemes exist.
    position_hypervectors = {}
    patch_id = 0
    for r in range(0, img_height - patch_h + 1, patch_h):
        for c in range(0, img_width - patch_w + 1, patch_w):
            position_hypervectors[patch_id] = create_random_hypervector(dimensions)
            patch_id += 1

    image_hypervector = np.zeros(dimensions)
    patch_id = 0
    for r in range(0, img_height - patch_h + 1, patch_h):
        for c in range(0, img_width - patch_w + 1, patch_w):
            patch = img_array[r:r+patch_h, c:c+patch_w]
            # Simple average of pixel values in the patch for encoding
            # More sophisticated methods would encode the patch content itself
            avg_pixel_value = int(np.mean(patch))

            # Create a hypervector for the patch content (simplified)
            # In a real scenario, this would be more robust, e.g., using intensity_hypervectors
            content_hv = create_random_hypervector(dimensions) # Placeholder

            # Bind content and position hypervectors
            bound_hv = content_hv * position_hypervectors[patch_id] # Element-wise multiplication for binding
            image_hypervector += bound_hv
            patch_id += 1

    image_hypervector = np.where(image_hypervector > 0, 1, -1)
    return image_hypervector





if __name__ == "__main__":
    # Dummy data for demonstration. Replace with actual dataset loading.
    # Assume you have a list of (image_path, label) tuples
    dummy_dataset = [
        ("/home/ubuntu/upload/resized_test_fire_frame0.jpg", "Fire"),
        ("/home/ubuntu/upload/resized_test_fire_frame0.jpg", "Fire"),
        # Add more dummy data for 'Not_Fire' class
        # For a real scenario, you would load your actual dataset here
        # For now, let's simulate a 'Not_Fire' image by using a different image or a modified version
        ("/home/ubuntu/upload/resized_test_fire_frame0.jpg", "Not_Fire"), # Placeholder for a 'Not_Fire' image
        ("/home/ubuntu/upload/resized_test_fire_frame0.jpg", "Not_Fire"), # Placeholder for a 'Not_Fire' image
    ]

    # --- Pixel-based Encoding Example ---
    print("\n--- Pixel-based Encoding and Classification ---")
    encoded_pixel_data = []
    total_pixel_encoding_time = 0
    total_pixel_encoding_memory_current = 0
    total_pixel_encoding_memory_peak = 0

    for img_path, label in dummy_dataset:
        (hv, current_mem, peak_mem), encoding_time = measure_memory_usage(measure_execution_time, pixel_based_encoding, img_path)
        encoded_pixel_data.append((hv, label))
        total_pixel_encoding_time += encoding_time
        total_pixel_encoding_memory_current += current_mem
        total_pixel_encoding_memory_peak += peak_mem

    print(f"Pixel-based Encoding Time: {total_pixel_encoding_time:.4f} seconds")
    print(f"Pixel-based Encoding Memory (Current/Peak): {total_pixel_encoding_memory_current / (1024 * 1024):.2f}MB / {total_pixel_encoding_memory_peak / (1024 * 1024):.2f}MB")

    # Train the model
    pixel_prototypes = train(encoded_pixel_data)
    print("Pixel-based prototypes trained.")

    # Classify a dummy image
    test_image_path = "/home/ubuntu/upload/resized_test_fire_frame0.jpg"
    (test_hv_pixel, current_mem, peak_mem), classification_time = measure_memory_usage(measure_execution_time, pixel_based_encoding, test_image_path)
    predicted_class_pixel = classify(test_hv_pixel, pixel_prototypes)
    print(f"Pixel-based Classification Time: {classification_time:.4f} seconds")
    print(f"Pixel-based Classification Memory (Current/Peak): {current_mem / (1024 * 1024):.2f}MB / {peak_mem / (1024 * 1024):.2f}MB")
    print(f"Test image classified as: {predicted_class_pixel} (Pixel-based)")

    # --- Patch-based Encoding Example ---
    print("\n--- Patch-based Encoding and Classification ---")
    encoded_patch_data = []
    total_patch_encoding_time = 0
    total_patch_encoding_memory_current = 0
    total_patch_encoding_memory_peak = 0

    for img_path, label in dummy_dataset:
        (hv, current_mem, peak_mem), encoding_time = measure_memory_usage(measure_execution_time, patch_based_encoding, img_path)
        encoded_patch_data.append((hv, label))
        total_patch_encoding_time += encoding_time
        total_patch_encoding_memory_current += current_mem
        total_patch_encoding_memory_peak += peak_mem

    print(f"Patch-based Encoding Time: {total_patch_encoding_time:.4f} seconds")
    print(f"Patch-based Encoding Memory (Current/Peak): {total_patch_encoding_memory_current / (1024 * 1024):.2f}MB / {total_patch_encoding_memory_peak / (1024 * 1024):.2f}MB")

    # Train the model
    patch_prototypes = train(encoded_patch_data)
    print("Patch-based prototypes trained.")

    # Classify a dummy image
    test_hv_patch = patch_based_encoding(test_image_path)
    (test_hv_patch, current_mem, peak_mem), classification_time = measure_memory_usage(measure_execution_time, patch_based_encoding, test_image_path)
    predicted_class_patch = classify(test_hv_patch, patch_prototypes)
    print(f"Patch-based Classification Time: {classification_time:.4f} seconds")
    print(f"Patch-based Classification Memory (Current/Peak): {current_mem / (1024 * 1024):.2f}MB / {peak_mem / (1024 * 1024):.2f}MB")
    print(f"Test image classified as: {predicted_class_patch} (Patch-based)")

    print("\nNote: For actual classification, you would need a diverse dataset with both 'Fire' and 'Not_Fire' images, and proper splitting into training and testing sets.")




import time

def measure_execution_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time




import tracemalloc

def measure_memory_usage(func, *args, **kwargs):
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, current, peak

