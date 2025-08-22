import os
import numpy as np
from PIL import Image
import time
import tracemalloc
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming hdc_image_classification.py contains the necessary functions
# from hdc_image_classification import pixel_based_encoding, patch_based_encoding, create_random_hypervector, hamming_distance, classify, train

# --- Functions from hdc_image_classification.py (copied for self-containment) ---
def create_random_hypervector(dimensions):
    return np.random.randint(0, 2, dimensions) * 2 - 1  # Bipolar hypervector (-1, 1)

def hamming_distance(hv1, hv2):
    return np.sum(hv1 != hv2)

def pixel_based_encoding(image_path, dimensions=10000):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)

    intensity_hypervectors = {i: create_random_hypervector(dimensions) for i in range(256)}

    image_hypervector = np.zeros(dimensions)

    for pixel_value in img_array.flatten():
        image_hypervector += intensity_hypervectors[pixel_value]

    image_hypervector = np.where(image_hypervector > 0, 1, -1)

    return image_hypervector

def patch_based_encoding(image_path, patch_size=(16, 16), dimensions=10000):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)

    img_height, img_width = img_array.shape
    patch_h, patch_w = patch_size

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
            avg_pixel_value = int(np.mean(patch))

            content_hv = create_random_hypervector(dimensions)

            bound_hv = content_hv * position_hypervectors[patch_id]
            image_hypervector += bound_hv
            patch_id += 1

    image_hypervector = np.where(image_hypervector > 0, 1, -1)
    return image_hypervector

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
    for label in class_prototypes:
        class_prototypes[label] = np.where(class_prototypes[label] > 0, 1, -1)
    return class_prototypes

# --- Measurement Functions ---
def measure_execution_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time

def measure_memory_usage(func, *args, **kwargs):
    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, current, peak

# --- Dataset Loading ---
def load_dataset(base_path):
    data = []
    classes = ["Fire", "Not_Fire"]
    for class_name in classes:
        class_path = os.path.join(base_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Directory {class_path} not found. Skipping.")
            continue
        for img_name in os.listdir(class_path):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_path, img_name)
                data.append((img_path, class_name))
    return data

if __name__ == "__main__":
    # Define dataset paths (adjust as per your actual dataset structure)
    # For demonstration, we'll use the uploaded image as a placeholder for both classes
    # In a real scenario, you would have separate directories for 'Fire' and 'Not_Fire' images
    # e.g., dataset_base_path = "/path/to/your/dataset/train"
    # and test_dataset_base_path = "/path/to/your/dataset/test"

    # Create dummy directories and copy the uploaded image for demonstration
    dummy_train_fire_dir = "./dummy_dataset/train/Fire"
    dummy_train_not_fire_dir = "./dummy_dataset/train/Not_Fire"
    dummy_test_fire_dir = "./dummy_dataset/test/Fire"
    dummy_test_not_fire_dir = "./dummy_dataset/test/Not_Fire"

    os.makedirs(dummy_train_fire_dir, exist_ok=True)
    os.makedirs(dummy_train_not_fire_dir, exist_ok=True)
    os.makedirs(dummy_test_fire_dir, exist_ok=True)
    os.makedirs(dummy_test_not_fire_dir, exist_ok=True)

    # Copy the uploaded image to simulate a dataset
    uploaded_image_path = "/home/ubuntu/upload/resized_test_fire_frame0.jpg"
    import shutil
    shutil.copy(uploaded_image_path, os.path.join(dummy_train_fire_dir, "fire_001.jpg"))
    shutil.copy(uploaded_image_path, os.path.join(dummy_train_fire_dir, "fire_002.jpg"))
    shutil.copy(uploaded_image_path, os.path.join(dummy_train_not_fire_dir, "not_fire_001.jpg"))
    shutil.copy(uploaded_image_path, os.path.join(dummy_train_not_fire_dir, "not_fire_002.jpg"))
    shutil.copy(uploaded_image_path, os.path.join(dummy_test_fire_dir, "fire_test_001.jpg"))
    shutil.copy(uploaded_image_path, os.path.join(dummy_test_not_fire_dir, "not_fire_test_001.jpg"))

    train_data = load_dataset("./dummy_dataset/train")
    test_data = load_dataset("./dummy_dataset/test")

    print(f"Loaded {len(train_data)} training images and {len(test_data)} testing images.")

    encoding_strategies = {
        "pixel_based": pixel_based_encoding,
        "patch_based": patch_based_encoding,
    }

    results = {}

    for strategy_name, encoding_func in encoding_strategies.items():
        print(f"\n--- Evaluating {strategy_name} encoding ---")
        encoded_train_data = []
        total_encoding_time = 0
        total_encoding_memory_peak = 0

        # Encoding training data
        for img_path, label in train_data:
            (hv_result, encoding_time), current_mem, peak_mem = measure_memory_usage(measure_execution_time, encoding_func, img_path)
            hv = hv_result
            encoded_train_data.append((hv, label))
            total_encoding_time += encoding_time
            total_encoding_memory_peak = max(total_encoding_memory_peak, peak_mem) # Max peak memory

        print(f"Training Encoding Time: {total_encoding_time:.4f} seconds")
        print(f"Training Encoding Peak Memory: {total_encoding_memory_peak / (1024 * 1024):.2f}MB")

        # Training the model
        train_start_time = time.perf_counter()
        prototypes = train(encoded_train_data)
        train_end_time = time.perf_counter()
        training_time = train_end_time - train_start_time
        print(f"Training Time: {training_time:.4f} seconds")

        # Encoding and classifying test data
        true_labels = []
        predicted_labels = []
        total_inference_encoding_time = 0
        total_inference_encoding_memory_peak = 0
        total_classification_time = 0

        for img_path, true_label in test_data:
            true_labels.append(true_label)

            # Encoding
            (test_hv_result, encoding_time), current_mem, peak_mem = measure_memory_usage(measure_execution_time, encoding_func, img_path)
            test_hv = test_hv_result
            total_inference_encoding_time += encoding_time
            total_inference_encoding_memory_peak = max(total_inference_encoding_memory_peak, peak_mem)

            # Classification
            classification_start_time = time.perf_counter()
            predicted_label = classify(test_hv, prototypes)
            classification_end_time = time.perf_counter()
            total_classification_time += (classification_end_time - classification_start_time)

            predicted_labels.append(predicted_label)

        accuracy = accuracy_score(true_labels, predicted_labels)
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Inference Encoding Time (per image avg): {total_inference_encoding_time / len(test_data):.4f} seconds")
        print(f"Inference Encoding Peak Memory (per image avg): {total_inference_encoding_memory_peak / (1024 * 1024):.2f}MB")
        print(f"Classification Time (per image avg): {total_classification_time / len(test_data):.4f} seconds")

        results[strategy_name] = {
            "accuracy": accuracy,
            "training_encoding_time": total_encoding_time,
            "training_encoding_peak_memory": total_encoding_memory_peak,
            "training_time": training_time,
            "inference_encoding_time_avg": total_inference_encoding_time / len(test_data),
            "inference_encoding_memory_peak_avg": total_inference_encoding_memory_peak / (1024 * 1024),
            "classification_time_avg": total_classification_time / len(test_data),
        }

    print("\n--- Summary of Results ---")
    for strategy_name, data in results.items():
        print(f"\nStrategy: {strategy_name}")
        for key, value in data.items():
            if "time" in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.4f} seconds")
            elif "memory" in key:
                print(f"  {key.replace('_', ' ').title()}: {value:.2f} MB")
            else:
                print(f"  {key.replace('_', ' ').title()}: {value:.4f}")

    # Clean up dummy dataset
    shutil.rmtree("./dummy_dataset")
    print("\nDummy dataset cleaned up.")

    print("\nEvaluation complete. Please note that for a meaningful comparison, you would need to run this script with your actual dataset and compare these metrics with your existing deep learning example.")
