''' From paper: 
        Classification and Recall With Binary
        Hyperdimensional Computing: Tradeoffs in
        Choice of Density and Mapping Characteristics
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import time
from PIL import Image
import os
from tqdm import tqdm

class HyperdimensionalEncoder:
    def __init__(self, D=10000, spatial_encoding=True, quantization_levels=8):
        """
        Initialize the Hyperdimensional Computing encoder
        
        Parameters:
        D (int): Dimension of hypervectors
        spatial_encoding (bool): Whether to use spatial encoding
        quantization_levels (int): Number of levels for value quantization
        """
        self.D = D
        self.spatial_encoding = spatial_encoding
        self.quantization_levels = quantization_levels
        
        # Initialize base hypervectors
        self.value_vectors = {}  # Will store value hypervectors
        self.position_vectors = {}  # Will store position hypervectors
        self.class_prototypes = {}  # Will store class prototype hypervectors
        
        # Initialize permutation vector for spatial encoding
        self.permutation_vector = None
        if self.spatial_encoding:
            self.permutation_vector = np.random.choice([-1, 1], size=D)
        
    def generate_base_vector(self):
        """Generate a random bipolar hypervector"""
        return np.random.choice([-1, 1], size=self.D)
    
    def quantize_value(self, value):
        """Quantize a continuous value to discrete levels"""
        return int(value * (self.quantization_levels - 1))
    
    def get_value_vector(self, value, channel=None):
        """
        Get or create a hypervector for a quantized value
        
        Parameters:
        value: The quantized value (0 to quantization_levels-1)
        channel: Color channel ('R', 'G', 'B') for value-specific encoding
        """
        key = f"{channel}_{value}" if channel else str(value)
        
        if key not in self.value_vectors:
            # Create base channel vector if it doesn't exist
            if channel and channel not in self.value_vectors:
                base_vec = self.generate_base_vector()
                # Apply slight variations based on channel for 'G' and 'B'
                if channel == 'G':
                    base_vec = self.permute_vector(base_vec, 1)
                elif channel == 'B':
                    base_vec = self.permute_vector(base_vec, 2)
                self.value_vectors[channel] = base_vec
            
            # Create value-specific vector using fractional power encoding
            if channel:
                base_channel_vec = self.value_vectors[channel]
                self.value_vectors[key] = self.permute_vector(base_channel_vec, value)
            else:
                self.value_vectors[key] = self.generate_base_vector()
                
        return self.value_vectors[key]
    
    def permute_vector(self, vector, n):
        """Permute a vector n times using circular shift"""
        return np.roll(vector, n)
    
    def get_position_vector(self, x, y, max_x, max_y):
        """Get or create a hypervector for a position (x, y)"""
        key = f"{x}_{y}"
        
        if key not in self.position_vectors:
            # Create base vectors for x and y axes if they don't exist
            if "x_base" not in self.position_vectors:
                self.position_vectors["x_base"] = self.generate_base_vector()
            if "y_base" not in self.position_vectors:
                self.position_vectors["y_base"] = self.generate_base_vector()
            
            # Create position vectors using fractional power encoding
            x_vec = self.permute_vector(self.position_vectors["x_base"], x)
            y_vec = self.permute_vector(self.position_vectors["y_base"], y)
            
            # Bind x and y vectors to represent the position
            self.position_vectors[key] = x_vec * y_vec
            
        return self.position_vectors[key]
    
    def encode_pixel(self, r, g, b, x, y, img_width, img_height):
        """Encode a single pixel into a hypervector"""
        # Quantize values
        q_r = self.quantize_value(r)
        q_g = self.quantize_value(g)
        q_b = self.quantize_value(b)
        
        # Get value vectors
        r_vec = self.get_value_vector(q_r, 'R')
        g_vec = self.get_value_vector(q_g, 'G')
        b_vec = self.get_value_vector(q_b, 'B')
        
        # Bind color channels
        color_vec = r_vec * g_vec * b_vec
        
        if self.spatial_encoding:
            # Get position vector and bind with color
            pos_vec = self.get_position_vector(x, y, img_width, img_height)
            pixel_vec = color_vec * pos_vec
        else:
            pixel_vec = color_vec
            
        return pixel_vec
    
    def encode_image(self, image):
        """Encode an entire image into a hypervector"""
        height, width = image.shape[:2]
        image_hv = np.zeros(self.D)
        
        # Sum all pixel hypervectors
        for y in range(height):
            for x in range(width):
                if len(image.shape) == 3:  # Color image
                    r, g, b = image[y, x] / 255.0
                else:  # Grayscale
                    r = g = b = image[y, x] / 255.0
                
                pixel_hv = self.encode_pixel(r, g, b, x, y, width, height)
                image_hv += pixel_hv
        
        # Binarize the resulting vector
        image_hv = np.sign(image_hv)
        return image_hv
    
    def train(self, images, labels):
        """Train the model by creating class prototypes"""
        class_vectors = {}
        class_counts = {}
        
        print("Training HDC model...")
        start_time = time.time()
        
        for image, label in tqdm(zip(images, labels), total=len(images), desc="Training images"):
            # Encode the image
            image_hv = self.encode_image(image)
            
            # Add to class bundle
            if label not in class_vectors:
                class_vectors[label] = np.zeros(self.D)
                class_counts[label] = 0
                
            class_vectors[label] += image_hv
            class_counts[label] += 1
        
        # Create prototypes by bundling and binarizing
        for label in class_vectors:
            self.class_prototypes[label] = np.sign(class_vectors[label])
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        return self.class_prototypes
    
    def predict(self, image):
        """Predict the class of a single image"""
        image_hv = self.encode_image(image)
        
        # Find the class with the smallest Hamming distance
        best_class = None
        min_distance = float('inf')
        
        for label, prototype in self.class_prototypes.items():
            # Hamming distance for bipolar vectors: (D - dot_product) / 2
            distance = (self.D - np.dot(image_hv, prototype)) / 2
            
            if distance < min_distance:
                min_distance = distance
                best_class = label
                
        return best_class
    
    def evaluate(self, test_images, test_labels):
        """Evaluate the model on a test set"""
        predictions = []
        
        print("Evaluating model...")
        start_time = time.time()
        
        for image in tqdm(test_images, desc="Evaluating images"):
            pred = self.predict(image)
            predictions.append(pred)
        
        eval_time = time.time() - start_time
        accuracy = accuracy_score(test_labels, predictions)
        
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Plot confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return accuracy, predictions

# Example usage with synthetic data
def create_synthetic_dataset(num_classes=5, images_per_class=20, img_size=(32, 32)):
    """Create a synthetic dataset for demonstration"""
    images = []
    labels = []
    
    for class_id in range(num_classes):
        for i in range(images_per_class):
            # Create a simple synthetic image with class-specific patterns
            if class_id == 0:  # Vertical lines
                img = np.zeros(img_size)
                for x in range(0, img_size[1], 4):
                    img[:, x] = 1.0
            elif class_id == 1:  # Horizontal lines
                img = np.zeros(img_size)
                for y in range(0, img_size[0], 4):
                    img[y, :] = 1.0
            elif class_id == 2:  # Checkerboard
                img = np.zeros(img_size)
                for y in range(img_size[0]):
                    for x in range(img_size[1]):
                        if (x // 4 + y // 4) % 2 == 0:
                            img[y, x] = 1.0
            elif class_id == 3:  # Diagonal
                img = np.zeros(img_size)
                for y in range(img_size[0]):
                    for x in range(img_size[1]):
                        if abs(x - y) < 3:
                            img[y, x] = 1.0
            else:  # Random dots
                img = np.random.rand(*img_size) > 0.8
            
            # Add some noise
            img = img + np.random.normal(0, 0.1, img_size)
            img = np.clip(img, 0, 1)
            
            images.append(img)
            labels.append(class_id)
    
    return np.array(images), np.array(labels)

def load_images_from_directory(directory, label, img_size=(100, 100), max_images=None):
    """Load images from directory and assign label"""
    images = []
    labels = []
    
    # Get list of image files
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    for filename in tqdm(image_files, desc=f"Loading {os.path.basename(directory)}"):
        img_path = os.path.join(directory, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  # Normalize to [0, 1]
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    return np.array(images), np.array(labels)

def load_fire_dataset(img_size=(100, 100), max_images_per_class=None):
    """Load fire detection dataset"""
    print("Loading training data...")
    
    # Load fire images (label 1)
    fire_train_images, fire_train_labels = load_images_from_directory(
        "Training/Fire", 1, img_size, max_images_per_class
    )
    
    # Load no-fire images (label 0)
    nofire_train_images, nofire_train_labels = load_images_from_directory(
        "Training/No_Fire", 0, img_size, max_images_per_class
    )
    
    # Load test data
    print("Loading test data...")
    fire_test_images, fire_test_labels = load_images_from_directory(
        "Test/Fire", 1, img_size, max_images_per_class
    )
    
    nofire_test_images, nofire_test_labels = load_images_from_directory(
        "Test/No_Fire", 0, img_size, max_images_per_class
    )
    
    # Combine training data
    train_images = np.concatenate([fire_train_images, nofire_train_images])
    train_labels = np.concatenate([fire_train_labels, nofire_train_labels])
    
    # Combine test data
    test_images = np.concatenate([fire_test_images, nofire_test_images])
    test_labels = np.concatenate([fire_test_labels, nofire_test_labels])
    
    # Shuffle the data
    train_indices = np.random.permutation(len(train_images))
    test_indices = np.random.permutation(len(test_images))
    
    train_images = train_images[train_indices]
    train_labels = train_labels[train_indices]
    test_images = test_images[test_indices]
    test_labels = test_labels[test_indices]
    
    print(f"Dataset loaded:")
    print(f"- Training: {len(train_images)} images ({np.sum(train_labels == 1)} fire, {np.sum(train_labels == 0)} no-fire)")
    print(f"- Test: {len(test_images)} images ({np.sum(test_labels == 1)} fire, {np.sum(test_labels == 0)} no-fire)")
    
    return train_images, train_labels, test_images, test_labels

# Main demonstration with real fire dataset
if __name__ == "__main__":
    # Load real fire detection dataset
    train_images, train_labels, test_images, test_labels = load_fire_dataset(
        img_size=(100, 100),  # Resize to 100x100 for consistency
        max_images_per_class=None 
    )
    
    # Initialize and train the HDC encoder
    encoder = HyperdimensionalEncoder(D=10000)  # Using 10000 dimensions for better accuracy
    
    # Train the model
    prototypes = encoder.train(train_images, train_labels)
    
    # Evaluate the model
    accuracy, predictions = encoder.evaluate(test_images, test_labels)
    
    # Display some test examples with predictions
    plt.figure(figsize=(12, 8))
    for i in range(min(12, len(test_images))):
        plt.subplot(3, 4, i+1)
        plt.imshow(test_images[i])
        plt.title(f'True: {test_labels[i]}, Pred: {predictions[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
