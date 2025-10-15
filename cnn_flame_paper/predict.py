import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse

# Import our custom modules
from cnn_flame_paper.model_cnn import FireDetectionModel
from dataset import IMG_HEIGHT, IMG_WIDTH # Use same image size as training

# --- Configuration ---
MODEL_PATH = 'checkpoints/fire_detection_best.pth'
CLASS_NAMES = ['Fire', 'No_Fire'] # Make sure this order matches ImageFolder

def predict(model, image_path, device):
    """
    Makes a prediction on a single image.
    
    Args:
        model (nn.Module): The trained PyTorch model.
        image_path (str): The path to the input image.
        device (torch.device): The device to run inference on.

    Returns:
        tuple: A tuple containing (predicted_class, confidence).
    """
    model.eval() # Set model to evaluation mode

    # Define the same transformations as used in training, but for a single image
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load and transform the image
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None

    # Add a batch dimension and send to device
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        # Apply sigmoid to get probability
        probability = torch.sigmoid(output).item()

    # Determine class based on probability > 0.5
    # IMPORTANT: Assumes 'Fire' is class 0 and 'No_Fire' is class 1.
    # If ImageFolder assigned them differently, this logic must change.
    # Check the output from dataset.py to be sure.
    if probability < 0.5:
        predicted_class = CLASS_NAMES[0] # Fire
        confidence = 1 - probability
    else:
        predicted_class = CLASS_NAMES[1] # No_Fire
        confidence = probability

    return predicted_class, confidence

def main():
    """
    Main function to run prediction from command line.
    """
    parser = argparse.ArgumentParser(description="Fire Detection Prediction")
    parser.add_argument('image_path', type=str, help='Path to the image for prediction.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the trained model checkpoint.')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model structure
    model = FireDetectionModel().to(device)
    
    # Load the trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_path}")
        print("Please run train.py first to generate the model checkpoint.")
        return

    # Make prediction
    predicted_class, confidence = predict(model, args.image_path, device)
    
    if predicted_class:
        print(f"\nPrediction for '{args.image_path}':")
        print(f"-> Class: {predicted_class}")
        print(f"-> Confidence: {confidence:.2%}")

if __name__ == '__main__':
    main() 