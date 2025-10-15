import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

# Import Captum and its visualization tools
from captum.attr import IntegratedGradients, visualization as viz

# Import our custom modules
from cnn_flame_paper.model_cnn import FireDetectionModel
from dataset import IMG_HEIGHT, IMG_WIDTH # Use same image size as training

# --- Configuration ---
MODEL_PATH = 'checkpoints/fire_detection_best.pth'
CLASS_NAMES = ['Fire', 'No_Fire'] # Ensure this order matches ImageFolder

def get_prediction(model, image_tensor):
    """Helper function to get model prediction."""
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
    if probability < 0.5:
        predicted_class_index = 0 # Fire
        confidence = 1 - probability
    else:
        predicted_class_index = 1 # No_Fire
        confidence = probability
    return predicted_class_index, confidence


def explain_prediction(model, image_tensor, device, target_class_index):
    """
    Uses Captum to generate attributions for a prediction.
    
    Args:
        model (nn.Module): The trained PyTorch model.
        image_tensor (torch.Tensor): The preprocessed input image tensor.
        device (torch.device): The device to run inference on.
        target_class_index (int): The index of the class to explain (0 for Fire, 1 for No_Fire).

    Returns:
        np.ndarray: The attribution map.
    """
    model.eval()
    
    # We need a baseline to compare against. A black image is a common choice.
    baseline = torch.zeros_like(image_tensor).to(device)

    # Initialize the Integrated Gradients algorithm
    ig = IntegratedGradients(model)

    # Calculate attributions. The target is the class index we want to explain.
    # For a binary model with 1 output, the target is always 0. The output value itself
    # determines the class, but we are explaining the output neuron itself.
    attributions, delta = ig.attribute(image_tensor, baseline, target=0, return_convergence_delta=True)
    
    print(f'Convergence Delta: {delta.item():.4f}') # Should be close to 0

    # Convert the attributions tensor to a numpy array for visualization
    attribution_map = attributions.squeeze(0).cpu().detach().numpy()
    
    # The output is (channels, height, width). We sum across channels for a single heatmap.
    return np.transpose(attribution_map, (1, 2, 0))


def main():
    """
    Main function to run explainability from command line.
    """
    parser = argparse.ArgumentParser(description="Fire Detection Explainability with Captum")
    parser.add_argument('image_path', type=str, help='Path to the image to explain.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the trained model checkpoint.')
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model and load weights
    model = FireDetectionModel().to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_path}")
        return

    # Load and transform the image
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    original_img = Image.open(args.image_path).convert('RGB')
    # For visualization, we use a version of the image before normalization
    vis_transform = transforms.Compose([transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor()])
    original_img_tensor = vis_transform(original_img)
    original_img_np = np.transpose(original_img_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0))
    
    image_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # Get prediction
    predicted_class_index, confidence = get_prediction(model, image_tensor)
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    print(f"Prediction: {predicted_class_name} with {confidence:.2%} confidence.")

    # Get attributions
    attribution_map = explain_prediction(model, image_tensor, device, predicted_class_index)

    # Visualize the results
    _ = viz.visualize_image_attr(
        attribution_map,
        original_img_np,
        method="blended_heat_map",
        sign="all", # You can use "positive" to see only what supports the decision
        show_colorbar=True,
        title=f"Attributions for '{predicted_class_name}'"
    )

    # Save the figure
    output_filename = f"explanation_{predicted_class_name}_{os.path.basename(args.image_path)}.png"
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Explicação salva em: {output_filename}")

if __name__ == '__main__':
    main() 