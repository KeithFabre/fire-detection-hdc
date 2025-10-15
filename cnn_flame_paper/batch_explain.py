import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
from captum.attr import IntegratedGradients, visualization as viz
from cnn_flame_paper.model_cnn import FireDetectionModel
from dataset import IMG_HEIGHT, IMG_WIDTH

MODEL_PATH = 'checkpoints/fire_detection_best.pth'
CLASS_NAMES = ['Fire', 'No_Fire']


def get_prediction(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probability = torch.sigmoid(output).item()
    if probability < 0.5:
        predicted_class_index = 0  # Fire
        confidence = 1 - probability
    else:
        predicted_class_index = 1  # No_Fire
        confidence = probability
    return predicted_class_index, confidence


def explain_prediction(model, image_tensor, device, target_class_index):
    model.eval()
    baseline = torch.zeros_like(image_tensor).to(device)
    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(image_tensor, baseline, target=0, return_convergence_delta=True)
    attribution_map = attributions.squeeze(0).cpu().detach().numpy()
    return np.transpose(attribution_map, (1, 2, 0))


def batch_explain(input_folder, output_folder, model_path=MODEL_PATH, num_images=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare output directory
    os.makedirs(output_folder, exist_ok=True)

    # Load model
    model = FireDetectionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    vis_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor()
    ])

    # Find all images (jpg, png, jpeg)
    image_paths = glob(os.path.join(input_folder, '*.jpg')) + \
                  glob(os.path.join(input_folder, '*.jpeg')) + \
                  glob(os.path.join(input_folder, '*.png'))
    image_paths = sorted(image_paths)
    if num_images is not None:
        image_paths = image_paths[:num_images]
    print(f"Found {len(image_paths)} images to process in {input_folder}")

    for img_path in image_paths:
        print(f"\nProcessing: {img_path}")
        original_img = Image.open(img_path).convert('RGB')
        original_img_np = np.array(original_img.resize((IMG_WIDTH, IMG_HEIGHT)))
        image_tensor = transform(original_img).unsqueeze(0).to(device)

        predicted_class_index, confidence = get_prediction(model, image_tensor)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        print(f"Prediction: {predicted_class_name} ({confidence:.2%})")

        attribution_map = explain_prediction(model, image_tensor, device, predicted_class_index)

        # Create a figure with two subplots for side-by-side comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original image on the left
        axes[0].imshow(original_img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Use Captum's visualization tool on the right subplot
        viz.visualize_image_attr(
            attribution_map,
            original_img_np,
            method="blended_heat_map",
            sign="all",
            plt_fig_axis=(fig, axes[1]), # Pass the specific axis
            show_colorbar=True,
            title=f"Attribution for '{predicted_class_name}'"
        )
        
        fig.suptitle(f"Explanation for: {os.path.basename(img_path)}", fontsize=16)

        # Save the figure
        output_filename = os.path.join(
            output_folder,
            f"side_by_side_{predicted_class_name}_{os.path.basename(img_path)}.png"
        )
        plt.savefig(output_filename, bbox_inches='tight')
        plt.close(fig) # Close the figure to free memory
        print(f"Saved: {output_filename}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Batch explainability for fire detection.")
    parser.add_argument('input_folder', type=str, help='Folder with images to explain.')
    parser.add_argument('--output_folder', type=str, default='batch_explanations', help='Folder to save explanations.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to the trained model checkpoint.')
    parser.add_argument('--num_images', type=int, default=None, help='Number of images to process (default: all).')
    args = parser.parse_args()

    batch_explain(args.input_folder, args.output_folder, args.model_path, args.num_images) 