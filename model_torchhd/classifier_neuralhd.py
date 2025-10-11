import torch
import torchhd

def load_neuralhd_model(model_path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Recreate the model
    model = torchhd.classifiers.NeuralHD(
        checkpoint['num_features'], 
        checkpoint['dimensions'], 
        checkpoint['num_classes'], 
        device=device
    )
    
    # Load the saved state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded: {checkpoint['model_name']}")
    print(f"Accuracy: {checkpoint['accuracy']:.2f}%")
    print(f"Run: {checkpoint['run_number']}")
    
    return model, checkpoint

# Exemplo de uso:
# model, info = load_neuralhd_model("saved_models/BEST_NeuralHD_acc_95.50.pth")