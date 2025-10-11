import torch
import torch.nn as nn
import torchvision.models as models

class VGG16FireDetection(nn.Module):
    """
    VGG16-based fire detection model using transfer learning.
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(VGG16FireDetection, self).__init__()
        
        # Load pre-trained VGG16
        self.vgg16 = models.vgg16(pretrained=pretrained)
        
        # Freeze the feature extraction layers (optional, for fine-tuning)
        # Uncomment the next line if you want to freeze the backbone
        # for param in self.vgg16.features.parameters():
        #     param.requires_grad = False
        
        # Get the number of features from VGG16's classifier
        num_features = self.vgg16.classifier[0].in_features
        
        # Replace the classifier with our custom one for binary classification
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)  # 1 output for binary classification
        )
        
    def forward(self, x):
        return self.vgg16(x)

def get_vgg16_model(num_classes=2, pretrained=True):
    """
    Factory function to create a VGG16-based fire detection model.
    
    Args:
        num_classes (int): Number of classes (2 for binary classification)
        pretrained (bool): Whether to use pre-trained weights
        
    Returns:
        VGG16FireDetection: The configured model
    """
    model = VGG16FireDetection(num_classes=num_classes, pretrained=pretrained)
    return model

if __name__ == '__main__':
    # Test the model
    model = get_vgg16_model()
    print(model)
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)  # VGG16 expects 224x224
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output value: {output.item()}")
    
    # Test prediction
    probability = torch.sigmoid(output).item()
    predicted_class = "Fire" if probability < 0.5 else "No_Fire"
    confidence = 1 - probability if probability < 0.5 else probability
    
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.2%}")
