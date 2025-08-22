import torch
import torch.nn as nn
import torch.nn.functional as F

class FireDetectionModel(nn.Module):
    """
    PyTorch implementation of the fire detection model, translated from the original Keras model.
    """
    def __init__(self, num_classes=2):
        super(FireDetectionModel, self).__init__()

        # Feature extractor part, equivalent to the Keras model's convolutional base
        self.features = nn.Sequential(
            # Keras: layers.Conv2D(8, 3, strides=2, padding="same")
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            # Keras: layers.BatchNormalization()
            nn.BatchNorm2d(8),
            # Keras: layers.Activation("relu")
            nn.ReLU(),
        )

        # Residual block equivalent
        # In Keras: for size in [8]...
        size = 8
        self.residual_block = nn.Sequential(
            # Block structure
            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=3, padding=1, groups=size), # SeparableConv2D depthwise
            nn.Conv2d(size, size, kernel_size=1), # SeparableConv2D pointwise
            nn.BatchNorm2d(size),

            nn.ReLU(),
            nn.Conv2d(size, size, kernel_size=3, padding=1, groups=size), # SeparableConv2D depthwise
            nn.Conv2d(size, size, kernel_size=1), # SeparableConv2D pointwise
            nn.BatchNorm2d(size),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Keras: residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
        self.residual_projection = nn.Conv2d(in_channels=8, out_channels=size, kernel_size=1, stride=2)


        # Final convolutional layers before the classifier
        self.final_conv = nn.Sequential(
            nn.Conv2d(size, size, kernel_size=3, padding=1, groups=size), # SeparableConv2D depthwise
            nn.Conv2d(size, size, kernel_size=1), # SeparableConv2D pointwise
            nn.BatchNorm2d(size),
            nn.ReLU(),
        )

        # Classifier part, equivalent to the Keras model's head
        self.classifier = nn.Sequential(
            # Keras: layers.GlobalAveragePooling2D()
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # Keras: layers.Dropout(0.5)
            nn.Dropout(0.5),
            # Keras: layers.Dense(units, activation=activation)
            # For num_classes=2 (binary), Keras uses 1 unit with sigmoid.
            # PyTorch's CrossEntropyLoss for multiclass (even binary) expects num_classes outputs.
            # We'll output 1 unit and use BCEWithLogitsLoss in the training script.
            nn.Linear(in_features=8, out_features=1 if num_classes == 2 else num_classes)
        )


    def forward(self, x):
        # Pass through the initial feature extractor
        previous_block_activation = self.features(x)

        # Pass through the main block
        x = self.residual_block(previous_block_activation)

        # Get the projected residual
        residual = self.residual_projection(previous_block_activation)

        # Add the residual connection
        x = x + residual

        # Pass through the final conv layers
        x = self.final_conv(x)

        # Pass through the classifier
        x = self.classifier(x)

        return x

if __name__ == '__main__':
    # Verify the model architecture and output shape
    model = FireDetectionModel()
    print(model)

    # Create a dummy input tensor to test the forward pass
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(32, 3, 256, 256)
    output = model(dummy_input)

    print(f"\nDummy input shape: {dummy_input.shape}")
    print(f"Model output shape: {output.shape}")
    # For binary classification, we expect (batch_size, 1)
    assert output.shape == (32, 1)
    print("Model architecture and forward pass verified successfully.") 