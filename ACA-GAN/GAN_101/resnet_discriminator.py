import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights

class ResNetDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Load a ResNet-18 model with the updated weights parameter
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Adjust the first convolutional layer to accept 3 channels instead of 64
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

       
        num_features = self.backbone.inplanes
        self.backbone.fc = nn.Identity()

        # Classifier for real/fake
        self.validity_classifier = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )

        # Classifier for class labels
        self.class_classifier = nn.Sequential(
            nn.Linear(num_features, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Extract features with the backbone
        features = self.backbone(x)

        # Classify for real/fake and class labels
        validity = self.validity_classifier(features).view(-1, 1)
        classes = self.class_classifier(features)

        return validity, classes

if __name__ == '__main__':
    # Example initialization
    discriminator = ResNetDiscriminator(num_classes=3)
    # Create a dummy input tensor of the shape [batch_size, channels, height, width]
    dummy_input = torch.randn(4, 3, 560, 640)
    # Forward pass of the dummy input through the discriminator
    # This will also print the shapes of features, validity, and classes
    validity, classes = discriminator(dummy_input)
    print(f"Validity shape: {validity.shape}")
    print(f"Classes shape: {classes.shape}")
    print(validity, classes)
