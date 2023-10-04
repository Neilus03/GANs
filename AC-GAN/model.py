import torch
import torch.nn as nn

# Generator
class Generator(nn.Module):
    def __init__(self, noise_dim, class_dim, output_shape):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + class_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, output_shape),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], -1)
        return self.model(x)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_shape, class_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, class_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        validity = self.discriminator(x)
        label = self.classifier(x)
        return validity, label
