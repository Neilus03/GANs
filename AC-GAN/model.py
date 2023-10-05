import torch
import torch.nn as nn

import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        
        self.softmax = nn.Softmax(dim=-2)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        
        attention = self.softmax(torch.bmm(query, key))
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, width, height)
        
        return out

class Generator(nn.Module):
    def __init__(self, noise_dim, class_dim):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + class_dim, 512, kernel_size=(70, 80)), 
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            SelfAttention(512),
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Final layer to generate image
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        x = torch.cat([noise, labels], 1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, class_dim):
        super(Discriminator, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            SelfAttention(128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 35 * 40, 512),  # Adapt these dimensions
            nn.LeakyReLU(0.2),
            nn.Linear(512, class_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        validity = self.discriminator(x)
        validity = validity.view(validity.size(0), -1)
        label = self.classifier(validity)
        return validity, label
