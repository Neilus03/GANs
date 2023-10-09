import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        
        # Define convolution layers for query, key, and value
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        
        # Softmax for the attention mechanism
        self.softmax = nn.Softmax(dim=-2)
        
    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Creating the Query, Key, and Value using the convolution layers
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        
        # Computing the attention weights
        attention = self.softmax(torch.bmm(query, key))
        
        # Output after applying the attention mechanism
        out = torch.bmm(value, attention.permute(0, 2, 1)).view(batch_size, C, width, height)
        
        return out

class Generator(nn.Module):
    def __init__(self, noise_dim, class_dim):
        super(Generator, self).__init__()
        
        # Fully connected layer to upscale the concatenated noise and label vectors
        self.fc = nn.Linear(noise_dim + class_dim, 512 * 70 * 80)
        
        # Generator's architecture
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(70, 80)), 
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # Adding self-attention mechanism
            #SelfAttention(512), let it commented for easier computation while trying to know if matches sizes
            
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # Final layer to generate the image
            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Flatten the noise tensor
        noise = noise.view(noise.size(0), -1)
        
        # Concatenate noise and label tensors
        x = torch.cat([noise, labels], dim=1)
        
        # Pass through the fully connected layer and reshape
        x = self.fc(x)
        x = x.view(x.size(0), 512, 70, 80)
        
        # Generate the image
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, class_dim):
        super(Discriminator, self).__init__()
        
        # Discriminator's architecture
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # Adding self-attention mechanism
            #SelfAttention(128), let it commented for easier computation while trying to know if matches sizes
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0),
        )
        
        # Classifier to determine the class of the image
        self.classifier = nn.Sequential(
            nn.Linear (31*36, 512), #as the output from discriminator before flattening is: ([4, 1, 31, 36])
            nn.LeakyReLU(0.2),
            nn.Linear(512, class_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # Calculate the validity score of the image
        validity = self.discriminator(x)
        print("Shape after Discriminator conv layers:", validity.shape)

        # Flatten for the classifier
        validity = validity.view(validity.size(0), -1)
        
        # Classify the image
        label = self.classifier(validity)
        
        return validity, label
