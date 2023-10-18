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
        
        # Fully connected layer
        self.fc = nn.Linear(noise_dim + class_dim, 512 * 35 * 40)  # Output: [B, 512 * 35 * 40]
        
        # Generator's architecture
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 70, 80]
            nn.BatchNorm2d(256),  # [B, 256, 70, 80]
            nn.ReLU(True),  # [B, 256, 70, 80]
            
            SelfAttention(256),  #  [B, 256, 70, 80] , let it commented for easier computation while trying to know if matches sizes
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  #[B, 128, 140, 160] 
            nn.BatchNorm2d(128),  #[B, 128, 140, 160]
            nn.ReLU(True),  # [B, 128, 140, 160]

            SelfAttention(128),  #  [B, 128, 140, 160] , let it commented for easier computation while trying to know if matches sizes
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [B, 64, 280, 320] 
            nn.BatchNorm2d(64), # [B, 64, 280, 320]
            nn.ReLU(True),  # [B, 64, 280, 320]

            SelfAttention(64),  # [B, 64, 280, 320] , let it commented for easier computation while trying to know if matches sizes
            
            #Last layer 
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 560, 640]
            nn.Tanh()  # Output: [B, 3, 560, 640] in range [-1,1]
        )

    def forward(self, noise, labels):
        # Flatten the noise tensor
        noise = noise.view(noise.size(0), -1)  # [B, noise_dim]
        
        # Concatenate noise and label tensors
        x = torch.cat([noise, labels], dim=1)  # [B, noise_dim + class_dim]
        
        # Pass through the fully connected layer and reshape
        x = self.fc(x)  #  [N, 512 * 35 * 40]
        #print(f"After fc layer, shape of x: {x.shape}")
        
        x = x.view(x.size(0), 512, 35, 40)  # [B, 512, 35, 40]
        #print(f"After view reshape, shape of x: {x.shape}")
        
        # Generate the image
        generated_img = self.generator(x)  # [B, 3, 560, 640]
        #print(f"shape of the generated img: {generated_img.shape}")
        return generated_img

class Discriminator(nn.Module):
    def __init__(self, class_dim):
        super(Discriminator, self).__init__()

        # We´ll apply GAP at the end so it must be instantiated.
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        #We´ll apply the sigmoid to validity as in training
        #it will use BCE loss which doesn't include sigmoid, unlike
        #general cross-entropy loss
        self.sigmoid = nn.Sigmoid()
        
        # Discriminator's architecture
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1), # [B, 64, 280, 320]  
            nn.BatchNorm2d(64),  # [B, 64, 280, 320]   #comment or uncomment batch norm of discriminator based on experiments
            nn.LeakyReLU(0.2), # [B, 64, 280, 320] 
            
            SelfAttention(64),  # [B, 64, 280, 320]   # let it commented for easier computation while trying to know if matches sizes
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), #[B, 128, 140, 160] 
            nn.BatchNorm2d(128), #[B, 128, 140, 160]  #comment or uncomment batch norm of discriminator based on experiments
            nn.LeakyReLU(0.2), #[B, 128, 140, 160] 
            
            SelfAttention(128), #[B, 128, 140, 160] # let it commented for easier computation while trying to know if matches sizes
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [B, 256, 70, 80]
            nn.BatchNorm2d(256), # [B, 256, 70, 80]  #comment or uncomment batch norm of discriminator based on experiments
            nn.LeakyReLU(0.2), # [B, 256, 70, 80]

            SelfAttention(256), # [B, 256, 70, 80] # let it commented for easier computation while trying to know if matches sizes
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # [B, 512 * 35 * 40]
            nn.BatchNorm2d(512),  # [B, 512 * 35 * 40]  #comment or uncomment batch norm of discriminator based on experiments
            nn.LeakyReLU(0.2),  # [B, 512 * 35 * 40]

            SelfAttention(512),  # [B, 512 * 35 * 40] # let it commented for easier computation while trying to know if matches sizes
            
            
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0) # [B, 512 * 32 * 37]
        )
        
        # Classifier to determine the class of the image
        self.classifier = nn.Sequential(
            nn.Linear (32*37, 512), #as the output from discriminator before flattening is: ([B, 1, 31, 36])
            nn.LeakyReLU(0.2),
            nn.Linear(512, class_dim) #[B, 3]
        )

    def forward(self, x):
        #print(f"Before discriminator, shape of x: {x.shape}")  # Debugging print
        # Calculate the validity score of the image
        validity = self.discriminator(x) #[B, 1, 32, 37]

        #print(f"After discriminator, shape of validity: {validity.shape}")  # Debugging print
        # Perform global average pooling over the validity feature map
        validity_avg = self.global_avg_pool(validity) #[B, 1, 1, 1]
        #print(f"After global average pooling, shape of validity_avg: {validity_avg.shape}")  # Debugging print        
        # Flatten for the classifier
        validity_flat = validity.view(validity.size(0), -1) #[B, 1, 32*37] == [B, 1184]
        #print(f"After flattening, shape of validity_flat: {validity_flat.shape}")  # Debugging print
        # Classify the image
        label = self.classifier(validity_flat) #[B, 3]
        #print(f"After classifier, shape of label: {label.shape}") 
        
        validity_avg = self.sigmoid(validity_avg)

        return validity_avg, label

#TESTING AND DEBUGGING
import matplotlib.pyplot as plt
def plot_images(images, num_images=8):
    fig = plt.figure(figsize=(14, 14))
    for i in range(num_images):
        ax = fig.add_subplot(1, num_images, i+1)
        ax.axis("off")
        img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
        img = (img + 1) / 2  # Rescale pixel values from [-1, 1] to [0, 1]
        plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    # Define dimensions and hyperparameters
    noise_dim = 100
    class_dim = 3
    batch_size = 8

    # Initialize models
    generator = Generator(noise_dim, class_dim)
    discriminator = Discriminator(class_dim)

    # Create fake data
    fake_noise = torch.randn(batch_size, noise_dim)  # Noise for the generator
    fake_labels = torch.randint(0, class_dim, (batch_size,))  # Random labels
    fake_labels_onehot = torch.nn.functional.one_hot(fake_labels, class_dim).float()  # One-hot encoding

    fake_images = torch.randn(batch_size, 3, 560, 640)  # Fake images to feed the discriminator

    # Forward pass through the generator
    generated_images = generator(fake_noise, fake_labels_onehot)
    print(f"Generated images shape: {generated_images.shape}")  # Should be [batch_size, 3, 560, 640]

    # Plot generated images
    plot_images(generated_images)

    # Forward pass through the discriminator
    validity, label = discriminator(fake_images)
    print(f"Validity shape: {validity.shape}")  # Should be [batch_size, 1, 1, 1]
    print(f"Label shape: {label.shape}")  # Should be [batch_size, class_dim]
