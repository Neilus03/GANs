import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_module101 import SelfAttention

class Generator(nn.Module):
    def __init__(self, noise_dim, class_dim):
        super(Generator, self).__init__()
        
        # Fully connected layer
        self.fc = nn.Linear(noise_dim + class_dim, 64 * 35 * 40)  # Output: [B, 64 * 35 * 40]
        print(f"shape of the fc layer: {self.fc}")
        # Generator's architecture
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # [B, 32, 70, 80]
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 70, 80]
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 70, 80]
            #nn.BatchNorm2d(32),  # [B, 32, 70, 80]
            nn.ReLU(True),  # [B, 32, 70, 80]
            
            #SelfAttention(32),  #  [B, 32, 70, 80] , let it commented for easier computation while trying to know if matches sizes
            
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  #[B, 16, 140, 160] 
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  #[B, 16, 140, 160]
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),  #[B, 16, 140, 160]
            #nn.BatchNorm2d(16),  #[B, 16, 140, 160]
            nn.ReLU(True),  # [B, 16, 140, 160]

            SelfAttention(16),  #  [B, 16, 140, 160] , let it commented for easier computation while trying to know if matches sizes
            
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # [B, 8, 280, 320] 
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # [B, 8, 280, 320]
            nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1),  # [B, 8, 280, 320]
            #nn.BatchNorm2d(64), # [B, 8, 280, 320]
            nn.ReLU(True),  # [B, 8, 280, 320]

            #SelfAttention(8),  # [B, 8, 280, 320] , let it commented for easier computation while trying to know if matches sizes

            #Last layer 
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),  # [B, 3, 560, 640]
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),  # [B, 3, 560, 640]
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),  # [B, 3, 560, 640]
            nn.Tanh()  # Output: [B, 3, 560, 640] in range [-1,1]
        )

    def forward(self, noise, labels):
        # Flatten the noise tensor
        noise = noise.view(noise.size(0), -1)  # [B, noise_dim]
        
        # Concatenate noise and label tensors
        x = torch.cat([noise, labels], dim=1)  # [B, noise_dim + class_dim]
        
        # Pass through the fully connected layer and reshape
        x = self.fc(x)  #  [N, 64 * 35 * 40]
        #print(f"After fc layer, shape of x: {x.shape}")
        
        x = x.view(x.size(0), 64, 35, 40)  # [B, 64, 35, 40]
        #print(f"After view reshape, shape of x: {x.shape}")
        
        # Generate the image
        generated_img = self.generator(x)  # [B, 3, 560, 640]
        #print(f"shape of the generated img: {generated_img.shape}")
        return generated_img

#TESTING AND DEBUGGING


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import matplotlib.pyplot as plt
    def plot_images(images, num_images=4):
        fig = plt.figure(figsize=(14, 14))
        for i in range(num_images):
            ax = fig.add_subplot(1, num_images, i+1)
            ax.axis("off")
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = (img + 1) / 2  # Rescale pixel values from [-1, 1] to [0, 1]
            plt.imshow(img)
        plt.show()
    
    noise_dim = 100
    generator = Generator(noise_dim=noise_dim, class_dim=3).to(device)
    noise = torch.randn(4, noise_dim).to(device)
    labels = torch.randint(0, 3, (4,)).to(device)
    labels_onehot = torch.nn.functional.one_hot(labels, 3).float().to(device)
    generated_images = generator(noise, labels_onehot).detach().cpu()
    print(f"Generated images shape: {generated_images.shape}")  # Should be [8, 3, 560, 640]
    plot_images(generated_images)
