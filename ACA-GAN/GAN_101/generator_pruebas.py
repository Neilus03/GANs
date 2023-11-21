import torch
from torch import nn
import matplotlib.pyplot as plt
from attention_module101 import SelfAttention

# Define the Unflatten module
class Unflatten(nn.Module):
    def __init__(self, in_features, out_channels, out_height, out_width):
        super().__init__()
        self.out_channels = out_channels
        self.out_height = out_height
        self.out_width = out_width

    def forward(self, input):
        return input.view(-1, self.out_channels, self.out_height, self.out_width)
    
class Generator(nn.Module):
    def __init__(self, z_dim, class_dim, img_channels, img_size_x, img_size_y):
        super().__init__()
        self.img_size_x = img_size_x
        self.img_size_y = img_size_y
        self.img_channels = img_channels
        self.z_dim = z_dim
        self.class_dim = class_dim

        # Calculate the initial size after reshaping
        self.init_size_x = img_size_x // 16  # We divide by 16 instead of 32
        self.init_size_y = img_size_y // 16  # to match the upsampling size
        self.fc_out_features = 512 * self.init_size_x * self.init_size_y

        # Input is Z_dim + class_dim
        self.main = nn.Sequential(
            nn.Linear(self.z_dim + self.class_dim, self.fc_out_features),
            nn.BatchNorm1d(self.fc_out_features),
            nn.ReLU(True),
            Unflatten(self.fc_out_features, 512, self.init_size_x, self.init_size_y),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            SelfAttention(512),  # Uncomment when running locally to check the size
            
            # Adjusted kernel sizes and paddings
            nn.ConvTranspose2d(512, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            #SelfAttention(256),  # Uncomment when running locally to check the size
            
            nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            #SelfAttention(128),  # Uncomment when running locally to check the size
            
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, self.img_channels, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, z, labels):
        #print("Shape of z:", z.shape) # This should be [batch_size, noise_dim]
        #print("Shape of labels:", labels.shape) # This should be [batch_size, class_dim]
        # Reshape labels to have the same batch size dimension as z
        z = z.view(z.shape[0], -1) #reshape z to have the same batch size dimension as labels [batch_size, noise_dim]
        labels = labels.view(labels.shape[0], -1) #reshape labels to have the same batch size dimension as z [batch_size, class_dim]
        # Concatenate noise and labels along the feature dimension
        z = torch.cat([z, labels], 1)
        # Ensure the concatenated tensor has the correct shape
        #print("Shape after concatenation:", z.shape) # This should be [batch_size, noise_dim + class_dim]
        img = self.main(z)
        return img


# Initialize the generator with the corrected architecture
if __name__ == "__main__":
    noise_dim = 100
    class_dim = 3
    img_channels = 3
    img_size_x = 560
    img_size_y = 640

    generator = Generator(
        z_dim=noise_dim,
        class_dim=class_dim,
        img_channels=img_channels,
        img_size_x=img_size_x,
        img_size_y=img_size_y
    )

    # Forward pass to create an image
    noise = torch.randn(4, noise_dim)
    labels = torch.randint(0, class_dim, (4,))
    labels_onehot = torch.nn.functional.one_hot(labels, class_dim).float()
    generated_images = generator(noise, labels_onehot)

    # Check the shape of the generated images
    print(f"Generated images shape: {generated_images.shape}")

    # Plot the generated images
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
            
    plot_images(generated_images) # Uncomment when running locally to display the images

