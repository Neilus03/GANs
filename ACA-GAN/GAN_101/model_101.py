import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_discriminator import ResNetDiscriminator
from attention_module101 import SelfAttention
from generator_pruebas import  Generator


generator = Generator(z_dim=100, class_dim=3, img_channels=3, img_size_x=560, img_size_y=640)
Discriminator = ResNetDiscriminator(num_classes=3)


#TESTING AND DEBUGGING
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

if __name__ == "__main__":
    # Define dimensions and hyperparameters
    noise_dim = 100
    class_dim = 3
    batch_size = 4
    img_channels = 3
    img_size_x = 560
    img_size_y = 640

    # Initialize models
    generator = Generator(z_dim=noise_dim, class_dim=class_dim, img_channels=img_channels, img_size_x=img_size_x, img_size_y=img_size_y)
    discriminator = ResNetDiscriminator(class_dim)

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
