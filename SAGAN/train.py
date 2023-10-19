import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torch.nn import init
from model import Generator, Discriminator  
from dataloader import EGD_SAGAN_Dataset  
import os
import wandb



# Initialize wandb
wandb.init(project='train_SAGAN1', entity='neildlf')

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=90),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset instance
dataset = EGD_SAGAN_Dataset(root_folder="/home/ndelafuente/CVC/EGD_Barcelona/GANs/ACA-GAN/EGD-Barcelona/split_by_label/train/2",
                          transform=transform)

# Hyperparameters
batch_size = 8
lr = 3e-3
noise_dim = 200
num_epochs = 200

# Initialize models and move to device
generator = Generator(noise_dim).to(device)
wandb.watch(generator)
discriminator = Discriminator().to(device)
wandb.watch(discriminator)

''' Some problems with the initialization of the weights due to dimension of tensors
# Xavier initialization for generator weights
for name, param in generator.named_parameters():
    if 'weight' in name:
        init.xavier_uniform(param)

# Xavier initialization for discriminator weights
for name, param in discriminator.named_parameters():
    if 'weight' in name:
        init.xavier_uniform(param)
'''

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Loss function
adversarial_loss = torch.nn.BCELoss().to(device)

# DataLoader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Create directory to save generated images
save_dir = "/home/ndelafuente/CVC/EGD_Barcelona/GANs/ACA-GAN/generated_images"
os.makedirs(save_dir, exist_ok=True)

# Training loop
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        for _ in range(10): #train n times the generator for each time you train the discriminator

            # Get the current batch size
            cur_batch_size = len(imgs)
            #print(f"Current batch size: {cur_batch_size}")

            # Move data to the device 
            imgs, labels = imgs.to(device), labels.to(device)
            #print(f"Shapes after moving to device - imgs: {imgs.shape}, labels: {labels.shape}")

            # Create tensors for valid (real) and fake labels
            valid = torch.full((cur_batch_size, 1, 1, 1), 0.95, dtype=torch.float32, device=device) #0.95 to avoid the discriminator to be too confident
            fake = torch.full((cur_batch_size, 1, 1, 1), 0.0, dtype=torch.float32, device=device)
            
            # --------- Train Generator ---------
            #print("Training Generator...")

            # Zero the gradients for generator
            optimizer_G.zero_grad()

            # Generate random noise for GAN
            noise = torch.randn(cur_batch_size, noise_dim, 1, 1, device=device)
            
            # Generate fake images
            gen_imgs = generator(noise)
            #print(f"Generated images shape: {gen_imgs.shape}")

            # Discriminator's prediction on generated images
            validity_avg = discriminator(gen_imgs)
            #print(f"Discriminator's validity_avg shape: {validity_avg.shape}, pred_label shape: {pred_label.shape}")
            assert validity_avg.shape == valid.shape  # Shape check
            #print("Data type of validity_avg:", validity_avg.dtype)  # Data type check
            #print("Data type of valid:", valid.dtype)  # Data type check
            #print("Min and Max of validity_avg:", torch.min(validity_avg).item(), torch.max(validity_avg).item())

            # Calculate generator's loss
            g_loss = adversarial_loss(validity_avg, valid)
            print(f"Generator loss: {g_loss.item()}")

            # Backpropagate and update generator's weights
            g_loss.backward()
            optimizer_G.step()
        
        # --------- Train Discriminator ---------
        #print("Training Discriminator...")

        # Zero the gradients for discriminator
        optimizer_D.zero_grad()

        # Discriminator's prediction on real images
        real_pred = discriminator(imgs)
        #print(f"Real validity shape: {real_pred.shape}, real_aux shape: {real_aux.shape}")

        # Discriminator's loss on real images
        d_real_loss = adversarial_loss(real_pred, valid)
        
        # Discriminator's prediction on fake images
        fake_pred = discriminator(gen_imgs.detach())
        #print(f"Fake validity shape: {fake_pred.shape}, fake_aux shape: {fake_aux.shape}")

        # Discriminator's loss on fake images
        d_fake_loss = adversarial_loss(fake_pred, fake)
        
        # Average discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        print(f"Discriminator loss: {d_loss.item()}")

        # Backpropagate and update discriminator's weights
        d_loss.backward()
        optimizer_D.step()

        # Log to TensorBoard
        if i % 10 == 0:
            wandb.log({'Generator Loss/train': g_loss.item()}, step=epoch * len(train_loader) + i)
            wandb.log({'Discriminator Loss/train': d_loss.item()}, step=epoch * len(train_loader) + i)
            
            # Generate and save images for the one class
            label_tensor = torch.tensor([0]).repeat(cur_batch_size)
            label_onehot = F.one_hot(label_tensor).float().to(device)
            fixed_noise = torch.randn(cur_batch_size, noise_dim, 1, 1, device=device)
            with torch.no_grad():
                fixed_gen_images = generator(fixed_noise).detach().cpu()
            img_grid = torchvision.utils.make_grid(fixed_gen_images, normalize=True)
            wandb.log({"Generated Images/class_2": [wandb.Image(img_grid)]}, step=epoch * len(train_loader) + i)
            class_dir = os.path.join(save_dir, "class_2")
            os.makedirs(class_dir, exist_ok=True)
            for j in range(cur_batch_size):
                img_path = os.path.join(class_dir, f"epoch_{epoch}_batch_{i}_img_{j}.png")
                if (epoch * len(train_loader) * cur_batch_size + i * cur_batch_size + j) % 50 == 0:
                    torchvision.utils.save_image(fixed_gen_images[j], img_path)
    
    # Print epoch losses
    print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

wandb.finish()

import torch.nn.init as init


