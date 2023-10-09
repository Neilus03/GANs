import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model6 import Generator, Discriminator
from dataloader_gan import EGD_GAN_Dataset  
from torch.utils.data import DataLoader
from torchvision import transforms

# Define your transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create dataset instance
dataset = EGD_GAN_Dataset(root_folder="/content/drive/MyDrive/EGD-Barcelona/split_by_label/train",
                     transform=transform)  

# Hyperparameters
batch_size = 4
lr = 3e-4
noise_dim = 100
class_dim = 3 
num_epochs = 200

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator(noise_dim, class_dim).to(device)
discriminator = Discriminator(class_dim).to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr)

# Loss functions
adversarial_loss = torch.nn.BCELoss().to(device)
auxiliary_loss = torch.nn.CrossEntropyLoss().to(device)

# Load data here as train_loader
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        
        # Move data to the device
        imgs = imgs.to(device)
        labels = labels.to(device)

        valid = Variable(torch.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False).to(device)

        real_imgs = Variable(imgs.type(torch.FloatTensor)).to(device)
        labels = Variable(labels.type(torch.LongTensor)).to(device)
        
        print("After Variable Conversion:")
        print(f"real_imgs shape: {real_imgs.shape}")
        print(f"labels shape: {labels.shape}")

        # Train Generator
        optimizer_G.zero_grad()

        noise = Variable(torch.FloatTensor(torch.randn(batch_size, noise_dim, 1, 1))).to(device)
        print(f"Noise shape: {noise.shape}")
        
        gen_labels = Variable(torch.LongTensor(torch.randint(0, class_dim, (batch_size,)))).to(device)
        print(f"gen_labels shape: {gen_labels.shape}")
        
        gen_labels_onehot = F.one_hot(gen_labels, num_classes=class_dim).float()
        print(f"gen_labels_onehot shape: {gen_labels_onehot.shape}")

        gen_imgs = generator(noise, gen_labels_onehot)
        print(f"Generated Images shape: {gen_imgs.shape}")

        validity, pred_label = discriminator(gen_imgs)
        print(f"Discriminator validity shape: {validity.shape}, pred_label shape: {pred_label.shape}")

        g_loss = adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels)

        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)

        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)

        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(f"[Epoch {epoch}/{num_epochs}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
