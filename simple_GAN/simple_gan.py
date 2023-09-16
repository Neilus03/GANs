import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from math import e

#To improve performance we could try:
# 1. to build a larger network (currently trying.)
# 2. Better normalization with BatchNorm
# 3. What about changing architecture to a CNN?

class Discriminator(nn.Module):
    def __init__(self, img_dim): #img_dim = 784 (28x28) MNIST
        super().__init__()
        self.disc = nn.Sequential(
      		nn.Linear(img_dim, 256),
			nn.LeakyReLU(0.1),
			nn.Linear(256, 128),
			nn.LeakyReLU(0.1),
			nn.Linear(128, 64),
			nn.LeakyReLU(0.1),
			nn.Linear(64,1),
			nn.Sigmoid(),
		)
    def forward(self, x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.gen = nn.Sequential(
			nn.Linear(z_dim, 256),
			nn.LeakyReLU(0.1),
			nn.Linear(256, 128),
			nn.LeakyReLU(0.1),
			nn.Linear(128, 64),
   			nn.LeakyReLU(0.1),
			nn.Linear(64, img_dim), #img_dim = 28x28x1 --> 784 (when flattened)
			nn.Tanh(),
  		)
    def forward(self, x):
        return self.gen(x)

#Hyperparameters (GANs are very sensitive to them)
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 3e-4 # 0.0003
z_dim = 64 # maybe I'll try 128, 256 ...
img_dim = 28 * 28 * 1 #784
batch_size = 32
num_epochs = 50

disc = Discriminator(img_dim).to(device)
gen = Generator(z_dim, img_dim).to(device)
fixed_noise = torch.randn((batch_size, z_dim)).to(device)
transforms = T.Compose([T.ToTensor(), T.Normalize((0.5,),(0.5,))])

dataset = D.MNIST(root="dataset/", transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()
writer_fake = SummaryWriter(f"runs/GAN_MNIST/fake")
writer_real = SummaryWriter(f"runs/GAN_MNIST/real")
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.view(-1, 784).to(device)
        batch_size = real.shape[0]
        
        ##Train discriminator: maximize log(D(real)) + log(1 - D(G(z)))    (where z is random noise, D stands for Discriminator and G for Generator)
        noise = torch.randn(batch_size, z_dim).to(device)
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real)) #go to BCE documentation, your trying to minimize the first part as the second part gets cancelled bcs of the 1s, then we get : ln = -Wn * [y_n * log x_n], Wn is just going to be one, the - sign is key, bcs [maximizing the log(D(x)) == minimizing the negative of that expression], then we got the first part of the expression max log(D(real)), just that instead we have the equivalent min -log(D(real)).
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake)) #As imagined, this is the opposite part, then with this we cancel the first part of the equation and get: ln = -Wn * [(1-y_n)*log(1-x_n)] where y_n is 0, then finally we have: ln = -Wn * [log(1-x_n)] then: [maximizing the log(1 - D(G(z))) == minimizing the negative of that expression]
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward(retain_graph = True)
        opt_disc.step()
        
        ##Train generator min log(1 - D(G(z))) <---> max log(D(G(z)) # where the second option of maximizing doesn't suffer from saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion (output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
