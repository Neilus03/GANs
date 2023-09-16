'''
Training of DCGAN on MNIST dataset
'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import torchvision.datasets as D
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from model import Discriminator, Generator, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#All hyperparams == paper
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1 #bcs MNIST is grayscale
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = T.Compose(
	[
		T.Resize(IMAGE_SIZE),
		T.ToTensor(),
  		T.Normalize(
			[0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
	]	
)

dataset = D.MNIST(root="dataset/", transform=transforms, train=True, download=True)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr= LEARNING_RATE, betas=(0.5, 0.999)) #Betas are explicitly changed in the aper, just b1 changes, from 0.9 to 0.5
opt_disc = optim.Adam(disc.parameters(), lr= LEARNING_RATE, betas=(0.5, 0.999)) #Betas are explicitly changed in the aper, just b1 changes, from 0.9 to 0.5

criterion = nn.BCELoss()

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
disc.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader): #the underscore is bcs as dcgan is unsupervised learning we dont need actual labels
        real = real.to(device)
        noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
        fake = gen(noise)
        
        ### Train Discriminator max log(D(x)) + log (1 - D(G(z)))
        
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        
        loss_disc = (loss_disc_fake + loss_disc_real) / 2
        
        disc.zero_grad() 
        loss_disc.backward(retain_graph=True)
        opt_disc.step()
        
        ### Train Generator min log(1 - D(G(z))) <-->  max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward(retain_graph=True)
        opt_gen.step()
        
        #Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
				f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx} / {len(loader)} \
        				Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
			)
            
            with torch.no_grad():
                fake = gen(fixed_noise)
                #take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
					real[:32], normalize=True
				)
                img_grid_fake = torchvision.utils.make_grid(
					fake[:32], normalize=True
				)
                
                writer_real.add_image('Real', img_grid_real, global_step=step)
                writer_fake.add_image('Fake', img_grid_fake, global_step=step)
            
            step += 1