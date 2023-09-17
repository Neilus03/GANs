'''
Training of WGAN on MNIST dataset
'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import torchvision.datasets as D
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from model import Critic, Generator, initialize_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#All hyperparams == paper
LEARNING_RATE = 5e-5
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1 #bcs MNIST is grayscale
Z_DIM = 100
NUM_EPOCHS = 5
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
WEIGHT_CLIP = 0.01

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
critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(device)

initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.RMSprop(gen.parameters(), lr= LEARNING_RATE)
opt_critic = optim.RMSprop(critic.parameters(), lr= LEARNING_RATE)


fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)

writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

for epoch in range(NUM_EPOCHS):
    for batch_idx, (real, _) in enumerate(loader): #the underscore is bcs as dcgan is unsupervised learning we dont need actual labels
        real = real.to(device)
        
        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((BATCH_SIZE, Z_DIM, 1, 1)).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            loss_critic = - (torch.mean(critic_real) - torch.mean(critic_fake)) #as rmsprop minimizes, and we want to maximize, lets minimize the negative of the term which is equivalent to maximiing the positive, as (-)*(-) = +
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
            for p in critic.parameters():
                p.data.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP) #CLIPPING WEIGHTS
        
        ### Train Generator: min -E[critic(gen_fake(z))] #where z is a batch of noise normally distributed 
        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        
        #Print losses occasionally and print to tensorboard
        if batch_idx % 100 == 0:
            print(
				f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx} / {len(loader)} \
        				Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f}"
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