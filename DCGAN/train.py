'''
Training of DCGAN on MNIST dataset
'''

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from model import Discriminator, Generator, initialize_weights

