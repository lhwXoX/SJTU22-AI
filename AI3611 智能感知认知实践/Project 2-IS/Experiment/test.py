import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from models.VAE import VAE
from config.config import cfg
from utils.function import *

print(BASE_DIR)
model = VAE(cfg.d_input, cfg.d_hidden, cfg.d_latent)