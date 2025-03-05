import os
import sys
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from models.VAE import VAE
from config import config
from utils.function import *

# load configuration
args = config.parse_args()

# select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create dataloader (test only)
data_path = os.path.join(BASE_DIR, 'data')
test_dataset = MNIST(data_path, train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# create model
model = VAE(args.d_input, args.d_hidden, args.d_latent).to(device)

# load state
path_checkpoint = os.path.join(BASE_DIR, args.path_checkpoint)
checkpoint = torch.load(path_checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# plot 3d figure (demonstrate only)
plot_16_32_64d_3(test_loader, model, None, device, args.d_latent, False)