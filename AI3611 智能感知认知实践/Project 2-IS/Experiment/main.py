import os
import sys
from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from models.VAE import VAE
from config.config import cfg

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # select device
    
    # create dataloader
    train_dataset = MNIST('./data/', train=True, transform=transforms.ToTensor(), download=True) # without other transforms for systhesis task
    test_dataset = MNIST('./data/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # create model
    model = VAE(cfg.d_input, cfg.d_hidden, cfg.d_latent)
    if cfg.pretrain:
        checkpoint = torch.load(cfg.path_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    # create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cfg.factor, patience=cfg.patience, threshold=cfg.threshold, min_lr=cfg.min_lr)
    if cfg.pretrain:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])