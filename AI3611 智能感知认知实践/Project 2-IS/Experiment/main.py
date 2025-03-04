import os
import sys
from tqdm import tqdm
import numpy as np
from datetime import datetime
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

if __name__ == '__main__':
    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # path to save model and figure
    time_str = datetime.strftime(datetime.now(), '%m-%d_%H-%M')
    save_path = os.path.join(BASE_DIR, 'output', time_str)
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'checkpoint')
    figure_path = os.path.join(save_path, 'figure')
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    
    # create dataloader
    train_dataset = MNIST('./data/', train=True, transform=transforms.ToTensor(), download=True) # without other transforms for systhesis task
    test_dataset = MNIST('./data/', train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    # create model
    model = VAE(cfg.d_input, cfg.d_hidden, cfg.d_latent)
    
    # create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=cfg.factor, patience=cfg.patience, threshold=cfg.threshold, min_lr=cfg.min_lr)
    
    # load model and optimizer if pretrain
    strat_epoch = 1
    if cfg.pretrain:
        checkpoint = torch.load(cfg.path_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        strat_epoch = checkpoint['epoch'] + 1
    
    # initialize recorder
    regularization_loss = {'train': [], 'test': []}
    reconstruction_loss = {'train': [], 'test': []}
    total_loss = {'train': [], 'test': []}
    best_epoch = 0
    best_reconstruction = checkpoint['best_reconstruction_loss'] if cfg.pretrain else 1e5
    
    # train and test
    for epoch in range(strat_epoch, cfg.max_epochs + 1):
        # train VAE
        train_loss, train_loss_rec, train_loss_reg = [], [], []
        model.train()
        for _, data in enumerate(tqdm(train_loader)):
            inputs, _ = data
            inputs = inputs.to(device)
            outputs, _, mu, logvar = model(inputs)
            optimizer.zero_grad()
            
            # compute loss and back propagation
            loss_regularization, loss_reconstruction = loss_funtion(inputs, outputs, logvar, mu)
            total_loss = loss_reconstruction + cfg.epsilon * loss_regularization
            total_loss.backward()
            optimizer.stop()
            
            # record loss
            train_loss.append(total_loss.item())
            train_loss_rec.append(loss_reconstruction.item())
            train_loss_reg.append(loss_regularization.item())
            
        # compute mean loss
        train_loss_mean = np.mean(train_loss)
        train_loss_rec_mean = np.mean(train_loss_rec)
        train_loss_reg_mean = np.mean(train_loss_reg)
        
        print('Training: Epoch {} / {} Reconstruction Loss: {:.4f} Regularization Loss: {:.4f} Total Loss: {:.4f}'.format(epoch, cfg.max_epochs + 1, train_loss_rec_mean, train_loss_reg_mean, train_loss_mean))
        
        # test VAE
        test_loss, test_loss_rec, test_loss_reg = [], [], []
        model.eval()
        for _, data in enumerate(tqdm(test_loader)):
            inputs, _ = data
            inputs = inputs.to(device)
            outputs, _, mu, logvar = model(inputs)
            optimizer.zero_grad()
            
            # compute loss and back propagation
            loss_regularization, loss_reconstruction = loss_funtion(inputs, outputs, logvar, mu)
            total_loss = loss_reconstruction + cfg.epsilon * loss_regularization
            
            # record loss
            test_loss.append(total_loss.item())
            test_loss_rec.append(loss_reconstruction.item())
            test_loss_reg.append(loss_regularization.item())
            
        # compute mean loss
        test_loss_mean = np.mean(test_loss)
        test_loss_rec_mean = np.mean(test_loss_rec)
        test_loss_reg_mean = np.mean(test_loss_reg)
        
        print('Testing: Epoch {} / {} Reconstruction Loss: {:.4f} Regularization Loss: {:.4f} Total Loss: {:.4f}'.format(epoch, cfg.max_epochs + 1, test_loss_rec_mean, test_loss_reg_mean, test_loss_mean))
        
        # record loss
        regularization_loss['train'].append(train_loss_reg_mean)
        reconstruction_loss['train'].append(train_loss_rec_mean)
        total_loss['train'].append(train_loss_mean)
        regularization_loss['test'].append(test_loss_reg_mean)
        reconstruction_loss['test'].append(test_loss_rec_mean)
        total_loss['test'].append(test_loss_mean)
        
        # plot loss curve
        plot_line(regularization_loss, reconstruction_loss, total_loss, figure_path)

        # save model
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch,
                      'best_reconstruction_loss': best_reconstruction}
        if epoch > 100 and best_reconstruction > test_loss_rec_mean:
            best_reconstruction = test_loss_rec_mean
            best_epoch = epoch
            path_checkpoint = os.path.join(model_path, 'checkpoint_best.pkl')
            torch.save(checkpoint, path_checkpoint)
        if epoch % cfg.save_interval == 0:
            path_checkpoint = os.path.join(model_path, 'checkpoint_epoch_{}.pkl'.format(epoch))
            torch.save(checkpoint, path_checkpoint)
    
    # complete trainig
    time_end = datetime.strftime(datetime.now(), '%m-%d_%H-%M')
    print('Training Done at {}, best reconstruction loss: {:.4f} in epoch {}'.format(time_end, best_reconstruction, best_epoch))
    
    # plot latent and output
    assert cfg.d_latent in [1, 2, 32]
    if cfg.d_latent == 1:
        plot_1d(test_loader, model, figure_path, device)
    elif cfg.d_latent == 2:
        plot_2d(test_loader, model, figure_path, device)
    else:
        plot_32d(test_loader, model, figure_path, device)
    print('Plot Done')