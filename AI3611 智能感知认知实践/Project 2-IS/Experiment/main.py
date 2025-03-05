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
from config import config
from utils.function import *

if __name__ == '__main__':
    # load configuration
    args = config.parse_args()

    # select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # path to save model and figure
    save_path = os.path.join(BASE_DIR, 'output', args.save_name)
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, 'checkpoint')
    figure_path = os.path.join(save_path, 'figure')
    loss_path = os.path.join(save_path, 'loss.txt') # record final loss
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    
    # create dataloader
    data_path = os.path.join(BASE_DIR, 'data')
    train_dataset = MNIST(data_path, train=True, transform=transforms.ToTensor(), download=True) # without other transforms for systhesis task
    test_dataset = MNIST(data_path, train=False, transform=transforms.ToTensor(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # create model
    model = VAE(args.d_input, args.d_hidden, args.d_latent)
    
    # create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # load state if pretrain
    strat_epoch = 1
    if args.pretrain:
        checkpoint = torch.load(args.path_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        strat_epoch = checkpoint['epoch'] + 1
    model.to(device)
    
    # initialize recorder
    regularization_loss = {'train': [], 'test': []}
    reconstruction_loss = {'train': [], 'test': []}
    total_losses = {'train': [], 'test': []}
    best_epoch = 0
    best_reconstruction = checkpoint['best_reconstruction_loss'] if args.pretrain else 1e5
    
    # train and test
    for epoch in range(strat_epoch, args.max_epochs + 1):
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
            total_loss = loss_reconstruction + args.epsilon * loss_regularization
            total_loss.backward()
            optimizer.step()
            
            # record loss
            train_loss.append(total_loss.item())
            train_loss_rec.append(loss_reconstruction.item())
            train_loss_reg.append(loss_regularization.item())
            
        # compute mean loss
        train_loss_mean = np.mean(train_loss) / args.batch_size
        train_loss_rec_mean = np.mean(train_loss_rec) / args.batch_size
        train_loss_reg_mean = np.mean(train_loss_reg) / args.batch_size
        
        print('Training: Epoch {} / {} Reconstruction Loss: {:.4f} Regularization Loss: {:.4f} Total Loss: {:.4f}'.format(epoch, args.max_epochs, train_loss_rec_mean, train_loss_reg_mean, train_loss_mean))
        
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
            total_loss = loss_reconstruction + args.epsilon * loss_regularization
            
            # record loss
            test_loss.append(total_loss.item())
            test_loss_rec.append(loss_reconstruction.item())
            test_loss_reg.append(loss_regularization.item())
            
        # compute mean loss
        test_loss_mean = np.mean(test_loss) / args.batch_size
        test_loss_rec_mean = np.mean(test_loss_rec) / args.batch_size
        test_loss_reg_mean = np.mean(test_loss_reg) / args.batch_size
        
        print('Testing: Epoch {} / {} Reconstruction Loss: {:.4f} Regularization Loss: {:.4f} Total Loss: {:.4f}'.format(epoch, args.max_epochs, test_loss_rec_mean, test_loss_reg_mean, test_loss_mean))
        
        # record loss
        regularization_loss['train'].append(train_loss_reg_mean)
        reconstruction_loss['train'].append(train_loss_rec_mean)
        total_losses['train'].append(train_loss_mean)
        regularization_loss['test'].append(test_loss_reg_mean)
        reconstruction_loss['test'].append(test_loss_rec_mean)
        total_losses['test'].append(test_loss_mean)
        
        # plot loss curve
        plot_line(regularization_loss, reconstruction_loss, total_losses, figure_path)

        # save model
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch,
                      'best_reconstruction_loss': best_reconstruction}
        if epoch >= 100 and best_reconstruction > test_loss_rec_mean:
            best_reconstruction = test_loss_rec_mean
            best_epoch = epoch
            path_checkpoint = os.path.join(model_path, 'checkpoint_best.pkl')
            torch.save(checkpoint, path_checkpoint)
        if epoch % args.save_interval == 0:
            path_checkpoint = os.path.join(model_path, 'checkpoint_epoch_{}.pkl'.format(epoch))
            torch.save(checkpoint, path_checkpoint)
            
    # record final loss
    file = open(loss_path, 'a')
    file.write('Training:\nReconstruction Loss: {:.2f}\nRegularization Loss: {:.2f}\nTotal Loss: {:.2f}\nTesting:\nReconstruction Loss: {:.2f}\nRegularization Loss: {:.2f}\nTotal Loss: {:.2f}'.format(train_loss_rec_mean, train_loss_reg_mean, train_loss_mean, test_loss_rec_mean, test_loss_reg_mean, test_loss_mean))
    file.close
    
    # complete trainig
    time_end = datetime.strftime(datetime.now(), '%m-%d_%H-%M')
    print('Training Done at {}, best reconstruction loss: {:.4f} in epoch {}'.format(time_end, best_reconstruction, best_epoch))
    
    # plot latent and output
    assert args.d_latent in [1, 2, 32, 64]
    if args.d_latent == 1:
        plot_1d(test_loader, model, figure_path, device)
    elif args.d_latent == 2:
        plot_2d(test_loader, model, figure_path, device)
    else:
        plot_32_64d(test_loader, model, figure_path, device, args.d_latent)
    print('Plot Done')