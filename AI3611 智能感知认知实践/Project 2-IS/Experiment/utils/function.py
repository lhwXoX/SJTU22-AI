import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def loss_funtion(inputs, outputs, logvar, mu):
    # compute BCELoss and KL distance
    # inputs, outputs: (batch_size, channel, height, width)
    loss_regularization = -0.5 * torch.sum(-torch.exp(logvar) + 1 + logvar - mu ** 2)
    loss_reconstruction = F.binary_cross_entropy(outputs, inputs, reduction='sum')
    return loss_regularization, loss_reconstruction

def plot_line(regularization_loss, reconstruction_loss, total_loss, figure_path):
    epoch = np.arange(1, len(total_loss['train']) + 1)
    plt.plot(epoch, regularization_loss['train'], label='Train Regularization Loss')
    plt.plot(epoch, reconstruction_loss['train'], label='Train Reconstruction Loss')
    plt.plot(epoch, total_loss['train'], label='Train Total Loss')
    plt.plot(epoch, regularization_loss['test'], label='Test Regularization Loss')
    plt.plot(epoch, reconstruction_loss['test'], label='Test Reconstruction Loss')
    plt.plot(epoch, total_loss['test'], label='Test Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, 'loss.png'))
    plt.close()