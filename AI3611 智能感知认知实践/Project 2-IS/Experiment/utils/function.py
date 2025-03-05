import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
    
def plot_1d(data_loader, model, figure_path, device):
    # plot output on latent space [-5, 5]
    latent = torch.arange(-5, 5, 0.025).unsqueeze(1).to(device) # 400 samples (400, 1)
    output = model.latent2output(latent)
    save_image(output, os.path.join(figure_path, 'output_1d.png'), nrow=20)
    
    # plot latent on MNIST dataset
    latents, labels = [], []
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            latent = model.input2latent(img)
        latents.append(latent)
        labels.append(label)
    latents, labels = torch.cat(latents).detach().cpu().numpy(), torch.cat(labels).detach().cpu().numpy() # (N, 1), (N, )
    
    for label in range(10):
        x_latent = latents[labels == label] # (M, 1) M/N samples
        y_label = label * np.ones_like(x_latent) # (M, 1)
        plt.scatter(x_latent, y_label, s=10, alpha=1.0, label=str(label))
    plt.xlabel('Latent')
    plt.ylabel('Label')
    plt.title('1D Latent Distribution')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, 'latent_1d.png'))
    plt.close()
        
def plot_2d(data_loader, model, figure_path, device):
    # plot output on latent 2d space [[-5, 5], [-5, 5]]
    x, y = torch.arange(-5, 5, 0.5), torch.arange(-5, 5, 0.5)
    latent = torch.cartesian_prod(x, y).to(device) # 400 samples (400, 2)
    output = model.latent2output(latent)
    save_image(output, os.path.join(figure_path, 'output_2d.png', nrow=20))
    
    # plot latent on MNIST dataset
    latents, labels = [], []
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            latent = model.input2latent(img)
        latents.append(latent)
        labels.append(label)
    latents, labels = torch.cat(latents).detach().cpu().numpy(), torch.cat(labels).detach().cpu().numpy() # (N, 2), (N, )
    
    for label in range(10):
        x_latent, y_latent = latents[labels == label].transpose(0, 1)
        plt.scatter(x_latent, y_latent, s=5, alpha=1.0, label=str(label))
    plt.xlabel('Latent x')
    plt.ylabel('Latent y')
    plt.title('2D Latent Distribution')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, 'latent_2d.png'))
    plt.close()

def plot_32_64d(data_loader, model, figure_path, device, dimension: int):
    assert dimension in [32, 64]
    
    # plot output on latent 32/64d space
    latent = torch.randn(400, dimension).to(device) # use (0, 1) gaussion distribution instead
    output = model.latent2output(latent)
    save_image(output, os.path.join(figure_path, f'output_{dimension}d.png', nrow=20))
    
    # plot latent on MNIST dataset
    latents, labels = [], []
    for img, label in data_loader:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            latent = model.input2latent(img)
        latents.append(latent)
        labels.append(label)
    latents, labels = torch.cat(latents).detach().cpu().numpy(), torch.cat(labels).detach().cpu().numpy() # (N, 32/64), (N, )
    
    # decomposition latents 32/64d -> 2d for visualization
    latents_2d_PCA = PCA(n_components=2).fit_transform(latents)
    latents_2d_TSNE = TSNE(n_components=2).fit_transform(latents)
    
    # PCA visualization
    for label in range(10):
        x_latent_PCA, y_latent_PCA = latents_2d_PCA[labels == label].transpose(0, 1)
        plt.scatter(x_latent_PCA, y_latent_PCA, s=5, alpha=1.0, label=str(label))
    plt.xlabel('Latent x')
    plt.ylabel('Latent y')
    plt.title(f'{dimension}D Latent Distribution (PCA)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, f'latent_{dimension}d_pca.png'))
    plt.close()
    
    # t-SNE visualization
    for label in range(10):
        x_latent_TSNE, y_latent_TSNE = latents_2d_TSNE[labels == label].transpose(0, 1)
        plt.scatter(x_latent_TSNE, y_latent_TSNE, s=5, alpha=1.0, label=str(label))
    plt.xlabel('Latent x')
    plt.ylabel('Latent y')
    plt.title(f'{dimension}D Latent Distribution (t-SNE)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(figure_path, f'latent_{dimension}d_tsne.png'))
    plt.close()