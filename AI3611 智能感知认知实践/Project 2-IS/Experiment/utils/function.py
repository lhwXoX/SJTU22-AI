import torch
import torch.nn.functional as F

def loss_funtion(inputs, outputs, logvar, mu):
    # compute BCELoss and KL distance
    # inputs, outputs: (batch_size, channel, height, width)
    loss_regularization = -0.5 * torch.sum(-torch.exp(logvar) + 1 + logvar - mu ** 2)
    loss_reconstruction = F.binary_cross_entropy(outputs, inputs, reduction='sum')
    return loss_regularization, loss_reconstruction