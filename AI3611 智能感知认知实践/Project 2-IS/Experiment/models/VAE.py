import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, d_latent: int):
        super(VAE, self).__init__()
        self.d_input = d_input
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_latent * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_input),
            nn.Sigmoid(),
        )
        
    def reparameterize(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)
    
    def forward(self, inputs):
        batch_size, channel, height, width = inputs.shape
        assert self.d_input == channel * height * width
        inputs = inputs.view(batch_size, -1)
        mu, logvar = self.encoder(inputs).chunk(2, dim=1)
        latent = self.reparameterize(mu, logvar)
        outputs = self.decoder(latent).view(batch_size, channel, height, width)
        return outputs, latent, mu, logvar