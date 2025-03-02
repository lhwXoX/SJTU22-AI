import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_encoder, d_decoder, d_hidden):
        super(Attention, self).__init__()
        self.encoder_attention = nn.Linear(d_encoder, d_hidden)
        self.decoder_attention = nn.Linear(d_decoder, d_hidden)
        self.attention = nn.Linear(d_hidden, 1)
        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, decoder_hidden):
        f_attn = self.encoder_attention(encoder_out) # (batch_size, size, d_hidden)
        h_attn = self.decoder_attention(decoder_hidden).unsqueeze(1) # (batch_size, 1, d_hidden)
        attn = self.attention(self.relu(f_attn + h_attn)) # (batch_size, size)
        weight = self.softmax(attn)
        output = (encoder_out * weight).sum(dim=1)
        return output, weight.squeeze(2)