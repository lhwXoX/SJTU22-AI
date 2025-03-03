# Reference: https://github.com/parksunwoo/show_attend_and_tell_pytorch/blob/master/model.py
import torch
import torch.nn as nn
from torchvision.models import resnet101
import random

class Attention(nn.Module):
    def __init__(self, d_encoder, d_decoder, d_hidden):
        super(Attention, self).__init__()
        self.encoder_attention = nn.Linear(d_encoder, d_hidden)
        self.decoder_attention = nn.Linear(d_decoder, d_hidden)
        self.attention = nn.Linear(d_hidden, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, encoder_out, decoder_hidden):
        f_attn = self.encoder_attention(encoder_out) # (batch_size, size, d_hidden)
        h_attn = self.decoder_attention(decoder_hidden).unsqueeze(1) # (batch_size, 1, d_hidden)
        attn = self.attention(self.relu(f_attn + h_attn)) # (batch_size, size)
        weight = self.softmax(attn)
        output = (encoder_out * weight).sum(dim=1)
        return output, weight.squeeze(2)
    
class Encoder(nn.Module):
    def __init__(self, size=14):
        super(Encoder, self).__init__()
        resnet = resnet101(pretraind=True)
        modules = list(resnet.children())[:-2] # exclude linear layers
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((size, size))
        self.set_grad()
        
    def set_grad(self, fine_tune=False):
        for params in self.resnet.parameters():
            params.requires_grad = False
        for layer in list(self.resnet.children())[5:]:
            for params in layer.parameters():
                params.requires_grad = fine_tune
                
    def forward(self, x):
        output = self.resnet(x)
        output = self.adaptive_pool(output).permute(0, 2, 3, 1) # (batch_size, c, size, size) -> (batch_size, size, size, c)
        return output
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_hidden, d_encoder, d_decoder, d_embedding, drop_prob=0.5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab_size = vocab_size
        self.attention = Attention(d_encoder, d_decoder, d_hidden)
        self.embedding = nn.Embedding(vocab_size, d_embedding)
        self.dropout = nn.Dropout(p=drop_prob)
        self.decode_step = nn.LSTMCell(d_embedding + d_encoder, d_decoder)
        self.init_cell = nn.Linear(d_encoder, d_decoder)
        self.init_hidden = nn.Linear(d_encoder, d_decoder)
        self.gate = nn.Linear(d_decoder, d_encoder)
        self.out = nn.Linear(d_decoder, vocab_size)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        
    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.out.bias.data.fill_(0)
        self.out.weight.data.uniform_(-0.1, 0.1)
        
    def init_hidden_state(self, encoder_output):
        mean_encoder_output = encoder_output.mean(dim=1) # (batch_size, d_encoder)
        hidden = self.init_hidden(mean_encoder_output)
        cell = self.init_cell(mean_encoder_output)
        return hidden, cell

    def forward(self, encoder_output, encoded_caption, caption_length, p=1.0):
        '''
        encoder_output: (batch_size, size, size, d_encoder)
        encoded_caption: (batch_size, max_len)
        caption_length: (batch_size, 1)
        '''
        batch_size, _, _, d_encoder = encoder_output.shape
        encoder_output = encoder_output.view(batch_size, -1, d_encoder)
        num_pixels = encoder_output.shape[1]
        caption_length, sort_idx = caption_length.squeeze(1).sort(dim=0, descending=True) # (batch_size, )
        encoder_output, encoded_caption = encoder_output[sort_idx], encoded_caption[sort_idx]
        
        embedding = self.embedding(encoded_caption) # (batch_size, max_length, d_embedding)
        hidden, cell = self.init_hidden_state(encoder_output) # h, c: (batch_size, d_endcoder)
        target_length = (caption_length - 1).tolist()
        max_length = max(target_length)
        
        caption_predict = torch.zeros(batch_size, max_length, self.vocab_size, device=self.device)
        weights = torch.zeros(batch_size, max_length, num_pixels, device=self.device)
        
        for step in range(max_length):
            batch_size_step = sum([length > step for length in target_length])
            output, weight = self.attention(encoder_output[:batch_size_step], hidden[:batch_size_step]) # output, weight: (batch_size, d_encoder), (batch_size, d_encoder)
            gate = self.sigmoid(self.gate(hidden[:batch_size_step])) # (batch_size, d_encoder)
            output = output * gate
            
            # scheduled sampling
            if random.random() <= p:
                embedding = embedding[:batch_size_step, step, :] # (batch_size, d_embedding)
            else:
                embedding = self.embedding(predict_token[:batch_size_step]) # (batch_size, d_embedding)
            
            hidden, cell = self.decode_step(
                torch.cat([embedding, output], dim=1), # (batch_size_step, d_embedding + d_encoder)
                (hidden[:batch_size_step], cell[:batch_size_step]) # (batch_size_step, d_decoder)
            )
            prediction = self.out(self.dropout(hidden)) # (batch_size_step, vocab_size)
            predict_token = prediction.argmax(dim=1) # (batch_size_step, )
            caption_predict[:batch_size_step, step, :] = prediction
            weights[:batch_size_step, step, :] = weight
            
        return caption_predict, encoded_caption, target_length, weights, sort_idx