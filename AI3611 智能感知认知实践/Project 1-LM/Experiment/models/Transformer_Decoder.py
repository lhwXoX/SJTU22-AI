'''
Implementation of GPT
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ScaleDotProductAttention(nn.Module):
    def __init__(self, max_len, drop_prob):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(max_len).view(1, 1, max_len, max_len))
        self.dropout = nn.Dropout(p=drop_prob)
        
    def forward(self, q, k, v):
        # k: [Batch_size, n_heads, seq_len, d_k]
        seq_len = k.shape[2]
        k_t = k.transpose(2, 3)
        scores = q @ k_t / math.sqrt(k.shape[-1])
        scores += self.mask[:, :, :seq_len, :seq_len]
        scores = self.dropout(self.softmax(scores))
        output = scores @ v
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_embedding, d_hidden, n_heads, max_len, drop_prob):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_embedding, d_hidden)
        self.w_k = nn.Linear(d_embedding, d_hidden)
        self.w_v = nn.Linear(d_embedding, d_hidden)
        self.w_o = nn.Linear(d_hidden, d_embedding)
        self.dropout = nn.Dropout(p=drop_prob)
        self.attention = ScaleDotProductAttention(max_len, drop_prob)
    
    def split(self, x):
        '''
        split x: [batch_size, seq_len, d_model] -> [batch_size, n_heads, seq_len, d_model // n_heads]
        '''
        batch_size, seq_len, d_model = x.shape
        x = x.view(batch_size, seq_len, self.n_heads, d_model // self.n_heads)
        x = x.transpose(1, 2)
        return x
    
    def concate(self, x):
        '''
        concate x: [batch_size, n_heads, seq_len, d] -> [batch_size, seq_len, d * n_heads]
        '''
        batch_size, n_heads, seq_len, d = x.shape
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, n_heads * d)
        return x
    
    def forward(self, q, k, v):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        o = self.attention(q, k, v)
        o = self.w_o(self.concate(o))
        o = self.dropout(o)
        return o
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_embedding, d_hidden, drop_prob):
        super(PositionwiseFeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_embedding, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_embedding)
        self.relu = nn.ReLU(inplace=True)
        self.dropout_1 = nn.Dropout(p=drop_prob)
        self.dropout_2 = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(self.relu(x))
        x = self.dropout_2(self.linear_2(x))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_embedding, d_hidden, n_heads, max_len, drop_prob):
        super(DecoderLayer, self).__init__()
        self.SelfAttention = MultiHeadAttention(d_embedding, d_hidden, n_heads, max_len, drop_prob)
        self.FeedForward = PositionwiseFeedForward(d_embedding, d_hidden, drop_prob)
        self.LayerNorm_1 = nn.LayerNorm(d_embedding)
        self.LayerNorm_2 = nn.LayerNorm(d_embedding)
        
    def forward(self, x):
        x_1 = self.LayerNorm_1(x)
        x = self.SelfAttention(x_1, x_1, x_1) + x
        x_2 = self.LayerNorm_2(x)
        x = self.FeedForward(x_2) + x
        return x
    
class GPT(nn.Module):
    def __init__(self, vocab_size, d_embedding, d_hidden, n_heads, max_len, drop_prob, n_layers):
        super(GPT, self).__init__()
        self.TokenEmb = nn.Embedding(vocab_size, d_embedding)
        self.LayerNorm = nn.LayerNorm(d_embedding)
        self.dropout = nn.Dropout(p=drop_prob)
        self.linear = nn.Linear(d_embedding, vocab_size)
        self.positionalEncoding = nn.Parameter(torch.rand(1, max_len, d_embedding))
        self.layers = nn.ModuleList([DecoderLayer(d_embedding, d_hidden, n_heads, max_len, drop_prob) for _ in range(n_layers)])
        self.init_weights()
        
    def forward(self, x):
        print(x.shape)
        print(self.positionalEncoding[:, :x.shape[1]].shape)
        embedding = self.TokenEmb(x) + self.positionalEncoding[:, :x.shape[1]]
        output = self.dropout(embedding)
        for layer in self.layers:
            output = layer(output)
        output = self.linear(self.LayerNorm(output))
        output = F.log_softmax(output, dim=-1)
        return output
    
    def init_weights(self):
        nn.init.uniform_(self.TokenEmb.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear.bias)
        
# Test
vocab_size = 16
d_embedding = 256
d_hidden = 512
n_heads = 8
max_len = 1024
drop_prob = 0.2
n_layers = 2
inputs = torch.randint(low=0, high=3, size=(2, 6), dtype=torch.int) # [batch_size, seq_len, vocab_size]
GPT = GPT(vocab_size, d_embedding, d_hidden, n_heads, max_len, drop_prob, n_layers)
output = GPT(inputs)
print(output.shape)