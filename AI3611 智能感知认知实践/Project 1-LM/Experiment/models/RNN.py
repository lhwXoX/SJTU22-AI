import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentNeuralNetwork(nn.Module):
    def __init__(self, vocab_size, d_embedding, d_hidden, n_layers, drop_prob=0.2):
        super(RecurrentNeuralNetwork, self).__init__()
        self.dropout = nn.Dropout(p=drop_prob)
        self.TokenEmb = nn.Embedding(vocab_size, d_embedding)
        self.rnn = nn.RNN(d_embedding, d_hidden, n_layers, dropout=drop_prob)
        self.linear = nn.Linear(d_hidden, vocab_size)
        self.init_weight()
        
    def forward(self, inputs, hidden):
        embedding = self.dropout(self.TokenEmb(inputs))
        output, hidden = self.rnn(embedding, hidden)
        output = self.linear(self.dropout(output))
        output = F.log_softmax(output, dim=-1)
        return output, hidden
    
    def init_weight(self):
        nn.init.uniform_(self.TokenEmb.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear.bias, -0.1, 0.1)