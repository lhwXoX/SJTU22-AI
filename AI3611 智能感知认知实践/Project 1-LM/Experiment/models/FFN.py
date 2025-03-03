import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNetwork(nn.Module):
    def __init__(self, vocab_size, d_embedding, n_step, d_hidden):
        super(FeedForwardNetwork, self).__init__()
        self.TokenEmb = nn.Embedding(vocab_size, d_embedding)
        self.linear1 = nn.Linear(d_embedding * n_step, d_hidden)
        self.linear2 = nn.Linear(d_hidden, vocab_size)
        self.init_weights()
        
    def forward(self, inputs):
        # input: [batch_size, n_step, vocab_size]
        embedding = self.TokenEmb(inputs)
        hidden = F.relu(self.linear1(embedding.view(embedding.shape[0], -1)))
        output = F.log_softmax(self.linear2(hidden), dim=-1)
        return output # [batch_size, vocab_size]
        
    def init_weights(self):
        nn.init.uniform_(self.TokenEmb.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear1.weight, -0.1, 0.1)
        nn.init.uniform_(self.linear2.weight, -0.1, 0.1)
        nn.init.zeros_(self.linear1.bias)
        nn.init.zeros_(self.linear2.bias)