import jittor as jt
from jittor import nn, Module

#定义了两个模型，最终选择了LSTM
class RNNModel(Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def execute(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out
#LSTM
class LSTMModel(Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers)
        self.out = nn.Sequential(
            nn.Linear(256,512), 
            nn.BatchNorm(512),
            nn.Relu(),
            nn.Linear(512,10)
        )
        
    def execute(self, x):
        out, _ = self.rnn(x)
        out = self.out(out[:, -1, :])
        return out