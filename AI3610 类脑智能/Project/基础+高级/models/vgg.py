import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, T: int, use_cupy=False):
        super().__init__()
        self.T = T
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
        layer.Flatten(),
        layer.Linear(512, 128, bias=True),
        neuron.LIFNode(surrogate_function=surrogate.ATan()),
        layer.Dropout(0.2),
        layer.Linear(128, 10, bias=True),
        #neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, step_mode='m')
        
        if use_cupy:
            functional.set_backend(self, backend='cupy')
        
    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.features(x_seq)
        out = self.classifier(out)
        out = out.mean(0)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [layer.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           layer.BatchNorm2d(x),
                           neuron.LIFNode(surrogate_function=surrogate.ATan(), tau=2.0)]
                in_channels = x
        layers += [layer.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def spiking_encoder(self):
        return self.features[0:3]


def cal_firing_rate(s_seq: torch.Tensor):
    return s_seq.flatten(1).mean(1)
if __name__ == '__main__':
    from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor
    model = VGG("VGG16", T=6)
    model = model.to('cuda')
    input = torch.randn(32, 1, 32, 32)
    input = input.to('cuda')
    fr_monitor = monitor.OutputMonitor(model, neuron.LIFNode, cal_firing_rate)
    with torch.no_grad():
        fr_monitor.enable()
        model(input)
        #print(fr_monitor.records)
        res = 0
        for tensor in fr_monitor.records:
            res += torch.mean(tensor)
        res /= len(fr_monitor.records)
        print(res)
        functional.reset_net(model)
        #0.0379*0.9*6*