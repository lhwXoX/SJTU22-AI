import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from spikingjelly import visualizing
from torch.autograd import Variable
from spikingjelly.activation_based.model import spiking_resnet
__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, layer.Linear) or isinstance(m, layer.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.conv1 = layer.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = layer.BatchNorm2d(planes)
        self.conv2 = layer.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(planes)
        self.IFNode = neuron.LIFNode(tau = 2.0, surrogate_function=surrogate.ATan(), detach_reset=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     layer.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     layer.BatchNorm2d(self.expansion * planes)
                )
                
    def forward(self, x: torch.Tensor):
        out = self.bn1(self.conv1(x))
        out = self.IFNode(out)
        out = self.bn2(self.conv2(out))
        out = self.IFNode(out)
        #print('out:', out.shape)
        #print('x:', x.shape)
        out += self.shortcut(x)
        #out += x
        #out = self.IFNode(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, T=6, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.T = T
        self.primary = nn.Sequential(
        layer.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
        layer.BatchNorm2d(16),
        neuron.LIFNode(tau=2.0 ,detach_reset=True, surrogate_function=surrogate.ATan()),
        )
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.classifier = nn.Sequential(
        layer.AdaptiveAvgPool2d(1),
        layer.Flatten(start_dim=1, end_dim=-1),
        layer.Linear(64, num_classes, bias=True),
        )
        #self.AvgPool = layer.AdaptiveAvgPool2d(output_size=(1, 1))
        #self.apply(_weights_init)
        functional.set_step_mode(self, step_mode='m')
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1) 
        out = self.primary(x_seq)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #out = self.AvgPool(out)
        out = self.classifier(out)
        out = out.mean(0)
        return out
    def spiking_encoder(self):
        return self.primary[0:3]

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])

def cal_firing_rate(s_seq: torch.Tensor):
    return s_seq.flatten(1).mean(1)

if __name__ == '__main__':
    from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor
    model = resnet32()
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
#FLOPs: 72.5M
#Firing Rate: 0.0378
#Energy: E * T * R * FLOPs = 0.9 * 6 * 0.0378 * 72.5M = 14.80 muJ
#ANN Energy: E * FLOPs = 5.4 * 72.5M = 333.5muj