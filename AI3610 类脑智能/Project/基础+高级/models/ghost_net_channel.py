import torch
import torch.nn as nn
import math
from spikingjelly.activation_based import neuron, functional, surrogate, layer, monitor

__all__ = ['ghost_net']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = layer.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                layer.Linear(channel, channel // reduction),
                neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
                layer.Linear(channel // reduction, channel),        )
        functional.set_step_mode(self, step_mode='m')
        
    def forward(self, x):
        t, b, c, _, _ = x.size()
        y = self.avg_pool(x).view(t, b, c)
        y = self.fc(y).view(t, b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


def depthwise_conv(inp, oup, kernel_size=3, stride=2, relu=False):
    return nn.Sequential(
        layer.Conv2d(inp, oup, kernel_size, stride, kernel_size//2, groups=inp, bias=False),
        layer.BatchNorm2d(oup),
        #neuron.IFNode(surrogate_function=surrogate.ATan()) if relu else nn.Sequential(),
    )


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, fmap_order=None):
        super().__init__()
        self.fmap_order = fmap_order
        self.oup = oup
        init_channels = int(math.ceil(oup / ratio))
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            layer.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            layer.BatchNorm2d(init_channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            layer.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            layer.BatchNorm2d(new_channels),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True) if relu else nn.Sequential(),
        )
        functional.set_step_mode(self, step_mode='m')
        
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=2)
        return out[:, :, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            depthwise_conv(hidden_dim, hidden_dim, kernel_size, stride, relu=False) if stride==2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
            SELayer(oup) if use_se else nn.Sequential(),    
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                depthwise_conv(inp, inp, 3, stride, relu=False),
                layer.Conv2d(inp, oup, 1, 1, 0, bias=False),
                layer.BatchNorm2d(oup),
                #neuron.LIFNode(surrogate_function=surrogate.ATan()),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostNet(nn.Module):
    def __init__(self, cfgs, num_classes=10, width_mult=1., T=6):
        super(GhostNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        self.T = T
        # building first layer
        output_channel = _make_divisible(16 * width_mult, 4)
        layers = [nn.Sequential(
            layer.Conv2d(3, output_channel, 3, 2, 1, bias=False),
            layer.BatchNorm2d(output_channel),
            neuron.LIFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
        )]
        input_channel = output_channel

        # building inverted residual blocks
        block = GhostBottleneck
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4)
            hidden_channel = _make_divisible(exp_size * width_mult, 4)
            layers.append(block(input_channel, hidden_channel, output_channel, k, s, use_se))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(exp_size * width_mult, 4)
        self.squeeze = nn.Sequential(
            layer.AdaptiveAvgPool2d((1, 1)),
            layer.Conv2d(input_channel, output_channel, 1, 1, 0, bias=False),
            #neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
        )
        input_channel = output_channel

        output_channel = 1024
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(input_channel, output_channel, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
            layer.Dropout(0.2),
            layer.Linear(output_channel, num_classes),
        )

        #self._initialize_weights()
        functional.set_step_mode(self, step_mode='m')
        
    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x_seq = self.features(x_seq)
        x_seq = self.squeeze(x_seq)
        out = self.classifier(x_seq)
        out = out.mean(0)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, layer.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def spiking_encoder(self):
        return self.features[0:3]


def ghost_net_channelMLP(**kwargs):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # kernelsize, t = ecp, c=out
        # k, t, c, SE, s 
        [3,  16,  16, 0, 1],    # conv1
        [3,  48,  24, 0, 2],    #

        [3,  72,  24, 0, 1],
        [5,  72,  40, 1, 2],

        [5, 120,  40, 1, 1],
        [3, 240,  80, 0, 2],

        [3, 200,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 184,  80, 0, 1],
        [3, 480, 112, 1, 1],
        [3, 672, 112, 1, 1],
        [5, 672, 160, 1, 2],

        [5, 960, 160, 1, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 1, 1],
        [5, 960, 160, 1, 1]
    ]
    return GhostNet(cfgs, **kwargs)

def cal_firing_rate(s_seq: torch.Tensor):
    return s_seq.flatten(1).mean(1)

if __name__ == '__main__':
    model = ghost_net_channelMLP()
    model = model.to('cuda')
    input = torch.randn(32, 3, 32, 32)
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
        #Firing Rate = 0.03
        #FLOPs = 141M (Paper)
        #SNN Energy = E * T * R * FLOPs = 0.9pJ * 6 * 0.03 * 141M = 22.84muJ
        #ANN Energy = E * FLOPs = 4.6pJ * 141M = 648 muj
