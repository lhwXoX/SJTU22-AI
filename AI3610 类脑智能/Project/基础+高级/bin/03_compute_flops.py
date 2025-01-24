import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from torchstat import stat
from models.ghost_net import GhostModule
from models.resnet_snn import resnet32
from models.vgg import VGG
from tools.common_tools import replace_conv
from models.ghost_net_dfc import ghost_net_dfc
from models.ghost_net import ghost_net
from models.ghost_net_channel import ghost_net_channelMLP
from spikingjelly.activation_based import neuron, functional, surrogate, layer
import torch
import torch.nn as nn
import math
from models.resnet_snn_ms import resnet32_ms

def cal_firing_rate(s_seq: torch.Tensor):
    return s_seq.flatten(1).mean(1)
if __name__ == '__main__':
    img_shape = (3, 32, 32)
    model = VGG("VGG16", T=6)
    #model = ghost_net_channelMLP()
    #model = resnet32_ms()
    #model = ghost_net_dfc()
    #model = resnet32()
    #model = ghost_net()
    stat(model, img_shape)
    # params, FLOPs





