import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.activation_based import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from torch.cuda import amp
import sys
import datetime
from PIL import Image
import matplotlib.image as plimg
from spikingjelly import visualizing
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
from torch.utils.data import DataLoader
from tools.cifar10_dataset import CifarDataset
from tools.model_trainer import ModelTrainer
from tools.common_tools import *
from models.ghost_net_dfc import ghost_net_dfc
from models.ghost_net import ghost_net
from models.ghost_net_channel import ghost_net_channelMLP
from models.resnet_snn import resnet32
from models.resnet_snn_ms import resnet32_ms
from models.vgg import VGG
from config.config import cfg
if __name__ == '__main__':
    test_dir = os.path.join(BASE_DIR, "..", "data", "cifar10_test")
    path_checkpoint = "E:/Paper's code/GhostNet/ghostnet_cifar10/results/ghostnet_T=6/checkpoint_best.pkl"
    path_save = "E:/Paper's code/GhostNet/ghostnet_cifar10/results/ghostnet_T=6/Visualization"
    #os.makedirs(path_save,exist_ok=True)
    model = ghost_net()
    check_p = torch.load(path_checkpoint, map_location="cpu")
    pretrain_dict = check_p["model_state_dict"]
    pretrain_dict_opti = check_p["optimizer_state_dict"]
    state_dict_cpu = state_dict_to_cpu(pretrain_dict)
    state_dict_cpu_opti = state_dict_to_cpu(pretrain_dict_opti)
    model.load_state_dict(state_dict_cpu)
    print("load model done")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    valid_data = CifarDataset(data_dir=test_dir, transform=cfg.transforms_valid)
    valid_data2 = CifarDataset(data_dir=test_dir, transform=cfg.transforms_valid2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4, num_workers=cfg.workers)
    valid_loader2 = DataLoader(dataset=valid_data2, batch_size=4, num_workers=cfg.workers)
    encoder = model.spiking_encoder()
    with torch.no_grad():
        print('done')
        for (i, data), (j, data2) in zip(enumerate(valid_loader), enumerate(valid_loader2)):
            print('start')
            img, label = data
            img, label = img.to(device), label.to(device)
            img2, label2 = data2
            img2, label2 = img2.to(device), label2.to(device)
            print('done1')
            img_seq = img.unsqueeze(0).repeat(model.T, 1, 1, 1, 1)
            spike_seq = encoder(img_seq)
            functional.reset_net(encoder)
            to_pil_img = torchvision.transforms.ToPILImage()
            os.mkdir(path_save)
            img2 = img2.cpu()
            spike_seq = spike_seq.cpu()
            img2 = F.interpolate(img2, scale_factor=4, mode='bilinear')
            for i in range(label.shape[0]):
                path_save_i = os.path.join(path_save, f"{i}")
                os.mkdir(path_save_i)
                to_pil_img(img2[i]).save(os.path.join(path_save_i,f'input.png'))
                for t in range(model.T):
                    print(f'saving {i}-th sample with t={t}...')
                    visualizing.plot_2d_feature_map(spike_seq[t][i], 8, spike_seq.shape[2] // 8, 2, f'$S[{t}]$')
                    plt.savefig(os.path.join(path_save_i, f's_{t}.png'))
                    plt.clf()
            exit()