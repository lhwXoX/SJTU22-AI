import torchvision.transforms as transforms
from easydict import EasyDict
from datetime import datetime

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')

class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cfg = EasyDict()

cfg.cls_names = class_names
cfg.class_num = len(cfg.cls_names)

# batch size
cfg.train_bs = 128
cfg.valid_bs = 128
cfg.workers = 8

# 学习率下降
cfg.patience = 5
cfg.factor = 0.1
cfg.milestones = [92, 136]
cfg.weight_decay = 1e-4
cfg.momentum = 0.9

cfg.log_interval = 10

# 数据预处理设置
norm_mean = [0.4914, 0.4822, 0.4465]
norm_std = [0.2023, 0.1994, 0.2010]

normTransform = transforms.Normalize(norm_mean, norm_std)
cfg.transforms_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    normTransform
])

cfg.transforms_valid = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    normTransform
])

cfg.transforms_valid2 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])







