from collections import OrderedDict
from functools import partial
from typing import Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torchsummary import summary

def _make_divisible(ch, divisor=8, min_ch=None):
    '''
    int(ch + divisor / 2) // divisor * divisor)
    目的是为了让new_ch是divisor的整数倍
    类似于四舍五入:ch超过divisor的一半则加1保留;不满一半则归零舍弃
    '''
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

# 反残差结构随机失活:batchsize个样本随机失活,应用于反残差结构的主路径
class DropConnect(nn.Module):
    def __init__(self, drop_prob=0.5):
        super(DropConnect, self).__init__()
        self.keep_prob = None
        self.set_rate(drop_prob)

    # 反残差结构的保留率
    def set_rate(self, drop_prob):
        if not 0 <= drop_prob < 1:
            raise ValueError("rate must be 0<=rate<1, got {} instead".format(drop_prob))
        self.keep_prob = 1 - drop_prob

    def forward(self, x):
        # 训练阶段随机丢失特征
        if self.training:
            # 是否保留取决于固定保留概率+随机概率
            random_tensor = self.keep_prob + torch.rand([x.size(0), 1, 1, 1],
                                                        dtype=x.dtype,
                                                        device=x.device)
            # 0表示丢失 1表示保留
            binary_tensor = torch.floor(random_tensor)
            # self.keep_prob个人理解对保留特征进行强化,概率越低强化越明显
            return torch.mul(torch.div(x, self.keep_prob), binary_tensor)
        else:
            return x
        
# 卷积块:3×3/5×5卷积层+BN层+Swish激活函数(可选)
class ConvBNAct(nn.Sequential):
    def __init__(self,
                 in_planes,                 # 输入通道
                 out_planes,                # 输出通道
                 kernel_size=3,             # 卷积核大小
                 stride=1,                  # 卷积核步长
                 groups=1,                  # 卷积层组数
                 norm_layer=None,           # 归一化层
                 activation_layer=None):    # 激活层
        super(ConvBNAct, self).__init__()
        # 计算padding
        padding = (kernel_size - 1) // 2
        # BN层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Swish激活函数
        if activation_layer is None:
            # nn.SiLU 等价于  x * torch.sigmoid(x)
            activation_layer = nn.SiLU
        super(ConvBNAct, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                  out_channels=out_planes,
                                                  kernel_size=kernel_size,
                                                  stride=stride,
                                                  padding=padding,
                                                  groups=groups,
                                                  bias=False),
                                        norm_layer(out_planes),
                                        activation_layer())

# SE注意力模块:对各通道的特征分别强化
class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c,           # 1×1卷积输出通道数(降维)
                 expand_c,          # se模块整体输入输出通道
                 se_ratio=0.25):    # 降维系数
        super(SqueezeExcitation, self).__init__()
        # 降维通道数=降维卷积输出通道数*降维系数
        squeeze_c = int(input_c * se_ratio)
        # 1×1卷积(降维)
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        # Swish激活函数
        self.ac1 = nn.SiLU()
        # 1×1卷积(特征提取)
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        # Sigmoid激活函数(0~1重要性加权)
        self.ac2 = nn.Sigmoid()

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        # 特征按通道分别进行加权
        return scale * x

# 反残差结构MBConv:1×1膨胀卷积层+BN层+Swish激活函数+3×3深度卷积层+BN层+Swish激活函数+1×1点卷积层+BN层
class MBConvBlock(nn.Module):
    def __init__(self,
                 kernel,                # 卷积核大小 3
                 input_c,               # 输入通道数
                 out_c,                 # 输出通道数
                 expand_ratio,          # 膨胀系数 4 or 6
                 stride,                # 卷积核步长
                 se_ratio,              # 启用se注意力模块
                 drop_rate,             # 通道随机失活率
                 norm_layer):           # 归一化层
        super(MBConvBlock, self).__init__()
        # 膨胀通道数 = 输入通道数*膨胀系数
        expanded_c = input_c * expand_ratio
        # 步长必须是1或者2
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        # 深度卷积步长为2则没有shortcut连接
        self.use_res_connect = (stride == 1 and input_c == out_c)

        layers = OrderedDict()
        # Swish激活函数
        activation_layer = nn.SiLU  # alias Swish


        # 在EfficientNetV2中，MBConvBlock中不存在expansion=1的情况
        assert expand_ratio != 1
        # 1×1膨胀卷积(膨胀系数>1) 升维
        layers.update({"expand_conv": ConvBNAct(input_c,
                                                expanded_c,
                                                kernel_size=1,
                                                norm_layer=norm_layer,
                                                activation_layer=activation_layer)})
        # 3×3深度卷积
        layers.update({"dwconv": ConvBNAct(expanded_c,
                                           expanded_c,
                                           kernel_size=kernel,
                                           stride=stride,
                                           groups=expanded_c,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer)})

        # 启用se注意力模块
        if se_ratio > 0:
            layers.update({"se": SqueezeExcitation(input_c,
                                                   expanded_c,
                                                   se_ratio)})

        # 1×1点卷积
        layers.update({"project_conv": ConvBNAct(expanded_c,
                                                 out_c,
                                                 kernel_size=1,
                                                 norm_layer=norm_layer,
                                                 activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)

        # 只有在使用shortcut连接时才使用drop_connect层
        if self.use_res_connect and drop_rate > 0:
            self.drop_connect = DropConnect(drop_rate)
        else:
            self.drop_connect = nn.Identity()

    def forward(self, x):
        result = self.block(x)
        result = self.drop_connect(result)
        # 反残差结构随机失活
        if self.use_res_connect:
            result += x
        print('finish MBConv')
        return result

# 反残差结构FusedMBConv:3×3膨胀卷积层+BN层+Swish激活函数+1×1点卷积层+BN层
class FusedMBConvBlock(nn.Module):
    def __init__(self,
                 kernel,                # 卷积核大小
                 input_c,               # 输入通道数
                 out_c,                 # 输出通道数
                 expand_ratio,          # 膨胀系数 1 or 4
                 stride,                # 卷积核步长
                 se_ratio,              # 启用se注意力模块
                 drop_rate,             # 通道随机失活率
                 norm_layer):           # 归一化层
        super(FusedMBConvBlock, self).__init__()
        # 膨胀通道数 = 输入通道数*膨胀系数
        expanded_c = input_c * expand_ratio
        # 步长必须是1或者2
        assert stride in [1, 2]
        # 没有se注意力模块
        assert se_ratio == 0
        # 深度卷积步长为2则没有shortcut连接
        self.use_res_connect = stride == 1 and input_c == out_c
        self.drop_rate = drop_rate

        self.has_expansion = expand_ratio != 1

        layers = OrderedDict()
        # Swish激活函数
        activation_layer = nn.SiLU

        # 只有当expand ratio不等于1时才有膨胀卷积
        if self.has_expansion:
            # 3×3膨胀卷积(膨胀系数>1) 升维
            layers.update({"expand_conv": ConvBNAct(input_c,
                                                    expanded_c,
                                                    kernel_size=kernel,
                                                        norm_layer=norm_layer,
                                                    activation_layer=activation_layer)})
            # 1×1点卷积
            layers.update({"project_conv": ConvBNAct(expanded_c,
                                               out_c,
                                               kernel_size=1,
                                               norm_layer=norm_layer,
                                               activation_layer=nn.Identity)})  # 注意没有激活函数
        else:
            # 当没有膨胀卷积时,3×3点卷积
            layers.update({"project_conv": ConvBNAct(input_c,
                                                     out_c,
                                                     kernel_size=kernel,
                                                     norm_layer=norm_layer,
                                                     activation_layer=activation_layer)})  # 注意有激活函数
        self.block = nn.Sequential(layers)
        if self.use_res_connect and drop_rate > 0:
            self.drop_connect = DropConnect(drop_rate)
        # 只有在使用shortcut连接时才使用drop_connect层
        if self.use_res_connect and drop_rate > 0:
            self.drop_connect = DropConnect(drop_rate)
        else:
            self.drop_connect = nn.Identity()
    def forward(self, x):
        result = self.block(x)
        result = self.drop_connect(result)
        # 反残差结构随机失活
        if self.use_res_connect:
            result += x
        print('finish Fused')
        return result


class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf,                 # 配置参数
                 num_classes=1000,          # 输出类别
                 num_features=1280,
                 dropout_rate=0.2,          # 通道随机失活率
                 drop_connect_rate=0.2):    # 反残差结构随机失活概率
        super(EfficientNetV2, self).__init__()

        # 配置参数无误
        for cnf in model_cnf:
            assert len(cnf) == 8
        # 配置bn层参数
        norm_layer = partial(nn.BatchNorm2d, eps=1e-3, momentum=0.1)

        self.stem = ConvBNAct(in_planes=1,
                              out_planes=_make_divisible(model_cnf[0][4]),
                              kernel_size=3,
                              stride=2,
                              norm_layer=norm_layer)  # 激活函数默认是SiLU
        # 当前反残差结构序号
        block_id = 0
        # 反残差结构总数
        total_blocks = sum([i[0] for i in model_cnf])
        blocks = []
        for cnf in model_cnf:
            # 选择反残差结构 0是MBConvBlock 1是FusedMBConvBlock
            op = FusedMBConvBlock if cnf[-2] == 0 else MBConvBlock
            # 每个stage的反残差结构数数
            repeats = cnf[0]
            for i in range(repeats):
                # 反残差结构随机失活概率随着网络深度等差递增,公差为drop_connect_rate/total_blocks，范围在[0,drop_connect_rate)
                blocks.append(op(kernel=cnf[1],
                                 input_c=_make_divisible(cnf[4] if i == 0 else cnf[5]),
                                 out_c=_make_divisible(cnf[5]),
                                 expand_ratio=cnf[3],
                                 stride=cnf[2] if i == 0 else 1,
                                 se_ratio=cnf[-1],
                                 drop_rate=drop_connect_rate * block_id / total_blocks,
                                 norm_layer=norm_layer))
                block_id += 1
        head_input_c = _make_divisible(model_cnf[-1][-3])
        # 主干网络
        self.blocks = nn.Sequential(*blocks)
        
        head = OrderedDict()
        head.update({"project_conv": ConvBNAct(head_input_c,
                                               _make_divisible(num_features),
                                               kernel_size=1,
                                               norm_layer=norm_layer)})  # 激活函数默认是SiLU
        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p=dropout_rate, inplace=True)})
        head.update({"classifier": nn.Linear(_make_divisible(num_features), num_classes)})
        
        # 分类器
        self.head = nn.Sequential(head)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # 主干网络
        x = self.stem(x)
        x = self.blocks(x)
        
        return x

# 不同的网络模型对应不同的分辨率
def efficientnetv2_s(num_classes = 1000):
    # train_size: 300, eval_size: 384

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[1, 3, 1, 1, 24, 24, 0, 0],
                    [2, 3, 2, 4, 24, 48, 0, 0],
                    [2, 3, 2, 4, 48, 64, 0, 0],
                    [3, 3, 2, 4, 64, 128, 1, 0.25],
                    [3, 3, 1, 6, 128, 160, 1, 0.25],
                    [4, 3, 2, 6, 160, 256, 1, 0.25],
                    [1, 3, 2, 6, 256, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.2)
    return model

def efficientnetv2_m(num_classes = 1000):
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[1, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]



    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.3)
    return model

def efficientnetv2_l(num_classes = 1000):
    # train_size: 384, eval_size: 480

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model

def efficientnetv2_xl(num_classes = 1000):
    # train_size: 384, eval_size: 512

    # repeat, kernel, stride, expansion, in_c, out_c, operator, se_ratio
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [8, 3, 2, 4, 32, 64, 0, 0],
                    [8, 3, 2, 4, 64, 96, 0, 0],
                    [16, 3, 2, 4, 96, 192, 1, 0.25],
                    [24, 3, 1, 6, 192, 256, 1, 0.25],
                    [32, 3, 2, 6, 256, 512, 1, 0.25],
                    [8, 3, 1, 6, 512, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf=model_config,
                           num_classes=num_classes,
                           dropout_rate=0.4)
    return model

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = efficientnetv2_s().to(device)
    summary(model, input_size=(1, 300, 300))
