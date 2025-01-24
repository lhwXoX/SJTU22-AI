## Project：脉冲神经网络

### 中级部分
* [output](./中级部分/output): 结果图(报告中均已呈现)
* [main.py](./中级部分/main.py): 将输出路径改为目标路径后可直接运行，可以更改**115**行选择Similarity1/Similarity2/二者的加权和作为目标相似度损失:

```python
snn_penalty = (snn_penalty1 * 0.3 + snn_penalty2 * 0.7)
```

### 基础+高级部分
- **bin**
    - [01_parse_cifar10_to_png.py](./基础+高级/bin/01_parse_cifar10_to_png.py): 用于数据准备，首先在[http://www.cs.toronto.edu/~kriz/cifar.html](http://www.cs.toronto.edu/~kriz/cifar.html)下载得到`cifar-10-python.tar.gz`，解压到[data](./基础+高级/data)文件夹下，最后直接运行[01_parse_cifar10_to_png.py](./基础+高级/bin/01_parse_cifar10_to_png.py)即可
    - [02_main.py](./基础+高级/bin/02_main.py): 修改输出路径后，可直接运行在CIFAR-10数据集上，可以修改**73-78**行更改使用的网络结构
    - [02_main_mnist.py](./基础+高级/bin/02_main_mnist.py): 修改输出路径并更改指定网络输入通道数为1后，可直接运行在MNIST数据集上，可以修改**89-94**行更改使用的网络结构
    - [03_compute_flops.py](./基础+高级/bin/03_compute_flops.py): 可以修改**23-29**行计算指定网络的参数、FLOPs，但由于本人为了适应Spiking jelly的库而修改了计算的库中的源码可能导致无法运行或算出的值不正确。

```python
#以下六种网络结构可供选择，2-6网络的T默认为6
1. model = VGG("VGG16", T=6)
2. model = resnet32()
3. model = resnet32_ms()
4. model = ghost_net()
5. model = ghost_net_dfc()
6. model = ghost_net_channelMLP()
```
- **config**
    - [config.py](./基础+高级/config/config.py): 存放部分训练设置
- **data**
    用于存放数据集
- **models**
    - [ghost_net.py](./基础+高级/models/ghost_net.py): Spiking Ghost Net架构
    - [ghost_net_dfc.py](./基础+高级/models/ghost_net_channel.py): Ghost Net + DFC架构
    - [ghost_net_channel.py](./基础+高级/models/ghost_net_dfc.py): Ghost Net + Channel MLP架构
    - [resnet_snn.py](./基础+高级/models/resnet_snn.py): SEW-ResNet的多种架构
    - [resnet_snn_ms.py](./基础+高级/models/resnet_snn_ms.py): MS-ResNet的多种架构
    - [vgg.py](./基础+高级/models/vgg.py): Spiking VGG的多种架构
    - [visualizing.py](./基础+高级/models/visualizing.py): 修改checkpoint路径并选择对应模型可将第一层的输出可视化
- **results**
    存放6个网络对应的训练结果图，可视化结果，模型的`checkpoint_best.pkl`可以通过[交大jbox](https://jbox.sjtu.edu.cn/l/o1sI6X)下载
- **tools**
    一些被调用的python函数
### 报告
- **[Report.pdf](./Report/Report.pdf)** 
包含创新点和分工等
### PPT
- **[类脑智能课堂展示.pptx](./Report/Slides.pdf)**

