#rsv-ultra
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes // 4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes // 4, planes // 4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes // 4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes // 4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)

        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes // 4)
            self.norm2 = nn.BatchNorm2d(planes // 4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)

        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes // 4)
            self.norm2 = nn.InstanceNorm2d(planes // 4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None

        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x + y)


# 修改通道数后的编码
class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        # 选择归一化方式
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # 修改卷积核的大小，从7x7改为5x5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)  # 增加卷积核的大小
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # 使用 LeakyReLU  0.2-0.01

        # 初始化输入通道数
        self.in_planes = 64

        # 增加卷积层的深度，并增加每个层的通道数
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)  # 增加通道数
        self.layer3 = self._make_layer(256, stride=2)  # 再次增加通道数

        # 输出卷积层，通道数由256调整为output_dim
        self.conv2 = nn.Conv2d(256, output_dim, kernel_size=1)

        # Dropout层，如果dropout > 0则使用
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #使用了新的激活函数
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu',
                                        a=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        # 创建每层的两个ResidualBlock
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # 处理输入为列表时，将其拼接为一个batch
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # 通过卷积、归一化和激活函数
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)


        # 经过多个ResidualBlock层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 通过1x1卷积将通道数调整为output_dim
        x = self.conv2(x)

        # 如果启用训练并且dropout层存在，则应用dropout
        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # 如果输入是列表，则将输出拆分为原始的batch
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


# SmallEncoder 类继承自 nn.Module，使得该类成为一个可以在 PyTorch 中训练和评估的神经网络模块。
# output_dim=128：表示编码器的输出特征图的通道数，默认是 128。
# norm_fn='batch'：指定使用的归一化方法，可以是 batch、group、instance 或 none。
# dropout=0.0：指定在网络中使用的 dropout 比率，默认为 0，即不使用 dropout。
class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        # 根据 norm_fn 参数的不同，选择不同的归一化方法：
        # GroupNorm：使用组归一化，分成 8 组。
        # BatchNorm2d：使用批归一化，对 32 个通道进行归一化。
        # InstanceNorm2d：使用实例归一化。
        # 'none'：不使用归一化，直接跳过。

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        # self.conv1：卷积层，输入通道数为 3（假设是 RGB 图像），输出通道数为 32，卷积核大小为 7x7，步长为 2，填充为 3（保持输出的空间尺寸）。
        # self.relu1：ReLU 激活函数，inplace=True 表示在原地修改输入张量，节省内存。
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # self.in_planes = 32：初始化输入通道数为 32。
        # self.layer1、self.layer2、self.layer3：调用 _make_layer 函数，构建 3 层不同尺寸的残差块（ResidualBlock），每一层的输出通道数分别为 32、64 和 96
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        # self.dropout：如果 dropout 参数大于 0，则在网络中加入 dropout 层（用于防止过拟合），p=dropout 指定丢弃的比例
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # self.conv2：一个卷积层，用于将通道数从 96 转换为 output_dim（默认 128）
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        # 对网络中的卷积层和归一化层进行初始化：
        # nn.init.kaiming_normal_：对卷积层的权重进行 He 正态分布初始化，适用于 ReLU 激活函数。
        # nn.init.constant_：对归一化层的权重初始化为 1，偏置初始化为 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # _make_layer 函数用于构建由两个 BottleneckBlock 组成的网络层。每个 BottleneckBlock 是一个包含多个卷积层和归一化的残差块（参考 BottleneckBlock 类）。
    # 第一个块使用给定的 stride，第二个块的 stride 为 1，保持空间维度不变。
    # 返回的 layers 是一个包含两个 BottleneckBlock 的顺序容器 nn.Sequential
    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    # forward 方法定义了网络的前向传播过程
    # 如果输入 x 是一个元组或列表（可能用于多视角输入），则将其合并成一个批次。
    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # 首先，通过卷积层 conv1，然后使用归一化层 norm1，最后通过 ReLU 激活函数 relu1。
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # 将输入数据通过三层网络 layer1、layer2、layer3。
        # 最后通过一个 1x1 的卷积层将特征图的通道数转换为目标输出通道数 output_dim
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        # 如果在训练阶段且 dropout 不为 0，则应用 dropout 层
        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # 如果输入是多个视角的拼接（即列表或元组），则在批次维度上拆分成原来的两个部分
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        # 返回经过所有卷积、归一化、激活和 dropout（如果有的话）处理后的特征图。
        return x



