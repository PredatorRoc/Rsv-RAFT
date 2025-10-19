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


# �޸�ͨ������ı���
class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        # ѡ���һ����ʽ
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)
        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)
        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # �޸ľ���˵Ĵ�С����7x7��Ϊ5x5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2)  # ���Ӿ���˵Ĵ�С
        # self.relu1 = nn.ReLU(inplace=True)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)  # ʹ�� LeakyReLU  0.2-0.01

        # ��ʼ������ͨ����
        self.in_planes = 64

        # ���Ӿ�������ȣ�������ÿ�����ͨ����
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(128, stride=2)  # ����ͨ����
        self.layer3 = self._make_layer(256, stride=2)  # �ٴ�����ͨ����

        # �������㣬ͨ������256����Ϊoutput_dim
        self.conv2 = nn.Conv2d(256, output_dim, kernel_size=1)

        # Dropout�㣬���dropout > 0��ʹ��
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # Ȩ�س�ʼ��
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #ʹ�����µļ����
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
        # ����ÿ�������ResidualBlock
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)

        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # ��������Ϊ�б�ʱ������ƴ��Ϊһ��batch
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # ͨ���������һ���ͼ����
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)


        # �������ResidualBlock��
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # ͨ��1x1�����ͨ��������Ϊoutput_dim
        x = self.conv2(x)

        # �������ѵ������dropout����ڣ���Ӧ��dropout
        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # ����������б���������Ϊԭʼ��batch
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


# SmallEncoder ��̳��� nn.Module��ʹ�ø����Ϊһ�������� PyTorch ��ѵ����������������ģ�顣
# output_dim=128����ʾ���������������ͼ��ͨ������Ĭ���� 128��
# norm_fn='batch'��ָ��ʹ�õĹ�һ�������������� batch��group��instance �� none��
# dropout=0.0��ָ����������ʹ�õ� dropout ���ʣ�Ĭ��Ϊ 0������ʹ�� dropout��
class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        # ���� norm_fn �����Ĳ�ͬ��ѡ��ͬ�Ĺ�һ��������
        # GroupNorm��ʹ�����һ�����ֳ� 8 �顣
        # BatchNorm2d��ʹ������һ������ 32 ��ͨ�����й�һ����
        # InstanceNorm2d��ʹ��ʵ����һ����
        # 'none'����ʹ�ù�һ����ֱ��������

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)

        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()
        # self.conv1������㣬����ͨ����Ϊ 3�������� RGB ͼ�񣩣����ͨ����Ϊ 32������˴�СΪ 7x7������Ϊ 2�����Ϊ 3����������Ŀռ�ߴ磩��
        # self.relu1��ReLU �������inplace=True ��ʾ��ԭ���޸�������������ʡ�ڴ档
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # self.in_planes = 32����ʼ������ͨ����Ϊ 32��
        # self.layer1��self.layer2��self.layer3������ _make_layer ���������� 3 �㲻ͬ�ߴ�Ĳв�飨ResidualBlock����ÿһ������ͨ�����ֱ�Ϊ 32��64 �� 96
        self.in_planes = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        # self.dropout����� dropout �������� 0�����������м��� dropout �㣨���ڷ�ֹ����ϣ���p=dropout ָ�������ı���
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # self.conv2��һ������㣬���ڽ�ͨ������ 96 ת��Ϊ output_dim��Ĭ�� 128��
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        # �������еľ����͹�һ������г�ʼ����
        # nn.init.kaiming_normal_���Ծ�����Ȩ�ؽ��� He ��̬�ֲ���ʼ���������� ReLU �������
        # nn.init.constant_���Թ�һ�����Ȩ�س�ʼ��Ϊ 1��ƫ�ó�ʼ��Ϊ 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # _make_layer �������ڹ��������� BottleneckBlock ��ɵ�����㡣ÿ�� BottleneckBlock ��һ��������������͹�һ���Ĳв�飨�ο� BottleneckBlock �ࣩ��
    # ��һ����ʹ�ø����� stride���ڶ������ stride Ϊ 1�����ֿռ�ά�Ȳ��䡣
    # ���ص� layers ��һ���������� BottleneckBlock ��˳������ nn.Sequential
    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    # forward ���������������ǰ�򴫲�����
    # ������� x ��һ��Ԫ����б��������ڶ��ӽ����룩������ϲ���һ�����Ρ�
    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # ���ȣ�ͨ������� conv1��Ȼ��ʹ�ù�һ���� norm1�����ͨ�� ReLU ����� relu1��
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # ����������ͨ���������� layer1��layer2��layer3��
        # ���ͨ��һ�� 1x1 �ľ���㽫����ͼ��ͨ����ת��ΪĿ�����ͨ���� output_dim
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        # �����ѵ���׶��� dropout ��Ϊ 0����Ӧ�� dropout ��
        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # ��������Ƕ���ӽǵ�ƴ�ӣ����б��Ԫ�飩����������ά���ϲ�ֳ�ԭ������������
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        # ���ؾ������о������һ��������� dropout������еĻ�������������ͼ��
        return x



