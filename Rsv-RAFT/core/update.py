#rsv-ultra
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.ops import deform_conv2d  # 可变形卷积核心计算函数

#New FlowHead
class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        # 使用深度可分离卷积：内部已包含 SEBlock
        self.conv1 = DepthwiseSeparableConv(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(hidden_dim, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

#original
# class BasicUpdateBlock(nn.Module):
#     def __init__(self, args, hidden_dim=128, input_dim=128):
#         super(BasicUpdateBlock, self).__init__()
#         self.args = args
#         self.encoder = BasicMotionEncoder(args)
#         self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
#         self.flow_head = FlowHead(hidden_dim, hidden_dim=256)
#
#         self.mask = nn.Sequential(
#             nn.Conv2d(128, 256, 3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 64*9, 1, padding=0))
#
#     def forward(self, net, inp, corr, flow, upsample=True):
#         motion_features = self.encoder(flow, corr)
#         inp = torch.cat([inp, motion_features], dim=1)
#
#         net = self.gru(net, inp)
#         delta_flow = self.flow_head(net)
#
#         # scale mask to balence gradients
#         mask = .25 * self.mask(net)
#         return net, mask, delta_flow

#add DCNV+CoordAttention
class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        # 原mask生成器替换为改进版
        self.mask = nn.Sequential(
            DeformConvBlock(128, 256),  # 可变形卷积
            CoordAttention(256),        # 坐标注意力
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0)
        )
    def forward(self, net, inp, corr, flow, upsample=True):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balance gradients
        mask = .25 * self.mask(net)
        return net, mask, delta_flow



class CoordAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction

        # 坐标编码分支
        self.x_conv = nn.Conv2d(1, channels, kernel_size=7, padding=3, bias=False)
        self.y_conv = nn.Conv2d(1, channels, kernel_size=7, padding=3, bias=False)

        # 通道注意力分支
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

        # 空间注意力分支
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape

        # ----------------------------
        # 坐标编码
        # ----------------------------
        x_coord = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        y_coord = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)

        x_feat = self.x_conv(x_coord)  # (B,C,H,W)
        y_feat = self.y_conv(y_coord)  # (B,C,H,W)

        # ----------------------------
        # 通道注意力
        # ----------------------------
        channel_att = self.avg_pool(x + x_feat + y_feat).view(B, C)
        channel_att = self.fc(channel_att).view(B, C, 1, 1)  # (B,C,1,1)

        # ----------------------------
        # 空间注意力
        # ----------------------------
        spatial_avg = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        spatial_att = torch.cat([spatial_avg, spatial_max], dim=1)  # (B,2,H,W)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_att))  # (B,1,H,W)

        return x * channel_att * spatial_att


class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 偏移量预测分支（输出2*3*3=18通道，对应3x3卷积核的偏移量）
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)

        # 可变形卷积主体
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # 生成偏移量
        offset = self.offset_conv(x)  # (B,18,H,W)

        # 应用可变形卷积
        return deform_conv2d(
            input=x,
            offset=offset,
            weight=self.conv.weight,
            bias=self.conv.bias,
            padding=(1, 1))

#SE 模块（Squeeze-and-Excitation） 可以补偿跨通道信息损失：
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        reduced_channels = max(in_channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.fc(x)
        return x * scale

#可分离卷积
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels)  # 添加 SE 模块

    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.se(x)  # SE 重新标定通道权重
        return x

