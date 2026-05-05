import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.ratio = ratio
        activation = 'relu'
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)

        if (activation == 'leakyrelu'):
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif (activation == 'gelu'):
            self.activation = nn.GELU()
        elif (activation == 'relu6'):
            self.activation = nn.ReLU6(inplace=True)
        elif (activation == 'hardswish'):
            self.activation = nn.Hardswish(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)

        self.fc2 = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))
        # print(x.shape)
        max_pool_out = self.max_pool(x)

        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))
        out = avg_out + max_out
        return x*self.sigmoid(out)

class SPA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SPA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out1 = torch.cat([avg_out, max_out], dim=1)
        out1 = self.conv1(out1)
        return x*self.sigmoid(out1)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, dilation=1, match=True):
        super().__init__()
        if match:
            assert in_channels == out_channels, "in_channels and out_channels must match!"
            group = in_channels
        else:
            group = 1
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, groups=group, kernel_size=kernel_size, padding=pad, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.single_conv(x)

class Hoblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, ratio, pad, down=False):
        super().__init__()
        self.conv1 = SingleConv(in_channels=in_channels // 2, out_channels=out_channels//2, kernel_size=kernel_size, pad=pad, dilation=dilation)
        self.conv2 = SingleConv(in_channels=in_channels // 2, out_channels=out_channels//2, kernel_size=kernel_size, pad=pad, dilation=dilation)
        # self.conv3 = SingleConv(in_channels=out_channels // 2, out_channels=out_channels // 2, kernel=1, pad=0)
        self.down = down
        self.p = nn.MaxPool2d(2)
        self.cam0 = ChannelAttention(in_channels//2, ratio=ratio)
        self.cam1 = ChannelAttention(in_channels // 2, ratio=ratio)
        self.sam0 = SPA()
        self.sam1 = SPA()

    def forward(self, x):
        y = list(torch.chunk(x, 2, dim=1))
        res0, res1 = y[0], y[1]
        y0 = self.conv1(y[0])
        branch = self.conv2(y[1])
        y0 = self.sam0(self.cam0(y0+res0))
        branch = self.sam1(self.cam1(res1+branch))
        # branch = y[1]
        # branch2 = self.conv3(branch)
        y2 = torch.cat([y0, branch], dim=1)

        if self.down:
            y2 = self.p(y2)
        return y2, branch

class ChannelShuffle(nn.Module):
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.groups == 0, 'The number of channels must be divisible by the number of groups'

        channels_per_group = num_channels // self.groups

        # reshape
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # transpose
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batch_size, -1, height, width)

        return x

class HoEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, ratio, pad, shuffle=True):
        super(HoEncoder, self).__init__()
        self.channels = []
        self.channels.append(in_channels)
        self.channels.extend(out_channels)
        self.cs = ChannelShuffle(groups=2)
        self.shuffle = shuffle
        for i in range(1, len(self.channels)):
            setattr(self, f"c{i}", Hoblock(self.channels[i-1], self.channels[i], kernel_size, dilation, ratio, pad, down=True))

    def forward(self, x):
        shortcuts = []
        for i in range(1, len(self.channels)):
            x, fea = getattr(self, f"c{i}")(x)
            shortcuts.append(fea)
            if self.shuffle:
                x = self.cs(x)
        return x, shortcuts

class Up(nn.Module):
    """Upscaling"""

    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return x

class HoDecoder(nn.Module):
    def __init__(self, in_channels, depth, kernel_size, dilation):
        super(HoDecoder, self).__init__()
        self.depth = depth
        for i in range(1, depth+1):
            setattr(self, f"c{i}", SingleConv(in_channels=in_channels, out_channels=in_channels//2, kernel_size=kernel_size, dilation=dilation, pad=dilation*(kernel_size-1) // 2, match=False))

        self.up = Up()
    def forward(self, x, fea, deep_supervision):
        # x = self.up(x)
        assert len(fea) <= self.depth, 'The length of features exceeds the depth of encoder!'
        deep_out = []
        if len(fea) == self.depth:
            for i in range(1, self.depth+1):
                # y = torch.cat([x, fea[self.depth-i]], dim=1)
                y = self.up(x, fea[self.depth-i])
                x = getattr(self, f"c{i}")(y)
                if deep_supervision:
                    deep_out.append(x)
        else:
            mdepth = min([len(fea), self.depth])
            for i in range(1, mdepth+1):
                y = torch.cat([x, fea[self.depth-i]], dim=1)
                x = getattr(self, f"c{i}")(y)
                if deep_supervision:
                    deep_out.append(y)
            for j in range(mdepth+1, self.depth+1):
                y = torch.cat([x, fea[self.depth - j]], dim=1)
                x = getattr(self, f"c{j}")(y)
        if deep_supervision:
            return deep_out #[x].extend(deep_out)
        else:
            return [x]

class SimStem(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=2, ratio=2):
        super(SimStem, self).__init__()
        pad = dilation*(kernel_size-1) // 2
        self.input_layer = nn.Sequential(
                    nn.Conv2d(3, channels, kernel_size, padding=pad, dilation=dilation),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.depthwise = nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, dilation=dilation, groups=channels, padding=pad),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(channels))
        self.cam = ChannelAttention(channels, ratio=ratio)
        self.sam = SPA()

        # self.pointwise = nn.Sequential(
        #             nn.Conv2d(channels, channels, kernel_size=1),
        #             nn.ReLU(inplace=True),
        #             nn.BatchNorm2d(channels))
    def forward(self, x):
        x = self.input_layer(x)
        residual = x
        x = self.depthwise(x)
        x += residual
        # x = self.pointwise(x)
        x = self.sam(self.cam(x))
        return x

class ESEUNet(nn.Module):
    def __init__(self, img_channels=3, out_channels=2, dim=32, depth=4, kernel_size=3, dilation=2, ratio=2, pad=2, shuffle=True, deep_supervision=False, deep_out=1):
        super(ESEUNet, self).__init__()
        # self.stem = PFC(dim)
        self.stem= SimStem(dim, kernel_size, dilation, ratio)
        self.img_channels = img_channels
        self.out_channels = out_channels

        self.depth = depth
        self.encoder = HoEncoder(dim, self.depth*[dim], kernel_size, dilation, ratio, pad, shuffle=shuffle)
        # self.TransBlock = TransBlock(dim=dim * 9)
        self.bottleneck = SingleConv(dim, dim // 2, kernel_size, pad, dilation=dilation, match=False)
        self.decoder = HoDecoder(dim, self.depth, kernel_size, dilation)
        # self.decoder = HoAttDecoder(dim, self.depth)

        if self.depth >= 5:
            self.stride = 1
        else:
            self.stride = pow(2, 4-depth)

        # self.seg_head = nn.Conv2d(dim//2, self.out_channels, kernel_size=1)
        self.deep_supervision = deep_supervision
        self.deep_out = deep_out

        if self.deep_supervision:
            assert self.deep_out <= depth and self.deep_out >= 1
            for i in range(0, self.deep_out):
                setattr(self, f"seg_head{i+1}", nn.Conv2d(dim//2, self.out_channels, kernel_size=1))
        else:
            self.seg_head1 = nn.Conv2d(dim//2, self.out_channels, kernel_size=1)


    def forward(self, x):
        x = self.stem(x)
        x, fea = self.encoder(x)
        x = self.bottleneck(x)

        tmp_out = self.decoder(x, fea, deep_supervision=self.deep_supervision)
        if self.deep_supervision:
            tmp_out = tmp_out[::-1]
            out = []
            for i in range(0, self.deep_out):
                out.append(getattr(self, f"seg_head{i+1}")(tmp_out[i]))
            return out   # 深度监督时返回列表
        else:
            out = getattr(self, f"seg_head{1}")(tmp_out[-1])   # 直接返回张量
            return out

    def inference(self, x):
        x = self.stem(x)
        x, fea = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, fea, deep_supervision=False)
        out = self.seg_head1(x[0])
        return out

from thop import profile

if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256).cuda()
    model = ESEUNet(img_channels=3, out_channels=2, dim=16, depth=5, kernel_size=3, dilation=2, ratio=2, pad=2, shuffle=True, deep_supervision=True, deep_out=5).cuda()
    flops, params = profile(model, inputs=(input,))
    print(flops/1e9)
    print(params/1e6)