#import attention
#from layer import *
import torch
from torch import nn
from timm.models.layers import DropPath
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Compress(nn.Module):
    def forward(self, x):
        x=torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )
        return x
class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    import torch
    def __init__(self, in_channels, out_channels,k):
        super().__init__()

        self.conv = nn.Sequential(
            # nn.Conv2d(in_channels, out_channels, kernel_size=1),

            nn.Conv2d(in_channels, out_channels, kernel_size=k, padding=k//2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DWConv(nn.Module):
    def __init__(self, in_ch,out_ch,k):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,k, padding=k//2,groups=out_ch),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class CSPF(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool2d(1)
        self.conv1 = Conv(in_channels, in_channels, 1)
        self.conv2 = Conv(in_channels, in_channels, 1)
        self.conv3 = Conv(in_channels*2, in_channels*2, 1)

    def forward(self, x0,y0):
        x = self.gap1(x0)
        y = self.gap2(y0)
        x1, x2 = torch.chunk(x0, 2, dim=1)
        y1, y2 = torch.chunk(y0, 2, dim=1)
        z1 = self.conv1(torch.cat([x1, y1],dim=1))
        z2 = self.conv2(torch.cat([x2, y2], dim=1))
        z1 = z1 * x
        z2 = z2 * y
        z = self.conv3(torch.cat([z1,z2],dim=1))
        return z

class CFGC(nn.Module):
   """(convolution => [BN] => ReLU) * 2"""
   def  __init__(self, in_channels, out_channels):
       super().__init__()
       self.conv1 = DWConv(in_channels , in_channels, 3)
       self.conv2 = DWConv(in_channels , in_channels, 7)
       self.H1 = Compress()
       self.W1 = Compress()
       self.H2 = Compress()
       self.W2 = Compress()
       self.conv4 = Conv(2, 1, 3)
       self.conv5 = Conv(2, 1, 3)
       self.conv7 = Conv(in_channels * 2 , out_channels , 1)
       self.conv8 = Conv(out_channels, out_channels, 3)

   def forward(self, x):
       # print(x.shape)
       x1 = self.conv1(x)
       x2 = self.conv2(x)
       xh1 = x1.permute(0, 2, 1, 3).contiguous()
       xh1 = self.H1(xh1)
       # print(xh1.shape)
       xw1 = x1.permute(0, 3, 2, 1).contiguous()
       xw1 = self.W1(xw1)
       # print(xw1.shape)
       xh2 = x2.permute(0, 2, 1, 3).contiguous()
       xh2 = self.H2(xh2)
       # print(xh2.shape)
       xw2 = x2.permute(0, 3, 2, 1).contiguous()
       xw2 = self.W2(xw2)
       # print(xw2.shape)
       xh = torch.cat([xh1, xh2], dim=2)
       # print(xh.shape)
       xh = self.conv4(xh)
       # print(xh.shape)
       xw = torch.cat([xw1,xw2], dim=3)
       # print(xw.shape)
       xw = self.conv5(xw)
       # print(xw.shape)
       xp = xw @ xh
       # print(xp.shape)
       xs = torch.sigmoid(xp)

       x5 = self.conv7(torch.cat([x1,x2],dim=1))
       x5 = x5 * xs
       x6 = self.conv8(x5)
       return x6

class XUNet(nn.Module):
    def __init__(self,  num_classes=1, input_channels=3, deep_supervision=False,):
        super().__init__()
        cn = [32, 64, 128, 256, 512]
        self.conv1 = Conv(input_channels, cn[0],3)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = CFGC(cn[0], cn[1])
        self.mp2 = nn.MaxPool2d(2)
        self.conv3 = CFGC(cn[1], cn[2])
        self.mp3 = nn.MaxPool2d(2)
        self.conv4 = CFGC(cn[2], cn[3])
        self.mp4 = nn.MaxPool2d(2)
        self.conv5 = CFGC(cn[3], cn[4])


        self.sa1 = CSPF(cn[0])
        self.sa2 = CSPF(cn[1])
        self.sa3 = CSPF(cn[2])
        self.sa4 = CSPF(cn[3])

        self.up1 = nn.ConvTranspose2d(cn[4], cn[3], kernel_size=2, stride=2)
        self.conv6 = CFGC(cn[4], cn[3])

        self.up2 = nn.ConvTranspose2d(cn[3], cn[2], kernel_size=2, stride=2)
        self.conv7 = CFGC(cn[3], cn[2])

        self.up3 = nn.ConvTranspose2d(cn[2], cn[1], kernel_size=2, stride=2)
        self.conv8 = CFGC(cn[2], cn[1])

        self.up4 = nn.ConvTranspose2d(cn[1], cn[0], kernel_size=2, stride=2)
        self.conv9 = CFGC(cn[1], cn[0])

        self.last_conv = nn.Conv2d(cn[0],num_classes,1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.mp1(x1))
        x3 = self.conv3(self.mp2(x2))
        x4 = self.conv4(self.mp3(x3))
        x5 = self.conv5(self.mp4(x4))


        x5 = self.up1(x5)
        x4 = self.sa4(x5, x4)
        x6 = self.conv6(x4)
        x6 = self.up2(x6)
        x3 = self.sa3(x6, x3)
        x7 = self.conv7(x3)

        x7 = self.up3(x7)
        x2 = self.sa2(x7, x2)
        x8 = self.conv8(x2)

        x8 = self.up4(x8)
        x1 = self.sa1(x8, x1)
        x9 = self.conv9(x1)
        x9 = self.last_conv(x9)
        return  x9

if __name__ == '__main__':
    from ptflops import get_model_complexity_info

    model = XUNet(1)

    # net.load_from(weights=np.load(config_vit.pretrained_path))
    # model = ViT_seg(num_classes=1).to('cuda')

    flops, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True, print_per_layer_stat=True)
    print('flops:',flops)
    print('params:',params)
