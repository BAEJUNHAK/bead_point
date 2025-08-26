# DW-UNet-Tiny (Conv/BN/ReLU/Pool/Upsample/Concat)
import torch.nn as nn
import torch

class DWConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))

class DWUNetTiny(nn.Module):
    """ONNX-safe U-Net Tiny (≤ ~0.35M params @ base=24)"""
    def __init__(self, in_ch=3, base=24, out_ch=2):
        super().__init__()
        self.e1 = nn.Sequential(DWConvBNReLU(in_ch, base),   DWConvBNReLU(base, base))
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(DWConvBNReLU(base, base*2),  DWConvBNReLU(base*2, base*2))
        self.p2 = nn.MaxPool2d(2)
        self.e3 = nn.Sequential(DWConvBNReLU(base*2, base*3),DWConvBNReLU(base*3, base*3))
        self.p3 = nn.MaxPool2d(2)
        self.e4 = nn.Sequential(DWConvBNReLU(base*3, base*4),DWConvBNReLU(base*4, base*4))
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d3 = nn.Sequential(DWConvBNReLU(base*4+base*3, base*3), DWConvBNReLU(base*3, base*3))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d2 = nn.Sequential(DWConvBNReLU(base*3+base*2, base*2), DWConvBNReLU(base*2, base*2))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d1 = nn.Sequential(DWConvBNReLU(base*2+base, base),     DWConvBNReLU(base, base))
        self.head = nn.Conv2d(base, out_ch, 1)
    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.p1(e1)); e3 = self.e3(self.p2(e2)); e4 = self.e4(self.p3(e3))
        d3 = self.d3(torch.cat([self.up3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)
