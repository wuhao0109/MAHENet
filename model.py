import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from timm import create_model

#############################################
# Auxiliary module
#############################################
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        rates = [1, 6, 12, 18]
        self.convs = nn.ModuleList([
            ConvBNReLU(in_ch, out_ch, 1, padding=0),
            ConvBNReLU(in_ch, out_ch, 3, padding=r, dilation=r) for r in rates[1:]
        ])
        self.project = ConvBNReLU(len(rates) * out_ch, out_ch, 1, padding=0)

    def forward(self, x):
        out = [conv(x) for conv in self.convs]
        return self.project(torch.cat(out, dim=1))

class SEBlock(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_ch, in_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_ch // reduction, in_ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#############################################
# Step 1: Semantic-Texture Hybrid Encoder
#############################################
class TextureEncoder(nn.Module):
    def __init__(self, in_ch=3, mid_ch=64):
        super().__init__()
        self.layer1 = nn.Sequential(
            ConvBNReLU(in_ch, mid_ch),
            ConvBNReLU(mid_ch, mid_ch)
        )
        self.layer2 = ASPP(mid_ch, mid_ch)
        self.down = nn.Sequential(
            nn.AvgPool2d(2),
            ASPP(mid_ch, mid_ch)
        )

    def forward(self, x):
        x1 = self.layer1(x)  # 1/2
        x2 = self.layer2(x1)
        x3 = self.down(x2)  # 1/4
        return x1, x3

class SemanticEncoder(nn.Module):
    def __init__(self, backbone='swin_tiny_patch4_window7_224'):
        super().__init__()
        self.backbone = create_model(backbone, pretrained=True, features_only=True)

    def forward(self, x):
        feats = self.backbone(x)  # S2(1/4), S3(1/8), S4(1/16), S5(1/32)
        return feats

#############################################
# Step 2: Material-aware attention module
#############################################
class GaborAttention(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.gabor_conv = ConvBNReLU(in_ch, out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        x = self.gabor_conv(x)
        x = self.se(x)
        return x

#############################################
# Step 3: Multi-scale texture fusion module
#############################################
class FusionModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ppm = ASPP(in_ch, out_ch)
        self.fpn_conv = nn.ModuleList([
            ConvBNReLU(out_ch, out_ch) for _ in range(3)
        ])

    def forward(self, features):
        # features: S5, S4, S3, F2 (1/32 to 1/4)
        p5 = self.ppm(features[0])
        p4 = self.fpn_conv[0](F.interpolate(p5, scale_factor=2) + features[1])
        p3 = self.fpn_conv[1](F.interpolate(p4, scale_factor=2) + features[2])
        p2 = self.fpn_conv[2](F.interpolate(p3, scale_factor=2) + features[3])
        return p2

#############################################
# Step 4: Edge Enhancement Decoder
#############################################
class Decoder(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, in_ch)
        self.conv2 = ConvBNReLU(in_ch, in_ch // 2)
        self.out_conv = nn.Conv2d(in_ch // 2, num_classes, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.out_conv(x)
        return x

#############################################
# Overall Model
#############################################
class MaterialSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.texture_encoder = TextureEncoder()
        self.semantic_encoder = SemanticEncoder()
        self.gabor_attention = GaborAttention(64, 64)
        self.fusion = FusionModule(768, 64)
        self.decoder = Decoder(64, num_classes)

    def forward(self, x):
        T1a, T2a = self.texture_encoder(x)
        S2, S3, S4, S5 = self.semantic_encoder(x)
        TG2 = self.gabor_attention(T2a)

        F2 = S2 + TG2
        p2 = self.fusion([S5, S4, S3, F2])
        out = self.decoder(p2)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        return out
