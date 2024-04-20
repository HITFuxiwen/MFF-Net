import torch.nn as nn
import torch
import torch.nn.functional as F
from t2t_vit import T2t_vit_t_14
from Transformer import Transformer
from Transformer import token_Transformer
from Decoder4 import Decoder4
from Decoder5 import Decoder5
from DecoderC import DecoderC
import numpy as np
import cv2
from cmt import cmt_s
import transforms 
from pvtv2 import pvt_v2_b2
from torch import Tensor
# from C_Resnet_model import CPD_ResNet

class ImageDepthNet6(nn.Module):
    

    def __init__(self, args):
        super(ImageDepthNet6, self).__init__()


        # VST Encoder
        self.rgb_backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.rgb_backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        
        self.rgb_backbone.load_state_dict(model_dict)

        self.edge_backbone = cmt_s(pretrained=True, args=args)
        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = DecoderC()
        self.concatFuse = nn.Sequential(
                nn.Linear(320+256, 384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
        self.concatFuse2 = nn.Sequential(
                nn.Linear(128+128, 128),
                nn.GELU(),
                nn.Linear(128, 128),
            )
        self.concatFuse3 = nn.Sequential(
                nn.Linear(64*2, 64),
                nn.GELU(),
                nn.Linear(64, 64),
            )

        self.SWSAM1 = SWSAM(128)
        self.dirConv1 = DirectionalConvUnit(128)

        self.SWSAM2 = SWSAM(64)
        self.dirConv2 = DirectionalConvUnit(64)

    def forward(self, image_Input):

        B, _, _, _ = image_Input.shape
        
        # VST Encoder
        rgb_fea_1_4, rgb_fea_1_8, rgb_fea_1_16, _ = self.rgb_backbone(image_Input)
        cmt_1_4, cmt_1_8, cmt_1_16 = self.edge_backbone(image_Input)
        #df[B, 28*28, 64]
        #at[B, 28*28, 1]
        #fs[B, 14*14, 384]
        # VST Convertor
        # print( rgb_fea_1_16.shape)
        rgb_fea_1_16 = self.concatFuse(torch.cat([rgb_fea_1_16.flatten(2).permute(0, 2, 1), cmt_1_16], dim=2))
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]
        B,C,H,W = rgb_fea_1_8.shape
        rgb_fea_1_8 = self.concatFuse2(torch.cat([rgb_fea_1_8.flatten(2).permute(0, 2, 1), cmt_1_8], dim=2)).permute(0, 2, 1).reshape(B, C, H, W)
        rgb_fea_1_8 = self.dirConv1(rgb_fea_1_8)
        rgb_fea_1_8 = self.SWSAM1(rgb_fea_1_8).flatten(2).permute(0, 2, 1)

        B,C,H,W = rgb_fea_1_4.shape
        rgb_fea_1_4 = self.concatFuse3(torch.cat([rgb_fea_1_4.flatten(2).permute(0, 2, 1), cmt_1_4], dim=2)).permute(0, 2, 1).reshape(B, C, H, W)
        rgb_fea_1_4 = self.dirConv2(rgb_fea_1_4)
        rgb_fea_1_4 = self.SWSAM2(rgb_fea_1_4).flatten(2).permute(0, 2, 1)

        
        masks = self.decoder(saliency_fea_1_16, rgb_fea_1_8, rgb_fea_1_4)

        return masks
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
        
class SWSAM(nn.Module):
    def __init__(self, channel=32): # group=8, branch=4, group x branch = channel
        super(SWSAM, self).__init__()

        self.SA1 = SpatialAttention()
        self.SA2 = SpatialAttention()
        self.SA3 = SpatialAttention()
        self.SA4 = SpatialAttention()
        self.weight = nn.Parameter(torch.ones(4, dtype=torch.float32), requires_grad=True)
        self.sa_fusion = nn.Sequential(BasicConv2d(1, 1, 3, padding=1),
                                       nn.Sigmoid()
                                       )

    def forward(self, x):
        x = channel_shuffle(x, 4)
        B,C,H,W = x.shape
        x1, x2, x3, x4 = torch.split(x, int(C/4), dim = 1)
        s1 = self.SA1(x1)
        s2 = self.SA1(x2)
        s3 = self.SA1(x3)
        s4 = self.SA1(x4)
        nor_weights = F.softmax(self.weight, dim=0)
        s_all = s1 * nor_weights[0] + s2 * nor_weights[1] + s3 * nor_weights[2] + s4 * nor_weights[3]
        x_out = self.sa_fusion(s_all) * x + x

        return x_out

class DirectionalConvUnit(nn.Module):
    def __init__(self, channel):
        super(DirectionalConvUnit, self).__init__()

        self.h_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))
        self.w_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # leading diagonal
        self.dia19_conv = nn.Conv2d(channel, channel // 4, (5, 1), padding=(2, 0))
        # reverse diagonal
        self.dia37_conv = nn.Conv2d(channel, channel // 4, (1, 5), padding=(0, 2))

    def forward(self, x):

        x1 = self.h_conv(x)
        x2 = self.w_conv(x)
        x3 = self.inv_h_transform(self.dia19_conv(self.h_transform(x)))
        x4 = self.inv_v_transform(self.dia37_conv(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)

        return x

    # Code from "CoANet- Connectivity Attention Network for Road Extraction From Satellite Imagery", and we modified the code
    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-2]]
        x = x.reshape(shape[0], shape[1], shape[2], shape[2]+shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[2], shape[3]+1)
        x = x[..., 0: shape[3]-shape[2]+1]
        return x.permute(0, 1, 3, 2)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x