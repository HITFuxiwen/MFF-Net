import torch.nn as nn
import torch
import torch.nn.functional as F
from t2t_vit import T2t_vit_t_14
from Transformer import Transformer
from Transformer import token_Transformer
from Decoder import Decoder
from Decoder3 import Decoder3
from Decoder4 import Decoder4
from cmt import cmt_s
import numpy as np
import cv2
import transforms 
from CPD_ResNet_models import CPD_ResNet
from DAhead import _DAHead
# from C_Resnet_model import CPD_ResNet

def dct2(image_Input):
    image_Input = image_Input.cpu().numpy()
    res = []
    for img in image_Input:
        channel = img
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)
        rows, cols, channels = channel.shape
        crow,ccol = rows//2 , cols//2
        fshift[crow-3:crow+4, ccol-3:ccol+4] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        new_img=np.array(img_back).astype(np.float32)
        res.append(np.array(new_img))
    return np.array(res)

class ImageDepthNet4(nn.Module):
    

    def __init__(self, args):
        super(ImageDepthNet4, self).__init__()


        # VST Encoder
        self.rgb_backbone = T2t_vit_t_14(pretrained=True, args=args)
        self.edge_backbone = cmt_s(pretrained=True, args=args)
        # VST Convertor
        self.transformer = Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)

        # VST Decoder
        self.token_trans = token_Transformer(embed_dim=384, depth=4, num_heads=6, mlp_ratio=3.)
        self.decoder = Decoder3()
        self.concatFuse = nn.Sequential(
                nn.Linear(384+256, 384),
                nn.GELU(),
                nn.Linear(384, 384),
            )
        self.concatFuse2 = nn.Sequential(
                nn.Linear(64+128, 128),
                nn.GELU(),
                nn.Linear(128, 128),
            )
        self.concatFuse3 = nn.Sequential(
                nn.Linear(64*2, 64),
                nn.GELU(),
                nn.Linear(64, 64),
            )


    def forward(self, image_Input):

        B, _, _, _ = image_Input.shape
        
        # VST Encoder
        rgb_fea_1_16, rgb_fea_1_8, rgb_fea_1_4 = self.rgb_backbone(image_Input)
        cmt_1_4, cmt_1_8, cmt_1_16 = self.edge_backbone(image_Input)
        #df[B, 28*28, 64]
        #at[B, 28*28, 1]
        #fs[B, 14*14, 384]
        # VST Convertor
        # print( rgb_fea_1_16.shape)
        rgb_fea_1_16 = self.concatFuse(torch.cat([rgb_fea_1_16, cmt_1_16], dim=2))
        rgb_fea_1_16 = self.transformer(rgb_fea_1_16)
        # rgb_fea_1_16 [B, 14*14, 384]

        # VST Decoder
        saliency_fea_1_16, fea_1_16, saliency_tokens, contour_fea_1_16, contour_tokens = self.token_trans(rgb_fea_1_16)
        # saliency_fea_1_16 [B, 14*14, 384]
        # fea_1_16 [B, 1 + 14*14 + 1, 384]
        # saliency_tokens [B, 1, 384]
        # contour_fea_1_16 [B, 14*14, 384]
        # contour_tokens [B, 1, 384]
        rgb_fea_1_8 = self.concatFuse2(torch.cat([rgb_fea_1_8, cmt_1_8], dim=2))

        rgb_fea_1_4 = self.concatFuse3(torch.cat([rgb_fea_1_4, cmt_1_4], dim=2))
        
        masks = self.decoder(saliency_fea_1_16, rgb_fea_1_8, rgb_fea_1_4)

        return masks