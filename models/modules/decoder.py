# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch
from torch import nn

from config import cfg


class ConvLayer(nn.Module):
    def __init__(self):
        super(ConvLayer, self).__init__()


        self.convT_d = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.convT1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.convT2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.convT3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(192, 192, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(192),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv3d(192, 1, kernel_size=3, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )

        self.image_global = torch.nn.Sequential(
            torch.nn.Conv3d(8, 64, kernel_size=1, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.common = torch.nn.Sequential(
            torch.nn.Conv3d(64, 64, kernel_size=1, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )

    # todo: 消融: 暂时只用FFM的值
    def forward(self, common_voxel, image_global_voxel):

        common_voxel = self.common(common_voxel)    # (-1, 64, 4, 4, 4)

        image_global_voxel = self.image_global(image_global_voxel) # (-1, 64, 8, 8, 8)

        voxel = self.convT1(common_voxel) # (-1, 64, 8, 8, 8)

        voxel = self.convT2(torch.cat([voxel, image_global_voxel], dim=1))
        d_voxel = self.convT_d(image_global_voxel)

        voxel = torch.cat([voxel, d_voxel], dim=1)
        voxel = self.convT3(voxel)

        voxel = self.conv3(voxel)

        return voxel

class UnFlatten(nn.Module):
    # NOTE: (size, x, x, x) are being computed manually as of now (this is based on output of encoder)
    def forward(self, input, size=128): # size=128
        return input.view(input.size(0), 64, 4, 4, 4)
        # return input.view(input.size(0), size, 2, 2, 2)

class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        self.convLayers = ConvLayer()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, common_voxel, image_global_voxel):


        voxel = self.convLayers(common_voxel, image_global_voxel)


        return voxel

if __name__ == "__main__":
    x = torch.randn(16, 64, 4, 4, 4)
    decoder = Decoder(cfg)
    output = decoder(x,x,x)
    print(output.shape)

# -*- coding: utf-8 -*-
import torch


class Decoder2(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder2, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer = torch.nn.Sequential(
            torch.nn.Conv3d(128, 64, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),

            torch.nn.Conv3d(64, 64, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),

            torch.nn.Conv3d(64, 64, kernel_size=1, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE),

            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),

            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),

            torch.nn.Conv3d(64, 1, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU(),

            torch.nn.Conv3d(64, 64, kernel_size=1, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(8, 64, kernel_size=1, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )

    def forward(self, common_feature, detail_feature):

        common_feature = self.layer2(common_feature)
        detail_feature = self.layer3(detail_feature)

        volume_features = torch.cat([common_feature, detail_feature], dim=1)

        gen_volume = self.layer(volume_features)

        return gen_volume

class Decoder3(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder3, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, volume_features):
        gen_volume = volume_features.view(-1, 64, 4, 4, 4)
        # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
        gen_volume = self.layer1(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 256, 8, 8, 8])
        gen_volume = self.layer2(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 128, 16, 16, 16])
        gen_volume = self.layer3(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 32, 32, 32, 32])
        gen_volume = self.layer4(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])

        # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])

        return gen_volume
