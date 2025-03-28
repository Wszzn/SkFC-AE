import torch.nn
from torch import nn

from config import cfg
from models.transformer.layer_norm import LayerNorm
from models.transformer.multi_head_attention import MultiHeadAttention
from models.transformer.position_wise_feed_forward import PositionwiseFeedForward

class UnFlatten(nn.Module):
    # NOTE: (size, x, x, x) are being computed manually as of now (this is based on output of encoder)
    def forward(self, input, size=128): # size=128
        return input.view(input.size(0), -1, 8, 8, 8)

class FeatureFusionModule(nn.Module):
    def __init__(self, cfg):
        super(FeatureFusionModule, self).__init__()
        self.image_layer = torch.nn.Sequential(
            UnFlatten(),

            torch.nn.Conv3d(2, 2, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(2),
            torch.nn.ReLU(),

            # torch.nn.Conv3d(16, 32, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            # torch.nn.BatchNorm3d(32),
            # torch.nn.ReLU(),

            torch.nn.Conv3d(2, 8, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.global_layer = torch.nn.Sequential(
            UnFlatten(),

            torch.nn.Conv3d(2, 2, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(2),
            torch.nn.ReLU(),

            # torch.nn.Conv3d(16, 32, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            # torch.nn.BatchNorm3d(32),
            # torch.nn.ReLU(),

            torch.nn.Conv3d(2, 8, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.point_layer = torch.nn.Sequential(
            UnFlatten(),

            torch.nn.Conv3d(2, 2, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(2),
            torch.nn.ReLU(),

            # torch.nn.Conv3d(16, 32, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            # torch.nn.BatchNorm3d(32),
            # torch.nn.ReLU(),

            torch.nn.Conv3d(2, 8, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.common_layer = torch.nn.Sequential(
            UnFlatten(),

            torch.nn.Conv3d(16, 16, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),

            # torch.nn.Conv3d(16, 32, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            # torch.nn.BatchNorm3d(32),
            # torch.nn.ReLU(),

            torch.nn.Conv3d(16, 64, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )

        self.image_global_layer = torch.nn.Sequential(
            torch.nn.Conv3d(16, 16, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),

            torch.nn.Conv3d(16, 8, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.norm = torch.nn.BatchNorm3d(8)
        self.softmax = torch.nn.Softmax(dim=1)
        self.weight1 = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float), requires_grad=True)
        self.weight2 = torch.nn.Parameter(torch.tensor([0.2], dtype=torch.float), requires_grad=True)
        self.weight3 = torch.nn.Parameter(torch.tensor([0.4], dtype=torch.float), requires_grad=True)


    def forward(self, image_features, global_features, points_features):
        # features.shape: (-1, 16, 4, 4, 4)
        image_voxel = self.image_layer(image_features)  # (-1, 64, 4, 4, 4)
        global_voxel = self.global_layer(global_features)  # (-1, 64, 4, 4, 4)
        points_voxel = self.point_layer(points_features)  # (-1, 64, 4, 4, 4)
        # common_voxel = self.common_layer(common_features)  # (-1, 64, 4, 4, 4)

        image_global_voxel = torch.cat([image_voxel, global_voxel], dim=1)
        image_global_voxel = self.image_global_layer(image_global_voxel)    # (-1, 64, 4, 4, 4)
        points_score = self.softmax(points_voxel)
        image_global_voxel = torch.mul(image_global_voxel, points_score)  # (-1, 64, 4, 4, 4)
        image_global_voxel = self.norm(image_global_voxel) # (-1, 64, 4, 4, 4)

        # voxel = self.weight1*common_voxel + self.weight2*image_global_voxel + self.weight3*points_voxel
        # voxel = self.norm(voxel)
        # voxel = self.softmax(voxel)
        return image_global_voxel, points_voxel

if __name__ == '__main__':
    features = torch.randn(16, 16, 4, 4, 4)
    encoder = MFFusion(cfg)
    output = encoder(features, features, features, features)
    print(output.shape)
