import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
# import models.pointnet.pointnet_utils as utils

class PointNetEncoder(torch.nn.Module):
    def __init__(self, channel=2):
        super(PointNetEncoder, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=channel, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=128, out_channels=256, kernel_size=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=256, out_channels=512, kernel_size=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU()
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=1),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU()
        )
        self.norm = torch.nn.BatchNorm1d(1)

    def forward(self, points):
        B, N, C = points.size()
        points = points.transpose(2, 1)
        # print(points.size())    # torch.Size([batch_size, 2, point_num])
        local_fetures_1 = self.layer1(points)
        # print(features.size())    # torch.Size([batch_size, 64, point_num])
        local_fetures_2 = self.layer2(local_fetures_1)
        # print(features.size())    # torch.Size([batch_size, 64, point_num])
        local_fetures_3 = self.layer3(local_fetures_2)
        # print(features.size())    # torch.Size([batch_size, 128, point_num])
        global_fetures_1 = self.layer4(local_fetures_3)
        # print(features.size())    # torch.Size([batch_size, 256, point_num])
        global_fetures_2 = self.layer5(global_fetures_1)
        # print(features.size())    # torch.Size([batch_size, 512, point_num])
        global_fetures_3 = self.layer6(global_fetures_2)
        # print(features.size())    # torch.Size([batch_size, 1024, point_num])
        features = torch.cat([local_fetures_1, local_fetures_2, local_fetures_3, global_fetures_1, global_fetures_2], 1)
        # print(features.size())    # torch.Size([batch_size, 2048, point_num])
        features = torch.max(features, 2, keepdim=True)[0].squeeze(2).unsqueeze(1)
        features = self.norm(features).squeeze(1)

        return features


# segmentation
#
if __name__ == '__main__':
    x = torch.ones(16, 256, 2)
    pointnet = PointNetEncoder(channel=2)
    output = pointnet(x)
    print(output)