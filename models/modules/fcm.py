import torch
from torch import nn

from config import cfg
#from models.transformer.encoder_layer import EncoderLayer
from models.transformer.layer_norm import LayerNorm
from models.transformer.multi_head_attention import MultiHeadAttention
from models.transformer.position_wise_feed_forward import PositionwiseFeedForward
from models.external_attention.ExternalAttention import ExternalAttention

class UnFlatten(nn.Module):
    # NOTE: (size, x, x, x) are being computed manually as of now (this is based on output of encoder)
    def forward(self, input, size=128): # size=128
        return input.view(input.size(0), 16, 4, 4, 4)

class FeatureComplementModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model=cfg.CONST.D_MODEL,
                                                  drop_prob=cfg.CONST.DROP_PROB,
                                                  n_head=cfg.CONST.N_HEAD)
                                     for _ in range(cfg.CONST.N_LAYERS)])

        self.common_layer = torch.nn.Sequential(
            UnFlatten(),

            torch.nn.Conv3d(16, 16, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU(),

            torch.nn.Conv3d(16, 64, kernel_size=3, stride=1, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )

        self.ea = ExternalAttention(d_model=64, S=256)

    def forward(self, x, y, s_mask):
        x = self.ea(x)
        for layer in self.layers:
            x = layer(x, y, s_mask)

        common_voxel = x.reshape(-1, 16, 4, 4, 4)
        common_voxel = self.common_layer(common_voxel)
        return common_voxel

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention1 = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.attention2 = MultiHeadAttention(d_model=d_model, n_head=n_head)

        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)



        '''self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)'''

    def forward(self, x, y, s_mask):
        # 1. compute self attention
        _x = x
        x, score = self.attention1(q=y, k=x, v=x, mask=s_mask)
        x, score = self.attention2(q=x, k=x, v=x, mask=s_mask)
        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        '''# 3. positionwise feed forward network
        _x = x
        x = self.ffn(x)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)'''
        return x

if __name__ == '__main__':
    x = torch.randn(1, 1, 2048)
    y = torch.randn(1, 1, 2048)
    fusion = SAFusion2(cfg)
    features = fusion(x, y, s_mask=None)
    print(features.shape)
