import numpy as np
import torch
from torch import nn
from torch.nn import init


class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=256):
        super().__init__()
        self.mk = nn.Linear(d_model, S, bias=True)
        self.mv = nn.Linear(S, d_model, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.norm = nn.BatchNorm1d(16)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        attn = self.mk(queries)  # bs,n,S
        attn = self.softmax(attn)  # bs,n,S
        attn = attn / torch.sum(attn, dim=2, keepdim=True)  # bs,n,S
        out = self.mv(attn)  # bs,n,d_model
        out = self.norm(out)
        return out


if __name__ == '__main__':
    input = torch.randn(16, 1, 1024)
    ea = ExternalAttention(d_model=1024, S=512)
    output = ea(input)
    print(output.shape)

