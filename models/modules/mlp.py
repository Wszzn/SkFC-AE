import torch
from torch import nn
from models.transformer.layer_norm import LayerNorm

class Adaptation(nn.Module):
    def __init__(self, dim):
        super(Adaptation, self).__init__()
        self.mlp1 = Mlp(dim, dim)
        self.mlp2 = Mlp(dim, dim)
        self.mlp3 = Mlp(dim, dim)
    def forward(self, image_features, global_features, point_features):
        image_features = self.mlp1(image_features)
        global_features = self.mlp2(global_features)
        point_features = self.mlp3(point_features)
        return image_features.reshape(-1, 16, 64), global_features.reshape(-1, 16, 64), point_features.reshape(-1, 16, 64)

class Mlp(nn.Module):
    def __init__(self,
                 embed_dim,
                 inner_dim,
                 dropout=0.2):
        super(Mlp, self).__init__()
        self.fc = nn.Linear(inner_dim, embed_dim)
        self.relu = torch.nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(inner_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.norm(x)
        x = self.dropout(x)

        return x

if __name__ == "__main__":
    x = torch.rand(16, 1024)
    mlp = Mlp(1024, 4096)
    output = mlp(x)
    print(output.shape)
    # x = torch.ones(0,2).squeeze()
    # print(x)
