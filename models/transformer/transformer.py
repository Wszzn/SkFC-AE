from torch import nn

from models.transformer.layer_norm import LayerNorm
from models.transformer.multi_head_attention import MultiHeadAttention
from models.transformer.position_wise_feed_forward import PositionwiseFeedForward


class Transformer(nn.Module):

    def __init__(self,cfg):
        super(Transformer, self).__init__()
        self.cfg = cfg

        self.attention = MultiHeadAttention(d_model=self.cfg.CONST.D_MODEL, n_head=self.cfg.CONST.N_HEAD)
        self.norm1 = LayerNorm(d_model=self.cfg.CONST.D_MODEL)
        self.dropout1 = nn.Dropout(p=self.cfg.CONST.DROP_PROB)

        '''self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)'''

    def forward(self, x, s_mask=None):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=s_mask)
        
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
