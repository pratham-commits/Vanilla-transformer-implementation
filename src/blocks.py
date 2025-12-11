import torch.nn as nn
from src.attention import MultiHeadAttention
from src.layers import PositionwiseFeedForward, ResidualConnection, LayerNormalization

class EncoderLayer(nn.Module):
    """
    contains Self-attention -> residual -> feefdorward -> residual
    """
    def __init__(self, d_model : int, heads : int, d_ff:int,dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads,dropout)
        self.sublayer_1 = ResidualConnection(d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model,d_ff,dropout)
        self.sublayer_2 = ResidualConnection(d_model, dropout)
    
    def forward(self,x, mask):
        attention_function = lambda x: self.self_attention(x,x,x,mask)
        x = self.sublayer_1(x, attention_function)
        x = self.sublayer_2(x,self.feed_forward)
        return x
    
class DecoderLayer(nn.Module):
    """contains : -
    masked self attention -> residual -> cross attention -> residual -> feedforward -> residual
    """
    def __init__(self,d_model:int,heads: int,d_ff:int,dropout:int):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model,heads,dropout)
        self.sublayer_1 = ResidualConnection(d_model,dropout)
        self.src_attention = MultiHeadAttention(d_model,heads,dropout)
        self.sublayer_2 = ResidualConnection(d_model,dropout)
        self.feed_forward = PositionwiseFeedForward(d_model,d_ff,dropout)
        self.sublayer_3 = ResidualConnection(d_model,dropout)
    
    def forward(self,x, memory,src_mask,tgt_mask):
        """
        x: Input to the decoder (target sentence so far)
        memory: Output from the Encoder (source sentence info)
        src_mask: Mask for the encoder output (ignore padding in source)
        tgt_mask: Mask for the decoder input (ignore future words + padding)
        """
        x = self.sublayer_1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayer_2(x, lambda x: self.src_attention(x, memory, memory, src_mask))
        x = self.sublayer_3(x, self.feed_forward)
        return x
        
        