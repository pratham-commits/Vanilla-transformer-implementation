import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(query,key,value,mask=None):
    """
    Compute 'Scaled Dot Product Attention'.
    Args:
        query: (Batch, Heads, Seq_Len_Q, d_k)
        key:   (Batch, Heads, Seq_Len_K, d_k)
        value: (Batch, Heads, Seq_Len_K, d_k)
        mask:  (Batch, 1, 1, Seq_Len_K) - Optional mask to hide future tokens or padding.
        
    Returns:
        output: (Batch, Heads, Seq_Len_Q, d_k)
        attention_weights: (Batch, Heads, Seq_Len_Q, Seq_Len_K)
    """
    #step 1 : dimension of query
    d_k = query.size(-1)
    
    #step 2 : matrix multiplication - Q*K^T
    scores = torch.matmul(query,key.transpose(-2,-1))
    
    #step 3 : scale by sqrt(d_k)
    scores = scores / math.sqrt(d_k)
    
    # step 4: if in case of decoder, mask is set true else false
    if mask is not None:
        scores = scores.masked_fill(mask== 0, -1e9)
        
    #step 5: softmax to get probabilities and in case of negative values it will  be zero
    # that's why we set negative values in masking
    attention_weights = scores.softmax(dim = -1)
    
    # step 6 : weights*V
    output = torch.matmul(attention_weights,value)
    
    return output, attention_weights  

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model = int, h= int, dropout = float):
        super().__init__()
        self.d_model = d_model
        self.h = h #number of heads
        
        #we ensure d_model divides evenly by h
        #512/8 = 64, so each head processes vectors of size 64
        assert d_model % h ==0,"d_model is not divisible by h"
        
        self.d_k = d_model // h
        # the three linear layers for Q, K, V
        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)
        
        #output linear layer
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,q,k,v,mask=None):
        batch_size = q.size(0)
        # 1. Linear Projections + Split into Heads
        # Reshape to (Batch, Seq_Len, Heads, d_k) -> Transpose to (Batch, Heads, Seq_Len, d_k)
        query = self.w_q(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key   = self.w_k(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
         
        x, self.attn_scores = scaled_dot_product_attention(query, key, value, mask)
        
        # can also add dropout here
        
        # concatenate heads
        # Transpose back : (Batch, seq_len,Heads, d_k)
        # Flatten : (batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.d_model)
        return self.w_o(x)
        