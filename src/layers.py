import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Construct a layernorm module
    Formula: y = (x - mean) / std * gamma + beta
    """
    def __init__(self, d_model : int , eps : float = 1e-6):
        super().__init__()
        # trainable params
        # gamma : it muliplies the normalised data
        # As the network trains, if it realizes "Hey, I need the values to be spread out more," it will change these numbers from 1.0 to 2.0 or 0.5. It learns the perfect "Volume" for the data.
        self.a_2 = nn.Parameter(torch.ones(d_model))
        # Beta : It adds to the normalized data.
        # If the network realizes "Hey, the average shouldn't be 0, it should be 10," it learns to change b_2 to 10. It learns the perfect "Baseline" for the data.
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
        
    def forward(self, x):
        
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1,keepdim=True)
        
        return self.a_2 * (x-mean)/(std +self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2
    """
    def __init__(self , d_model : int, d_ff : int, dropout : float = 0.1):
        
        #w_1 : expansion layer , takes input of 512 and projects it to 2048 space
        self.w_1 = nn.Linear(d_model,d_ff)
        
        #w_2 : compression layer , takes large vector and squashes it back to 512
        self.w_2 = nn.Linear(d_ff,d_model)
        
        #dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,x):
        #step 1: expand to 2048
        exp=self.w_1(x)
        # step2: applying relu to replace all negative numbers with 0, adds 'non linearity'
        after_relu = torch.relu(exp)
        #step3: dropout
        after_drop = self.dropout(after_relu)
        #step4 : compress back to 512
        return self.w_2(after_drop)
    
    
class ResidualConnection(nn.Module):
    def __init__(self, d_model : int, dropout : float):
        super().__init__()
        self.norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))