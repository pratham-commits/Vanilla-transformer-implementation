import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    """
    Convert tokens to vectors here
    """
    def __init__(self, d_model:int, vocab_size:int):
        # The embedding layer: Input (Batch, Seq_Len) -> Output (Batch, Seq_Len, d_model)
        #d_model represents the numerical dimension to which the word is to be converted into
        #seq_len is the number of tokens in each batch
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)
    
    def forward(self, x):
        #as d_model gets larger, the values inside vectors gets smaller, so we multiply
        # them by using the root of d_model to ensure that they are comparable to positional encodings
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    inject the essence of positions of each respective token in the embeddings by using sine or cosine
    """
    def __init__(self,d_model : int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        # helps prevent overfitting, in paper a dropout of 0.1 is used
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(seq_len,d_model)
        
        # first find the exact indexes for the position of each token
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        
        # second we need to find the speed at which the sine /cosine wave will move
        # that is basically we are calcualting the omega inside the sin(omega*x)
        omega = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        
        # sine encoding at even pos
        pe[:,0::2] = torch.sin(position*omega)
        
        #cos encoding at odd pos
        pe[:,1::2] = torch.cos(position*omega)
        
        # now we want to store these values when we save the model
        # but we don't want the optimiser to alter them, we use register_buffer for that
        self.register_buffer('pe',pe.unsqueeze(0))
    
    def forward(self,x):
        # x is the output from prev step
        # here we add position vectors to the word vectors
        # x + pe
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        
        
        
        