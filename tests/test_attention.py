import torch
from src.attention import MultiHeadAttention

def test_multi_head_attention_shape():
    d_model = 512
    heads = 8
    seq_len = 10
    batch_size = 2
    dropout = 0.1
    
    mha = MultiHeadAttention(d_model, heads, dropout)
    
    # Create dummy input (Batch, Seq_Len, d_model)
    # In self-attention, Q, K, and V are usually the same tensor
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Forward pass
    output = mha(x, x, x, mask=None)
    
    # Check output shape
    # It must return the exact same shape as input
    assert output.shape == (batch_size, seq_len, d_model)
    print("MultiHeadAttention Shape Test Passed!")

if __name__ == "__main__":
    test_multi_head_attention_shape()