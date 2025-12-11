import torch
from src.blocks import EncoderLayer, DecoderLayer

def test_encoder_layer():
    d_model = 512
    heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_len = 10
    
    layer = EncoderLayer(d_model, heads, d_ff, dropout)
    
    # Input
    x = torch.randn(batch_size, seq_len, d_model)
    # Mask (1, 1, 1, seq_len) - No masking for this test
    mask = torch.ones(1, 1, 1, seq_len)
    
    output = layer(x, mask)
    
    assert output.shape == (batch_size, seq_len, d_model)
    print("EncoderLayer Shape Test Passed!")

def test_decoder_layer():
    d_model = 512
    heads = 8
    d_ff = 2048
    dropout = 0.1
    batch_size = 2
    seq_len_tgt = 10
    seq_len_src = 15 # Encoder output might be different length
    
    layer = DecoderLayer(d_model, heads, d_ff, dropout)
    
    # Decoder Input (Target)
    x = torch.randn(batch_size, seq_len_tgt, d_model)
    # Encoder Output (Memory)
    memory = torch.randn(batch_size, seq_len_src, d_model)
    
    # Masks
    src_mask = torch.ones(1, 1, 1, seq_len_src)
    tgt_mask = torch.ones(1, 1, seq_len_tgt, seq_len_tgt)
    
    output = layer(x, memory, src_mask, tgt_mask)
    
    # Output must match the Decoder Input length (seq_len_tgt), NOT source
    assert output.shape == (batch_size, seq_len_tgt, d_model)
    print("DecoderLayer Shape Test Passed!")

if __name__ == "__main__":
    test_encoder_layer()
    test_decoder_layer()