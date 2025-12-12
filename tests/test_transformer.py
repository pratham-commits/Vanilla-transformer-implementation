import torch
from src.transformer import build_transformer

def test_transformer_build():
    src_vocab = 100
    tgt_vocab = 100
    src_seq_len = 10
    tgt_seq_len = 10
    d_model = 512
    
    model = build_transformer(src_vocab, tgt_vocab, src_seq_len, tgt_seq_len, d_model)
    
    # Create dummy inputs
    src = torch.randint(0, src_vocab, (1, src_seq_len))
    src_mask = torch.ones(1, 1, 1, src_seq_len)
    
    tgt = torch.randint(0, tgt_vocab, (1, tgt_seq_len))
    tgt_mask = torch.ones(1, 1, tgt_seq_len, tgt_seq_len)
    
    # Forward Pass Steps
    # 1. Encode
    encoder_output = model.encode(src, src_mask)
    assert encoder_output.shape == (1, src_seq_len, d_model)
    
    # 2. Decode
    decoder_output = model.decode(tgt, encoder_output, src_mask, tgt_mask)
    assert decoder_output.shape == (1, tgt_seq_len, d_model)
    
    # 3. Project
    final_output = model.project(decoder_output)
    assert final_output.shape == (1, tgt_seq_len, tgt_vocab)
    
    print("Transformer Architecture Build & Forward Pass: SUCCESS")

if __name__ == "__main__":
    test_transformer_build()