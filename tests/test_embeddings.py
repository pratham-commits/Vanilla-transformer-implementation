import torch
from src.embeddings import InputEmbeddings, PositionalEncoding

def test_embeddings_shape():
    vocab_size = 100
    d_model = 512
    batch_size = 2
    seq_len = 10
    
    embeddings = InputEmbeddings(d_model=d_model,vocab_size=vocab_size)
    
    #dummy
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    word_embeddings = embeddings(x)
    
    assert word_embeddings.shape == (batch_size,seq_len,d_model)
    print("Input embeddings passed")

def test_positional_encoding_shape():
    d_model = 512
    seq_len = 50
    dropout = 0.1
    batch_size = 2
    
    pe = PositionalEncoding(d_model,seq_len,dropout)
    
    #dummy
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Pass through layer
    output = pe(x)
    
    # Check shape: Should remain (Batch, Seq_Len, d_model)
    assert output.shape == (batch_size, seq_len, d_model)
    print("PositionalEncoding Shape Test Passed!")
    
if __name__ == "__main__":
    test_embeddings_shape()
    test_positional_encoding_shape()
    