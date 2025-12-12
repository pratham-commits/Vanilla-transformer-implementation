import torch
import torch.nn as nn
from src.blocks import EncoderLayer,DecoderLayer
from src.embeddings import InputEmbeddings, PositionalEncoding
from src.layers import LayerNormalization

class Encoder(nn.Module):
    """
    Stacking N encoder modules
    """
    def __init__(self,layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(layers[0].self_attention.d_model)
    
    def forward(self, x, mask):
        # pass x through each layer in the stack
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
class Decoder(nn.Module):
    """
    Stack of N Decoder layers
    """
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(layers[0].self_attention.d_model)
    
    def forward(self, x , memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,memory,src_mask,tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    """
    The Final Layer: Converts the vector sequence into vocabulary probabilities.
    Input: (Batch, Seq_Len, d_model)
    Output: (Batch, Seq_Len, Vocab_Size)
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # We don't use Softmax here because we will use CrossEntropyLoss later,
        # which includes Softmax/LogSoftmax internally.
        # This is more numerically stable.
        return self.proj(x)
        
class Transformer(nn.Module):
    """
    Complete architecture     
    """
    
    def __init__(self, 
                 encoder : Encoder,
                 decoder : Decoder,
                 src_embed : nn.Sequential,
                 tgt_embed : nn.Sequential,
                 projection_layer : ProjectionLayer
                 ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        #step 1: embed + position encode
        x = self.src_embed(src)
        #step 2: run encoder stack
        return self.encoder(x, src_mask)
    
    def decode(self,tgt, encoder_output, src_mask, tgt_mask):
        #step 1: embed + position encode
        x = self.tgt_embed(tgt)
        # run decoder stack
        return self.decoder(x, encoder_output, src_mask, tgt_mask)
    def project(self,x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, 
                      src_seq_len: int, tgt_seq_len: int, 
                      d_model: int = 512, N: int = 6, h: int = 8, 
                      dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    
    # 1. Create the Embedding blocks
    src_embed = nn.Sequential(
        InputEmbeddings(d_model, src_vocab_size),
        PositionalEncoding(d_model, src_seq_len, dropout)
    )
    
    tgt_embed = nn.Sequential(
        InputEmbeddings(d_model, tgt_vocab_size),
        PositionalEncoding(d_model, tgt_seq_len, dropout)
    )
    
    # 2. Create the Stacks
    # We use list comprehension to create N copies of the layer
    encoder_blocks = []
    for _ in range(N):
        encoder_blocks.append(EncoderLayer(d_model, h, d_ff, dropout))
        
    decoder_blocks = []
    for _ in range(N):
        decoder_blocks.append(DecoderLayer(d_model, h, d_ff, dropout))
        
    # 3. Create the Containers
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # 4. Create Projection
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # 5. Assemble
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, projection_layer)
    
    # 6. Initialize Parameters (Xavier Uniform)
    # This is a research best practice for starting training smoothly.
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer