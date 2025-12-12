import torch
import numpy as np

def causal_mask(size):
    """
    Creates a 'Lower Triangular' mask for the Decoder.
    It allows looking at current and past words, but blocks future words.
    
    Example for size=3:
    [[1, 0, 0],
     [1, 1, 0],
     [1, 1, 1]]
    """
    # np.triu means "Triangle Upper". We make it 1s, then ignore them (k=1).
    # This creates the upper triangle of 1s (the future) which we want to hide.
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    
    # We invert it: 0 becomes 1 (visible), 1 becomes 0 (hidden).
    return torch.from_numpy(mask) == 0