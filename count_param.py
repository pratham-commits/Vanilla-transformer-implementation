import torch
from src.transformer import build_transformer
from tokenizers import Tokenizer

# 1. Load Tokenizers to get Vocab Size
# (The model size depends heavily on how many words it knows)
try:
    tokenizer_src = Tokenizer.from_file("tokenizer_en.json")
    tokenizer_tgt = Tokenizer.from_file("tokenizer_fr.json")
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
except:
    # Fallback if you don't have json files handy
    print("⚠️ Could not load tokenizers, assuming 30,000 for estimation.")
    src_vocab_size = 30000
    tgt_vocab_size = 30000

# 2. Build the Empty Model (No weights loaded, just the skeleton)
# d_model = 512 (Standard from your config)
model = build_transformer(src_vocab_size, tgt_vocab_size, 150, 150, 512)

# 3. Count the Parameters
# p.numel() returns the total number of elements in a tensor
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"--- 📊 Model Audit ---")
print(f"English Vocab: {src_vocab_size}")
print(f"French Vocab:  {tgt_vocab_size}")
print(f"d_model:       512")
print(f"----------------------")
print(f"🔥 Total Parameters: {total_params:,}")
print(f"----------------------")

# 4. Explain the File Size
print(f"Floats are 4 bytes.")
print(f"Model Size in RAM: {total_params * 4 / (1024**2):.2f} MB")
print(f"Why is your .pt file 1GB? Because it includes the Optimizer states (Adam).")
print(f"Adam stores 2 extra copies of every parameter for momentum.")
print(f"Total Saved Size ≈ {total_params * 4 * 3 / (1024**2):.2f} MB")