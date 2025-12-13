import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Hugging Face libraries
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm
import warnings

# Our modules
from src.dataset import BilingualDataset, causal_mask
from src.transformer import build_transformer
from src.config import get_config, get_weights_file_path

def get_all_sentences(ds, lang):
    """
    Generator function to yield sentences one by one.
    Needed for the tokenizer to scan the dataset.
    """
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    """
    1. Check if a tokenizer file exists.
    2. If not, build a new one by scanning the dataset.
    """
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # We use a simple WordLevel tokenizer for this research project
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        
        # [UNK] = Unknown word
        # [PAD] = Padding
        # [SOS] = Start of Sentence
        # [EOS] = End of Sentence
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
        print(f"Tokenizer saved to {tokenizer_path}")
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # 1. Load raw data from Hugging Face
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # 2. Build Tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # 3. Split Train/Validation (90% / 10%)
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])
    
    # 4. Create the Dataset objects (from src/dataset.py)
    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    # 5. Create DataLoaders (The things that feed batches to the GPU)
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return model

def train_model(config):
    # 1. Setup Device (GPU is preferred)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Make sure weights folder exists
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    # 3. Prepare Data
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # 4. Build Model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    
    # 5. Tensorboard Logging
    writer = SummaryWriter(config['experiment_name'])
    
    # 6. Optimizer (Adam is standard for Transformers)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # 7. Loss Function
    # We ignore the [PAD] token so the model doesn't get punished for predicting padding
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1)

    # 8. The Loop
    initial_epoch = 0
    global_step = 0
    
    # Preload weights if defined
    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        
        for batch in batch_iterator:
            
            # --- A. DATA TO DEVICE ---
            encoder_input = batch['encoder_input'].to(device) # (B, Seq_Len)
            decoder_input = batch['decoder_input'].to(device) # (B, Seq_Len)
            encoder_mask = batch['encoder_mask'].to(device)   # (B, 1, 1, Seq_Len)
            decoder_mask = batch['decoder_mask'].to(device)   # (B, 1, Seq_Len, Seq_Len)
            
            # --- B. FORWARD PASS ---
            # 1. Run Encoder
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_Len, d_model)
            # 2. Run Decoder
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (B, Seq_Len, d_model)
            # 3. Project to Vocabulary
            proj_output = model.project(decoder_output) # (B, Seq_Len, Vocab_Size)
            
            # --- C. LOSS CALCULATION ---
            # We flatten the output to (Batch * Seq_Len, Vocab_Size) for CrossEntropy
            label = batch['label'].to(device) # (B, Seq_Len)
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            
            # Update Progress Bar
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Log to Tensorboard
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()
            
            # --- D. BACKPROPAGATION ---
            loss.backward()       # Calculate Gradients
            optimizer.step()      # Update Weights
            optimizer.zero_grad() # Reset Gradients
            
            global_step += 1
            
        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)

if __name__ == '__main__':
    # Filter warnings to keep output clean
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
