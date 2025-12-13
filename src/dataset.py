import torch
import torch.nn as nn
from torch.utils.data import Dataset
from src.utils import causal_mask

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # We pre-calculate special token IDs to speed up the loop
        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # 1. Get the raw text pair
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # 2. Convert text to Integer IDs (Tokenization)
        # We assume the tokenizer has an .encode() method
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # 3. Calculate Padding needed
        # We enforce a fixed sequence length (e.g., 350)
        # Source Padding = Max_Len - Actual_Len - 2 (for SOS and EOS)
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        # Decoder Padding = Max_Len - Actual_Len - 1 (for SOS only)
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Validation: If sentence is too long, we might crash. 
        # In a real generic repo, we would truncate. For now, we assume data fits.
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # 4. Build the Tensors
        
        # Encoder Input: [SOS] + Text + [EOS] + [PAD]...
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Decoder Input: [SOS] + Text + [PAD]...
        # Note: No EOS here! We teach the decoder to START with SOS.
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        # Label (The Truth): Text + [EOS] + [PAD]...
        # This is what the output should look like.
        # Note: Shifted by 1 compared to Decoder Input.
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )
        
        # 5. Verify Sizes (Researcher's Sanity Check)
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # 6. Return the dict
        # We create the masks "on the fly" using the logical expressions (val != pad)
        return {
            "encoder_input": encoder_input,  # (Seq_Len)
            "decoder_input": decoder_input,  # (Seq_Len)
            
            # Encoder Mask: Hide PAD tokens
            # (1, 1, Seq_Len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            
            # Decoder Mask: Hide PAD tokens AND Future tokens
            # (1, Seq_Len) & (1, Seq_Len, Seq_Len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            
            "label": label,  # (Seq_Len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

