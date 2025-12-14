import streamlit as st
import torch
from tokenizers import Tokenizer
from pathlib import Path

# Import your model architecture
# (Make sure the 'src' folder is in the same directory as app.py)
from src.transformer import build_transformer

# --- CONFIGURATION (Must match your training!) ---
SEQ_LEN = 150
D_MODEL = 512
SRC_LANG = "en"
TGT_LANG = "fr"
MODEL_PATH = "weights/tmodel_03.pt"  # <--- Verify this name matches your file

# --- LOADERS ---
@st.cache_resource
def load_tokenizers():
    try:
        tokenizer_src = Tokenizer.from_file("tokenizer_en.json")
        tokenizer_tgt = Tokenizer.from_file("tokenizer_fr.json")
        return tokenizer_src, tokenizer_tgt
    except Exception as e:
        st.error(f"❌ Could not load tokenizers! Did you download .json files? Error: {e}")
        return None, None

@st.cache_resource
def load_model():
    tokenizer_src, tokenizer_tgt = load_tokenizers()
    if not tokenizer_src: return None, None, None

    device = torch.device("cpu") # Force CPU for laptop demo
    
    # Build the empty shell
    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        SEQ_LEN, SEQ_LEN, D_MODEL
    )

    # Load the trained 1GB brain
    try:
        print(f"Loading weights from {MODEL_PATH}...")
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        print("✅ Model loaded!")
    except FileNotFoundError:
        st.error(f"❌ Could not find {MODEL_PATH}. Check your folder structure.")
        return None, None, None

    return model, tokenizer_src, tokenizer_tgt

# --- INFERENCE LOGIC ---
def translate(text, model, tokenizer_src, tokenizer_tgt):
    model.eval()
    device = torch.device("cpu")
    
    # 1. Encode Input
    sos_token = tokenizer_tgt.token_to_id('[SOS]')
    eos_token = tokenizer_tgt.token_to_id('[EOS]')
    pad_token = tokenizer_tgt.token_to_id('[PAD]')

    enc_input_tokens = tokenizer_src.encode(text).ids
    enc_input = torch.tensor([sos_token] + enc_input_tokens + [eos_token]).unsqueeze(0)
    enc_mask = (enc_input != pad_token).unsqueeze(0).unsqueeze(0).int()

    # 2. Decode Loop
    decoder_input = torch.tensor([[sos_token]])
    
    output_text = ""
    progress_bar = st.progress(0)
    
    with torch.no_grad():
        # Pre-calculate Encoder output (Running only once saves huge time)
        encoder_output = model.encode(enc_input, enc_mask)
        
        for i in range(SEQ_LEN):
            # Update progress bar
            progress_bar.progress((i + 1) / 20) # Normalize to ~20 words expected
            
            # Create Mask
            dec_mask = (decoder_input != pad_token).unsqueeze(0).int() & torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1).type(torch.int) == 0

            # Run Decoder
            out = model.decode(decoder_input, encoder_output, enc_mask, dec_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
            # Stop if EOS
            if next_word.item() == eos_token:
                break
                
            # Append next word
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(enc_input).fill_(next_word.item())], dim=1)
            
    output_text = tokenizer_tgt.decode(decoder_input[0].tolist())
    progress_bar.progress(100)
    return output_text

# --- THE UI ---
st.set_page_config(page_title="Neural Translator", page_icon="🧠")

st.title("🧠 NeuralMachineTranslator")
st.markdown("*Built from Scratch with PyTorch | 75M Parameters*")
st.divider()

# Left Column (Input), Right Column (Output)
col1, col2 = st.columns(2)

with col1:
    st.subheader("🇺🇸 English")
    input_text = st.text_area("Enter text:", height=200, placeholder="I am reading a book.")

with col2:
    st.subheader("🇫🇷 French")
    if st.button("Translate ✨", use_container_width=True):
        if not input_text:
            st.warning("Please type something first.")
        else:
            model, t_src, t_tgt = load_model()
            if model:
                with st.spinner("Processing Attention layers..."):
                    translation = translate(input_text, model, t_src, t_tgt)
                    st.success(translation)
                    st.caption("Inference run on CPU")

st.divider()
st.markdown("### How it works")
st.image("https://jalammar.github.io/images/t/transformer_decoding_2.gif", caption="Transformer Decoding Process")