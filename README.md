# 🍦 Vanilla Transformer: NMT from First Principles

> *A PyTorch-native implementation of the original "Attention Is All You Need" architecture, built and trained from scratch without pre-trained weights.*

### 🎯 The "Why": Learning over Performance
**The entire point of this project was to learn how these massive structures are built from the bottom up.** In an era of simple API calls and pre-trained libraries, it is easy to treat AI as a "black box." This project breaks that box open. I deliberately avoided high-level abstractions to force myself to implement every matrix multiplication, attention mask, and gradient step manually. The goal was not to beat Google Translate, but to intimately understand the mathematics that makes it possible.

---

### 🚀 Project Overview
**Vanilla Transformer** is a full Sequence-to-Sequence Transformer model implemented entirely in PyTorch. It demonstrates the fundamental mathematics of Large Language Models—including Multi-Head Attention, Positional Encodings, and Encoder-Decoder blocks—trained on the Opus Books (English-to-French) dataset.

---

### 🛠️ Technical Architecture
I implemented the architecture described in the Vaswani et al. (2017) paper with the following components:

* **Custom Tokenization:** Implemented **Word-Level Tokenization** to map specific words to IDs. Unlike BPE (Byte Pair Encoding), this requires the model to learn exact word mappings, making the task harder but more mathematically transparent.
* **Embeddings & Positional Encoding:** Implemented sinusoidal positional encodings to give the model a sense of sequence order.
* **Multi-Head Attention:** Built the "Scaled Dot-Product Attention" mechanism manually to allow the model to focus on different parts of the sentence simultaneously.
* **Encoder-Decoder Stack:** A 6-layer stack with Feed-Forward Networks and Layer Normalization.
* **Inference Pipeline:** A greedy decoding loop that generates translations autoregressively.

**Tech Stack:** `Python`, `PyTorch`, `HuggingFace Datasets` (loading only), `Streamlit` (Frontend).

---

### 📊 Training & Performance
* **Dataset:** Opus Books (English-French subset).
* **Model Size:** ~90 Million Parameters. (run count_param.py, to verify it)
* **Hardware:** Single T4 GPU (Google Colab / Kaggle).
* **Training Time:** ~3.5 Hours (4 Epochs).
* **Current Loss:** ~5.5 (Cross Entropy).

#### ✅ Engineering Wins (The "Good")
1.  **Grammar Mapping:** The model has successfully learned to map English sentence structures to French ones (e.g., SVO order).
2.  **Pronoun Resolution:** It correctly identifies relationships like "I" $\to$ "Je" and "You" $\to$ "Vous".
3.  **Zero-Dependency:** The model runs 100% offline. The inference engine requires no internet connection and runs on a standard CPU.

#### ⚠️ Critical Limitations (The "Honest Truth")
* **Vocabulary Gaps (Word-Level Issue):** Since this uses Word-Level tokenization trained on old books, modern names (e.g., "Pratham") or tech terms result in Unknown tokens.
    * *Example:* Input: "My name is Pratham" $\to$ Output: "Mon ' ." (It recognized "My" $\to$ "Mon", but failed the name).
* **Repetition Loops:** Due to limited training time (4 epochs vs standard 100+), the model sometimes struggles with stopping criteria (e.g., *"Vous vous vous"*).
* **Compute Constraint:** A Transformer typically requires 24+ hours on A100 GPUs to reach fluency. With more compute time, the current loss of 5.5 would drop below 3.0, resolving the fluency issues.

---

### 🧪 Live Demo Results (Epoch 4)

**Case 1: Grammar Success**
* **Input:** *"She is good"*
* **Output:** *"Elle est ."*
* *Analysis:* The model correctly mapped "She" to "Elle" and "is" to "est". It missed the adjective due to training sparsity.

**Case 2: Pronoun Success**
* **Input:** *"Me is me"*
* **Output:** *"Je ."*
* *Analysis:* Successfully identified the first-person subject pronoun.

**Case 3: Vocabulary Failure**
* **Input:** *"My name is Pratham"*
* **Output:** *"Mon ' ."*
* *Analysis:* "Pratham" is Out-Of-Vocabulary (OOV). Future work will implement BPE (Byte Pair Encoding) to handle unknown sub-words.

---

### 👨‍💻 Run It Locally

```bash
# 1. Clone the repo
git clone [https://github.com/pratham-commits/vanilla-transformer.git](https://github.com/pratham-commits/vanilla-transformer.git)

# 2. Install dependencies
pip install torch streamlit tokenizers datasets

# 3. Train your model - change the epochs as per requirements
python train.py
# downloads the best .pt file, store  in the `weights/` directory.

# 4. Run the App
streamlit run app.py