# FinexaGPT: Transformer Implementation
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
A clean, readable GPT implementation that actually makes sense. No magic, no bloat - just the transformer architecture you can understand in an afternoon.

# ⚡ What This Is
A minimal GPT-like transformer implementation

134M parameter model (like a tiny GPT-2)

Clean code you can actually read

Production-ready data pipeline

Everything fits in one file (almost)

# 🎯 Why FinexaGPT?

While there are many transformer implementations, FinexaGPT is:
- **Financial-domain ready** (optimized for earnings calls, SEC filings, market data)
- **Actually readable** (most implementations are abstracted into oblivion)
- **Built for extension** (clean architecture for adding financial-specific features)

# 🚀 What This Is NOT
❌ Not a 1-trillion parameter model

❌ Not a framework with 1000 dependencies

❌ Not abstracted into oblivion

❌ Not pretending to be something it's not

# 📁 Files 
text
.
├── model.py              # The transformer (134M params)
├── trainer.py            # Training loop with cosine decay
├── train.py              # Main training script  
├── data.py               # Data loading & tokenization
├── config.yaml           # Hyperparameters
└── README.md             # This file
🎯 Quick Start
1. Install
bash
git clone https://github.com/franciskamande-ai/FinexaGPT.git
cd FinexaGPT
pip install torch tiktoken omegaconf
2. Train
bash
# Get some text
echo "Hello world! This is training data." > data.txt

# Train (30 seconds on CPU)
python train.py --data data.txt --max-steps 100
3. Generate
python
from model import Transformer

model = Transformer(vocab_size=50000)
# Load checkpoint
model.load_state_dict(torch.load("checkpoint.pt"))

output = model.generate(
    idx=torch.tensor([[1, 2, 3]]),  # Your tokens
    max_new_tokens=50,
    temperature=0.8
)
print(output)
🧠 The Model (model.py)
python
# What you get:
#### - Multi-head attention with causal masking
#### - Feed-forward networks (GELU)
#### - Layer normalization & dropout
#### - Learned positional embeddings
#### - Next-token prediction
#### - Temperature/top-k sampling

# What's TODO (help wanted!):
#### - GQA (Grouped Query Attention)
#### - Flash Attention optimization  
#### - RMSNorm instead of LayerNorm
#### - Rotary positional embeddings
#### - SwiGLU activation
# 📊 Specs
Parameters: 134M (configurable)

Context: 512 tokens (extendable)

Layers: 12

Heads: 12

Embedding: 768-dim

Vocab: 50K (tiktoken's cl100k_base)

# 🔧 Configuration
yaml
# config.yaml (simple, not overwhelming)
model:
  num_layers: 12
  num_heads: 12  
  d_model: 768
  vocab_size: 50000
  max_seq_length: 512
  
training:
  batch_size: 32
  learning_rate: 3e-4
  warmup_steps: 2000
  total_steps: 100000
# 🤝 Want to Help?
See the TODO in model.py:

GQA - Grouped Query Attention (memory efficient)

Flash Attention - Faster attention computation

RMSNorm - Better than LayerNorm

RoPE - Rotary positional embeddings

SwiGLU - Better activation function

How to contribute:

Fork the repo

Pick a TODO item

Implement it

Submit a PR

Get credit in the README

# ❓ FAQ
Q: Can this beat GPT-4?
A: No. It's 134M params vs 1.7T. This is for learning.

Q: Why not use HuggingFace?
A: This is for understanding, not just using.

Q: What GPU do I need?
A: Trains on a single GPU (8GB+ VRAM). Runs on CPU (slowly).

Q: Is this production ready?
A: The code is clean and works. Add your own monitoring/CI.

# 📈 Performance
Training: ~500 tokens/sec on RTX 3090

Memory: ~3GB for 134M model

Accuracy: Learns language patterns (not SOTA)

# 🎓 Learning Path
Read model.py (understand attention)

Read trainer.py (understand training)

Read data.py (understand tokenization)

Modify something

Train it

See what breaks

Learn

## 📄 License

FinexaGPT is released under the [MIT License](LICENSE).

- **You can:** Use commercially, modify, distribute, private use
- **You must:** Include original copyright notice
- **You cannot:** Hold the author liable

See the [LICENSE](LICENSE) file for full terms.

# 🙏 Credits
Architecture from "Attention Is All You Need"

Implementation inspired by minGPT (Karpathy)

Tokenization using tiktoken (OpenAI)

Funding from FKNLABS

Built with PyTorch

# 📞 Contact
Found a bug? Want to contribute?
Email researcher : fkamande264@gmail.com
Open an issue or submit a PR. No bureaucracy.

FinexaGPT - Transformer, simplified.
For developers who want to understand, not just import.

Star the repo if it helped you learn something. ⭐

# RESEARCHERS
- Francis Nyambura(FKNLABS-Founder)
       -Email ; fkamande264@gmail.com 
