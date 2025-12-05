# test_tiny.py
import torch
from model import Transformer

# Smallest possible model
model = Transformer(
    num_heads=2,
    d_model=64,
    num_layers=2,
    dropout=0.0,
    vocab_size=100,  # Very small vocab
    max_seq_length=32
)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Test forward pass
x = torch.randint(0, 100, (1, 10))  # Batch of 1, sequence of 10
logits, loss = model(x)
print(f"Output shape: {logits.shape}")
print("✓ Model works on CPU!")

# Test generation
with torch.no_grad():
    generated = model.generate(x, max_new_tokens=5)
    print(f"Generated: {generated.shape}")