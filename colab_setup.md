# Google Colab Setup Guide for Part 4

## Option 1: Using Google Drive (Recommended)

### Step 1: Upload Your Code to Google Drive

1. Zip your entire project folder:
   ```bash
   cd ~/Desktop
   zip -r cs288-sp26-a2.zip cs288-sp26-a2/ -x "*.git*" "*.pyc" "*__pycache__*" "*/venv/*"
   ```

2. Upload `cs288-sp26-a2.zip` to your Google Drive

### Step 2: Create a Colab Notebook

Create a new notebook in Colab with these setup cells:

**Cell 1: Mount Google Drive and Extract**
```python
from google.colab import drive
drive.mount('/content/drive')

# Extract your project
!unzip -q "/content/drive/MyDrive/cs288-sp26-a2.zip" -d /content/
%cd /content/cs288-sp26-a2
```

**Cell 2: Install Dependencies**
```python
# Install required packages
!pip install -q torch tiktoken

# Verify installation
import sys
import torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

**Cell 3: Test Your Implementations**
```python
# Quick test that your implementations work
import sys
sys.path.insert(0, '/content/cs288-sp26-a2')

# Test Part 1
from part1.tokenizer import get_tokenizer
from part1.train_bpe import train_bpe

# Test Part 2
from part2.model import TransformerLM

# Test Part 3
from part3.nn_utils import cross_entropy, gradient_clipping

print("✅ All imports successful!")
```

### Step 3: Download TinyStories Dataset

**Cell 4: Download Dataset**
```python
# Option A: Use the small sample for quick testing
# (already in part1/fixtures/tinystories_sample_5M.txt - 52KB)

# Option B: Download larger dataset for better training
import urllib.request
import os

# Create directory
os.makedirs('/content/cs288-sp26-a2/part4/fixtures', exist_ok=True)

# Download a reasonable subset (you can adjust size)
# For this assignment, 100MB should be sufficient
print("Downloading TinyStories subset...")

# Using HuggingFace datasets
!pip install -q datasets
from datasets import load_dataset

dataset = load_dataset("roneneldan/TinyStories", split="train[:10000]")  # First 10k stories
print(f"Loaded {len(dataset)} stories")

# Save to text file
output_path = '/content/cs288-sp26-a2/part4/fixtures/tinystories_100k.txt'
with open(output_path, 'w', encoding='utf-8') as f:
    for story in dataset:
        f.write(story['text'])
        f.write('\n<|endoftext|>\n')

print(f"✅ Saved to {output_path}")
print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
```

### Step 4: Run Pre-training

**Cell 5: Train Tokenizer**
```python
from pathlib import Path
from part1.train_bpe import train_bpe
from part1.tokenizer import get_tokenizer

# Configuration
PRETRAIN_DATA = Path('/content/cs288-sp26-a2/part4/fixtures/tinystories_100k.txt')
# Or use the small sample for quick testing:
# PRETRAIN_DATA = Path('/content/cs288-sp26-a2/part1/fixtures/tinystories_sample_5M.txt')

VOCAB_SIZE = 2048
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]

print("Training BPE tokenizer...")
vocab, merges = train_bpe(
    input_path=PRETRAIN_DATA,
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
)

tokenizer = get_tokenizer(vocab, merges, SPECIAL_TOKENS)
print(f"✅ Tokenizer trained! Vocab size: {len(vocab)}")

# Test it
test_text = "Once upon a time, there was a little girl."
tokens = tokenizer.encode(test_text)
decoded = tokenizer.decode(tokens)
print(f"\nTest: '{test_text}'")
print(f"Tokens: {tokens}")
print(f"Decoded: '{decoded}'")
```

**Cell 6: Create Model**
```python
from part2.model import TransformerLM
import torch

# Model configuration (adjust based on compute/time constraints)
CONFIG = {
    "vocab_size": len(tokenizer.vocab),
    "context_length": 256,
    "d_model": 256,           # Hidden dimension
    "num_layers": 6,          # Transformer blocks
    "num_heads": 8,           # Attention heads
    "d_ff": 1024,             # FFN dimension
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = TransformerLM(
    vocab_size=CONFIG["vocab_size"],
    context_length=CONFIG["context_length"],
    d_model=CONFIG["d_model"],
    num_layers=CONFIG["num_layers"],
    num_heads=CONFIG["num_heads"],
    d_ff=CONFIG["d_ff"],
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"✅ Model created!")
print(f"Parameters: {num_params:,}")
print(f"Size: ~{num_params * 4 / 1024 / 1024:.1f} MB (fp32)")
```

**Cell 7: Create Dataloader**
```python
from part4.datasets import create_pretraining_dataloader

dataloader = create_pretraining_dataloader(
    file_path=PRETRAIN_DATA,
    tokenizer=tokenizer,
    batch_size=32,  # Adjust based on GPU memory
    max_length=CONFIG["context_length"],
    stride=CONFIG["context_length"] // 2,  # 50% overlap
    shuffle=True,
)

print(f"✅ Dataloader created!")
print(f"Number of sequences: {len(dataloader.dataset)}")
print(f"Batches per epoch: {len(dataloader)}")
print(f"Total tokens per epoch: ~{len(dataloader) * 32 * 256:,}")
```

**Cell 8: Train the Model**
```python
from part4.trainer import Trainer, TrainingConfig

train_config = TrainingConfig(
    num_epochs=3,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=100,
    max_grad_norm=1.0,
    batch_size=32,
    device=device,
    log_interval=50,  # Log every 50 batches
)

trainer = Trainer(
    model=model,
    config=train_config,
    train_dataloader=dataloader,
)

print("🚀 Starting training...")
print(f"Epochs: {train_config.num_epochs}")
print(f"Learning rate: {train_config.learning_rate}")
print("-" * 60)

results = trainer.train()

print("\n✅ Training complete!")
print(f"Final loss: {trainer.train_losses[-1]:.4f}")
```

**Cell 9: Test Generation**
```python
from part4.sampling import generate_text

model.eval()

prompts = [
    "Once upon a time",
    "The little dog",
    "There was a princess",
]

print("=" * 60)
print("GENERATED TEXT SAMPLES")
print("=" * 60)

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 40)
    
    # Greedy decoding
    greedy_text = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=50,
        method="greedy"
    )
    print(f"Greedy: {greedy_text}")
    
    # Top-k sampling
    topk_text = generate_text(
        model, tokenizer, prompt,
        max_new_tokens=50,
        method="top_k",
        k=50,
        temperature=0.8
    )
    print(f"Top-k:  {topk_text}")
```

**Cell 10: Save Model**
```python
# Save model for fine-tuning later
import torch

save_path = '/content/drive/MyDrive/cs288_pretrained_model.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'config': CONFIG,
    'vocab': vocab,
    'merges': merges,
    'special_tokens': SPECIAL_TOKENS,
}, save_path)

print(f"✅ Model saved to: {save_path}")
```

---

## Option 2: Using GitHub (Alternative)

If you have your code in a GitHub repo:

```python
# Cell 1: Clone repo
!git clone https://github.com/yourusername/cs288-sp26-a2.git
%cd cs288-sp26-a2

# Cell 2: Install dependencies
!pip install -q torch tiktoken

# Then continue with cells 3-10 above
```

---

## Tips for Colab

1. **Enable GPU**: Runtime → Change runtime type → GPU (T4)

2. **Monitor Resources**: 
   - Click the RAM/Disk icon in the top right
   - GPU usage: `!nvidia-smi`

3. **Prevent Disconnection**:
   - Keep the tab active
   - Use Colab Pro for longer sessions
   - Save checkpoints frequently

4. **Batch Size Tuning**:
   - If you get OOM errors, reduce batch_size
   - Start with 16-32 and adjust

5. **Time Estimates**:
   - Small model + small data (quick test): ~2-5 min
   - Medium model + 10k stories: ~10-30 min
   - Full training: ~1-2 hours

---

## Next Steps

After completing **Part 4A Step 1** (Pre-training):
- **Step 2**: Fine-tune on MCQA (add classification head)
- **Step 3**: Generate predictions and save to JSON

Let me know when you want to move to the next step!
