# ============================================================
# Small LM from scratch — full pretraining script for Colab.
# Paste this into a single Colab cell (or split by the "# ---" markers
# into separate cells), set Runtime > Change runtime type > T4 GPU
# BEFORE running anything.
# ============================================================

# --- Setup ---
# In a Colab cell, run this line by itself first:
# !pip install -q tiktoken datasets

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import tiktoken
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
assert device == "cuda", "GPU not detected — check Runtime > Change runtime type > T4 GPU"

# --- Tokenizer ---
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
eot_token = enc.eot_token

# --- Model classes ---
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class GQA(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.group_size = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, self.d_k * num_heads)
        self.W_k = nn.Linear(d_model, self.d_k * num_kv_heads)
        self.W_v = nn.Linear(d_model, self.d_k * num_kv_heads)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)

        K = torch.repeat_interleave(K, self.group_size, dim=1)
        V = torch.repeat_interleave(V, self.group_size, dim=1)

        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn_out = weights @ V

        output = attn_out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.W_o(output)
        return output


class TransfomerBlock(nn.Module):
    def __init__(self, d_model, num_heads, num_kv_heads, d_ff):
        super().__init__()
        self.attn = GQA(d_model, num_heads, num_kv_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class SmallLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_kv_heads, d_ff, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_embedding = nn.Embedding(max_seq_len, d_model)
        self.full_block = nn.ModuleList(
            TransfomerBlock(d_model, num_heads, num_kv_heads, d_ff) for _ in range(num_layers)
        )
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        batch, seq_len = token_ids.shape

        token_emb = self.token_embedding(token_ids)
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_emb = self.positional_embedding(positions)

        x = token_emb + pos_emb

        for block in self.full_block:
            x = block(x)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits


# --- Dataset ---
class TokenDataset(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) // (self.block_size + 1)

    def __getitem__(self, idx):
        start = idx * self.block_size
        chunk = self.tokens[start: start + self.block_size + 1]
        input_ids = chunk[:-1]
        target_ids = chunk[1:]
        return input_ids, target_ids


# --- Config ---
block_size = 128
d_model = 128
num_heads = 4
num_kv_heads = 2
d_ff = 512
num_layers = 4
batch_size = 32
num_epochs = 1
warmup_steps = 200
save_every = 500

checkpoint_path = "/content/drive/MyDrive/small_llm_checkpoint_500k.pt"

# --- Mount Drive ---
from google.colab import drive
drive.mount('/content/drive')

# --- Load + tokenize dataset ---
dataset = load_dataset("roneneldan/TinyStories")
subset = dataset["train"].select(range(500000))

all_tokens = []
for line in subset:
    ids = enc.encode(line["text"])
    all_tokens.extend(ids)
    all_tokens.append(eot_token)

all_tokens = torch.tensor(all_tokens, dtype=torch.long)
print("total tokens:", all_tokens.shape)

train_ds = TokenDataset(all_tokens, block_size)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
print("steps per epoch:", len(train_loader))

# --- Model, optimizer, resume-from-checkpoint ---
model = SmallLM(
    vocab_size=vocab_size, d_model=d_model, num_heads=num_heads,
    num_kv_heads=num_kv_heads, d_ff=d_ff, num_layers=num_layers, max_seq_len=block_size,
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

total_steps = len(train_loader) * num_epochs

def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    return max(0.1, (total_steps - step) / (total_steps - warmup_steps))

scheduler = LambdaLR(optimizer, lr_lambda)

start_epoch = 0
start_step = 0

if os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    start_epoch = ckpt["epoch"]
    start_step = ckpt["step"] + 1
    print(f"resumed from epoch {start_epoch} step {start_step}")
else:
    print("no checkpoint found, starting fresh")

print("model device:", next(model.parameters()).device)

# --- Training loop ---
model.train()

for epoch in range(start_epoch, num_epochs):
    for step, (x, y) in enumerate(train_loader):
        if epoch == start_epoch and step < start_step:
            continue

        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 50 == 0:
            print(f"epoch {epoch} step {step}/{len(train_loader)} loss {loss.item():.4f} lr {scheduler.get_last_lr()[0]:.6f}")

        if step % save_every == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": {
                    "vocab_size": vocab_size, "d_model": d_model, "num_heads": num_heads,
                    "num_kv_heads": num_kv_heads, "d_ff": d_ff, "num_layers": num_layers,
                    "max_seq_len": block_size,
                },
                "epoch": epoch,
                "step": step,
                "loss": loss.item(),
            }, checkpoint_path)

print("training complete")

# --- Generation ---
@torch.no_grad()
def generate(model, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    model.eval()
    ids = enc.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        ids_cond = ids[:, -block_size:]
        logits = model(ids_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature
        top_vals, top_idx = torch.topk(logits, top_k)
        probs = torch.softmax(top_vals, dim=-1)
        next_token = top_idx[0, torch.multinomial(probs[0], 1)]

        ids = torch.cat([ids, next_token.view(1, 1)], dim=1)

    model.train()
    return enc.decode(ids[0].tolist())


print(generate(model, "Once upon a time"))
