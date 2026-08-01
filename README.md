# small-llm-from-scratch

A small language model built entirely from scratch in PyTorch — no `transformers` library, no pre-built attention layers. Every piece (tensors → autograd → attention → grouped-query attention → transformer block → full model → training loop → generation) was implemented by hand, then pretrained on GPU and deployed as a live demo.

**Live demo:** https://huggingface.co/spaces/maanas1234321/small-llm-from-scratch

## What's in here

- Tensor/autograd fundamentals, `nn.Module` mechanics
- Scaled dot-product self-attention with causal masking, implemented from raw matmuls
- Multi-Head Attention (MHA)
- **Grouped-Query Attention (GQA)** — the attention variant used in the final model, generalizing to Multi-Query Attention (MQA) as the `num_kv_heads=1` special case
- A pre-norm Transformer decoder block (GQA + feedforward, residual connections, LayerNorm)
- A full GPT-style language model: token + positional embeddings, stacked transformer blocks, output projection to vocabulary logits
- Batched training with `tiktoken` (GPT-2 BPE tokenizer), a packed-sequence dataset/dataloader, cross-entropy loss, AdamW, and a warmup+linear-decay LR schedule
- Resumable checkpointing (survives Colab disconnects)
- Autoregressive text generation with temperature and top-k sampling
- Deployment as a Gradio app on Hugging Face Spaces (ZeroGPU hardware)

## Model

| | |
|---|---|
| Attention | Grouped-Query Attention, 4 query heads, 2 KV heads |
| Layers | 4 transformer blocks |
| `d_model` | 128 |
| Feedforward dim | 512 |
| Context length | 128 tokens |
| Tokenizer | GPT-2 BPE (`tiktoken`, vocab size 50,257) |
| Parameters | ~small (millions, not billions — this is a learning-scale model) |

## Training

- **Dataset:** [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), 500,000 examples (~112M tokens after tokenization)
- **Hardware:** single T4 GPU, Google Colab (free tier)
- **Schedule:** 1 epoch, batch size 32, AdamW (`lr=3e-4`), 200-step linear warmup then linear decay
- **Result:** loss dropped from ~10.97 (untrained baseline, ≈ `ln(vocab_size)`, i.e. random guessing) to ~2.5

Sample output after training (prompt: *"Once upon a time"*):

> Once upon a time, there was a little girl named Lily. Lily loved to watch in the forest when she could see the trees. One day, she saw a big, scary tree with a hole in it. She asked her mom, "What is that sound?"
>
> Her mom smiled and said, "My name won't fly like," said Lily. "I am here to help you up," said her mom.

## Repo structure

```
001.ipynb            — main build notebook: architecture built and tested step by step
pretrain_colab.py    — standalone, self-contained script for full pretraining on Colab (resumable checkpointing included)
space/               — deployed Gradio app (app.py + requirements.txt), mirrors the live HF Space
```

## Running it yourself

**Pretraining:** open `pretrain_colab.py` in Google Colab (set `Runtime > Change runtime type > T4 GPU`), run top to bottom. Checkpoints save to Google Drive every 500 steps and auto-resume on reconnect.

**Local inference/demo:** see `space/app.py` — loads a checkpoint and serves generation through a Gradio interface. Requires a trained `small_llm_checkpoint_500k.pt` (or your own checkpoint with matching config) in the same directory.

## Why this exists

Built as a from-scratch learning project to go from "can read PyTorch code" to "can implement transformers and attention mechanisms (MHA/MQA/GQA) from raw tensor operations, with real understanding of every shape and design decision" — then proved that understanding by shipping a working, deployed model end to end.
