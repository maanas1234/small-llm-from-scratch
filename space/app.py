import gradio as gr
import torch
import torch.nn as nn
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"

enc = tiktoken.get_encoding("gpt2")


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


ckpt = torch.load("small_llm_checkpoint_500k.pt", map_location=device)
cfg = ckpt["config"]

model = SmallLM(
    vocab_size=cfg["vocab_size"],
    d_model=cfg["d_model"],
    num_heads=cfg["num_heads"],
    num_kv_heads=cfg["num_kv_heads"],
    d_ff=cfg["d_ff"],
    num_layers=cfg["num_layers"],
    max_seq_len=cfg["max_seq_len"],
).to(device)

model.load_state_dict(ckpt["model_state_dict"])
model.eval()

block_size = cfg["max_seq_len"]


@torch.no_grad()
def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    ids = enc.encode(prompt)
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(int(max_new_tokens)):
        ids_cond = ids[:, -block_size:]
        logits = model(ids_cond)
        logits = logits[:, -1, :]

        logits = logits / temperature
        top_vals, top_idx = torch.topk(logits, int(top_k))
        probs = torch.softmax(top_vals, dim=-1)
        next_token = top_idx[0, torch.multinomial(probs[0], 1)]

        ids = torch.cat([ids, next_token.view(1, 1)], dim=1)

    return enc.decode(ids[0].tolist())


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", value="Once upon a time"),
        gr.Slider(10, 300, value=100, step=10, label="Max new tokens"),
        gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature"),
        gr.Slider(1, 100, value=40, step=1, label="Top-k"),
    ],
    outputs=gr.Textbox(label="Generated story"),
    title="Small LM from scratch (TinyStories)",
    description="A tiny GQA transformer, pretrained from scratch on 500k TinyStories examples.",
)

demo.launch()
