import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
batch_size = 64       # sequences processed in parallel
block_size = 256      # maximum context length for predictions
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200      # iterations to average loss over when evaluating
embed_dim = 384       # embedding dimension
num_heads = 6
num_layers = 6
dropout = 0.2

torch.manual_seed(1337)

# -----------------------------------------------------------------------------
# Data: load text and build character-level tokenizer
# -----------------------------------------------------------------------------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


def encode(s: str) -> list[int]:
    """Encode string to list of token indices."""
    return [char_to_idx[c] for c in s]


def decode(indices: list[int]) -> str:
    """Decode list of token indices to string."""
    return ''.join([idx_to_char[i] for i in indices])


# Train/validation split (90% / 10%)
token_ids = torch.tensor(encode(text), dtype=torch.long)
num_train = int(0.9 * len(token_ids))
train_data = token_ids[:num_train]
val_data = token_ids[num_train:]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def get_batch(split: str):
    """Sample a batch of (input, target) sequences."""
    data = train_data if split == 'train' else val_data
    start_indices = torch.randint(len(data) - block_size, (batch_size,))
    inputs = torch.stack([data[i : i + block_size] for i in start_indices])
    targets = torch.stack([data[i + 1 : i + block_size + 1] for i in start_indices])
    return inputs.to(device), targets.to(device)


@torch.no_grad()
def estimate_loss():
    """Compute average loss over train and validation sets."""
    model.eval()
    loss_by_split = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            inputs, targets = get_batch(split)
            _, loss = model(inputs, targets)
            losses[i] = loss.item()
        loss_by_split[split] = losses.mean()
    model.train()
    return loss_by_split

# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------
class AttentionHead(nn.Module):
    """Single head of self-attention."""

    def __init__(self, head_dim):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        self.register_buffer('causal_mask', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size_, seq_len, _ = x.shape
        key = self.key(x)
        query = self.query(x)
        # attention scores (scaled dot-product)
        attention_weights = query @ key.transpose(-2, -1) * (x.shape[-1] ** -0.5)
        attention_weights = attention_weights.masked_fill(
            self.causal_mask[:seq_len, :seq_len] == 0, float('-inf')
        )
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # weighted sum of values
        value = self.value(x)
        return attention_weights @ value


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel with output projection."""

    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(head_dim) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        head_outputs = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(head_outputs))


class FeedForward(nn.Module):
    """MLP: two linear layers with ReLU (position-wise)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block: self-attention + feed-forward, with residual connections."""

    def __init__(self):
        super().__init__()
        head_dim = embed_dim // num_heads
        self.self_attention = MultiHeadAttention(num_heads, head_dim)
        self.feed_forward = FeedForward()
        self.norm_before_attn = nn.LayerNorm(embed_dim)
        self.norm_before_ff = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.self_attention(self.norm_before_attn(x))
        x = x + self.feed_forward(self.norm_before_ff(x))
        return x


class TransformerLanguageModel(nn.Module):
    """Character-level transformer language model (GPT-style)."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        batch_size_, seq_len = idx.shape

        token_emb = self.token_embedding(idx)
        position_emb = self.position_embedding(torch.arange(seq_len, device=device))
        x = token_emb + position_emb
        x = self.blocks(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            logits_flat = logits.view(-1, logits.shape[-1])
            targets_flat = targets.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Autoregressively generate new tokens given a context."""
        for _ in range(max_new_tokens):
            context = idx[:, -block_size:]
            logits, _ = self(context)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
model = TransformerLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):
    if step % eval_interval == 0 or step == max_iters - 1:
        loss_by_split = estimate_loss()
        print(
            f"step {step}: train loss {loss_by_split['train']:.4f}, "
            f"val loss {loss_by_split['val']:.4f}"
        )

    input_batch, target_batch = get_batch('train')
    _, loss = model(input_batch, target_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500)
print(decode(generated[0].tolist()))
# open('more.txt', 'w').write(decode(model.generate(context, max_new_tokens=10000)[0].tolist()))