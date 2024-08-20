
import torch
import torch.nn as nn
from torch.nn import functional as F

with open('pg142.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

st = "what are you doing"
print(encode(st))
print(decode(encode(st)))


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Hyper- Parameters
batch_size = 4
block_size = 8
vocab_size = len(chars)
n_embd = 32
dropout = 0.2
n_head = 6
n_layer = 6
print(vocab_size)

def get_batch(split):                                                           #Get Batch
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
      B,T,C = x.shape
      k = self.key(x)
      q = self.query(x)
      wei = (q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5)
      wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
      wei = F.softmax(wei, dim=-1)

      v = self.value(x)
      out = wei @ v
      return out


class MultiHeadAttention(nn.Module):
  def __init__(self, num_heads):
    super().__init__()
    head_size = n_embd // num_heads
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.dropout(self.proj(out))
    return out

class FeedForward(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n_embd, 4*n_embd),
        nn.ReLU(),
        nn.Linear(4*n_embd, n_embd),
        nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd, n_head):
    super().__init__()
    self.sa_head = MultiHeadAttention(n_head)
    self.ln1 = nn.LayerNorm(n_embd)

  def forward(self, x):
    x = x + self.sa_head(self.ln1(x))
    x = x + FeedForward(n_embd)(self.ln1(x))
    return x

from operator import pos
class Bigram(nn.Module):

  def __init__(self):
    super().__init__()
    self.embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_embed = self.embedding_table(idx) # (B, T, C)
    pos_embed = self.position_embedding(torch.arange(T))
    x = tok_embed + pos_embed
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm(x)

    if targets is None:
            loss = None
    else:
      B,T,C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):

    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx

model = Bigram()
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
for i in range(10000):
  xb, yb = get_batch('train')
  logits, loss = model(xb, yb)
  loss.backward()
  optimiser.step()
print(f'Final Loss ---> {loss}')

jst = "what do i say now?"
idx = encode(st)
idx = torch.tensor(idx).unsqueeze(0)
print(decode(model.generate(idx, max_new_tokens=200)[0].tolist()))

