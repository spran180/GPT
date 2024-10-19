# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

!nvidia-smi

!wget "https://drive.google.com/file/d/177ssqeCIKlAd3c4uKo824lKUTKhf262O/view?usp=sharing"

import torch
import torch.nn as nn
from torch.nn import functional as F

with open('/kaggle/input/textfile/pg142.txt', 'r', encoding='utf-8') as f:
    text = f.read()

len(text)



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
batch_size = 64
block_size = 258
vocab_size = len(chars)
n_embd = 384
dropout = 0.2
n_head = 6
n_layer = 6
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(vocab_size)

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
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
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)

        # Create the mask dynamically
        tril = torch.tril(torch.ones(T, T, device=x.device)) 
        wei = wei.masked_fill(tril == 0, float('-inf')) 
        wei = F.softmax(wei, dim=-1) 

        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, head_size)
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
    self.ff = FeedForward(n_embd)  

  def forward(self, x):
    x = x + self.sa_head(self.ln1(x))
    x = x + self.ff(self.ln1(x))  
    return x

from operator import pos
class Bigram(nn.Module):

  def __init__(self):
    super().__init__()
    self.embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding = nn.Embedding(block_size, n_embd).to(device)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)
    self.lm = nn.Linear(n_embd, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape
    tok_embed = self.embedding_table(idx) # (B, T, C)
    pos_embed = self.position_embedding(torch.arange(T, device=device))
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
model = model.to(device)
# optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
# for i in range(5000):
#   xb, yb = get_batch('train')
#   xb, yb = xb.to(device), yb.to(device)
#   logits, loss = model(xb, yb)
#   if i % 500 == 0 or i == 4999:
#         print(f'{i} ----> {loss}')
#   optimiser.zero_grad()
#   loss.backward()
#   optimiser.step()

# torch.save(model.state_dict(), 'model.pth')
state_dict = torch.load('/kaggle/input/practise-gpt/pytorch/default/1/model (2).pth') 
model.load_state_dict(state_dict) # Load the state dictionary
model.eval()

jst = "what do i say now?"
idx = encode(st)
idx = torch.tensor(idx).unsqueeze(0).to(device)
print(decode(model.generate(idx, max_new_tokens=200)[0].tolist()))




model = torch.load('model.pth')
model.eval()

jst = "what do i say now?"
idx = encode(st)
idx = torch.tensor(idx).unsqueeze(0).to(device)
print(decode(model.generate(idx, max_new_tokens=500)[0].tolist()))

torch.save(model.state_dict(), 'model.pth')

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
TORCH_USE_CUDA_DSA=True

