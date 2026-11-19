"""Compact SMILES transformer encoderle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
RES = '/home/fmclean/results'; FIG = '/home/fmclean/figs'; WORK = '/home/fmclean/work'
torch.set_num_threads(2)

SMI_RE = re.compile(r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])")
PAD, CLS, MASK, UNK = 0, 1, 2, 3
SPECIAL = ['<pad>', '<cls>', '<mas: tokenizer, model, MLM pretraining, PEFT adapters."""
import os, re, json, math, time, pickk>', '<unk>']

def tokenize(smi):
    return [t for t in SMI_RE.findall(smi) if t]

class Vocab:
    def __init__(self, toks):
        self.itos = SPECIAL + sorted(toks)
        self.stoi = {t: i for i, t in enumerate(self.itos)}
    def __len__(self): return len(self.itos)
    def encode(self, smi, max_len):
        ids = [CLS] + [self.stoi.get(t, UNK) for t in tokenize(smi)][:max_len - 1]
        return ids + [PAD] * (max_len - len(ids)), len(ids)
    def save(self, p): json.dump(self.itos, open(p, 'w'))
    @staticmethod
    def load(p):
        v = Vocab([]); v.itos = json.load(open(p)); v.stoi = {t: i for i, t in enumerate(v.itos)}; return v

def build_vocab(smiles, min_count=10):
    from collections import Counter
    c = Counter()
    for s in smiles: c.update(tokenize(s))
    return Vocab([t for t, n in c.items() if n >= min_count])

# ---------------- LoRA-capable linear ----------------
class LoRALinear(nn.Module):
    def __init__(self, lin: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.lin = lin
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.randn(r, lin.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(lin.out_features, r))
    def forward(self, x):
        return self.lin(x) + F.linear(F.linear(x, self.A), self.B) * self.scale

class Block(nn.Module):
    def __init__(self, d, h, ff, p=0.1):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.h, self.dh = h, d // h
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)
    def attn(self, x, mask):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(a.transpose(1, 2).reshape(B, T, D))
    def forward(self, x, mask):
        x = x + self.drop(self.attn(self.ln1(x), mask))
        return x + self.drop(self.ff(self.ln2(x)))

class MolEncoder(nn.Module):
    def __init__(self, vocab, d=192, L=4, h=6, ff=384, max_len=96, p=0.1):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h, ff, p) for _ in range(L)])
        self.ln = nn.LayerNorm(d)
        self.d, self.max_len = d, max_len
    def forward(self, ids):
        pad = ids.eq(PAD)
        x = self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device))[None]
        m = (~pad)[:, None, None, :]
        for b in self.blocks: x = b(x, m)
        x = self.ln(x)
        w = (~pad).float().unsqueeze(-1)
        mean = (x * w).sum(1) / w.sum(1).clamp(min=1)
        return x, torch.cat([x[:, 0], mean], -1)   # (tokens, [CLS];mean-pool) -> 2d

class MLMHead(nn.Module):
    def __init__(self, d, vocab):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, vocab))
    def forward(self, x): return self.net(x)

def encode_batch(vocab, smiles, max_len):
    ids = np.full((len(smiles), max_len), PAD, dtype=np.int64)
    for i, s in enumerate(smiles):
        e, n = vocab.encode(s, max_len); ids[i] = e
    return torch.from_numpy(ids)

@torch.no_grad()
def embed(model, vocab, smiles, max_len=96, bs=256):
    model.eval(); out = []
    for i in range(0, len(smiles), bs):
        ids = encode_batch(vocab, smiles[i:i + bs], max_len)
        _, pooled = model(ids)
        out.append(pooled.numpy())
    return np.concatenate(out).astype(np.float32)
