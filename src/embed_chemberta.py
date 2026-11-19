"""Extract frozen ChemBERTa embeddings for every dataset (CLS + mean pooling, as for our encoder)."""
import os, sys, time, pickle, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from get_chemberta import load, MODELS

WORK = '/home/claude/fmclean/work'
DATASETS = ['bbbp', 'bace', 'clintox', 'tox21', 'sider', 'hiv']
torch.set_num_threads(2)

@torch.no_grad()
def embed(tok, model, smiles, bs=64, max_len=256):
    out = []
    for i in range(0, len(smiles), bs):
        enc = tok(smiles[i:i + bs], return_tensors='pt', padding=True,
                  truncation=True, max_length=max_len)
        h = model(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).float()
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
        out.append(torch.cat([h[:, 0], mean], -1).numpy())
    return np.concatenate(out).astype(np.float32)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tags', default='chemberta_mlm,chemberta_mtr')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    a = ap.parse_args()
    loaded = {}
    for tag in a.tags.split(','):
        tok, m, cfg, missing, unexpected, _ = load(tag, MODELS[tag])
        n = sum(p.numel() for p in m.parameters())
        print(f'{tag}: params={n:,} layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} '
              f'missing={len(missing)} unexpected={len(unexpected)}', flush=True)
        loaded[tag] = (tok, m)
    for name in a.datasets.split(','):
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        path = f'{WORK}/emb_cb_{name}.npz'
        store = dict(np.load(path)) if os.path.exists(path) else {}
        for tag, (tok, m) in loaded.items():
            t0 = time.time()
            store[tag] = embed(tok, m, d['smiles'])
            print(f'{name} {tag} {store[tag].shape} {time.time()-t0:.0f}s', flush=True)
        np.savez_compressed(path, **store)
