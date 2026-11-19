"""Extract frozen embeddings from pretrained FM and random-init control for every dataset."""
import os, sys, pickle, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from model import MolEncoder, Vocab, embed

WORK = '/home/claude/fmclean/work'
DATASETS = ['bbbp', 'bace', 'clintox', 'tox21', 'sider', 'hiv']

def load(tag):
    ck = torch.load(f'{WORK}/fm_{tag}.pt', map_location='cpu', weights_only=False)
    v = Vocab([]); v.itos = ck['vocab']; v.stoi = {t: i for i, t in enumerate(v.itos)}
    m = MolEncoder(len(v), **ck['cfg']); m.load_state_dict(ck['sd']); m.eval()
    return m, v, ck['cfg']['max_len']

if __name__ == '__main__':
    models = {t: load(t) for t in ('pretrained', 'random')}
    for name in DATASETS:
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        store = {}
        for tag, (m, v, ml) in models.items():
            t0 = time.time()
            store[tag] = embed(m, v, d['smiles'], max_len=ml)
            print(name, tag, store[tag].shape, f'{time.time()-t0:.0f}s', flush=True)
        np.savez_compressed(f'{WORK}/emb_{name}.npz', **store)
