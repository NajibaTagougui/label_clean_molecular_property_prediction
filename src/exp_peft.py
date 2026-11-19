
import os, sys, json, time, pickle, copy, argparse
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from model import MolEncoder, Vocab, LoRALinear, encode_batch
from clean import inject_noise, reference_margin, oof_probs, clean_labels

RES = '/home/fmclean/results'; FIG = '/home/fmclean/figs'; WORK = '/home/fmclean/work'
torch.set_num_threads(2)

def load_fm(tag='pretrained'):
    ck = torch.load(f'{WORK}/fm_{tag}.pt', map_location='cpu', weights_only=False)
    v = Vocab([]); v.itos = ck['vocab']; v.stoi = {t: i for i, t in enumerate(v.itos)}
    m = MolEncoder(len(v), **ck['cfg']); m.load_state_dict(ck['sd'])
    return m, v, ck['cfg']

class Net(nn.Module):
    def __init__(self, enc, d):
        super().__init__(); self.enc = enc; self.head = nn.Linear(2 * d, 1)
    def forward(self, ids):
        _, pooled = self.enc(ids)
        return self.head(pooled).squeeze(-1)

def configure(net, mode, r=8):
    for p in net.parameters(): p.requires_grad = False
    for p in net.head.parameters(): p.requires_grad = True
    if mode == 'probe':
        pass
    elif mode == 'bitfit':
        for n, p in net.enc.named_parameters():
            if n.endswith('.bias') or 'ln' in n or 'LayerNorm' in n: p.requires_grad = True
    elif mode.startswith('lora'):
        rr = int(mode[4:]) if len(mode) > 4 else r
        for b in net.enc.blocks:
            b.q = LoRALinear(b.q, r=rr); b.v = LoRALinear(b.v, r=rr)
            for nm in ('q', 'v'):
                mod = getattr(b, nm)
                mod.A.requires_grad = True; mod.B.requires_grad = True
                for p in mod.lin.parameters(): p.requires_grad = False
    elif mode == 'full':
        for p in net.parameters(): p.requires_grad = True
    else: raise ValueError(mode)
    return net

def auc(y, p): return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan

def train_ft(ids_tr, y_tr, w_tr, ids_va, y_va, ids_te, y_te, mode, cfg, enc_sd, vocab_n,
             lr, epochs=8, bs=32, patience=3, seed=0):
    torch.manual_seed(seed)
    enc = MolEncoder(vocab_n, **cfg); enc.load_state_dict(enc_sd)
    net = configure(Net(enc, cfg['d']), mode)
    tp = sum(p.numel() for p in net.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    yt = torch.from_numpy(y_tr.astype(np.float32)); wt = torch.from_numpy(w_tr.astype(np.float32))
    n = len(y_tr); best, best_state, bad = -np.inf, None, 0
    t0 = time.time()
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            if len(b) < 2: continue
            logit = net(ids_tr[b])
            loss = (F.binary_cross_entropy_with_logits(logit, yt[b], reduction='none') * wt[b]).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(ids_va[i:i + 256]) for i in range(0, len(y_va), 256)]).sigmoid().numpy()
        a = auc(y_va, pv)
        if np.isnan(a): a = 0.5
        if a > best: best, best_state, bad = a, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience: break
    sec = time.time() - t0
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        pt = torch.cat([net(ids_te[i:i + 256]) for i in range(0, len(y_te), 256)]).sigmoid().numpy()
    return dict(auc=auc(y_te, pt), auprc=average_precision_score(y_te, pt) if len(np.unique(y_te)) > 1 else np.nan,
                val_auc=best, trainable=tp, sec=sec, epochs_run=ep + 1, lr=lr)

# selected on the pilot's *validation* AUC (ties broken toward the larger lr)
LRS = dict(probe=1e-2, bitfit=3e-3, lora8=3e-3, full=3e-4)

def main(pilot=False, datasets=('bbbp', 'bace', 'clintox'), splits=('scaffold_0', 'scaffold_1'),
         noises=(('sym', 0.0), ('sym', 0.3)), cleans=('none', 'fmclean-hybrid'),
         modes=('probe', 'bitfit', 'lora8', 'full'), out='exp_peft.csv', epochs=8):
    enc0, vocab, cfg = load_fm('pretrained')
    enc_sd = copy.deepcopy(enc0.state_dict()); vn = len(vocab)
    rows = []
    for name in datasets:
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        E = np.load(f'{WORK}/emb_{name}.npz')
        Zfm_all = StandardScaler().fit_transform(E['pretrained']).astype(np.float32)
        ids_all = encode_batch(vocab, d['smiles'], cfg['max_len'])
        y_all = d['y']
        for sp in splits:
            tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]
            trva = np.concatenate([tr, va])
            Zfm, Zfp = Zfm_all[trva], d['ecfp'][trva].astype(np.float32)
            y_trva = y_all[trva]
            refm = reference_margin(Zfp, y_trva)
            # keep an internal val split (noisy labels, realistic model selection)
            rs = np.random.RandomState(0); order = rs.permutation(len(trva))
            n_va = max(50, int(0.12 * len(trva)))
            vi, ti = order[:n_va], order[n_va:]
            for kind, rate in noises:
                yn, flipped = inject_noise(y_trva, rate, kind, seed=hash((name, sp, kind, rate)) % 10000,
                                           ref_margin=refm)
                p_fm = oof_probs(Zfm, yn, 0)
                for cl in cleans:
                    c = clean_labels(cl, Zfm, Zfp, yn, oracle_flip=flipped, p_fm=p_fm)
                    keep, yo = c['keep'], c['y_out']
                    sel = np.array([i for i in ti if keep[i]])
                    w = np.ones(len(sel), np.float32)
                    for mode in modes:
                        r = train_ft(ids_all[trva][sel], yo[sel], w,
                                     ids_all[trva][vi], yn[vi],
                                     ids_all[te], y_all[te],
                                     mode, cfg, enc_sd, vn, LRS[mode], epochs=epochs)
                        rows.append(dict(dataset=name, split=sp, noise=kind, rate=rate,
                                         clean=cl, mode=mode, n_train=len(sel), **r))
                        print(rows[-1], flush=True)
                        pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', action='store_true'); ap.add_argument('--out', default='exp_peft.csv')
    ap.add_argument('--epochs', type=int, default=8)
    a = ap.parse_args()
    if a.pilot:
        for mode, lr in [('probe', 1e-2), ('probe', 1e-3), ('bitfit', 3e-3), ('bitfit', 1e-3),
                         ('lora8', 3e-3), ('lora8', 1e-3), ('full', 3e-4), ('full', 1e-4)]:
            LRS[mode] = lr
            main(datasets=('bbbp',), splits=('scaffold_0',), noises=(('sym', 0.0),),
                 cleans=('none',), modes=(mode,), out=f'peft_pilot_{mode}_{lr}.csv', epochs=a.epochs)
    else:
        main(out=a.out, epochs=a.epochs)
