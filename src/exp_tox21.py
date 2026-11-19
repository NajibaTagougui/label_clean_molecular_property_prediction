"""Robustness extension: all 12 Tox21 endpoints.


"""
import os, sys, time, pickle, argparse
from collections import defaultdict
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from prep import canon, scaffold, scaffold_split, featurize
from clean import (inject_noise, reference_margin, oof_probs, clean_labels,
                   detection_metrics, knn_graph, desc_reference_matrix)
RDLogger.DisableLog('rdApp.*')

DATA = '/home/data'; WORK = '/home/fmclean/work'; RES = '/home/fmclean/results'
ENDPOINTS = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
# SIDER spans a far wider class-balance range (1.5%-92.4% positive) than Tox21 (3-19%),
# which makes it the decisive test of whether class balance drives the treatment result.
import pandas as _pd
def sider_endpoints():
    return [c for c in _pd.read_csv(f'{DATA}/sider.csv').columns if c != 'smiles']
DSETS = {'tox21': ('tox21.csv', None), 'sider': ('sider.csv', None)}
NOISE = [('sym', 0.0), ('sym', 0.3), ('asym', 0.3), ('inst', 0.3)]
PAIRS = [('none', '-'), ('smallloss', 'ecfp'), ('cl', 'ecfp'), ('knn', 'ecfp'),
         ('fmclean-remove', 'ecfp'), ('fmclean-hybrid', 'ecfp'),
         ('fmclean-reweight', 'ecfp'), ('fmclean-capped', 'ecfp'), ('oracle', '-')]
ALGO_MAP = {'cl': 'cl_fm', 'knn': 'knn_fm'}
K, ALPHA = 15, 0.5

def build_master(dataset='tox21', endpoints=None):
    """Canonicalise the pool once; featurise once; index per endpoint."""
    endpoints = endpoints or ENDPOINTS
    cache = f'{WORK}/{dataset}_master.pkl'
    if os.path.exists(cache):
        cached = pickle.load(open(cache, 'rb'))
        if set(endpoints).issubset(cached['labels']):   # stale partial caches must not be reused
            return cached
        print(f'cache covers {len(cached["labels"])} endpoints, {len(endpoints)} requested: rebuilding',
              flush=True)
    df = pd.read_csv(f'{DATA}/{DSETS[dataset][0]}')
    rows = []
    for _, r in df.iterrows():
        cs, m = canon(str(r['smiles']))
        if cs is None: continue
        rows.append((cs, scaffold(m), {e: r[e] for e in endpoints}))
  
    agg = defaultdict(lambda: (None, defaultdict(list)))
    order = []
    for cs, sc, lab in rows:
        if cs not in agg: order.append(cs)
        cur = agg[cs]; agg[cs] = (sc, cur[1])
        for e, v in lab.items():
            if pd.notna(v): agg[cs][1][e].append(float(v))
    smiles = order
    scaf = [agg[s][0] for s in smiles]
    labels = {}
    for e in endpoints:
        y = np.full(len(smiles), np.nan)
        for i, s in enumerate(smiles):
            v = agg[s][1].get(e)
            if v: y[i] = float(np.round(np.mean(v)))
        labels[e] = y
    print(f'master pool: {len(smiles)} unique molecules', flush=True)
    t0 = time.time()
    ecfp, maccs, desc = featurize(smiles)
    print(f'featurised in {time.time()-t0:.0f}s', flush=True)
    out = dict(smiles=smiles, scaffold=scaf, labels=labels, ecfp=ecfp, desc=desc)
    pickle.dump(out, open(cache, 'wb'))
    return out

def ev(Xtr, ytr, Xte, yte, w=None):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2: return (np.nan, np.nan)
    lr = LogisticRegression(max_iter=3000); lr.fit(Xtr, ytr, sample_weight=w)
    p = lr.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)

def run(out='exp_tox21.csv', endpoints=ENDPOINTS, splits=(0, 1, 2), noise=NOISE, dataset='tox21'):
    M = build_master(dataset, endpoints)
    rows = []
    for ep in endpoints:
        y_all_full = M['labels'][ep]
        idx = np.where(~np.isnan(y_all_full))[0]
        y_all = y_all_full[idx].astype(int)
        ecfp = M['ecfp'][idx].astype(np.float32)
        descm = desc_reference_matrix(M['desc'][idx])
        scaf = [M['scaffold'][i] for i in idx]
        t0 = time.time()
        for s in splits:
            tr, va, te = scaffold_split(scaf, seed=100 + s)
            tr, va, te = np.asarray(tr, int), np.asarray(va, int), np.asarray(te, int)
            if len(te) < 30: continue
            trva = np.concatenate([tr, va])
            Z, y_tr, y_te = ecfp[trva], y_all[trva], y_all[te]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2: continue
            g = knn_graph(Z, max(50, K))
            refm = reference_margin(descm[trva], y_tr)
            for kind, rate in noise:
                yn, flipped = inject_noise(y_tr, rate, kind,
                                           seed=abs(hash((ep, s, kind, rate))) % 10000,
                                           ref_margin=refm)
                p_ec = oof_probs(Z, yn, 0)
                for algo, space in PAIRS:
                    meth = ALGO_MAP.get(algo, algo)
                    c = clean_labels(meth, Z, Z, yn, k=K, alpha=ALPHA, oracle_flip=flipped,
                                     g_fm=g, g_fp=g, p_fm=p_ec, p_fp=p_ec)
                    keep, yo, w = c['keep'], c['y_out'], c.get('weight')
                    ww = None if w is None else w[keep]
                    a, pr = ev(Z[keep], yo[keep], ecfp[te], y_te, ww)
                    dm = detection_metrics(c['score'], flipped, c['flagged'])
                    act = (y_tr == 1); rm = ~keep
                    flip_to0 = act & keep & (yo == 0)
                    rows.append(dict(
                        endpoint=ep, split=f'scaffold_{s}', noise=kind, rate=rate,
                        algo=algo, space=space, method=f'{algo}@{space}',
                        n_train=int(keep.sum()), n_orig=len(y_tr),
                        base_pos=float(y_tr.mean()), n_actives=int(act.sum()),
                        frac_removed=float(rm.mean()),
                        actives_removed=float((rm & act).sum() / max(act.sum(), 1)),
                        actives_flipped=float(flip_to0.sum() / max(act.sum(), 1)),
                        pos_after=float(yo[keep].mean()) if keep.sum() else np.nan,
                        label_acc=float((yo[keep] == y_tr[keep]).mean()) if keep.sum() else np.nan,
                        auc=a, auprc=pr, **dm))
            print(f'{ep} split{s} {time.time()-t0:.0f}s rows={len(rows)}', flush=True)
        pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    print('wrote', out, len(rows), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='exp_tox21.csv')
    ap.add_argument('--dataset', default='tox21')
    ap.add_argument('--endpoints', default='')
    a = ap.parse_args()
    eps = (a.endpoints.split(',') if a.endpoints
           else (ENDPOINTS if a.dataset == 'tox21' else sider_endpoints()))
    run(out=a.out, endpoints=eps, dataset=a.dataset)
