"""Two robustness runs sharing one harness.

--what seeds : three independent noise draws per cell, so that noise-realisation variance can be
               separated from split variance (every cell elsewhere uses a single draw).
--what hiv   : HIV (40,950 molecules, 3.4% positive) -- a 27x scale jump over the main grid and the
               most extreme class imbalance available, which is where Section 5.8 predicts trouble.
"""
import os, sys, time, pickle, argparse, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from clean import (inject_noise, reference_margin, oof_probs, clean_labels,
                   detection_metrics, knn_graph, desc_reference_matrix)
from exp_main import ALGO_MAP, ev

WORK = '/home/claude/fmclean/work'; RES = '/home/claude/fmclean/results'
PAIRS = [('none', '-'), ('smallloss', 'ecfp'), ('cl', 'ecfp'), ('knn', 'ecfp'),
         ('fmclean-remove', 'ecfp'), ('fmclean-hybrid', 'ecfp'),
         ('fmclean-reweight', 'ecfp'), ('fmclean-capped', 'ecfp'), ('oracle', '-')]
K, ALPHA = 15, 0.5

def one_cell(Z, y_tr, Zte, y_te, kind, rate, seed, refm, g, rows, meta):
    yn, flipped = inject_noise(y_tr, rate, kind, seed=seed, ref_margin=refm)
    p = oof_probs(Z, yn, 0)
    for algo, space in PAIRS:
        meth = ALGO_MAP.get(algo, algo)
        c = clean_labels(meth, Z, Z, yn, k=K, alpha=ALPHA, oracle_flip=flipped,
                         g_fm=g, g_fp=g, p_fm=p, p_fp=p)
        keep, yo, w = c['keep'], c['y_out'], c.get('weight')
        ww = None if w is None else w[keep]
        a, pr = ev(Z[keep], yo[keep], Zte, y_te, ww)
        dm = detection_metrics(c['score'], flipped, c['flagged'])
        act = (y_tr == 1); rm = ~keep
        rows.append(dict(noise=kind, rate=rate, noise_seed=seed, algo=algo, space=space,
                         method=f'{algo}@{space}', n_train=int(keep.sum()), n_orig=len(y_tr),
                         base_pos=float(y_tr.mean()),
                         actives_removed=float((rm & act).sum() / max(act.sum(), 1)),
                         pos_after=float(yo[keep].mean()) if keep.sum() else np.nan,
                         label_acc=float((yo[keep] == y_tr[keep]).mean()),
                         true_err=int(flipped.sum()), auc=a, auprc=pr, **dm, **meta))

def run_seeds(out='exp_seeds.csv'):
    rows = []
    for name in ['bbbp', 'bace', 'clintox', 'tox21']:
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        y_all = d['y']; ec = d['ecfp'].astype(np.float32)
        descm = desc_reference_matrix(d['desc'])
        for sp in ['scaffold_0', 'scaffold_1', 'scaffold_2']:
            tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]
            trva = np.concatenate([tr, va])
            Z, y_tr, y_te = ec[trva], y_all[trva], y_all[te]
            g = knn_graph(Z, max(50, K)); refm = reference_margin(descm[trva], y_tr)
            t0 = time.time()
            for kind, rate in [('sym', 0.3), ('inst', 0.3)]:
                for seed in (11, 22, 33):
                    one_cell(Z, y_tr, ec[te], y_te, kind, rate, seed, refm, g, rows,
                             dict(dataset=name, split=sp))
            print(f'{name} {sp} {time.time()-t0:.0f}s rows={len(rows)}', flush=True)
            pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    print('wrote', out, len(rows), flush=True)

def run_hiv(out='exp_hiv.csv'):
    rows = []
    d = pickle.load(open(f'{WORK}/hiv.pkl', 'rb'))
    y_all = d['y']; ec = d['ecfp'].astype(np.float32)
    descm = desc_reference_matrix(d['desc'])
    print(f'HIV: {len(y_all)} molecules, {y_all.mean()*100:.2f}% positive', flush=True)
    for sp in ['scaffold_0', 'scaffold_1', 'scaffold_2']:
        tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]
        trva = np.concatenate([tr, va])
        Z, y_tr, y_te = ec[trva], y_all[trva], y_all[te]
        t0 = time.time()
        g = knn_graph(Z, max(50, K))
        print(f'  {sp}: knn graph on {len(y_tr)} molecules in {time.time()-t0:.0f}s', flush=True)
        refm = reference_margin(descm[trva], y_tr)
        for kind, rate in [('sym', 0.0), ('sym', 0.3), ('asym', 0.3), ('inst', 0.3)]:
            one_cell(Z, y_tr, ec[te], y_te, kind, rate,
                     abs(hash(('hiv', sp, kind, rate))) % 10000, refm, g, rows,
                     dict(dataset='hiv', split=sp))
            print(f'  {sp} {kind}{rate} done {time.time()-t0:.0f}s rows={len(rows)}', flush=True)
        pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    print('wrote', out, len(rows), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--what', default='seeds')
    a = ap.parse_args()
    if a.what == 'seeds': run_seeds()
    elif a.what == 'hiv': run_hiv()
