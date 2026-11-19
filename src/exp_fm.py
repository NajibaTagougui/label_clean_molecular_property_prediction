"""Representation study with real chemical foundation models.

Adds ChemBERTa spaces to the same cells as exp_main.py (identical deterministic noise seeds),
so results are directly pairable with the existing run.
"""
import os, sys, time, pickle, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from clean import (inject_noise, reference_margin, oof_probs, clean_labels,
                   detection_metrics, knn_graph, desc_reference_matrix)
from exp_main import DATASETS, SPLITS, NOISE, ALGO_MAP, ev, K, ALPHA
WORK = '/home/fmclean/work'; RES = '/home/fmclean/results'
ALGOS = ['cl', 'knn', 'fmclean-hybrid', 'fmclean-remove']

def run(datasets=DATASETS, splits=SPLITS, noise=NOISE, tags=('chemberta_mlm', 'chemberta_mtr'),
        out='exp_fm.csv', k=K, alpha=ALPHA):
    rows = []
    for name in datasets:
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        CB = np.load(f'{WORK}/emb_cb_{name}.npz')
        y_all = d['y']
        ec = d['ecfp'].astype(np.float32)
        SP_ALL = {t: StandardScaler().fit_transform(CB[t]).astype(np.float32) for t in tags}
        SP_ALL['ecfp'] = ec
        t0 = time.time()
        for sp in splits:
            tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]
            trva = np.concatenate([tr, va])
            S = {kk: vv[trva] for kk, vv in SP_ALL.items()}
            y_tr, y_te = y_all[trva], y_all[te]
            Ste_ecfp = SP_ALL['ecfp'][te]
            graphs = {kk: knn_graph(vv, max(50, k)) for kk, vv in S.items() if kk in tags}
            refm = (reference_margin(desc_reference_matrix(d['desc'])[trva], y_tr)
                    if any(n[0] == 'inst' for n in noise) else None)
            for kind, rate in noise:
                yn, flipped = inject_noise(y_tr, rate, kind,
                                           seed=abs(hash((name, sp, kind, rate))) % 10000,
                                           ref_margin=refm)
                probs = {kk: oof_probs(S[kk], yn, 0) for kk in tags}
                for space in tags:
                    for algo in ALGOS:
                        t1 = time.time()
                        meth = ALGO_MAP.get(algo, algo)
                        Z = S[space]
                        c = clean_labels(meth, Z, Z, yn, k=k, alpha=alpha, oracle_flip=flipped,
                                         g_fm=graphs[space], g_fp=graphs[space],
                                         p_fm=probs[space], p_fp=probs[space])
                        keep, yo, w = c['keep'], c['y_out'], c.get('weight')
                        ww = None if w is None else w[keep]
                        a_ec, p_ec = ev(S['ecfp'][keep], yo[keep], Ste_ecfp, y_te, ww)
                        dm = detection_metrics(c['score'], flipped, c['flagged'])
                        rows.append(dict(dataset=name, split=sp, noise=kind, rate=rate,
                                         algo=algo, space=space, method=f'{algo}@{space}',
                                         n_train=int(keep.sum()), n_orig=len(y_tr),
                                         label_acc=float((yo[keep] == y_tr[keep]).mean()),
                                         est_err=c.get('est_err'), true_err=int(flipped.sum()),
                                         auc=a_ec, auprc=p_ec, clean_sec=time.time() - t1, **dm))
            print(f'{name} {sp} {time.time()-t0:.0f}s rows={len(rows)}', flush=True)
            pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    print('wrote', out, len(rows), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='exp_fm.csv')
    ap.add_argument('--tags', default='chemberta_mlm,chemberta_mtr')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--noise', default='')
    a = ap.parse_args()
    nz = NOISE if not a.noise else [(k, r) for (k, r) in NOISE if k in a.noise.split(',')]
    run(datasets=a.datasets.split(','), tags=tuple(a.tags.split(',')), out=a.out, noise=nz)
