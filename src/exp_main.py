"""Experiment 1: controlled label-noise study.

Design: the *cleaning representation* is an explicit factor; the downstream learner is held
fixed (ECFP4 + logistic regression, primary; frozen-encoder probe, secondary) so that
differences in test AUC isolate the quality of label cleaning rather than representation power.
Noise is injected into train+validation; test labels stay clean.
"""
import os, sys, json, time, pickle, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from clean import (inject_noise, reference_margin, oof_probs, clean_labels,
                   detection_metrics, knn_graph, desc_reference_matrix)

WORK = '/home/claude/fmclean/work'; RES = '/home/claude/fmclean/results'
DATASETS = ['bbbp', 'bace', 'clintox', 'tox21', 'sider']
SPLITS = [f'{a}_{s}' for a in ('scaffold', 'random') for s in (0, 1, 2)]
NOISE = [('sym', 0.0), ('sym', 0.1), ('sym', 0.2), ('sym', 0.3), ('sym', 0.4),
         ('asym', 0.2), ('asym', 0.4), ('inst', 0.2), ('inst', 0.4)]
ALGOS = ['smallloss', 'cl', 'knn', 'fmclean-remove', 'fmclean-relabel',
         'fmclean-hybrid', 'fmclean-reweight']
PAIRS = ([('none', '-'), ('oracle', '-')]
         + [(a, s) for s in ('fm', 'ecfp') for a in ALGOS]
         + [(a, s) for s in ('fmcat', 'fmrand') for a in ('cl', 'knn', 'fmclean-hybrid')])
K = 15; ALPHA = 0.5
ALGO_MAP = {'cl': 'cl_fm', 'knn': 'knn_fm'}

def ev(Xtr, ytr, Xte, yte, w=None):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2: return (np.nan, np.nan)
    lr = LogisticRegression(max_iter=3000); lr.fit(Xtr, ytr, sample_weight=w)
    p = lr.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)

def run(datasets=DATASETS, out='exp_main.csv', splits=SPLITS, noise=NOISE,
        pairs=PAIRS, k=K, alpha=ALPHA):
    rows = []
    for name in datasets:
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        E = np.load(f'{WORK}/emb_{name}.npz')
        y_all = d['y']
        fm = StandardScaler().fit_transform(E['pretrained']).astype(np.float32)
        fr = StandardScaler().fit_transform(E['random']).astype(np.float32)
        ec = d['ecfp'].astype(np.float32)
        SP_ALL = dict(fm=fm, ecfp=ec, fmrand=fr,
                      fmcat=np.hstack([fm, StandardScaler().fit_transform(ec)]).astype(np.float32))
        t0 = time.time()
        for sp in splits:
            tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]
            trva = np.concatenate([tr, va])
            S = {kk: vv[trva] for kk, vv in SP_ALL.items()}
            Ste = {kk: vv[te] for kk, vv in SP_ALL.items()}
            y_tr, y_te = y_all[trva], y_all[te]
            graphs = {kk: knn_graph(vv, max(50, k)) for kk, vv in S.items()}
            refm = (reference_margin(desc_reference_matrix(d['desc'])[trva], y_tr)
                    if any(n[0] == 'inst' for n in noise) else None)
            for kind, rate in noise:
                yn, flipped = inject_noise(y_tr, rate, kind,
                                           seed=abs(hash((name, sp, kind, rate))) % 10000,
                                           ref_margin=refm)
                probs = {kk: oof_probs(vv, yn, 0) for kk, vv in S.items()}
                for algo, space in pairs:
                    t1 = time.time()
                    meth = ALGO_MAP.get(algo, algo)
                    Z = S.get(space, S['fm'])
                    c = clean_labels(meth, Z, Z, yn, k=k, alpha=alpha, oracle_flip=flipped,
                                     g_fm=graphs.get(space), g_fp=graphs.get(space),
                                     p_fm=probs.get(space), p_fp=probs.get(space))
                    keep, yo, w = c['keep'], c['y_out'], c.get('weight')
                    ww = None if w is None else w[keep]
                    a_ec, p_ec = ev(S['ecfp'][keep], yo[keep], Ste['ecfp'], y_te, ww)
                    a_fm, p_fm2 = ev(S['fm'][keep], yo[keep], Ste['fm'], y_te, ww)
                    dm = detection_metrics(c['score'], flipped, c['flagged'])
                    rows.append(dict(dataset=name, split=sp, noise=kind, rate=rate,
                                     algo=algo, space=space, method=f'{algo}@{space}',
                                     n_train=int(keep.sum()), n_orig=len(y_tr),
                                     n_relabeled=int((yo != yn).sum()),
                                     label_acc=float((yo[keep] == y_tr[keep]).mean()),
                                     est_err=c.get('est_err'), true_err=int(flipped.sum()),
                                     auc=a_ec, auprc=p_ec, auc_fmprobe=a_fm, auprc_fmprobe=p_fm2,
                                     clean_sec=time.time() - t1, **dm))
            print(f'{name} {sp} {time.time()-t0:.0f}s rows={len(rows)}', flush=True)
            pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    print('wrote', out, len(rows), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='exp_main.csv')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--noise', default='')
    a = ap.parse_args()
    nz = NOISE if not a.noise else [(k, r) for (k, r) in NOISE if k in a.noise.split(',')]
    run(datasets=a.datasets.split(','), out=a.out, noise=nz)
