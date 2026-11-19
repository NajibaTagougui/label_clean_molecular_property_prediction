"""Experiment 5: ablations -- neighbourhood size k, signal-fusion weight alpha, pruning budget."""
import os, sys, time, pickle
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from clean import (inject_noise, reference_margin, oof_probs, clean_labels,
                   detection_metrics, knn_graph)

WORK = '/home/claude/fmclean/work'; RES = '/home/claude/fmclean/results'
DS = ['bbbp', 'bace', 'clintox', 'tox21']
SPLITS = ['scaffold_0', 'scaffold_1', 'random_0']
NOISE = [('sym', 0.3), ('inst', 0.3), ('asym', 0.3)]

def ev(Ztr, ytr, Zte, yte, w=None):
    if len(np.unique(ytr)) < 2: return dict(auc=np.nan, auprc=np.nan)
    lr = LogisticRegression(max_iter=3000); lr.fit(Ztr, ytr, sample_weight=w)
    p = lr.predict_proba(Zte)[:, 1]
    return dict(auc=roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else np.nan,
                auprc=average_precision_score(yte, p) if len(np.unique(yte)) > 1 else np.nan)

rows = []
for name in DS:
    d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
    E = np.load(f'{WORK}/emb_{name}.npz')
    y_all = d['y']
    Zfm_all = StandardScaler().fit_transform(E['pretrained']).astype(np.float32)
    Zfp_all = d['ecfp'].astype(np.float32)
    for sp in SPLITS:
        tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]; trva = np.concatenate([tr, va])
        Zfm, Zfp, ytr = Zfm_all[trva], Zfp_all[trva], y_all[trva]
        g_fm = knn_graph(Zfm, 50); g_fp = knn_graph(Zfp, 50)
        refm = reference_margin(Zfp, ytr)
        for kind, rate in NOISE:
            yn, fl = inject_noise(ytr, rate, kind, seed=hash((name, sp, kind, rate)) % 10000,
                                  ref_margin=refm)
            p_fm = oof_probs(Zfm, yn, 0); p_fp = oof_probs(Zfp, yn, 0)
            base = dict(dataset=name, split=sp, noise=kind, rate=rate, n=len(ytr))
            # --- k sweep (neighbourhood consensus component)
            for k in (5, 10, 15, 25, 50):
                for meth in ('knn_fm', 'fmclean-hybrid'):
                    c = clean_labels(meth, Zfm, Zfp, yn, k=k, oracle_flip=fl,
                                     g_fm=g_fm, g_fp=g_fp, p_fm=p_fm, p_fp=p_fp)
                    m = ev(Zfm[c['keep']], c['y_out'][c['keep']], Zfm_all[te], y_all[te])
                    rows.append(dict(**base, ablation='k', value=k, method=meth, **m,
                                     **detection_metrics(c['score'], fl, c['flagged'])))
            # --- alpha sweep (0 = confidence only, 1 = neighbourhood only)
            for al in (0.0, 0.25, 0.5, 0.75, 1.0):
                c = clean_labels('fmclean-hybrid', Zfm, Zfp, yn, alpha=al, oracle_flip=fl,
                                 g_fm=g_fm, g_fp=g_fp, p_fm=p_fm, p_fp=p_fp)
                m = ev(Zfm[c['keep']], c['y_out'][c['keep']], Zfm_all[te], y_all[te])
                rows.append(dict(**base, ablation='alpha', value=al, method='fmclean-hybrid', **m,
                                 **detection_metrics(c['score'], fl, c['flagged'])))
            # --- budget sensitivity (fraction of training set pruned/relabelled)
            true_b = int(fl.sum())
            for tag, b in [('est', None), ('true', true_b), ('half', max(1, true_b // 2)),
                           ('double', min(len(ytr) - 10, 2 * true_b))]:
                for meth in ('cl_fm', 'fmclean-hybrid', 'fmclean-remove'):
                    c = clean_labels(meth, Zfm, Zfp, yn, budget=b, oracle_flip=fl,
                                     g_fm=g_fm, g_fp=g_fp, p_fm=p_fm, p_fp=p_fp)
                    m = ev(Zfm[c['keep']], c['y_out'][c['keep']], Zfm_all[te], y_all[te])
                    rows.append(dict(**base, ablation='budget', value=tag, method=meth, **m,
                                     est_err=c.get('est_err'), true_err=true_b,
                                     **detection_metrics(c['score'], fl, c['flagged'])))
        print(name, sp, len(rows), flush=True)
        pd.DataFrame(rows).to_csv(f'{RES}/exp_ablate.csv', index=False)
pd.DataFrame(rows).to_csv(f'{RES}/exp_ablate.csv', index=False)
print('done', len(rows))
