"""Experiment 3: classical baselines (ECFP/descriptors) and Experiment 4: real-world label noise.

Real-noise protocol: no synthetic corruption. Cleaning is applied to the *original* benchmark
labels; we (a) measure downstream test AUC change, and (b) evaluate the suspicion score against
duplicate-SMILES label conflicts, an independent, assay-level indicator of genuine annotation noise.
"""
import os, sys, time, pickle, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from clean import (inject_noise, reference_margin, oof_probs, clean_labels,
                   detection_metrics, knn_graph, knn_consensus, cl_prune_scores)

WORK = '/home/claude/fmclean/work'; RES = '/home/claude/fmclean/results'
DATASETS = ['bbbp', 'bace', 'clintox', 'tox21', 'sider']
SPLITS = [f'{a}_{s}' for a in ('scaffold', 'random') for s in (0, 1, 2)]

def sc(y, p):
    return dict(auc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                auprc=average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan)

def baselines(out='exp_baselines.csv'):
    rows = []
    for name in DATASETS:
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        E = np.load(f'{WORK}/emb_{name}.npz')
        y = d['y']; ecfp = d['ecfp'].astype(np.float32)
        desc = np.nan_to_num(d['desc'], posinf=0, neginf=0)
        desc = StandardScaler().fit_transform(np.clip(desc, -1e6, 1e6))
        feats = dict(ecfp=ecfp, maccs=d['maccs'].astype(np.float32), desc=desc,
                     fm=StandardScaler().fit_transform(E['pretrained']),
                     fm_rand=StandardScaler().fit_transform(E['random']))
        models = dict(lr=lambda: LogisticRegression(max_iter=3000),
                      rf=lambda: RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=0),
                      svm=lambda: SVC(random_state=0),   # ranked by decision_function (AUC needs no calibration)
                      gbm=lambda: HistGradientBoostingClassifier(random_state=0))
        for sp in SPLITS:
            tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]; trva = np.concatenate([tr, va])
            for fn, X in feats.items():
                for mn, mk in models.items():
                    if fn == 'ecfp' and mn == 'gbm': continue      # 2048 binary cols, slow & weak
                    t0 = time.time()
                    m = mk(); m.fit(X[trva], y[trva])
                    p = (m.predict_proba(X[te])[:, 1] if hasattr(m, 'predict_proba')
                         else m.decision_function(X[te]))
                    rows.append(dict(dataset=name, split=sp, feat=fn, model=mn,
                                     sec=time.time() - t0, **sc(y[te], p)))
            print(name, sp, 'baselines done', flush=True)
            pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)

def real_noise(out='exp_real.csv', methods=('none', 'cl_fp', 'cl_fm', 'knn_fm',
                                            'fmclean-remove', 'fmclean-hybrid',
                                            'fmclean-reweight', 'fmclean-capped'),
               budget_fracs=(None, 0.05, 0.10, 0.20)):
    rows, det = [], []
    for name in DATASETS:
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        E = np.load(f'{WORK}/emb_{name}.npz')
        y = d['y']; conflict = d['conflict']
        Zfm_all = StandardScaler().fit_transform(E['pretrained']).astype(np.float32)
        Zfp_all = d['ecfp'].astype(np.float32)
        for sp in SPLITS:
            tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]; trva = np.concatenate([tr, va])
            Zfm, Zfp, ytr = Zfm_all[trva], Zfp_all[trva], y[trva]
            g_fm = knn_graph(Zfm, 50); g_fp = knn_graph(Zfp, 50)
            p_fm = oof_probs(Zfm, ytr, 0); p_fp = oof_probs(Zfp, ytr, 0)
            conf_tr = conflict[trva].astype(bool)
            for meth in methods:
                for bf in budget_fracs:
                    b = None if bf is None else int(bf * len(ytr))
                    if meth == 'none' and bf is not None: continue
                    c = clean_labels(meth, Zfm, Zfp, ytr, budget=b, oracle_flip=np.zeros(len(ytr), bool),
                                     g_fm=g_fm, g_fp=g_fp, p_fm=p_fm, p_fp=p_fp)
                    keep, yo = c['keep'], c['y_out']
                    if len(np.unique(yo[keep])) < 2: continue
                    w = c.get('weight')
                    ww = None if w is None else w[keep]
                    lr = LogisticRegression(max_iter=3000)
                    lr.fit(Zfp[keep], yo[keep], sample_weight=ww)      # ECFP4 + LR, as in Section 5
                    p = lr.predict_proba(Zfp_all[te])[:, 1]
                    act = (ytr == 1); rm = ~keep
                    rows.append(dict(dataset=name, split=sp, method=meth,
                                     base_pos=float(ytr.mean()),
                                     actives_removed=float((rm & act).sum() / max(act.sum(), 1)),
                                     pos_after=float(yo[keep].mean()),
                                     budget_frac=bf, n_train=int(keep.sum()),
                                     n_removed=int((~keep).sum()),
                                     n_relabeled=int((yo != ytr).sum()),
                                     est_err=c.get('est_err'), **sc(y[te], p)))
                # suspicion score vs duplicate-conflict flags (independent noise indicator)
                if meth != 'none' and conf_tr.sum() >= 3:
                    c = clean_labels(meth, Zfm, Zfp, ytr, oracle_flip=np.zeros(len(ytr), bool),
                                     g_fm=g_fm, g_fp=g_fp, p_fm=p_fm, p_fp=p_fp)
                    det.append(dict(dataset=name, split=sp, method=meth,
                                    n_conflict=int(conf_tr.sum()),
                                    auroc=roc_auc_score(conf_tr, c['score']),
                                    auprc=average_precision_score(conf_tr, c['score']),
                                    base_rate=float(conf_tr.mean())))
            print(name, sp, 'real done', flush=True)
            pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
            pd.DataFrame(det).to_csv(f'{RES}/exp_real_detect.csv', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(det).to_csv(f'{RES}/exp_real_detect.csv', index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--what', default='both')
    a = ap.parse_args()
    if a.what in ('base', 'both'): baselines()
    if a.what in ('real', 'both'): real_noise()
