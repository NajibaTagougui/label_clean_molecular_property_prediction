"""Separating domain from detection quality.

Section 5.11 left one confound: the two domains differ both in subject matter and in how well
the neighbourhood signal detects corrupted labels (AUPRC 0.81 on CIFAR against 0.56 on
molecules), so the pooled regression cannot say which of the two drives the reversal of the
class-balance law.

This experiment removes the confound from inside a single domain. CIFAR-10 one-vs-rest tasks
are built at controlled minority fractions exactly as in exp_cifar.py, but the *cleaning*
representation is progressively degraded with isotropic Gaussian noise while the downstream
representation is held fixed (mirroring the molecular protocol, where the cleaning space was a
factor and the downstream model was always ECFP4+LR). Degradation sweeps detection AUPRC
across the molecular regime and beyond without changing the domain, the task, or the balance.

If detection quality is the moderator, deletion's benefit should fall with degradation and its
slope in the minority fraction should swing from negative (good detector, as in Section 5.11)
to positive (weak detector, as on molecules) within CIFAR alone.
"""
import os, sys, time, argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, os.path.dirname(__file__))
from clean import (inject_noise, oof_probs, clean_labels, detection_metrics, knn_graph)

RES = '/home/claude/fmclean/results'; CIF = '/home/claude/fmclean/cifar'
PAIRS = [('none', '-'), ('cl', 'feat'), ('knn', 'feat'), ('fmclean-remove', 'feat'),
         ('fmclean-hybrid', 'feat'), ('fmclean-reweight', 'feat'),
         ('fmclean-capped', 'feat'), ('oracle', '-')]
ALGO_MAP = {'cl': 'cl_fm', 'knn': 'knn_fm'}
MINORITY = [0.02, 0.10, 0.35]
POS_CLASSES = [0, 3, 8]            # airplane, cat, ship
LAMBDA = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0]   # std-dev of Gaussian noise added to the cleaning space
N_TOTAL = 3000
K, ALPHA = 15, 0.5


def ev(Xtr, ytr, Xte, yte, w=None):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return (np.nan, np.nan)
    lr = LogisticRegression(max_iter=3000); lr.fit(Xtr, ytr, sample_weight=w)
    p = lr.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), average_precision_score(yte, p)


def build_task(F, labels, pos_class, minority, seed):
    rng = np.random.RandomState(seed)
    pos = np.where(labels == pos_class)[0]; neg = np.where(labels != pos_class)[0]
    n_pos = int(round(N_TOTAL * minority)); n_neg = N_TOTAL - n_pos
    if n_pos > len(pos) or n_neg > len(neg):
        return None
    idx = np.concatenate([rng.choice(pos, n_pos, replace=False),
                          rng.choice(neg, n_neg, replace=False)])
    rng.shuffle(idx)
    y = (labels[idx] == pos_class).astype(int)
    return F[idx], y


def degrade(Z, lam, seed):
    """Add isotropic Gaussian noise of std lam to a standardised space, then re-standardise."""
    if lam <= 0:
        return Z
    rng = np.random.RandomState(seed)
    Zd = Z + rng.normal(0.0, lam, size=Z.shape).astype(np.float32)
    return StandardScaler().fit_transform(Zd).astype(np.float32)


def run(out='exp_decouple.csv', noise=(('sym', 0.3),)):
    F = np.load(f'{CIF}/feats.npy'); labels = np.load(f'{CIF}/labels.npy')
    F = StandardScaler().fit_transform(F).astype(np.float32)
    rows = []
    for minority in MINORITY:
        for pc in POS_CLASSES:
            for split_seed in (0, 1, 2):
                task = build_task(F, labels, pc, minority, seed=1000 + split_seed)
                if task is None:
                    continue
                X, y = task
                n_te = int(0.2 * len(y))
                te = np.arange(n_te); trva = np.arange(n_te, len(y))
                Xtr, y_tr, Xte, y_te = X[trva], y[trva], X[te], y[te]
                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue
                for lam in LAMBDA:
                    Z = degrade(Xtr, lam, seed=7000 + split_seed)   # cleaning space only
                    g = knn_graph(Z, max(50, K))
                    t0 = time.time()
                    for kind, rate in noise:
                        yn, flipped = inject_noise(
                            y_tr, rate, kind,
                            seed=abs(hash((pc, minority, split_seed, kind, rate))) % 10000)
                        p = oof_probs(Z, yn, 0)
                        for algo, space in PAIRS:
                            meth = ALGO_MAP.get(algo, algo)
                            c = clean_labels(meth, Z, Z, yn, k=K, alpha=ALPHA,
                                             oracle_flip=flipped, g_fm=g, g_fp=g,
                                             p_fm=p, p_fp=p)
                            keep, yo, w = c['keep'], c['y_out'], c.get('weight')
                            ww = None if w is None else w[keep]
                            # downstream representation is NOT degraded
                            a, pr = ev(Xtr[keep], yo[keep], Xte, y_te, ww)
                            dm = detection_metrics(c['score'], flipped, c['flagged'])
                            act = (y_tr == 1); rm = ~keep
                            rows.append(dict(
                                minority_target=minority, pos_class=int(pc),
                                split=split_seed, lam=lam, noise=kind, rate=rate,
                                algo=algo, space=space, method=f'{algo}@{space}',
                                base_pos=float(y_tr.mean()), n_orig=len(y_tr),
                                n_train=int(keep.sum()),
                                actives_removed=float((rm & act).sum() / max(act.sum(), 1)),
                                actives_flipped=float((act & keep & (yo == 0)).sum() / max(act.sum(), 1)),
                                pos_after=float(yo[keep].mean()) if keep.sum() else np.nan,
                                label_acc=float((yo[keep] == y_tr[keep]).mean()),
                                auc=a, auprc=pr, **dm))
                    print(f'minority={minority:.2f} class={pc} split={split_seed} '
                          f'lam={lam:g} {time.time()-t0:.0f}s rows={len(rows)}', flush=True)
        pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    print('wrote', out, len(rows), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='exp_decouple.csv')
    a = ap.parse_args(); run(out=a.out)
