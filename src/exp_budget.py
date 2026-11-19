"""Budget recalibration: rerun the primary comparison using cleanlab's error-count estimate.

Our calibrated-joint implementation estimates more label errors than the reference cleanlab
implementation (1.80x the true count on average against cleanlab's 1.47x). Because over-pruning
costs accuracy, every budget-driven method -- the confident-learning baseline included -- ran on an
inflated budget. Here the budget is taken from cleanlab and applied uniformly to all methods, so
the comparison is both fair and anchored to the reference implementation.
"""
import os, sys, time, pickle, argparse, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from clean import (inject_noise, reference_margin, oof_probs, clean_labels,
                   detection_metrics, knn_graph, desc_reference_matrix)
from exp_main import DATASETS, SPLITS, NOISE, ALGO_MAP, ev
from cleanlab.count import num_label_issues

WORK = '/home/claude/fmclean/work'; RES = '/home/claude/fmclean/results'
PAIRS = [('none', '-'), ('smallloss', 'ecfp'), ('cl', 'ecfp'), ('knn', 'ecfp'),
         ('fmclean-remove', 'ecfp'), ('fmclean-hybrid', 'ecfp'),
         ('fmclean-reweight', 'ecfp'), ('fmclean-capped', 'ecfp'), ('oracle', '-')]
K, ALPHA = 15, 0.5

def run(datasets=DATASETS, splits=SPLITS, noise=NOISE, out='exp_budget.csv'):
    rows = []
    for name in datasets:
        d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
        y_all = d['y']; ec = d['ecfp'].astype(np.float32)
        descm = desc_reference_matrix(d['desc'])
        t0 = time.time()
        for sp in splits:
            tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]
            trva = np.concatenate([tr, va])
            Z, y_tr, y_te = ec[trva], y_all[trva], y_all[te]
            g = knn_graph(Z, max(50, K))
            refm = reference_margin(descm[trva], y_tr)
            for kind, rate in noise:
                yn, flipped = inject_noise(y_tr, rate, kind,
                                           seed=abs(hash((name, sp, kind, rate))) % 10000,
                                           ref_margin=refm)
                p = oof_probs(Z, yn, 0)
                try:
                    b_cl = int(num_label_issues(yn, p))
                except Exception:
                    b_cl = None
                for algo, space in PAIRS:
                    meth = ALGO_MAP.get(algo, algo)
                    budget = None if algo in ('none', 'oracle') else b_cl
                    c = clean_labels(meth, Z, Z, yn, k=K, alpha=ALPHA, budget=budget,
                                     oracle_flip=flipped, g_fm=g, g_fp=g, p_fm=p, p_fp=p)
                    keep, yo, w = c['keep'], c['y_out'], c.get('weight')
                    ww = None if w is None else w[keep]
                    a, pr = ev(Z[keep], yo[keep], ec[te], y_te, ww)
                    dm = detection_metrics(c['score'], flipped, c['flagged'])
                    act = (y_tr == 1); rm = ~keep
                    rows.append(dict(dataset=name, split=sp, noise=kind, rate=rate,
                                     algo=algo, space=space, method=f'{algo}@{space}',
                                     budget_source='cleanlab', budget=b_cl,
                                     true_err=int(flipped.sum()), n_orig=len(y_tr),
                                     n_train=int(keep.sum()), base_pos=float(y_tr.mean()),
                                     actives_removed=float((rm & act).sum() / max(act.sum(), 1)),
                                     pos_after=float(yo[keep].mean()) if keep.sum() else np.nan,
                                     label_acc=float((yo[keep] == y_tr[keep]).mean()),
                                     auc=a, auprc=pr, **dm))
            print(f'{name} {sp} {time.time()-t0:.0f}s rows={len(rows)}', flush=True)
            pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    pd.DataFrame(rows).to_csv(f'{RES}/{out}', index=False)
    print('wrote', out, len(rows), flush=True)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default='exp_budget.csv')
    ap.add_argument('--datasets', default=','.join(DATASETS))
    a = ap.parse_args()
    run(datasets=a.datasets.split(','), out=a.out)
