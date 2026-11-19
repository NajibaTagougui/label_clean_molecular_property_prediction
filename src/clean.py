"""Label-noise injection, cleaning methods (baselines + ours), detection metrics."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-9

# ---------------------------------------------------------------- noise models
def inject_noise(y, rate, kind, seed, ref_margin=None):
    """Return (y_noisy, flipped_mask). Only train/val indices should be passed."""
    rng = np.random.RandomState(seed)
    y = y.copy(); yn = y.copy()
    n = len(y)
    if rate <= 0: return yn, np.zeros(n, bool)
    if kind == 'sym':
        f = rng.rand(n) < rate
    elif kind == 'asym':
        # bioassay-style: false negatives dominate (actives recorded inactive)
        f = np.zeros(n, bool)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        f[pos[rng.rand(len(pos)) < rate]] = True
        f[neg[rng.rand(len(neg)) < rate / 5.0]] = True
    elif kind == 'inst':
        # instance-dependent: corrupt the most ambiguous molecules (lowest margin)
        assert ref_margin is not None
        k = int(round(rate * n))
        f = np.zeros(n, bool)
        order = np.argsort(ref_margin)          # ascending margin = most ambiguous
        f[order[:k]] = True
    else:
        raise ValueError(kind)
    yn[f] = 1 - yn[f]
    return yn, f

def desc_reference_matrix(desc):
    """RDKit-descriptor matrix used to define instance-dependent noise.

    Deliberately NOT one of the representations under comparison (ECFP4, learned encoders),
    so that the corrupted set is not defined by a model living in any evaluated space.
    """
    X = np.asarray(desc, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(X, -1e6, 1e6)

def reference_margin(X, y, seed=0):
    """Clean-label reference model margin |p-0.5|, used only to define instance-dep. noise."""
    Xs = StandardScaler().fit_transform(X.astype(np.float64))
    p = cross_val_predict(LogisticRegression(max_iter=2000, C=1.0),
                          Xs, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method='predict_proba')[:, 1]
    return np.abs(p - 0.5)

# ------------------------------------------------------------- helper: oof probs
def oof_probs(Z, y, seed=0, folds=5, C=1.0):
    """Out-of-sample predicted probabilities from a linear probe (cheap: no backbone update)."""
    if len(np.unique(y)) < 2:
        return np.full((len(y), 2), 0.5)
    folds = int(min(folds, np.bincount(y).min())) if np.bincount(y).min() >= 2 else 2
    folds = max(2, folds)
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    p = cross_val_predict(LogisticRegression(max_iter=3000, C=C),
                          Z, y, cv=cv, method='predict_proba')
    return p

# ------------------------------------------------------- confident learning core
def confident_joint(y, probs):
    """Northcutt et al. (2021) confident joint, 2-class."""
    K = 2
    t = np.array([probs[y == j, j].mean() if (y == j).sum() else 0.5 for j in range(K)])
    C = np.zeros((K, K))
    above = probs >= t[None, :]
    for i in range(len(y)):
        cand = np.where(above[i])[0]
        if len(cand) == 0: continue
        j = cand[np.argmax(probs[i, cand])]
        C[y[i], j] += 1
    return C, t

def cl_prune_scores(y, probs):
    """Suspicion score + CL-estimated number of label errors (prune-by-noise-rate)."""
    C, t = confident_joint(y, probs)
    n = len(y)
    # calibrate joint to dataset size & class counts
    counts = np.bincount(y, minlength=2).astype(float)
    Cc = C.copy()
    for i in range(2):
        s = Cc[i].sum()
        if s > 0: Cc[i] *= counts[i] / s
    Q = Cc / max(Cc.sum(), EPS)
    n_err = int(round((Q.sum() - np.trace(Q)) * n))
    # self-confidence margin: lower = more suspicious
    sc = probs[np.arange(n), y]
    other = probs[np.arange(n), 1 - y]
    score = other - sc
    return score, max(n_err, 0), Q

# ----------------------------------------------------- neighbourhood consensus
def knn_graph(Z, k=15, block=2048):
    """Chunked cosine top-k neighbour graph (memory-safe for ~40k molecules)."""
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + EPS)
    n = Zn.shape[0]; k = int(min(k, n - 1))
    idx = np.zeros((n, k), dtype=np.int32); sim = np.zeros((n, k), dtype=np.float32)
    for s in range(0, n, block):
        e = min(s + block, n)
        S = Zn[s:e] @ Zn.T
        S[np.arange(e - s), np.arange(s, e)] = -np.inf
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        sv = np.take_along_axis(S, part, 1)
        o = np.argsort(-sv, axis=1)
        idx[s:e] = np.take_along_axis(part, o, 1)
        sim[s:e] = np.take_along_axis(sv, o, 1)
    return idx, sim

def knn_consensus(Z, y, k=15, temp=0.25, graph=None):
    """Similar-property principle: label disagreement with chemically-similar neighbours."""
    if graph is None:
        idx, sim = knn_graph(Z, k)
    else:
        idx, sim = graph
        k = int(min(k, idx.shape[1]))
        idx, sim = idx[:, :k], sim[:, :k]
    w = np.exp((sim - sim.max(1, keepdims=True)) / temp)
    w = w / (w.sum(1, keepdims=True) + EPS)
    nb = y[idx]
    agree = (nb == y[:, None]).astype(float)
    s_agree = (w * agree).sum(1)                       # weighted agreement in [0,1]
    p_nb1 = (w * (nb == 1)).sum(1)                     # neighbourhood P(y=1)
    return 1.0 - s_agree, p_nb1                        # disagreement score, nb prob

def _rank01(x):
    o = np.argsort(np.argsort(x))
    return o / max(len(x) - 1, 1)

# ------------------------------------------------------------------- cleaners
def clean_labels(method, Z_fm, Z_fp, y_noisy, seed=0, k=15, alpha=0.5,
                 budget=None, flip_tau=0.75, oracle_flip=None,
                 g_fm=None, g_fp=None, p_fm=None, p_fp=None):
    """Returns dict(score=suspicion score, y_out, keep_mask, n_flagged, mode)."""
    n = len(y_noisy)
    y = y_noisy.copy()
    out = dict(y_out=y.copy(), keep=np.ones(n, bool), score=np.zeros(n), est_err=None,
               flagged=np.zeros(n, bool))

    if method == 'none':
        return out
    if method == 'oracle':
        out['keep'] = ~oracle_flip
        out['score'] = oracle_flip.astype(float)
        out['flagged'] = oracle_flip.copy()
        return out

    if method == 'smallloss':                       # loss-based filtering on FM probe
        p = p_fm if p_fm is not None else oof_probs(Z_fm, y, seed)
        loss = -np.log(np.clip(p[np.arange(n), y], EPS, 1))
        score = loss
        b = budget if budget is not None else int(0.2 * n)
        out['score'] = score
        out['keep'] = np.ones(n, bool); out['keep'][np.argsort(-score)[:b]] = False
        out['flagged'] = ~out['keep']
        return out

    if method in ('cl_fp', 'cl_fm'):                # confident learning
        cached = p_fp if method == 'cl_fp' else p_fm
        Z = Z_fp if method == 'cl_fp' else Z_fm
        p = cached if cached is not None else oof_probs(Z, y, seed)
        score, n_err, _ = cl_prune_scores(y, p)
        b = budget if budget is not None else n_err
        out['score'] = score; out['est_err'] = n_err
        out['keep'] = np.ones(n, bool); out['keep'][np.argsort(-score)[:b]] = False
        out['flagged'] = ~out['keep']
        return out

    if method == 'knn_fp' or method == 'knn_fm':    # neighbourhood consensus only
        Z = Z_fp if method == 'knn_fp' else Z_fm
        g = g_fp if method == 'knn_fp' else g_fm
        dis, _ = knn_consensus(Z, y, k=k, graph=g)
        b = budget if budget is not None else int(0.2 * n)
        out['score'] = dis
        out['keep'] = np.ones(n, bool); out['keep'][np.argsort(-dis)[:b]] = False
        out['flagged'] = ~out['keep']
        return out

    if method.startswith('fmclean'):                # ---- ours ----
        mode = method.split('-')[1] if '-' in method else 'remove'
        p = p_fm if p_fm is not None else oof_probs(Z_fm, y, seed)
        cl_score, n_err, _ = cl_prune_scores(y, p)
        dis, p_nb1 = knn_consensus(Z_fm, y, k=k, graph=g_fm)
        score = alpha * _rank01(dis) + (1 - alpha) * _rank01(cl_score)
        b = budget if budget is not None else n_err
        out['score'] = score; out['est_err'] = n_err
        if mode == 'capped':
            # Per-class budget proportional to class size: cleaning cannot move the class
            # balance, which is the failure mode diagnosed in the imbalanced-data analysis.
            flagged = []
            for cls in (0, 1):
                idx_c = np.where(y == cls)[0]
                if len(idx_c) == 0: continue
                b_c = int(round(b * len(idx_c) / n))
                b_c = min(b_c, max(len(idx_c) - 2, 0))   # never empty a class
                if b_c <= 0: continue
                flagged.extend(idx_c[np.argsort(-score[idx_c])[:b_c]])
            flagged = np.array(flagged, dtype=int)
        else:
            flagged = np.argsort(-score)[:b]
        fm_mask = np.zeros(n, bool); fm_mask[flagged] = True
        out['flagged'] = fm_mask
        if mode in ('remove', 'capped'):
            out['keep'] = np.ones(n, bool); out['keep'][flagged] = False
        elif mode == 'relabel':
            yo = y.copy()
            for i in flagged: yo[i] = 1 - yo[i]
            out['y_out'] = yo
        elif mode == 'hybrid':                      # relabel when both signals agree, else drop
            yo = y.copy(); keep = np.ones(n, bool)
            p_alt = p[np.arange(n), 1 - y]
            nb_alt = np.where(y == 1, 1 - p_nb1, p_nb1)
            for i in flagged:
                if p_alt[i] >= flip_tau and nb_alt[i] >= 0.5:
                    yo[i] = 1 - yo[i]
                else:
                    keep[i] = False
            out['y_out'] = yo; out['keep'] = keep
        elif mode == 'reweight':
            w = 1.0 - np.clip(_rank01(score) - (1 - b / n), 0, None) / max(b / n, EPS)
            out['weight'] = np.clip(w, 0.05, 1.0)
        return out
    raise ValueError(method)

# ------------------------------------------------------------ detection metrics
def detection_metrics(score, flipped, flagged):
    m = dict()
    if flipped.sum() > 0 and flipped.sum() < len(flipped):
        m['det_auroc'] = roc_auc_score(flipped, score)
        m['det_auprc'] = average_precision_score(flipped, score)
    else:
        m['det_auroc'] = np.nan; m['det_auprc'] = np.nan
    rm = np.asarray(flagged, bool)
    tp = int((rm & flipped).sum()); fp = int((rm & ~flipped).sum())
    fn = int((~rm & flipped).sum())
    m['det_prec'] = tp / max(tp + fp, 1)
    m['det_rec'] = tp / max(tp + fn, 1)
    m['det_f1'] = 2 * m['det_prec'] * m['det_rec'] / max(m['det_prec'] + m['det_rec'], EPS)
    m['n_flagged'] = int(rm.sum())
    return m
