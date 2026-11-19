"""Reviewer M3: measure, rather than assert, why rare actives get flagged.

The claim in Section 5.3 is that a minority-class example sits in a neighbourhood
dominated by the majority class, so the neighbourhood signal and the probe both
mark it anomalous for the same reason. That is a statement about the local class
composition around flagged versus unflagged actives, and it is directly measurable.

For each dataset and scaffold split we clean the ORIGINAL labels (no injected
noise) in the ECFP4 space with FUSE-Hybrid, exactly as Section 5.3 does, and then
compare, among training actives:

    p_nb1 = similarity-weighted fraction of the k = 15 neighbourhood labelled active

for the actives the cleaner flags against the actives it leaves alone. The same
contrast is computed for inactives as a control: if the effect is about rarity
rather than about flagging, it should be far weaker there.
"""
import os, sys, pickle
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(__file__))
from clean import clean_labels, knn_graph, knn_consensus, oof_probs

WORK = '/home/claude/fmclean/work'; RES = '/home/claude/fmclean/results'
DATASETS = ['bbbp', 'bace', 'clintox', 'tox21', 'sider']
SPLITS = [f'scaffold_{s}' for s in (0, 1, 2)]
K = 15

rows = []
for name in DATASETS:
    d = pickle.load(open(f'{WORK}/{name}.pkl', 'rb'))
    E = np.load(f'{WORK}/emb_{name}.npz')
    y = d['y']
    Zfm_all = StandardScaler().fit_transform(E['pretrained']).astype(np.float32)
    Zfp_all = d['ecfp'].astype(np.float32)
    for sp in SPLITS:
        tr, va, te = [np.asarray(x, int) for x in d['splits'][sp]]
        trva = np.concatenate([tr, va])
        Zfm, Zfp, ytr = Zfm_all[trva], Zfp_all[trva], y[trva]
        g_fm = knn_graph(Zfm, 50); g_fp = knn_graph(Zfp, 50)
        p_fm = oof_probs(Zfm, ytr, 0); p_fp = oof_probs(Zfp, ytr, 0)

        c = clean_labels('fmclean-hybrid', Zfm, Zfp, ytr,
                         oracle_flip=np.zeros(len(ytr), bool),
                         g_fm=g_fm, g_fp=g_fp, p_fm=p_fm, p_fp=p_fp)
        flagged = c['flagged'].astype(bool)

        # local class composition in the ECFP4 space that guided the cleaning
        _, p_nb1 = knn_consensus(Zfp, ytr, k=K, graph=g_fp)

        for cls, lab in ((1, 'active'), (0, 'inactive')):
            m = ytr == cls
            a = p_nb1[m & flagged]        # flagged members of this class
            b = p_nb1[m & ~flagged]       # unflagged members of this class
            r = dict(dataset=name, split=sp, cls=lab,
                     base_pos=float(ytr.mean()),
                     n_flagged=int(len(a)), n_unflagged=int(len(b)),
                     # structural term: how active is the average member's
                     # neighbourhood, before any flagging decision is made
                     nb_active_frac_all=float(p_nb1[m].mean()),
                     flag_rate=float(flagged[m].mean()),
                     nb_active_frac_flagged=float(a.mean()) if len(a) else np.nan,
                     nb_active_frac_unflagged=float(b.mean()) if len(b) else np.nan)
            if len(a) >= 3 and len(b) >= 3:
                u = mannwhitneyu(a, b, alternative='less' if cls == 1 else 'greater')
                r['delta'] = float(a.mean() - b.mean()); r['p'] = float(u.pvalue)
            else:
                r['delta'] = np.nan; r['p'] = np.nan
            rows.append(r)
        print(name, sp, 'done', flush=True)

df = pd.DataFrame(rows)
df.to_csv(f'{RES}/T_mechanism_neighbourhood.csv', index=False)

print('\n=== neighbourhood composition around flagged vs unflagged, by class ===')
g = df.groupby('cls').agg(nb_flagged=('nb_active_frac_flagged', 'mean'),
                          nb_unflagged=('nb_active_frac_unflagged', 'mean'),
                          delta=('delta', 'mean'),
                          cells=('delta', 'size'),
                          cells_sig=('p', lambda s: int((s < 0.05).sum())))
print(g.round(4))

print('\n=== per dataset, actives only ===')
act = df[df.cls == 'active'].copy()
ina = df[df.cls == 'inactive'].set_index(['dataset', 'split']).flag_rate
act['enrich'] = act.flag_rate.values / act.set_index(['dataset', 'split']).index.map(ina).values
print(act.groupby('dataset').agg(base_pos=('base_pos', 'mean'),
                                 nb_all=('nb_active_frac_all', 'mean'),
                                 flag_rate=('flag_rate', 'mean'),
                                 enrich=('enrich', 'mean'),
                                 nb_flagged=('nb_active_frac_flagged', 'mean'),
                                 nb_unflagged=('nb_active_frac_unflagged', 'mean'),
                                 delta=('delta', 'mean')).round(4))
ok = act.delta.notna()
print('\nactives: flagged lower in %d of %d cells where both groups exist' %
      ((act.delta[ok] < 0).sum(), ok.sum()))

# the structural claim: the rarer the class, the less active a typical active's
# neighbourhood, and therefore the more of the class the cleaner marks anomalous
from scipy.stats import pearsonr
per = act.groupby('dataset').agg(base_pos=('base_pos', 'mean'),
                                 nb_all=('nb_active_frac_all', 'mean'),
                                 flag_rate=('flag_rate', 'mean'),
                                 enrich=('enrich', 'mean'))
print('\nbase rate vs mean neighbourhood activity of an active: r = %.3f (p = %.3f, n = %d)'
      % (*pearsonr(per.base_pos, per.nb_all), len(per)))
print('base rate vs flag-rate enrichment of actives:          r = %.3f (p = %.3f)'
      % pearsonr(per.base_pos, per.enrich))
print(per.round(4))
