import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RES = '/home/claude/fmclean/results'; FIGS = '/home/claude/fmclean/figs'
m = pd.read_csv(f'{RES}/decouple_cells.csv')
d = pd.read_csv(f'{RES}/exp_decouple.csv')
key = ['minority_target', 'pos_class', 'split', 'lam', 'noise', 'rate']
base = d[d.algo == 'none'].set_index(key)['auc']
det = d[d.algo == 'knn'].set_index(key)['det_auprc']

fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.1))
cols = {0.02: '#c0392b', 0.10: '#e67e22', 0.35: '#2980b9'}

# left: deletion benefit against detection quality, one series per minority fraction
for mi, g in m.groupby('minority_target'):
    gg = g.sort_values('det')
    ax[0].scatter(gg.det, gg.auc, s=16, alpha=.55, color=cols[mi],
                  label=f'{int(mi*100)}% minority')
    b = np.polyfit(gg.det, gg.auc, 1)
    xs = np.linspace(gg.det.min(), gg.det.max(), 50)
    r, _ = stats.pearsonr(gg.det, gg.auc)
    ax[0].plot(xs, np.polyval(b, xs), color=cols[mi], lw=2)
ax[0].axhline(0, color='k', lw=.8, ls=':')
ax[0].axvline(0.56, color='grey', lw=1.2, ls='--')
ax[0].text(0.565, ax[0].get_ylim()[0] + .02, 'molecular\nregime', fontsize=8, color='grey')
ax[0].set_xlabel('detection AUPRC of the neighbourhood signal')
ax[0].set_ylabel('AUC gain from deleting flagged labels')
ax[0].set_title('Within one domain, deletion tracks detection quality', fontsize=10)
ax[0].legend(fontsize=8, frameon=False, loc='upper left')
for mi, xc in [(0.02, 0.913), (0.10, 0.750), (0.35, 0.482)]:
    ax[0].plot([xc], [0], marker='v', ms=7, color=cols[mi], clip_on=False)
ax[0].text(0.60, -0.235, 'markers: detection AUPRC at which deletion breaks even\n'
                         '0.48 (35%) / 0.75 (10%) / 0.91 (2%)', fontsize=8, color='#444')

# right: treatments across the degradation sweep
algos = [('fmclean-remove', 'remove', '#c0392b'), ('fmclean-capped', 'capped', '#8e44ad'),
         ('fmclean-reweight', 'reweight', '#16a085'), ('cl', 'confident learning', '#7f8c8d')]
for algo, lab, c in algos:
    s = (d[d.algo == algo].set_index(key)['auc'] - base).dropna().reset_index()
    a = s.merge(det.reset_index().rename(columns={'det_auprc': 'det'}), on=key)
    agg = a.groupby('lam').agg(det=('det', 'mean'), delta=('auc', 'mean'),
                               se=('auc', lambda x: x.std() / np.sqrt(len(x)))).reset_index()
    agg = agg.sort_values('det')
    ax[1].errorbar(agg.det, agg.delta, yerr=agg.se, marker='o', ms=4, lw=1.8,
                   capsize=2, color=c, label=lab)
ax[1].axhline(0, color='k', lw=.8, ls=':')
ax[1].set_xlim(0.645, 0.90)
ax[1].set_xlabel('mean detection AUPRC (cleaning space degraded left to right)')
ax[1].set_ylabel('AUC gain over no cleaning')
ax[1].set_title('Deletion crosses zero near AUPRC 0.68; reweighting does not', fontsize=10)
ax[1].legend(fontsize=8, frameon=False)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(f'{FIGS}/fig12_decouple.png', dpi=190)
print('wrote fig12_decouple.png')
