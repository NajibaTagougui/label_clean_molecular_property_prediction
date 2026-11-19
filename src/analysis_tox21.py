"""Figure and tables for the 12-endpoint Tox21 robustness extension."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = '/home/claude/fmclean/results'; FIG = '/home/claude/fmclean/figs'
plt.rcParams.update({'font.size': 8.5, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 170,
                     'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'grid.linewidth': .5})

d = pd.read_csv(f'{RES}/exp_tox21.csv')
d['cell'] = d.endpoint + '|' + d.split + '|' + d.noise + '|' + d.rate.astype(str)
nz = d[d.rate > 0]
base = nz[nz.method == 'none@-'].groupby(['endpoint', 'split', 'noise', 'rate']).auc.mean().rename('base').reset_index()
j = nz.merge(base, on=['endpoint', 'split', 'noise', 'rate']); j['delta'] = j.auc - j.base

order = d.groupby('endpoint').base_pos.mean().sort_values().index.tolist()
fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9), layout='constrained')

# (a) per-endpoint benefit: reweight vs remove
x = np.arange(len(order)); w = 0.38
for i, (meth, lab, col) in enumerate([('fmclean-remove@ecfp', 'FUSE-Remove', '#d62728'),
                                      ('fmclean-reweight@ecfp', 'FUSE-Reweight', '#17becf')]):
    v = [j[(j.endpoint == e) & (j.method == meth)].delta.mean() for e in order]
    s = [j[(j.endpoint == e) & (j.method == meth)].delta.sem() for e in order]
    axes[0].bar(x + (i - .5) * w, v, w, yerr=s, capsize=1.6, label=lab, color=col)
axes[0].axhline(0, color='k', lw=.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(order, rotation=55, ha='right', fontsize=5.8)
axes[0].set_ylabel(r'$\Delta$ test ROC-AUC vs no cleaning')
axes[0].set_title('Benefit per Tox21 endpoint (ordered by positive rate)', fontsize=8)
axes[0].legend(fontsize=6.5)

# (b) the pathology at zero injected noise
z = d[(d.rate == 0) & (d.method == 'fmclean-hybrid@ecfp')]
t = z.groupby('endpoint').agg(pos=('base_pos', 'mean'), rm=('frac_removed', 'mean'),
                              ar=('actives_removed', 'mean'), af=('actives_flipped', 'mean'),
                              pa=('pos_after', 'mean'))
axes[1].scatter(t.pos * 100, (t.ar + t.af) * 100, s=26, color='#9467bd', label='actives lost')
axes[1].scatter(t.pos * 100, t.rm * 100, s=26, marker='^', color='#999999', label='training set removed')
for e in t.index:
    axes[1].annotate(e, (t.loc[e, 'pos'] * 100, (t.loc[e, 'ar'] + t.loc[e, 'af']) * 100),
                     fontsize=5, xytext=(3, 2), textcoords='offset points')
axes[1].set_xlabel('base positive rate (%)'); axes[1].set_ylabel('% of class')
axes[1].set_ylim(-5, 112)
axes[1].set_title('Minority-class loss at zero injected noise', fontsize=8)
axes[1].legend(fontsize=6.5, loc='lower left')
fig.savefig(f'{FIG}/fig9_tox21.png'); plt.close(fig)

# tables
rows = []
for e in order:
    g = j[j.endpoint == e]
    zz = t.loc[e]
    rows.append(dict(endpoint=e, pos_rate=zz.pos * 100,
                     d_remove=g[g.method == 'fmclean-remove@ecfp'].delta.mean(),
                     d_reweight=g[g.method == 'fmclean-reweight@ecfp'].delta.mean(),
                     d_cl=g[g.method == 'cl@ecfp'].delta.mean(),
                     removed_pct=zz.rm * 100, actives_lost_pct=(zz.ar + zz.af) * 100,
                     pos_after_pct=zz.pa * 100))
tt = pd.DataFrame(rows)
tt.to_csv(f'{RES}/T_tox21.csv', index=False)

stat = []
for a in ['fmclean-reweight@ecfp', 'cl@ecfp', 'smallloss@ecfp', 'fmclean-hybrid@ecfp',
          'fmclean-remove@ecfp', 'knn@ecfp']:
    x = nz[nz.method == a].set_index('cell').auc; y = nz[nz.method == 'none@-'].set_index('cell').auc
    k = x.index.intersection(y.index); xv, yv = x[k].values, y[k].values
    ok = ~(np.isnan(xv) | np.isnan(yv)); dd = xv[ok] - yv[ok]
    ep = j[j.method == a].groupby('endpoint').delta.mean()
    stat.append(dict(method=a, n=len(dd), delta=dd.mean(), win_pct=100 * np.mean(dd > 0),
                     p=stats.wilcoxon(xv[ok], yv[ok]).pvalue,
                     endpoints_positive=f'{int((ep>0).sum())}/{len(ep)}'))
st = pd.DataFrame(stat); st.to_csv(f'{RES}/T_tox21_stats.csv', index=False)
print(tt.round(4).to_string(index=False)); print(); print(st.round(4).to_string(index=False))
