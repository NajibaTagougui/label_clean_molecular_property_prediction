"""Figures and tables for the pre-submission additions: pooled class-balance law, capped
treatment, budget recalibration, HIV scale, seed variance."""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = '/home/fmclean/results'; FIG = '/home/fmclean/figs'
plt.rcParams.update({'font.size': 8.5, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 170,
                     'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'grid.linewidth': .5})

# ---- pooled 39-endpoint class-balance law
frames = []
for f, src in [('exp_sider.csv', 'SIDER'), ('exp_tox21_v2.csv', 'Tox21')]:
    d = pd.read_csv(f'{RES}/{f}'); d['source'] = src; frames.append(d)
d = pd.concat(frames, ignore_index=True)
nz = d[d.rate > 0]
base = (nz[nz.method == 'none@-'].groupby(['source', 'endpoint', 'split', 'noise', 'rate'])
        .auc.mean().rename('base').reset_index())
j = nz.merge(base, on=['source', 'endpoint', 'split', 'noise', 'rate']); j['delta'] = j.auc - j.base

fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), layout='constrained')
COL = {'fmclean-remove@ecfp': '#d62728', 'fmclean-reweight@ecfp': '#17becf'}
LBL = {'fmclean-remove@ecfp': 'FUSE-Remove (deletes)', 'fmclean-reweight@ecfp': 'FUSE-Reweight (keeps all)'}
rows = []
for meth in ['fmclean-remove@ecfp', 'fmclean-reweight@ecfp']:
    g = j[j.method == meth].groupby(['source', 'endpoint']).agg(dd=('delta', 'mean'), pos=('base_pos', 'mean'))
    g['minority'] = np.minimum(g.pos, 1 - g.pos)
    sl, ic, rv, pv, se = stats.linregress(g.minority, g.dd)
    rows.append(dict(method=meth, r=rv, p=pv, slope=sl, n=len(g)))
    axes[0].scatter(g.minority * 100, g.dd, s=20, alpha=.75, color=COL[meth],
                    label=f'{LBL[meth]}: r={rv:+.2f}, p={pv:.3f}')
    xs = np.linspace(g.minority.min(), g.minority.max(), 50)
    axes[0].plot(xs * 100, ic + sl * xs, color=COL[meth], lw=1.2)
axes[0].axhline(0, color='k', lw=.8, ls=':')
axes[0].set_xlabel('minority-class fraction (%)')
axes[0].set_ylabel(r'$\Delta$ test ROC-AUC vs no cleaning')
axes[0].set_title('Deletion depends on class balance; reweighting does not', fontsize=8)
axes[0].legend(fontsize=6.2, loc='lower right')

bands = [('<10%', lambda g: g.minority < .10), ('10-25%', lambda g: (g.minority >= .10) & (g.minority < .25)),
         ('>=25%', lambda g: g.minority >= .25)]
gr = {m: j[j.method == m].groupby(['source', 'endpoint']).agg(dd=('delta', 'mean'), pos=('base_pos', 'mean'))
      for m in COL}
for m in gr: gr[m]['minority'] = np.minimum(gr[m].pos, 1 - gr[m].pos)
x = np.arange(len(bands)); w = .38
for i, meth in enumerate(COL):
    v = [gr[meth][f(gr[meth])].dd.mean() for _, f in bands]
    e = [gr[meth][f(gr[meth])].dd.sem() for _, f in bands]
    axes[1].bar(x + (i - .5) * w, v, w, yerr=e, capsize=2.5, color=COL[meth], label=LBL[meth])
axes[1].axhline(0, color='k', lw=.8)
axes[1].set_xticks(x); axes[1].set_xticklabels([f'{b}\n(n={int(f(gr["fmclean-remove@ecfp"]).sum())})'
                                                for b, f in bands], fontsize=7)
axes[1].set_xlabel('minority-class fraction'); axes[1].set_ylabel(r'$\Delta$ test ROC-AUC')
axes[1].set_title('Pooled across 39 endpoints', fontsize=8); axes[1].legend(fontsize=6.2)
fig.savefig(f'{FIG}/fig10_class_balance_law.png'); plt.close(fig)
pd.DataFrame(rows).to_csv(f'{RES}/T_class_balance_law.csv', index=False)

# ---- budget recalibration table
b = pd.read_csv(f'{RES}/exp_budget.csv'); b['cell'] = b.dataset + '|' + b.split + '|' + b.noise + '|' + b.rate.astype(str)
m = pd.read_csv(f'{RES}/exp_main.csv'); m['cell'] = m.dataset + '|' + m.split + '|' + m.noise + '|' + m.rate.astype(str)
nzb, nzm = b[b.rate > 0], m[(m.rate > 0) & (m.space.isin(['ecfp', '-']))]
t = pd.concat([nzm.groupby('method').auc.mean().rename('ours_budget'),
               nzb.groupby('method').auc.mean().rename('cleanlab_budget')], axis=1).dropna()
t['change'] = t.cleanlab_budget - t.ours_budget
t.sort_values('cleanlab_budget', ascending=False).round(4).to_csv(f'{RES}/T_budget_recal.csv')

def paired(df, a, bm, metric='auc'):
    x = df[df.method == a].set_index('cell')[metric]; y = df[df.method == bm].set_index('cell')[metric]
    k = x.index.intersection(y.index); xv, yv = x[k].values, y[k].values
    ok = ~(np.isnan(xv) | np.isnan(yv)); dd = xv[ok] - yv[ok]
    return dict(a=a, b=bm, delta=dd.mean(), win=100 * np.mean(dd > 0),
                p=stats.wilcoxon(xv[ok], yv[ok]).pvalue, n=len(dd))
pd.DataFrame([paired(nzb, a, bm) for a, bm in
              [('fmclean-remove@ecfp', 'cl@ecfp'), ('fmclean-capped@ecfp', 'cl@ecfp'),
               ('fmclean-capped@ecfp', 'fmclean-remove@ecfp'), ('fmclean-remove@ecfp', 'none@-'),
               ('fmclean-reweight@ecfp', 'cl@ecfp'), ('knn@ecfp', 'cl@ecfp')]]
             ).to_csv(f'{RES}/T_budget_stats.csv', index=False)
print('wrote class-balance law figure and budget tables')
