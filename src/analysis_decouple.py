import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

RES = '/home/fmclean/results'
d = pd.read_csv(f'{RES}/exp_decouple.csv')
key = ['minority_target', 'pos_class', 'split', 'lam', 'noise', 'rate']

print('=== 1. Degradation sweeps detection quality (knn signal) ===')
kn = d[d.algo == 'knn']
print(kn.groupby('lam')[['det_auprc', 'det_f1']].mean().round(3).to_string())

base = d[d.algo == 'none'].set_index(key)['auc']
det = kn.set_index(key)['det_auprc']

rowsets = {}
for algo in ['fmclean-remove', 'fmclean-capped', 'fmclean-reweight', 'cl', 'fmclean-hybrid']:
    s = d[d.algo == algo].set_index(key)['auc']
    delta = (s - base).dropna()
    rowsets[algo] = delta

print('\n=== 2. Deletion benefit by degradation level and minority fraction ===')
rem = rowsets['fmclean-remove'].reset_index()
piv = rem.pivot_table(index='lam', columns='minority_target', values='auc', aggfunc='mean')
print((piv * 1).round(4).to_string())
print('\ndetection AUPRC by lam x minority:')
print(kn.pivot_table(index='lam', columns='minority_target',
                     values='det_auprc', aggfunc='mean').round(3).to_string())

print('\n=== 3. The class-balance slope swings sign WITHIN CIFAR as the detector weakens ===')
print(f"{'lam':>5} {'detAUPRC':>9} {'r(delta,minority)':>18} {'p':>9} "
      f"{'delta@2%':>9} {'delta@35%':>10}")
out = []
for lam, g in rem.groupby('lam'):
    dd = kn[kn.lam == lam]['det_auprc'].mean()
    r, p = stats.pearsonr(g.minority_target, g.auc)
    d2 = g[g.minority_target == 0.02].auc.mean()
    d35 = g[g.minority_target == 0.35].auc.mean()
    out.append(dict(lam=lam, det=dd, r=r, p=p, d2=d2, d35=d35))
    print(f'{lam:>5g} {dd:>9.3f} {r:>18.3f} {p:>9.2g} {d2:>9.4f} {d35:>10.4f}')
o = pd.DataFrame(out)

print('\n=== 4. Deletion benefit against detection quality (per-cell, within CIFAR) ===')
m = rem.merge(det.reset_index().rename(columns={'det_auprc': 'det'}), on=key)
r, p = stats.pearsonr(m.det, m.auc)
print(f'r(delta_remove, det_auprc) = {r:+.3f}  p = {p:.2g}  n = {len(m)}')
for mi, g in m.groupby('minority_target'):
    r2, p2 = stats.pearsonr(g.det, g.auc)
    print(f'  at minority={mi:.2f}: r = {r2:+.3f}  p = {p2:.2g}  n = {len(g)}')

print('\n=== 5. Moderation regression, single domain, orthogonal factors ===')
fit = smf.ols('auc ~ minority_target * det', data=m).fit()
print(fit.summary().tables[1])
print(f'R2 = {fit.rsquared:.3f}   n = {int(fit.nobs)}')
print('corr(minority, det) within this design = '
      f'{np.corrcoef(m.minority_target, m.det)[0,1]:+.3f}')

print('\n=== 6. Where deletion turns harmful ===')
agg = m.groupby('lam').agg(det=('det', 'mean'), delta=('auc', 'mean')).reset_index()
print(agg.round(4).to_string(index=False))
neg = agg[agg.delta < 0]
if len(neg):
    print(f'deletion first turns negative at lam={neg.lam.iloc[0]:g} '
          f'(detection AUPRC {neg.det.iloc[0]:.3f})')
else:
    print('deletion never turns negative in this sweep')

print('\n=== 7. Treatment ranking at the molecular detection regime ===')
# the lam whose detection AUPRC is closest to the molecular 0.56
tgt = o.iloc[(o.det - 0.56).abs().argsort()].iloc[0]
lam_star = tgt.lam
print(f'closest lam to molecular detection (0.56) is lam={lam_star:g} '
      f'(det={tgt.det:.3f})')
for algo, s in rowsets.items():
    g = s.reset_index()
    g = g[g.lam == lam_star]
    r, p = stats.pearsonr(g.minority_target, g.auc)
    print(f'  {algo:>18}: mean delta {g.auc.mean():+.4f}   '
          f'r(delta,minority) {r:+.3f} (p={p:.2g})')
print('\nfor comparison, at lam=0 (strong detector):')
for algo, s in rowsets.items():
    g = s.reset_index(); g = g[g.lam == 0]
    r, p = stats.pearsonr(g.minority_target, g.auc)
    print(f'  {algo:>18}: mean delta {g.auc.mean():+.4f}   '
          f'r(delta,minority) {r:+.3f} (p={p:.2g})')

print('\n=== 8. Minority-class destruction across the sweep ===')
dr = d[d.algo == 'fmclean-remove']
print(dr.pivot_table(index='lam', columns='minority_target',
                     values='actives_removed', aggfunc='mean').round(3).to_string())

o.to_csv(f'{RES}/decouple_slopes.csv', index=False)
m.to_csv(f'{RES}/decouple_cells.csv', index=False)
print('\nwrote decouple_slopes.csv, decouple_cells.csv')
