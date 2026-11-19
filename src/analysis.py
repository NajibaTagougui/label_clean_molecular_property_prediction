"""Final aggregation: tables (csv + markdown), paired statistics, publication figures."""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

RES = '/home/claude/fmclean/results'; FIG = '/home/claude/fmclean/figs'
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 8.5, 'axes.grid': True, 'grid.alpha': .25,
                     'figure.dpi': 170, 'savefig.bbox': 'tight', 'axes.spines.top': False,
                     'axes.spines.right': False, 'legend.frameon': False, 'grid.linewidth': .5})
ALG_LBL = {'none': 'No cleaning', 'smallloss': 'Small-loss filter', 'cl': 'Confident learning',
           'knn': 'Neighbour editing (ENN)', 'fmclean-remove': 'FUSE-Remove (ours)',
           'fmclean-relabel': 'FUSE-Relabel (ours)', 'fmclean-hybrid': 'FUSE-Hybrid (ours)',
           'fmclean-reweight': 'FUSE-Reweight (ours)', 'oracle': 'Oracle (upper bound)'}
SP_LBL = {'-': '', 'ecfp': 'ECFP4', 'fm': 'encoder', 'fmrand': 'random enc.', 'fmcat': 'ECFP4+enc.'}
NK_LBL = {'sym': 'Symmetric', 'asym': 'Class-conditional', 'inst': 'Instance-dependent'}
COL = {'none': '#9e9e9e', 'smallloss': '#8c564b', 'cl': '#1f77b4', 'knn': '#ff7f0e',
       'fmclean-remove': '#d62728', 'fmclean-hybrid': '#9467bd', 'fmclean-relabel': '#e377c2',
       'fmclean-reweight': '#17becf', 'oracle': '#2ca02c'}
ORDER = ['none', 'smallloss', 'cl', 'knn', 'fmclean-relabel', 'fmclean-reweight',
         'fmclean-hybrid', 'fmclean-remove', 'oracle']

def lbl(m):
    a, _, s = m.partition('@')
    return ALG_LBL.get(a, a) + (f' [{SP_LBL.get(s,s)}]' if SP_LBL.get(s, s) else '')

def md(df, path, fmt='%.3f', index=False):
    d = df.reset_index() if index else df.copy()
    for c in d.columns:
        if d[c].dtype.kind == 'f': d[c] = d[c].map(lambda v: '' if pd.isna(v) else fmt % v)
    lines = ['| ' + ' | '.join(map(str, d.columns)) + ' |',
             '| ' + ' | '.join('---' for _ in d.columns) + ' |']
    lines += ['| ' + ' | '.join(map(str, r)) + ' |' for r in d.values]
    open(path, 'w').write('\n'.join(lines) + '\n')

def L(f):
    p = f'{RES}/{f}'
    return pd.read_csv(p) if os.path.exists(p) else None

def paired(df, a, b, metric, key='cell'):
    x = df[df.method == a].set_index(key)[metric]; y = df[df.method == b].set_index(key)[metric]
    j = x.index.intersection(y.index); x, y = x[j].values, y[j].values
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    if len(x) < 8: return None
    d = x - y
    return dict(comparison=f'{a} vs {b}', metric=metric, n=len(x), delta=d.mean(),
                sd=d.std(ddof=1), win_pct=100 * np.mean(d > 0),
                p=stats.wilcoxon(x, y).pvalue, cohens_d=d.mean() / (d.std(ddof=1) + 1e-12))

# ============================================================ main noise study
def main_study():
    m = L('exp_main.csv')
    m['cell'] = m.dataset + '|' + m.split + '|' + m.noise + '|' + m.rate.astype(str)
    nz = m[m.rate > 0]
    PRIM = ['none@-', 'smallloss@ecfp', 'cl@ecfp', 'knn@ecfp', 'fmclean-relabel@ecfp',
            'fmclean-reweight@ecfp', 'fmclean-hybrid@ecfp', 'fmclean-remove@ecfp', 'oracle@-']

    # T2: AUC and detection by method x noise type (primary = ECFP cleaning space)
    for metric, tag in [('auc', 'auc'), ('det_auprc', 'detect')]:
        t = nz[nz.method.isin(PRIM)].pivot_table(index='method', columns='noise', values=metric)
        t = t.reindex([p for p in PRIM if p in t.index])
        t = t[[c for c in ['sym', 'asym', 'inst'] if c in t.columns]]
        t.columns = [NK_LBL[c] for c in t.columns]
        t['Mean'] = t.mean(1); t.insert(0, 'Method', [lbl(i) for i in t.index])
        md(t, f'{RES}/T_{tag}_by_noise.md'); t.to_csv(f'{RES}/T_{tag}_by_noise.csv', index=False)

    # T3: AUC vs noise rate
    t = nz[nz.method.isin(PRIM)].pivot_table(index='method', columns='rate', values='auc')
    t = t.reindex([p for p in PRIM if p in t.index]); t.insert(0, 'Method', [lbl(i) for i in t.index])
    md(t, f'{RES}/T_auc_by_rate.md'); t.to_csv(f'{RES}/T_auc_by_rate.csv', index=False)

    # T4: per-dataset at 30% symmetric
    s = m[(m.noise == 'sym') & (m.rate == 0.3) & (m.method.isin(PRIM))]
    t = s.pivot_table(index='method', columns='dataset', values='auc')
    t = t.reindex([p for p in PRIM if p in t.index]); t['Mean'] = t.mean(1)
    t.insert(0, 'Method', [lbl(i) for i in t.index])
    md(t, f'{RES}/T_perdataset_sym30.md'); t.to_csv(f'{RES}/T_perdataset_sym30.csv', index=False)

    # T5: representation comparison (same algorithm, different cleaning space)
    rows = []
    for alg in ['cl', 'knn', 'fmclean-hybrid']:
        for sp in ['ecfp', 'fm', 'fmrand', 'fmcat']:
            k = f'{alg}@{sp}'
            if k not in set(nz.method): continue
            g = nz[nz.method == k]
            rows.append(dict(algorithm=ALG_LBL[alg], space=SP_LBL[sp],
                             auc=g.auc.mean(), det_auprc=g.det_auprc.mean(),
                             det_f1=g.det_f1.mean(), label_acc=g.label_acc.mean()))
    t = pd.DataFrame(rows)
    md(t, f'{RES}/T_representation.md'); t.to_csv(f'{RES}/T_representation.csv', index=False)

    # T6: paired tests
    tests = []
    for a, b in [('fmclean-remove@ecfp', 'none@-'), ('fmclean-remove@ecfp', 'smallloss@ecfp'),
                 ('fmclean-remove@ecfp', 'cl@ecfp'), ('fmclean-remove@ecfp', 'knn@ecfp'),
                 ('fmclean-hybrid@ecfp', 'cl@ecfp'), ('fmclean-hybrid@ecfp', 'knn@ecfp'),
                 ('fmclean-remove@ecfp', 'fmclean-remove@fm'),
                 ('fmclean-hybrid@ecfp', 'fmclean-hybrid@fm'),
                 ('fmclean-hybrid@fm', 'fmclean-hybrid@fmrand'),
                 ('knn@fm', 'knn@fmrand'), ('cl@fm', 'cl@fmrand'),
                 ('knn@ecfp', 'knn@fm'), ('fmclean-hybrid@fmcat', 'fmclean-hybrid@ecfp')]:
        for metric in ['auc', 'det_auprc']:
            r = paired(nz, a, b, metric)
            if r: tests.append(r)
    t = pd.DataFrame(tests)
    # Holm correction within metric family
    t['p_holm'] = np.nan
    for met in t.metric.unique():
        idx = t.index[t.metric == met]; p = t.loc[idx, 'p'].values
        o = np.argsort(p); adj = np.empty_like(p); run = 0.0
        for rank, i in enumerate(o):
            run = max(run, (len(p) - rank) * p[i]); adj[i] = min(run, 1.0)
        t.loc[idx, 'p_holm'] = adj
    md(t, f'{RES}/T_stats.md', '%.4g'); t.to_csv(f'{RES}/T_stats.csv', index=False)

    # T7: oracle-gap recovery + clean-label penalty + cost
    g = nz.groupby('method').auc.mean(); n0, orc = g['none@-'], g['oracle@-']
    z = m[m.rate == 0].groupby('method').auc.mean()
    rows = []
    for k in PRIM:
        if k not in g: continue
        rows.append(dict(method=lbl(k), auc_noisy=g[k],
                         gap_recovered_pct=100 * (g[k] - n0) / (orc - n0),
                         auc_clean=z.get(k, np.nan), clean_penalty=z.get(k, np.nan) - z['none@-'],
                         label_acc=nz[nz.method == k].label_acc.mean(),
                         clean_sec=nz[nz.method == k].clean_sec.mean()))
    t = pd.DataFrame(rows)
    md(t, f'{RES}/T_recovery.md', '%.4f'); t.to_csv(f'{RES}/T_recovery.csv', index=False)

    # T8: noise-rate estimation by confident joint
    e = nz[nz.method == 'cl@ecfp'][['noise', 'rate', 'est_err', 'true_err', 'n_orig']].dropna()
    e['estimated'] = e.est_err / e.n_orig; e['true'] = e.true_err / e.n_orig
    t = e.groupby(['noise', 'rate'])[['true', 'estimated']].mean().reset_index()
    t['ratio'] = t.estimated / t['true'].clip(lower=1e-9)
    t['noise'] = t.noise.map(NK_LBL)
    md(t, f'{RES}/T_noise_estimation.md', '%.3f'); t.to_csv(f'{RES}/T_noise_estimation.csv', index=False)

    # --- Figures
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.5), sharey=True, layout='constrained')
    show = ['none@-', 'smallloss@ecfp', 'cl@ecfp', 'knn@ecfp', 'fmclean-hybrid@ecfp',
            'fmclean-remove@ecfp', 'oracle@-']
    for ax, nk in zip(axes, ['sym', 'asym', 'inst']):
        sub = m[(m.noise == nk) | (m.rate == 0)]
        for meth in show:
            q = sub[sub.method == meth].groupby('rate').auc.agg(['mean', 'sem'])
            if not len(q): continue
            a = meth.split('@')[0]
            ax.errorbar(q.index * 100, q['mean'], yerr=q['sem'], label=lbl(meth), color=COL[a],
                        marker='o', ms=2.6, lw=1.1, capsize=1.6,
                        ls='--' if a in ('none', 'oracle') else '-')
        ax.set_title(NK_LBL[nk]); ax.set_xlabel('injected noise rate (%)')
    axes[0].set_ylabel('test ROC-AUC')
    axes[2].legend(fontsize=6, loc='lower left')
    fig.savefig(f'{FIG}/fig2_auc_vs_noise.png'); plt.close(fig)

    sub = nz[(~nz.algo.isin(['none', 'oracle'])) & (nz.space == 'ecfp')]
    gg = sub.groupby(['algo', 'noise']).det_auprc.mean().unstack()
    gg = gg.reindex([a for a in ORDER if a in gg.index])
    fig, ax = plt.subplots(figsize=(5.6, 2.6), layout='constrained')
    x = np.arange(len(gg)); w = .26
    for i, c in enumerate(['sym', 'asym', 'inst']):
        ax.bar(x + (i - 1) * w, gg[c], w, label=NK_LBL[c])
    base = nz.groupby('noise').rate.mean()
    ax.set_xticks(x); ax.set_xticklabels([ALG_LBL[i] for i in gg.index], rotation=22,
                                         ha='right', fontsize=6.5)
    ax.set_ylabel('corrupted-label detection AUPRC'); ax.legend(fontsize=7)
    fig.savefig(f'{FIG}/fig3_detection.png'); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6), layout='constrained')
    MK = {'ecfp': 'o', 'fm': 's', 'fmrand': '^', 'fmcat': 'D'}
    CC = {'ecfp': '#1f77b4', 'fm': '#d62728', 'fmrand': '#2ca02c', 'fmcat': '#ff7f0e'}
    algs = ['cl', 'knn', 'fmclean-hybrid']
    for ax, metric, ylab in zip(axes, ['det_auprc', 'auc'],
                                ['detection AUPRC', 'test ROC-AUC']):
        for sp in ['ecfp', 'fm', 'fmrand', 'fmcat']:
            xs, ys, es = [], [], []
            for i, al in enumerate(algs):
                g = nz[(nz.algo == al) & (nz.space == sp)][metric].dropna()
                if not len(g): continue
                xs.append(i); ys.append(g.mean()); es.append(g.sem())
            ax.errorbar(xs, ys, yerr=es, marker=MK[sp], ms=5, lw=0, elinewidth=1,
                        capsize=3, color=CC[sp], label=SP_LBL[sp])
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels([ALG_LBL[a] for a in algs], rotation=12, ha='right', fontsize=7)
        ax.set_ylabel(ylab); ax.set_xlim(-.4, len(algs) - .6)
    axes[0].legend(fontsize=6.5, title='cleaning space', title_fontsize=6.5, ncol=2)
    fig.savefig(f'{FIG}/fig4_representation.png'); plt.close(fig)
    return m

def fig_pretrain():
    p = f'{RES}/pretrain_log.json'
    if not os.path.exists(p): return
    df = pd.DataFrame(json.load(open(p))['log'])
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    ax.plot(df.mols / 1000, df.loss, color='#1f77b4', lw=1.1)
    ax.set_xlabel('molecules seen (thousands)'); ax.set_ylabel('MLM cross-entropy')
    ax.set_title('Encoder pretraining (ZINC-250k)', fontsize=8.5)
    fig.savefig(f'{FIG}/fig1_pretrain.png'); plt.close(fig)

def ablations():
    a = L('exp_ablate.csv')
    if a is None: return
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.5), layout='constrained')
    k = a[a.ablation == 'k'].copy(); k['value'] = k.value.astype(float)
    for meth, c in [('knn_fm', '#ff7f0e'), ('fmclean-hybrid', '#9467bd')]:
        s = k[k.method == meth].groupby('value')[['auc', 'det_auprc']].mean()
        axes[0].plot(s.index, s.det_auprc, marker='o', ms=3, color=c,
                     label=ALG_LBL.get(meth.replace('_fm', ''), meth) + ' (AUPRC)')
        axes[0].plot(s.index, s.auc, marker='s', ms=3, ls='--', color=c, alpha=.6,
                     label=ALG_LBL.get(meth.replace('_fm', ''), meth) + ' (AUC)')
    axes[0].set_xlabel('neighbourhood size $k$'); axes[0].set_ylabel('score')
    axes[0].legend(fontsize=5.5)
    al = a[a.ablation == 'alpha'].copy(); al['value'] = al.value.astype(float)
    s = al.groupby('value')[['auc', 'det_auprc']].mean()
    axes[1].plot(s.index, s.det_auprc, marker='o', ms=3, color='#9467bd', label='detection AUPRC')
    axes[1].plot(s.index, s.auc, marker='s', ms=3, ls='--', color='#d62728', label='test ROC-AUC')
    axes[1].set_xlabel(r'fusion weight $\alpha$'); axes[1].set_ylabel('score'); axes[1].legend(fontsize=6.5)
    axes[1].annotate('confidence only', (0, s.det_auprc.iloc[0]), fontsize=6,
                     xytext=(2, 10), textcoords='offset points')
    axes[1].annotate('neighbourhood only', (1, s.det_auprc.iloc[-1]), fontsize=6,
                     ha='right', xytext=(-2, 10), textcoords='offset points')
    axes[1].set_ylim(0.53, 0.80)
    b = a[a.ablation == 'budget']
    pt = b.pivot_table(index='value', columns='method', values='auc')
    pt = pt.reindex([v for v in ['half', 'est', 'true', 'double'] if v in pt.index])
    x = np.arange(len(pt)); w = .26
    for i, c in enumerate(pt.columns):
        axes[2].bar(x + (i - 1) * w, pt[c], w,
                    label=ALG_LBL.get(c.replace('_fm', ''), c), color=COL.get(c.replace('_fm', '')))
    axes[2].set_xticks(x); axes[2].set_xticklabels(['½ est.', 'estimated', 'true', '2× est.'], fontsize=7)
    axes[2].set_ylim(0.70, None); axes[2].set_xlabel('pruning budget')
    axes[2].set_ylabel('test ROC-AUC'); axes[2].legend(fontsize=6)
    fig.savefig(f'{FIG}/fig6_ablations.png'); plt.close(fig)
    for name in ['k', 'alpha', 'budget']:
        s = a[a.ablation == name]
        t = s.pivot_table(index='value', columns='method', values=['auc', 'det_auprc'])
        t.columns = ['_'.join(c) for c in t.columns]
        t.reset_index().to_csv(f'{RES}/T_ablate_{name}.csv', index=False)
        md(t.reset_index(), f'{RES}/T_ablate_{name}.md', '%.4f')

def real_noise():
    r, dt = L('exp_real.csv'), L('exp_real_detect.csv')
    if r is not None:
        base = r[r.method == 'none'].groupby(['dataset', 'split']).auc.mean().rename('base')
        j = r.merge(base, on=['dataset', 'split']); j['delta'] = j.auc - j.base
        t = (j[j.method != 'none'].groupby(['method', 'budget_frac']).delta
             .agg(['mean', 'sem', 'count']).reset_index())
        t['t_p'] = [stats.ttest_1samp(j[(j.method == m) & (j.budget_frac == b)].delta.dropna(), 0).pvalue
                    if c > 2 else np.nan for m, b, c in zip(t.method, t.budget_frac, t['count'])]
        md(t, f'{RES}/T_real.md', '%.4f'); t.to_csv(f'{RES}/T_real.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5), layout='constrained')
        for meth in sorted(j.method.unique()):
            if meth == 'none': continue
            s = j[j.method == meth].groupby('budget_frac').delta.agg(['mean', 'sem'])
            a = meth.replace('_fm', '').replace('_fp', '')
            axes[0].errorbar(s.index.astype(float) * 100, s['mean'], yerr=s['sem'], marker='o',
                             ms=3, lw=1.1, capsize=2, label=meth, color=COL.get(a))
        axes[0].axhline(0, color='k', lw=.8, ls=':')
        axes[0].set_xlabel('training labels pruned / relabelled (%)')
        axes[0].set_ylabel(r'$\Delta$ test ROC-AUC')
        axes[0].set_title('Cleaning the original benchmark labels', fontsize=8)
        axes[0].legend(fontsize=6)
        if dt is not None and len(dt):
            g = dt.groupby('method').auroc.agg(['mean', 'sem'])
            x = np.arange(len(g))
            axes[1].bar(x, g['mean'], .55, yerr=g['sem'], capsize=2.5, color='#4c72b0')
            axes[1].axhline(0.5, color='k', ls=':', lw=.8)
            axes[1].set_xticks(x); axes[1].set_xticklabels(g.index, rotation=20, ha='right', fontsize=6.5)
            axes[1].set_ylabel('AUROC vs. duplicate-SMILES\nlabel conflicts'); axes[1].set_ylim(0.4, 0.9)
            axes[1].set_title('Agreement with real annotation conflicts', fontsize=8)
        fig.savefig(f'{FIG}/fig7_real_noise.png'); plt.close(fig)
    if dt is not None and len(dt):
        rows = []
        for m_, g in dt.groupby('method'):
            t_, p = stats.ttest_1samp(g.auroc.values, 0.5)
            rows.append(dict(method=m_, auroc=g.auroc.mean(),
                             sem=g.auroc.std(ddof=1) / np.sqrt(len(g)), auprc=g.auprc.mean(),
                             n_obs=len(g), mean_conflicts=g.n_conflict.mean(),
                             base_rate=g.base_rate.mean(), p_vs_chance=p))
        t = pd.DataFrame(rows)
        md(t, f'{RES}/T_real_detect.md', '%.4g'); t.to_csv(f'{RES}/T_real_detect.csv', index=False)

def baselines():
    b = L('exp_baselines.csv')
    if b is None: return
    for tag, sub in [('scaffold', b[b.split.str.startswith('scaffold')]), ('all', b)]:
        t = sub.pivot_table(index=['feat', 'model'], values='auc', columns='dataset')
        t['Mean'] = t.mean(1); t = t.sort_values('Mean', ascending=False).reset_index()
        md(t, f'{RES}/T_baselines_{tag}.md'); t.to_csv(f'{RES}/T_baselines_{tag}.csv', index=False)
    sc = b[b.split.str.startswith('scaffold')]
    t = sc.groupby(['feat', 'model']).agg(auc=('auc', 'mean'), sec=('sec', 'mean')).reset_index()
    t.to_csv(f'{RES}/T_baseline_cost.csv', index=False)

def peft():
    p = L('exp_peft.csv')
    if p is None or not len(p): return
    ML = {'probe': 'Linear probe', 'bitfit': 'BitFit', 'lora8': 'LoRA (r=8)', 'full': 'Full FT'}
    p['cell'] = p.dataset + '|' + p.split
    t = (p.groupby(['mode', 'clean', 'rate'])
         .agg(auc=('auc', 'mean'), sem=('auc', 'sem'), auprc=('auprc', 'mean'),
              trainable=('trainable', 'mean'), sec=('sec', 'mean'), n=('auc', 'size'))
         .reset_index())
    t['mode'] = t['mode'].map(ML)
    md(t, f'{RES}/T_peft.md', '%.4f'); t.to_csv(f'{RES}/T_peft.csv', index=False)
    rows = []
    for mode in p['mode'].unique():
        for rate in sorted(p.rate.unique()):
            s = p[(p['mode'] == mode) & (p.rate == rate)]
            a = s[s.clean == 'fmclean-hybrid'].set_index('cell').auc
            b = s[s.clean == 'none'].set_index('cell').auc
            j = a.index.intersection(b.index)
            if len(j) < 3: continue
            d = (a[j] - b[j]).values
            rows.append(dict(mode=ML.get(mode, mode), rate=rate, n=len(d), delta_auc=d.mean(),
                             sem=d.std(ddof=1) / np.sqrt(len(d)),
                             p=stats.ttest_1samp(d, 0).pvalue if len(d) > 2 else np.nan))
    if rows:
        tt = pd.DataFrame(rows)
        md(tt, f'{RES}/T_peft_cleaning_effect.md', '%.4f')
        tt.to_csv(f'{RES}/T_peft_cleaning_effect.csv', index=False)
    noisy = p[p.rate > 0]
    if not len(noisy):
        print('peft: no noisy runs yet, skipping figure'); return
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.6), layout='constrained')
    g = noisy.groupby(['mode', 'clean']).agg(auc=('auc', 'mean'), tp=('trainable', 'mean'),
                                             sec=('sec', 'mean')).reset_index()
    for cl, mk, cc in [('none', 'o', '#9e9e9e'), ('fmclean-hybrid', 's', '#d62728')]:
        s = g[g.clean == cl]
        axes[0].scatter(s.tp, s.auc, marker=mk, s=34, color=cc,
                        label='noisy labels' if cl == 'none' else 'FUSE-cleaned labels')
        for _, r in s.iterrows():
            axes[0].annotate(ML.get(r['mode'], r['mode']), (r.tp, r.auc), fontsize=5.5,
                             xytext=(3, 3), textcoords='offset points')
    axes[0].set_xscale('log'); axes[0].set_xlabel('trainable parameters')
    axes[0].set_ylabel('test ROC-AUC (30% noise)'); axes[0].legend(fontsize=6.5)
    pt = g.pivot_table(index='mode', columns='clean', values='auc').reindex(
        [m for m in ['probe', 'bitfit', 'lora8', 'full'] if m in set(g['mode'])])
    x = np.arange(len(pt)); w = .35
    for i, c in enumerate(pt.columns):
        axes[1].bar(x + (i - .5) * w, pt[c], w, label='noisy' if c == 'none' else 'cleaned',
                    color='#9e9e9e' if c == 'none' else '#d62728')
    axes[1].set_xticks(x); axes[1].set_xticklabels([ML[i] for i in pt.index], fontsize=7)
    axes[1].set_ylabel('test ROC-AUC (30% noise)'); axes[1].legend(fontsize=6.5)
    fig.savefig(f'{FIG}/fig5_peft.png'); plt.close(fig)

if __name__ == '__main__':
    fig_pretrain(); main_study(); ablations(); real_noise(); baselines(); peft()
    print('figures:', sorted(os.listdir(FIG)))
    print('tables:', sorted(f for f in os.listdir(RES) if f.startswith('T_')))
