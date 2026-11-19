"""Representation study including real chemical foundation models: table, figure, class-balance check."""
import os, sys, pickle
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from clean import inject_noise, oof_probs, clean_labels, knn_graph
from sklearn.preprocessing import StandardScaler

RES = '/home/fmclean/results'; FIG = '/home/fmclean/figs'; WORK = '/home/fmclean/work'
plt.rcParams.update({'font.size': 8.5, 'axes.grid': True, 'grid.alpha': .25, 'figure.dpi': 170,
                     'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False, 'grid.linewidth': .5})
SP_LBL = {'ecfp': 'ECFP4\n(no learning)', 'chemberta_mtr': 'ChemBERTa-77M\nMTR (3.4M)',
          'fm': 'ours\n(1.2M, 250k mol)', 'chemberta_mlm': 'ChemBERTa-77M\nMLM (3.4M)',
          'chemberta_zinc': 'ChemBERTa-zinc\n(44M)', 'fmrand': 'random init\n(1.2M)'}
ORDER = ['ecfp', 'chemberta_mtr', 'fm', 'fmrand', 'chemberta_mlm', 'chemberta_zinc']
ALG_LBL = {'cl': 'Confident learning', 'knn': 'Neighbour editing (ENN)',
           'fmclean-hybrid': 'FUSE-Hybrid', 'fmclean-remove': 'FUSE-Remove'}

def load_all():
    f = pd.read_csv(f'{RES}/exp_fm.csv'); m = pd.read_csv(f'{RES}/exp_main.csv')
    for d in (f, m): d['cell'] = d.dataset + '|' + d.split + '|' + d.noise + '|' + d.rate.astype(str)
    a = pd.concat([f, m], ignore_index=True)
    return a[a.rate > 0]

def paired(nz, a, b, metric, algo):
    x = nz[(nz.space == a) & (nz.algo == algo)].set_index('cell')[metric]
    y = nz[(nz.space == b) & (nz.algo == algo)].set_index('cell')[metric]
    j = x.index.intersection(y.index); x, y = x[j].values, y[j].values
    ok = ~(np.isnan(x) | np.isnan(y)); x, y = x[ok], y[ok]
    d = x - y
    return dict(space=a, reference=b, algo=algo, metric=metric, n=len(x), delta=d.mean(),
                win_pct=100 * np.mean(d > 0), p=stats.wilcoxon(x, y).pvalue)

def tables_and_figure(nz):
    rows = []
    for sp in ORDER:
        for algo in ['cl', 'knn', 'fmclean-hybrid', 'fmclean-remove']:
            g = nz[(nz.space == sp) & (nz.algo == algo)]
            if not len(g): continue
            rows.append(dict(space=sp, algo=ALG_LBL[algo], det_auprc=g.det_auprc.mean(),
                             auc=g.auc.mean(), label_acc=g.label_acc.mean()))
    t = pd.DataFrame(rows)
    t.to_csv(f'{RES}/T_fm_representation.csv', index=False)
    tests = []
    for metric in ['det_auprc', 'auc']:
        for algo in ['cl', 'knn', 'fmclean-hybrid', 'fmclean-remove']:
            for a, b in [('chemberta_mtr', 'ecfp'), ('chemberta_mlm', 'ecfp'), ('chemberta_zinc', 'ecfp'),
                         ('chemberta_mtr', 'fm'), ('chemberta_mtr', 'chemberta_mlm'),
                         ('chemberta_zinc', 'chemberta_mtr')]:
                try: tests.append(paired(nz, a, b, metric, algo))
                except Exception: pass
    st = pd.DataFrame(tests); st.to_csv(f'{RES}/T_fm_stats.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 2.9), layout='constrained')
    CC = {'cl': '#1f77b4', 'knn': '#ff7f0e', 'fmclean-remove': '#d62728'}
    MK = {'cl': 'o', 'knn': 's', 'fmclean-remove': 'D'}
    present = [s for s in ORDER if s in set(nz.space)]
    for ax, metric, ylab in zip(axes, ['det_auprc', 'auc'],
                                ['corrupted-label detection AUPRC', 'test ROC-AUC']):
        for algo in ['cl', 'knn', 'fmclean-remove']:
            xs, ys, es = [], [], []
            for i, sp in enumerate(present):
                g = nz[(nz.space == sp) & (nz.algo == algo)][metric].dropna()
                if not len(g): continue
                xs.append(i); ys.append(g.mean()); es.append(g.sem())
            ax.errorbar(xs, ys, yerr=es, marker=MK[algo], ms=5, lw=1.0, elinewidth=1, capsize=3,
                        color=CC[algo], label=ALG_LBL[algo])
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([SP_LBL.get(s, s) for s in present], fontsize=6, rotation=30, ha='right')
        ax.set_ylabel(ylab)
        ax.axvline(0.5, color='k', ls=':', lw=.7)
    axes[0].legend(fontsize=6.5)
    fig.savefig(f'{FIG}/fig8_fm_representation.png'); plt.close(fig)
    return t, st

def class_balance(spaces=('ecfp', 'chemberta_mtr', 'fm'), datasets=('clintox', 'tox21', 'sider')):
    """Does a better representation avoid eating the minority class? Original labels, no injected noise."""
    rows = []
    for ds in datasets:
        d = pickle.load(open(f'{WORK}/{ds}.pkl', 'rb'))
        E = np.load(f'{WORK}/emb_{ds}.npz'); C = np.load(f'{WORK}/emb_cb_{ds}.npz')
        SP = {'ecfp': d['ecfp'].astype(np.float32),
              'fm': StandardScaler().fit_transform(E['pretrained']).astype(np.float32),
              'chemberta_mtr': StandardScaler().fit_transform(C['chemberta_mtr']).astype(np.float32)}
        for sp_name in spaces:
            acc = []
            for split in ['scaffold_0', 'scaffold_1', 'scaffold_2']:
                tr, va, te = [np.asarray(x, int) for x in d['splits'][split]]
                t = np.concatenate([tr, va]); Z = SP[sp_name][t]; y = d['y'][t]
                g = knn_graph(Z, 50); p = oof_probs(Z, y, 0)
                c = clean_labels('fmclean-hybrid', Z, Z, y, oracle_flip=np.zeros(len(y), bool),
                                 g_fm=g, g_fp=g, p_fm=p, p_fp=p)
                rm = ~c['keep']; yo = c['y_out']; act = (y == 1)
                flip = act & c['keep'] & (yo == 0)
                acc.append((y.mean() * 100, rm.mean() * 100, (rm & act).sum() / act.sum() * 100,
                            flip.sum() / act.sum() * 100, yo[c['keep']].mean() * 100))
            r = np.array(acc).mean(0)
            rows.append(dict(dataset=ds, space=sp_name, base_pos=r[0], removed=r[1],
                             actives_removed=r[2], actives_flipped=r[3], pos_after=r[4],
                             enrichment=r[2] / r[1] if r[1] > 0 else np.nan))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows); t.to_csv(f'{RES}/T_fm_class_balance.csv', index=False)
    return t

if __name__ == '__main__':
    nz = load_all()
    t, st = tables_and_figure(nz)
    print(t.pivot_table(index='space', columns='algo', values='det_auprc').round(4).to_string())
    print()
    print(t.pivot_table(index='space', columns='algo', values='auc').round(4).to_string())
    print()
    class_balance()
