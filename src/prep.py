"""Data preparation: canonicalization, duplicate-conflict detection, splits, descriptors."""
import os, json, pickle, warnings
import numpy as np, pandas as pd
from collections import defaultdict
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*'); warnings.filterwarnings('ignore')

DATA = '/home/claude/data'
WORK = '/home/claude/fmclean/work'
os.makedirs(WORK, exist_ok=True)

TASKS = {
    'bbbp':    dict(file='BBBP.csv',    smiles='smiles', label='p_np'),
    'bace':    dict(file='bace.csv',    smiles='smiles', label='Class'),
    'clintox': dict(file='clintox.csv', smiles='smiles', label='CT_TOX'),
    'tox21':   dict(file='tox21.csv',   smiles='smiles', label='SR-MMP'),
    'sider':   dict(file='sider.csv',   smiles='smiles', label='Hepatobiliary disorders'),
    'hiv':     dict(file='HIV.csv',     smiles='smiles', label='HIV_active'),
}

def canon(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None, None
    # keep largest organic fragment (salt stripping) -- standard QSAR practice
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        frags = sorted(frags, key=lambda f: f.GetNumHeavyAtoms(), reverse=True)
        m = frags[0]
    if m.GetNumHeavyAtoms() < 3: return None, None
    try:
        s = Chem.MolToSmiles(m)
    except Exception:
        return None, None
    return s, m

def scaffold(mol):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ''

def load_task(name):
    cfg = TASKS[name]
    df = pd.read_csv(os.path.join(DATA, cfg['file']))
    df = df[[cfg['smiles'], cfg['label']]].dropna()
    df.columns = ['smiles_raw', 'y']
    recs, bad = [], 0
    for smi, y in zip(df.smiles_raw, df.y):
        cs, m = canon(str(smi))
        if cs is None: bad += 1; continue
        recs.append((cs, int(y), scaffold(m)))
    # duplicate-SMILES label conflicts = evidence of real annotation noise
    g = defaultdict(list)
    for cs, y, sc in recs: g[cs].append(y)
    dup_groups = {k: v for k, v in g.items() if len(v) > 1}
    conflict = {k: v for k, v in dup_groups.items() if len(set(v)) > 1}
    seen, out = set(), []
    for cs, y, sc in recs:
        if cs in seen: continue
        seen.add(cs)
        # majority vote for duplicates; ties -> first
        ys = g[cs]
        yv = int(np.round(np.mean(ys))) if len(ys) > 1 else y
        out.append((cs, yv, sc, int(cs in conflict)))
    smiles = [o[0] for o in out]; y = np.array([o[1] for o in out])
    scaf = [o[2] for o in out];   conf = np.array([o[3] for o in out])
    stats = dict(name=name, n_raw=len(df), n_invalid=bad, n_unique=len(out),
                 n_dup_groups=len(dup_groups), n_conflict_groups=len(conflict),
                 n_conflict_mols=int(conf.sum()), pos_rate=float(y.mean()),
                 n_scaffolds=len(set(scaf)))
    return dict(smiles=smiles, y=y, scaffold=scaf, conflict=conf, stats=stats)

def scaffold_split(scaf, seed=None, frac=(0.8, 0.1, 0.1)):
    """Bemis-Murcko scaffold split. seed=None -> deterministic (largest sets to train)."""
    sets = defaultdict(list)
    for i, s in enumerate(scaf): sets[s].append(i)
    groups = list(sets.values())
    if seed is None:
        groups.sort(key=lambda g: (len(g), g[0]), reverse=True)
    else:
        rng = np.random.RandomState(seed); rng.shuffle(groups)
    n = len(scaf); n_tr, n_va = frac[0] * n, (frac[0] + frac[1]) * n
    tr, va, te = [], [], []
    for g in groups:
        if len(tr) + len(g) <= n_tr: tr += g
        elif len(tr) + len(va) + len(g) <= n_va: va += g
        else: te += g
    return np.array(tr), np.array(va), np.array(te)

def random_split(n, seed, frac=(0.8, 0.1, 0.1)):
    rng = np.random.RandomState(seed); idx = rng.permutation(n)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return idx[:a], idx[a:b], idx[b:]

_mg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
DESC = [d for d in Descriptors._descList]

def featurize(smiles):
    ecfp = np.zeros((len(smiles), 2048), dtype=np.uint8)
    maccs = np.zeros((len(smiles), 167), dtype=np.uint8)
    desc = np.zeros((len(smiles), len(DESC)), dtype=np.float32)
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        fp = _mg.GetFingerprintAsNumPy(m); ecfp[i] = fp
        maccs[i] = np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)
        for j, (nm, fn) in enumerate(DESC):
            try: v = fn(m)
            except Exception: v = 0.0
            desc[i, j] = v if np.isfinite(v) else 0.0
    return ecfp, maccs, desc

if __name__ == '__main__':
    all_stats = []
    for name in TASKS:
        d = load_task(name)
        print(json.dumps(d['stats']), flush=True)
        all_stats.append(d['stats'])
        n = len(d['smiles'])
        d['splits'] = {'scaffold_det': scaffold_split(d['scaffold'], None)}
        for s in (0, 1, 2):
            d['splits'][f'scaffold_{s}'] = scaffold_split(d['scaffold'], seed=100 + s)
            d['splits'][f'random_{s}'] = random_split(n, seed=100 + s)
        e, mc, de = featurize(d['smiles'])
        d['ecfp'], d['maccs'], d['desc'] = e, mc, de
        with open(f'{WORK}/{name}.pkl', 'wb') as f: pickle.dump(d, f)
    pd.DataFrame(all_stats).to_csv('/home/claude/fmclean/results/dataset_stats.csv', index=False)
    # ZINC pretraining corpus
    z = pd.read_csv(f'{DATA}/zinc250k.csv')
    smis = [str(s).strip() for s in z.smiles]
    ok = []
    for s in smis:
        cs, m = canon(s)
        if cs is not None and len(cs) <= 110: ok.append(cs)
    ok = list(dict.fromkeys(ok))
    print('zinc canonical unique:', len(ok), flush=True)
    with open(f'{WORK}/zinc.txt', 'w') as f: f.write('\n'.join(ok))
