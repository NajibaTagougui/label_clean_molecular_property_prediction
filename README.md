# When is it safe to delete a suspect label?

Code for **"When is it safe to delete a suspect label? A break-even rule for label
cleaning in molecular property prediction"** (Tagougui & Ahmadi, submitted to the
*Journal of Computer-Aided Molecular Design*).

Label cleaning is a routine preprocessing step whose benefit is unstable: the same
procedure improves one assay and damages another. This study treats that instability
as the thing to be explained, and arrives at a break-even rule — deleting flagged
labels pays only once the detector's AUPRC clears a threshold that rises as the
active class gets rarer, from 0.48 at 35% minority to 0.91 at 2%.

Everything runs on CPU. The full study is 23,241 cleaning runs in under eight
CPU-hours, built and tested on 2 threads with 7 GB RAM and no GPU.

## What is here

Source only. The curated datasets, the per-run result tables and the trained encoder
weights are not in this repository; every script writes its own outputs to `results/`
when you run it.

```
src/
  prep.py              curation, scaffold/random splits, ECFP4 + MACCS + RDKit descriptors
  pretrain.py          MLM pretraining on ZINC-250k
  model.py             the 4-layer SMILES transformer
  embed_all.py         frozen embeddings, pretrained + random-init control
  embed_chemberta.py   ChemBERTa checkpoint embeddings
  get_chemberta.py     checkpoint download
  get_cifar.py         CIFAR-10 subset + frozen ImageNet ResNet-18 features
  clean.py             the four axes: representation, signal, budget, treatment
  exp_*.py             the experiments
  analysis*.py         tables and figures
  fit_breakeven.py     fits the closed form of the break-even curve
```

## Reproducing the study

```bash
python -I src/prep.py                      # curation and splits                 
python -I src/pretrain.py                  # PT_MIN sets the budget in minutes   
python -I src/embed_all.py                 # frozen embeddings                   
python -I src/exp_main.py                  # controlled noise grid, 5,940 runs   
python -I src/exp_base_real.py --what both # classical baselines + real labels    
python -I src/exp_ablate.py                # k / alpha / budget ablations         
python -I src/embed_chemberta.py --datasets bbbp,bace,clintox,tox21,side
python -I src/exp_fm.py                    # foundation-model grid, 3,000 runs   
python -I src/exp_tox21.py                 # 12 Tox21 endpoints, 1,056 runs      
python -I src/exp_tox21.py --dataset sider --out exp_sider.csv   # 27 SIDER endpoints
python -I src/exp_budget.py                # comparison under cleanlab's budget  
python -I src/exp_seeds_hiv.py --what seeds
python -I src/exp_seeds_hiv.py --what hiv  # HIV at 40,950 molecules
python -I src/exp_mechanism.py             # neighbourhood composition 
python -I src/get_cifar.py                 # CIFAR-10 features                    
python -I src/exp_cifar.py                 # controlled minority fractions 
python -I src/exp_decouple.py              # detector-degradation sweep 
python -I src/analysis.py                  # then analysis_fm, analysis_tox21,
                                           # analysis_final, analysis_decouple
python -I src/fit_breakeven.py             # the break-even curve
```


## Data

All datasets are public. MoleculeNet classification sets (BBBP, BACE, ClinTox,
Tox21, SIDER, HIV) come from the `GLambard/Molecules_Dataset_Collection` mirror,
ZINC-250k from `aspuru-guzik-group/chemical_vae`, and CIFAR-10 from its public
distribution. `prep.py` and `get_cifar.py` fetch and curate them.

}
```
