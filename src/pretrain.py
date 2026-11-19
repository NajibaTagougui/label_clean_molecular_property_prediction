"""MLM pretraining of the compact chemical encoder on ZINC-250k."""
import os, sys, time, json, math, pickle
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from model import (Vocab, build_vocab, MolEncoder, MLMHead, encode_batch,
                   PAD, CLS, MASK, UNK, tokenize)

WORK = '/home/claude/fmclean/work'
torch.set_num_threads(2)
MAX_LEN = 96

def main(minutes=float(os.environ.get('PT_MIN', 30)), bench=False):
    smiles = open(f'{WORK}/zinc.txt').read().split('\n')
    smiles = [s for s in smiles if s]
    rng = np.random.RandomState(0); rng.shuffle(smiles)
    vocab = build_vocab(smiles[:50000])
    vocab.save(f'{WORK}/vocab.json')
    print('vocab', len(vocab), 'corpus', len(smiles), flush=True)

    ids_all = encode_batch(vocab, smiles, MAX_LEN)
    torch.manual_seed(0)
    model = MolEncoder(len(vocab), d=192, L=4, h=6, ff=384, max_len=MAX_LEN)
    head = MLMHead(192, len(vocab))
    n_par = sum(p.numel() for p in model.parameters())
    print('encoder params', n_par, 'head', sum(p.numel() for p in head.parameters()), flush=True)

    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=3e-4, weight_decay=0.01)
    bs = 64
    t0, step, losses, log = time.time(), 0, [], []
    budget = minutes * 60
    n = ids_all.size(0)
    warm = 300
    model.train()
    while time.time() - t0 < budget:
        sel = torch.from_numpy(np.random.randint(0, n, bs))
        ids = ids_all[sel].clone()
        tgt = ids.clone()
        maskable = (ids != PAD) & (ids != CLS)
        pr = torch.rand(ids.shape)
        chosen = maskable & (pr < 0.15)
        if chosen.sum() == 0: continue
        r = torch.rand(ids.shape)
        ids[chosen & (r < 0.8)] = MASK
        rand_tok = torch.randint(4, len(vocab), ids.shape)
        repl = chosen & (r >= 0.8) & (r < 0.9)
        ids[repl] = rand_tok[repl]
        tgt[~chosen] = -100
        toks, _ = model(ids)
        logits = head(toks)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.view(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        for g in opt.param_groups:
            g['lr'] = 3e-4 * min(1.0, (step + 1) / warm)
        opt.step()
        losses.append(loss.item()); step += 1
        if step % 100 == 0:
            el = time.time() - t0
            m = float(np.mean(losses[-100:]))
            log.append(dict(step=step, loss=m, sec=el, mols=step * bs))
            print(f'step {step} loss {m:.4f} elapsed {el:.0f}s mols {step*bs}', flush=True)
        if bench and step == 100:
            print('BENCH sec/100steps', time.time() - t0, flush=True); return
    torch.save(dict(sd=model.state_dict(), vocab=vocab.itos, cfg=dict(d=192, L=4, h=6, ff=384, max_len=MAX_LEN)),
               f'{WORK}/fm_pretrained.pt')
    # random-init control with identical architecture/seed-different init
    torch.manual_seed(1234)
    ctrl = MolEncoder(len(vocab), d=192, L=4, h=6, ff=384, max_len=MAX_LEN)
    torch.save(dict(sd=ctrl.state_dict(), vocab=vocab.itos, cfg=dict(d=192, L=4, h=6, ff=384, max_len=MAX_LEN)),
               f'{WORK}/fm_random.pt')
    json.dump(dict(log=log, steps=step, final_loss=float(np.mean(losses[-200:])),
                   minutes=minutes, params=n_par, vocab=len(vocab), bs=bs,
                   mols_seen=step * bs, epochs_equiv=step * bs / len(smiles)),
              open('/home/claude/fmclean/results/pretrain_log.json', 'w'), indent=1)
    print('done steps', step, 'final loss', np.mean(losses[-200:]), flush=True)

SPECIAL_OFF = 4
if __name__ == '__main__':
    main(bench='--bench' in sys.argv)
