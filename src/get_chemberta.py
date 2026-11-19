"""Download ChemBERTa checkpoints and load their .bin weights with weights_only deserialization."""
import os, sys, json, urllib.request, shutil
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel

MODELS = {
    'chemberta_mlm': 'DeepChem/ChemBERTa-77M-MLM',
    'chemberta_mtr': 'DeepChem/ChemBERTa-77M-MTR',
    'chemberta_zinc': 'seyonec/ChemBERTa-zinc-base-v1',
}
ROOT = '/home/claude/fmclean/hf'   # untrusted downloads live only here

def fetch(repo, dest):
    os.makedirs(dest, exist_ok=True)
    meta = json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}'))
    files = [s['rfilename'] for s in meta['siblings']]
    want = [f for f in files if f in ('config.json', 'pytorch_model.bin', 'tokenizer.json',
                                      'tokenizer_config.json', 'vocab.json', 'merges.txt',
                                      'special_tokens_map.json', 'added_tokens.json')]
    for f in want:
        out = os.path.join(dest, f)
        if os.path.exists(out): continue
        urllib.request.urlretrieve(f'https://huggingface.co/{repo}/resolve/main/{f}', out)
    return want

def load(tag, repo):
    dest = os.path.join(ROOT, tag)
    got = fetch(repo, dest)
    tok = AutoTokenizer.from_pretrained(dest)
    cfg = AutoConfig.from_pretrained(dest)
    model = AutoModel.from_config(cfg)
    sd = torch.load(os.path.join(dest, 'pytorch_model.bin'), map_location='cpu', weights_only=True)
    clean = {}
    for k, v in sd.items():
        for pre in ('roberta.', 'bert.', 'model.'):
            if k.startswith(pre): k = k[len(pre):]; break
        clean[k] = v
    info = model.load_state_dict(clean, strict=False)
    missing = [k for k in info.missing_keys if 'position_ids' not in k and 'pooler' not in k]
    unexpected = [k for k in info.unexpected_keys
                  if not any(s in k for s in ('lm_head', 'classifier', 'regression', 'norm_head', 'pooler'))]
    model.eval()
    return tok, model, cfg, missing, unexpected, got

if __name__ == '__main__':
    for tag, repo in MODELS.items():
        try:
            tok, m, cfg, missing, unexpected, got = load(tag, repo)
            n = sum(p.numel() for p in m.parameters())
            enc = tok(['CCO', 'c1ccccc1C(=O)Nc1ccccc1'], return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                h = m(**enc).last_hidden_state
            print(f'{tag}: {repo} params={n:,} layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} '
                  f'max_pos={cfg.max_position_embeddings} out={tuple(h.shape)} '
                  f'missing={len(missing)} unexpected={len(unexpected)}', flush=True)
            if missing[:5]: print('   missing sample:', missing[:5], flush=True)
        except Exception as e:
            print(f'{tag}: FAILED {type(e).__name__}: {str(e)[:220]}', flush=True)
