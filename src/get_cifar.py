"""Fetch a CIFAR-10 subset and extract frozen ImageNet ResNet-18 features.

Writes cifar/images.npy (16,000 x 32 x 32 x 3 uint8), cifar/labels.npy and
cifar/feats.npy (16,000 x 512 float32, the global-average-pool layer of a frozen
torchvision ResNet-18 with ImageNet weights, images upsampled to 96 x 96 and
normalised with the ImageNet statistics). Used by exp_cifar.py and exp_decouple.py.
Requires egress to huggingface.co and *.hf.co for the dataset, and to
download.pytorch.org for the checkpoint.
"""
import os
import numpy as np
import torch, torch.nn as nn
import torchvision
from torchvision import transforms
from datasets import load_dataset

OUT = '/home/claude/fmclean/cifar'
N = 16000
SEED = 0


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = load_dataset('uoft-cs/cifar10', split='train')
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(ds), N, replace=False)
    sub = ds.select(idx)
    images = np.stack([np.asarray(r['img'], dtype=np.uint8) for r in sub])
    labels = np.asarray(sub['label'], dtype=np.int64)
    np.save(f'{OUT}/images.npy', images)
    np.save(f'{OUT}/labels.npy', labels)

    net = torchvision.models.resnet18(
        weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval()
    torch.set_num_threads(2)

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(96, antialias=True),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    feats = np.empty((N, 512), dtype=np.float32)
    bs = 256
    with torch.no_grad():
        for i in range(0, N, bs):
            batch = torch.stack([tf(im) for im in images[i:i + bs]])
            feats[i:i + bs] = net(batch).numpy()
            print(f'{min(i + bs, N)}/{N}', flush=True)
    np.save(f'{OUT}/feats.npy', feats)
    print('wrote', feats.shape)


if __name__ == '__main__':
    main()
