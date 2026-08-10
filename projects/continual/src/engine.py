"""Training engine for one (arm, lr, wd, partition, seed) run.

Records, per epoch of both phases: old-class accuracy (stability),
new-class accuracy (plasticity), and per-layer relative update norms
||dW||/||W|| averaged over steps (the Tier-1 effective-LR observable).
"""

import numpy as np
import torch
import torch.nn.functional as F

from models import build_model
from optim import LARS


def make_optimizer(name, model, lr, wd):
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if name == "lars":
        return LARS(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(name)


@torch.no_grad()
def accuracy(model, x, y, batch=4096):
    model.eval()
    correct = 0
    for i in range(0, len(x), batch):
        logits = model(x[i : i + batch])
        correct += (logits.argmax(1) == y[i : i + batch]).sum().item()
    model.train()
    return correct / len(x)


def augment_batch(x):
    """Random crop (pad-4 reflect) + horizontal flip, on flat (B, 3072) tensors."""
    B = x.shape[0]
    img = x.view(B, 3, 32, 32)
    img = torch.nn.functional.pad(img, (4, 4, 4, 4), mode="reflect")
    top = torch.randint(0, 9, (B,), device=x.device)
    left = torch.randint(0, 9, (B,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (top[:, None] + ar[None, :])[:, None, :, None]  # (B,1,32,1)
    cols = (left[:, None] + ar[None, :])[:, None, None, :]  # (B,1,1,32)
    ch = torch.arange(3, device=x.device)[None, :, None, None]
    img = img[torch.arange(B, device=x.device)[:, None, None, None],
              ch, rows, cols].squeeze(-1)  # (B,3,32,32)
    flip = torch.rand(B, device=x.device) < 0.5
    img[flip] = img[flip].flip(-1)
    return img.reshape(B, -1)


def train_epochs(model, opt, data, epochs, batch_size, log_updates=False,
                 augment=False):
    """Returns per-epoch rows of update norms (empty if not logging)."""
    n = len(data.xtr)
    upd_hist = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=data.xtr.device)
        upd_acc = None
        nsteps = 0
        nlogged = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            if log_updates and nsteps % 5 == 0:
                before = [w.detach().clone() for w in model.weight_matrices()]
            xb = augment_batch(data.xtr[idx]) if augment else data.xtr[idx]
            logits = model(xb)
            loss = F.cross_entropy(logits, data.ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if log_updates:
                with torch.no_grad():
                    rel = [
                        ((w - b).norm() / b.norm().clamp_min(1e-12)).item()
                        for w, b in zip(model.weight_matrices(), before)
                    ]
                upd_acc = rel if upd_acc is None else [a + r for a, r in zip(upd_acc, rel)]
                nlogged += 1
            nsteps += 1
        if log_updates:
            upd_hist.append([a / max(nlogged, 1) for a in upd_acc])
    return upd_hist


def run_config(*, arm, opt_name, lr, wd, partition, seed, partitions,
               data_dir, device, epochs_p1, epochs_p2, batch_size=256,
               augment=False):
    from data import SplitData  # local import: cache lives at module level

    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = partitions[partition]
    old_classes, new_classes = classes[:5], classes[5:]
    p1 = SplitData(data_dir, device, old_classes, offset=0)
    p2 = SplitData(data_dir, device, new_classes, offset=5)

    model = build_model(arm, num_classes=5).to(device)
    opt = make_optimizer(opt_name, model, lr, wd)
    train_epochs(model, opt, p1, epochs_p1, batch_size, log_updates=False,
                 augment=augment)

    rows = []
    old0 = accuracy(model, p1.xte, p1.yte)
    rows.append(dict(phase=1, epoch=epochs_p1, old_acc=old0, new_acc=np.nan,
                     avg_acc=np.nan, **{f"upd{k}": np.nan for k in range(4)}))

    model.extend_head(10)
    opt = make_optimizer(opt_name, model, lr, wd)
    for ep in range(epochs_p2):
        upd = train_epochs(model, opt, p2, 1, batch_size, log_updates=True,
                           augment=augment)
        old_a = accuracy(model, p1.xte, p1.yte)
        new_a = accuracy(model, p2.xte, p2.yte)
        row = dict(phase=2, epoch=ep + 1, old_acc=old_a, new_acc=new_a,
                   avg_acc=0.5 * (old_a + new_a))
        for k in range(4):
            row[f"upd{k}"] = upd[0][k] if upd and k < len(upd[0]) else np.nan
        rows.append(row)
    return rows
