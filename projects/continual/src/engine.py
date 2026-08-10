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


def train_epochs(model, opt, data, epochs, batch_size, log_updates=False):
    """Returns (per-epoch rows of update norms, none if not logging)."""
    n = len(data.xtr)
    upd_hist = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=data.xtr.device)
        upd_acc = None
        nsteps = 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            if log_updates:
                before = [w.detach().clone() for w in model.weight_matrices()]
            logits = model(data.xtr[idx])
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
            nsteps += 1
        if log_updates:
            upd_hist.append([a / max(nsteps, 1) for a in upd_acc])
    return upd_hist


def run_config(*, arm, opt_name, lr, wd, partition, seed, partitions,
               data_dir, device, epochs_p1, epochs_p2, batch_size=256):
    from data import SplitData  # local import: cache lives at module level

    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = partitions[partition]
    old_classes, new_classes = classes[:5], classes[5:]
    p1 = SplitData(data_dir, device, old_classes)
    p2 = SplitData(data_dir, device, new_classes)

    model = build_model(arm, num_classes=5).to(device)
    opt = make_optimizer(opt_name, model, lr, wd)
    train_epochs(model, opt, p1, epochs_p1, batch_size, log_updates=False)

    rows = []
    old0 = accuracy(model, p1.xte, p1.yte)
    rows.append(dict(phase=1, epoch=epochs_p1, old_acc=old0, new_acc=np.nan,
                     avg_acc=np.nan, **{f"upd{k}": np.nan for k in range(4)}))

    model.extend_head(10)
    opt = make_optimizer(opt_name, model, lr, wd)
    upd_hist = train_epochs(model, opt, p2, epochs_p2, batch_size, log_updates=True)
    for ep in range(epochs_p2):
        old_a = accuracy(model, p1.xte, p1.yte)
        new_a = accuracy(model, p2.xte, p2.yte)
        row = dict(phase=2, epoch=ep + 1, old_acc=old_a, new_acc=new_a,
                   avg_acc=0.5 * (old_a + new_a))
        for k in range(4):
            row[f"upd{k}"] = upd_hist[ep][k] if k < len(upd_hist[ep]) else np.nan
        rows.append(row)
    return rows
