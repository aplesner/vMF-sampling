"""Training runner for the LR-band sweeps.

Pinned protocol (pre-registered, identical across arms — thesis App. B.1):
  AdamW (weight_decay=0), cosine schedule, warmup 2%, batch size 32,
  no gradient accumulation, alpha = 2r.

Real runs: ViT-S/16 (timm) on VTAB-19 / FGVC-10shot, one run per card,
two arms in parallel across the two 3090s. ViT-S/VTAB fits comfortably on
one 3090 (see 00-shared-constraints).

Local smoke path: --synthetic builds a TinyViT and random-label data so the
full plumbing (adapters, effective-step logging, divergence detection, CSV
output) is testable on CPU. It is a plumbing check, not a result.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from adapters import AdapterLinear, adapter_parameters, inject_adapters
from bandwidth import GRID, RunRecord
from effstep import AdapterLogger

WARMUP_RATIO = 0.02  # pinned
BATCH_SIZE = 32      # pinned
WEIGHT_DECAY = 0.0   # pinned


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class _Block(nn.Module):
    """Minimal ViT-style block with timm-compatible names (qkv/proj/fc1/fc2)."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn_qkv = nn.Linear(dim, 3 * dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.mlp_fc1 = nn.Linear(dim, 4 * dim)
        self.mlp_fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        qkv = self.attn_qkv(x).chunk(3, dim=-1)
        attn = torch.softmax(qkv[0] @ qkv[1].transpose(-2, -1) / math.sqrt(x.shape[-1]), dim=-1)
        x = x + self.attn_proj(attn @ qkv[2])
        x = x + self.mlp_fc2(F.gelu(self.mlp_fc1(x)))
        return x


class TinyViT(nn.Module):
    """Smoke-test-only stand-in with timm-compatible module names.

    Names use attn_qkv etc. (underscores), so the injector's default filter
    is applied with name_filter=_tiny_filter below."""

    def __init__(self, dim: int = 64, depth: int = 2, n_classes: int = 10,
                 img: int = 32, patch: int = 8):
        super().__init__()
        n_tok = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.blocks = nn.ModuleList([_Block(dim) for _ in range(depth)])
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x.mean(dim=1))


def _tiny_filter(name: str) -> bool:
    return any(k in name for k in ("attn_qkv", "attn_proj", "mlp_fc1", "mlp_fc2"))


def build_model(arch: str, n_classes: int, device: torch.device):
    if arch == "tinyvit":
        return TinyViT(n_classes=n_classes).to(device), _tiny_filter
    if arch.startswith("vit"):
        import timm  # cluster path only
        model = timm.create_model("vit_small_patch16_224", pretrained=True,
                                  num_classes=n_classes).to(device)
        return model, None  # default filter matches timm qkv/proj/fc1/fc2
    raise ValueError(f"unknown arch {arch!r}")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def build_data(dataset: str, batch_size: int, seed: int, device: torch.device):
    if dataset == "synthetic":
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(256, 3, 32, 32, generator=g)
        y = torch.randint(0, 10, (256,), generator=g)
        train = torch.utils.data.TensorDataset(x[:192], y[:192])
        val = torch.utils.data.TensorDataset(x[192:], y[192:])
        n_classes = 10
    else:
        raise NotImplementedError(
            f"dataset {dataset!r}: VTAB/FGVC loaders run on the cluster; "
            "see sweep.py --datasets-dir")
    kw = dict(batch_size=batch_size, shuffle=True)
    return (torch.utils.data.DataLoader(train, **kw),
            torch.utils.data.DataLoader(val, batch_size=batch_size),
            n_classes)


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------

def lr_at(step: int, total: int, base_lr: float) -> float:
    """Warmup 2% then cosine to 0 — pinned identically across arms."""
    wu = max(1, int(WARMUP_RATIO * total))
    if step < wu:
        return base_lr * (step + 1) / wu
    t = (step - wu) / max(1, total - wu)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * t))


def train_run(arm: str, rank: int, lr: float, seed: int, arch: str,
              dataset: str, epochs: int, device: torch.device,
              log_every: int = 1) -> tuple[RunRecord, list[dict]]:
    torch.manual_seed(seed)
    train_ld, val_ld, n_classes = build_data(dataset, BATCH_SIZE, seed, device)
    model, name_filter = build_model(arch, n_classes, device)

    adapters = inject_adapters(model, arm=arm, r=rank, name_filter=name_filter, seed=seed)
    params = adapter_parameters(model)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=WEIGHT_DECAY)
    logger = AdapterLogger(adapters, log_every=log_every)

    total_steps = epochs * len(train_ld)
    step = 0
    losses: list[float] = []
    grad_norm_max = 0.0
    nan_encountered = False
    logger.record(0)  # reference dW at init (effective step 0)

    model.train()
    for _ in range(epochs):
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total_steps, lr)
            loss = F.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                nan_encountered = True
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, float("inf")).item()
            grad_norm_max = max(grad_norm_max, gn)
            opt.step()
            step += 1
            losses.append(loss.item())
            logger.record(step)
        if nan_encountered:
            break

    # Final val metric.
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_ld:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.numel()
    metric = 100.0 * correct / max(total, 1)
    if nan_encountered:
        metric = float("nan")

    n_tail = max(1, len(losses) // 10)
    rec = RunRecord(
        arm=arm, rank=rank, lr=lr, seed=seed,
        metric=metric,
        nan_encountered=nan_encountered,
        loss_init=losses[0] if losses else float("nan"),
        loss_final_window=sum(losses[-n_tail:]) / n_tail if losses else float("nan"),
        grad_norm_max=grad_norm_max,
        eff_step_run_mean=logger.summary()["eff_step_run_mean"],
    )
    return rec, logger.records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--lrs", type=float, nargs="+", default=list(GRID))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--arch", default="tinyvit")
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", required=True, help="append run rows to this CSV")
    p.add_argument("--steps-out", default="", help="optional JSONL for per-step logs")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    device = torch.device(args.device)
    new = not os.path.exists(args.out)
    with open(args.out, "a") as f, \
         open(args.steps_out, "a") if args.steps_out else open(os.devnull, "w") as sf:
        if new:
            f.write("arm,rank,lr,seed,metric,nan_encountered,loss_init,"
                    "loss_final_window,grad_norm_max,eff_step_run_mean\n")
        for lr in args.lrs:
            for seed in args.seeds:
                t0 = time.time()
                rec, steps = train_run(args.arm, args.rank, lr, seed, args.arch,
                                       args.dataset, args.epochs, device)
                f.write(f"{rec.arm},{rec.rank},{rec.lr},{rec.seed},{rec.metric},"
                        f"{int(rec.nan_encountered)},{rec.loss_init},"
                        f"{rec.loss_final_window},{rec.grad_norm_max},"
                        f"{rec.eff_step_run_mean}\n")
                f.flush()
                for s in steps:
                    sf.write(json.dumps({"arm": rec.arm, "rank": rec.rank,
                                         "lr": rec.lr, "seed": rec.seed, **s}) + "\n")
                print(f"[{rec.arm} r={rec.rank} lr={lr:.2e} seed={seed}] "
                      f"metric={rec.metric:.2f} div={rec.diverged()} "
                      f"({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
