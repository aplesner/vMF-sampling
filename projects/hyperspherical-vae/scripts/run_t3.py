"""T3 — one nViT/ViT VAE run on CIFAR-10 -> JSON row.

Arms (brief T3): nvit+vmf (the new experiment) vs
  nvit+gaussian, vit+vmf, vit+gaussian, vit+vmf+sqrt-d-rescale (arm 4, the
  control that decides whether the story is input scaling).

Example:
    uv run python projects/hyperspherical-vae/scripts/run_t3.py \
        --latent vmf --arch nvit --d 256 --lr 3e-4 --seed 0 --out results/t3_rows
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import get_cifar10  # noqa: E402
from nvit import ViTVAE, count_params  # noqa: E402

ARMS = {
    "nvit_vmf": dict(latent="vmf", arch="nvit", rescale_sqrt_d=False),
    "nvit_gaussian": dict(latent="gaussian", arch="nvit", rescale_sqrt_d=False),
    "vit_vmf": dict(latent="vmf", arch="vit", rescale_sqrt_d=False),
    "vit_gaussian": dict(latent="gaussian", arch="vit", rescale_sqrt_d=False),
    "vit_vmf_sqrtd": dict(latent="vmf", arch="vit", rescale_sqrt_d=True),
}


def _evaluate(model, loader, device, beta, max_batches=20):
    model.eval()
    tot, n = {"recon": 0.0, "kl": 0.0, "loss": 0.0}, 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            _, metrics = model.elbo_loss(x, beta=beta)
            for k in tot:
                tot[k] += metrics[k] * x.shape[0]
            n += x.shape[0]
    return {k: v / max(n, 1) for k, v in tot.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--iwae-samples", type=int, default=500)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "t3_rows")
    p.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "results" / "t3_logs")
    args = p.parse_args()

    tag = f"cifar10_{args.arm}_d{args.d}_lr{args.lr:g}_s{args.seed}"
    out_path = args.out / f"{tag}.json"
    if out_path.exists():
        print(f"skip (exists): {out_path}")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    train_ds, test_ds = get_cifar10()
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=6, drop_last=True
    )
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=4)

    torch.manual_seed(args.seed)
    model = ViTVAE(latent_dim=args.d, **ARMS[args.arm]).to(device)
    # nGPT recipe: no weight decay.  Per-arm LR comes from the sweep.
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)

    best_val, best_epoch, best_state = float("inf"), -1, None
    log_fh = open(args.log_dir / f"{tag}.jsonl", "a")
    t0 = time.time()
    for epoch in range(args.epochs):
        beta = min(1.0, (epoch + 1) / max(args.warmup_epochs, 1))
        model.train()
        agg, nb = {"recon": 0.0, "kl": 0.0, "loss": 0.0}, 0
        for x, _ in train_loader:
            x = x.to(device)
            loss, metrics = model.elbo_loss(x, beta=beta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss epoch {epoch}: {metrics}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.normalize_weights_()  # nGPT post-step normalization
            for k in agg:
                agg[k] += metrics[k]
            nb += 1
        val = _evaluate(model, test_loader, device, beta)
        rec = {"epoch": epoch, "beta": beta, **{f"train_{k}": v / nb for k, v in agg.items()},
               **{f"val_{k}": v for k, v in val.items()}}
        log_fh.write(json.dumps(rec) + "\n")
        log_fh.flush()
        # no early stopping during KL warm-up (see train.py)
        if epoch + 1 < args.warmup_epochs:
            continue
        if val["loss"] < best_val - 1e-4:
            best_val, best_epoch = val["loss"], epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch > args.patience:
            break
    log_fh.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    train_secs = time.time() - t0

    t0 = time.time()
    model.eval()
    lls = []
    with torch.no_grad():
        for x, _ in test_loader:
            lls.append(model.iwae_loglikelihood(x.to(device), n_samples=args.iwae_samples))
    ll = torch.cat(lls)
    iwae_secs = time.time() - t0

    record = dict(
        tag=tag, arm=args.arm, d=args.d, m=args.d + 1 if ARMS[args.arm]["latent"] == "vmf" else args.d,
        lr=args.lr, seed=args.seed, n_params=count_params(model),
        device=torch.cuda.get_device_name(0) if device == "cuda" else device,
        best_val_loss=best_val, best_epoch=best_epoch,
        iwae_ll_mean=ll.mean().item(), iwae_ll_sd=(ll.std() / (len(ll) ** 0.5)).item(),
        train_secs=train_secs, iwae_secs=iwae_secs,
    )
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.rename(out_path)
    print(json.dumps({k: record[k] for k in ["tag", "best_val_loss", "iwae_ll_mean"]}))


if __name__ == "__main__":
    main()
