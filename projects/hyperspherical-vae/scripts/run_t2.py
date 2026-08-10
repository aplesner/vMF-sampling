"""T1b/T2 — one training run: (dataset, family, d, lr, seed) -> JSON row.

Used for both the T1 MNIST reproduction (d <= 40, gaussian/vmf) and the T2
dimension sweep.  One run per GPU; the Slurm array runs two tasks at a time
(two 3090s as independent workers).

Example:
    uv run python projects/hyperspherical-vae/scripts/run_t2.py \
        --dataset mnist --family vmf --d 40 --lr 1e-3 --seed 0 \
        --epochs 500 --out results/t2_rows
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

from data import DATASETS  # noqa: E402
from models import make_vae  # noqa: E402
from train import (  # noqa: E402
    aggregate_mu_stats,
    evaluate_iwae,
    grad_variance_decomposition,
    train_vae,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--family", choices=["gaussian", "gaussian_tied", "vmf", "power_spherical"], required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--warmup-epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--iwae-samples", type=int, default=500)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "t2_rows")
    p.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "results" / "t2_logs")
    args = p.parse_args()

    tag = f"{args.dataset}_{args.family}_d{args.d}_lr{args.lr:g}_s{args.seed}"
    out_path = args.out / f"{tag}.json"
    if out_path.exists():
        print(f"skip (exists): {out_path}")
        return
    args.out.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    get_data, input_dim = DATASETS[args.dataset]
    train_ds, test_ds = get_data()

    torch.manual_seed(args.seed)
    model = make_vae(args.family, input_dim, args.d)
    n_params = sum(t.numel() for t in model.parameters())

    t0 = time.time()
    info = train_vae(
        model,
        train_ds,
        test_ds,
        device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience,
        seed=args.seed,
        log_path=args.log_dir / f"{tag}.jsonl",
        grad_var_batches=4,
        num_workers=4,
    )
    train_secs = time.time() - t0

    t0 = time.time()
    ll = evaluate_iwae(
        model, test_ds, device, n_samples=args.iwae_samples, chunk=100, num_workers=4
    )
    iwae_secs = time.time() - t0

    mu_stats = aggregate_mu_stats(model, DataLoader(test_ds, batch_size=256), device)
    x, _ = next(iter(DataLoader(test_ds, batch_size=256, shuffle=False)))
    gv = grad_variance_decomposition(model, x, device)
    gradvar = {}
    if gv is not None:
        gradvar = {
            "gradvar_pathwise": gv[0].var().item(),
            "gradvar_correction": gv[1].var().item(),
        }

    record = dict(
        tag=tag,
        dataset=args.dataset,
        family=args.family,
        d=args.d,
        m=args.d + 1,
        lr=args.lr,
        seed=args.seed,
        n_params=n_params,
        device=torch.cuda.get_device_name(0) if device == "cuda" else device,
        **info,
        iwae_ll_mean=ll.mean().item(),
        iwae_ll_sd=ll.std().item() / (len(ll) ** 0.5),
        train_secs=train_secs,
        iwae_secs=iwae_secs,
        **mu_stats,
        **gradvar,
    )
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1))
    tmp.rename(out_path)  # atomic so killed jobs never leave half-rows
    print(json.dumps({k: record[k] for k in ["tag", "best_val_loss", "best_epoch", "iwae_ll_mean"]}))


if __name__ == "__main__":
    main()
