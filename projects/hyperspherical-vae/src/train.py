"""Training and evaluation driver for T1/T2 arms.

Protocol (Davidson et al. 2018 App. F.1 unless overridden by the brief):
Adam, mini-batches of 64, linear KL warm-up over 100 epochs, early stopping
on validation ELBO with a 50-epoch look-ahead.  Per-arm LR is a first-class
argument: the brief mandates per-arm LR sweeps (no arm compared at a shared
LR).
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vmf_kl import bessel_ratio_A  # noqa: E402


def _evaluate(model, loader, device, beta, max_batches=None):
    model.eval()
    tot = {"recon": 0.0, "kl": 0.0, "loss": 0.0}
    n = 0
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            _, metrics = model.elbo_loss(x, beta=beta)
            bs = x.shape[0]
            for k in tot:
                tot[k] += metrics[k] * bs
            n += bs
    return {k: v / max(n, 1) for k, v in tot.items()}


def grad_variance_decomposition(model, x, device):
    """Pathwise vs correction gradient components w.r.t. kappa (S-VAE only).

    Rows are independent, so grad of the batch sum yields per-row gradients.
    Returns per-row (g_rep, g_cor) tensors; callers aggregate variances.
    """
    if model.latent_family != "vmf":
        return None
    model.train()
    x = x.view(x.shape[0], -1).to(device)
    h = model.encoder(x)
    mu, kappa = model._posterior(h)
    z, correction = __import__("vmf_vae").sample_vmf(mu, kappa)
    import torch.nn.functional as F

    logits = model.decoder(z)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(-1)
    g_rep = torch.autograd.grad(recon.sum(), kappa, retain_graph=True)[0]
    g_cor = torch.autograd.grad((recon.detach() * correction).sum(), kappa)[0]
    return g_rep.detach(), g_cor.detach()


@torch.no_grad()
def aggregate_mu_stats(model, loader, device, max_items=4096):
    """Aggregate-posterior mean-map dispersion (T5 diagnostic core).

    Returns effective rank and mean resultant length of the aggregate {mu(x)},
    plus mean kappa/rho.  Mean-map collapse (mu -> constant) shows as
    effective rank -> 1 with kappa untouched.
    """
    if model.latent_family not in ("vmf", "power_spherical"):
        return {}
    model.eval()
    mus, kappas = [], []
    for x, _ in loader:
        x = x.view(x.shape[0], -1).to(device)
        mu, kappa = model._posterior(model.encoder(x))
        mus.append(mu)
        kappas.append(kappa)
        if sum(t.shape[0] for t in mus) >= max_items:
            break
    mu = torch.cat(mus)[:max_items].cpu().double()
    kappa = torch.cat(kappas)[:max_items].cpu().double()
    centered = mu - mu.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(centered)
    p = (sv * sv) / (sv * sv).sum().clamp_min(1e-20)
    eff_rank = torch.exp(-(p * torch.log(p.clamp_min(1e-20))).sum()).item()
    m = model.m
    rho = bessel_ratio_A(kappa.cpu(), m)
    return {
        "mu_eff_rank": eff_rank,
        "mu_resultant": mu.mean(0).norm().item(),
        "mean_kappa": kappa.mean().item(),
        "mean_rho": rho.mean().item(),
        "kl_over_m": float("nan"),  # filled by caller from eval metrics if needed
    }


def train_vae(
    model,
    train_ds,
    val_ds,
    device,
    *,
    epochs: int = 400,
    lr: float = 1e-3,
    batch_size: int = 64,
    warmup_epochs: int = 100,
    patience: int = 50,
    seed: int = 0,
    log_path: Path | None = None,
    grad_var_batches: int = 0,
    num_workers: int = 2,
):
    torch.manual_seed(seed)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers)

    best_val = math.inf
    best_epoch = -1
    best_state = None
    log_fh = open(log_path, "a") if log_path else None

    for epoch in range(epochs):
        beta = min(1.0, (epoch + 1) / max(warmup_epochs, 1))
        model.train()
        t0 = time.time()
        agg = {"recon": 0.0, "kl": 0.0, "loss": 0.0}
        nb = 0
        for x, _ in train_loader:
            x = x.to(device)
            loss, metrics = model.elbo_loss(x, beta=beta)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at epoch {epoch}: {metrics}"
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            for k in agg:
                agg[k] += metrics[k]
            nb += 1
        train_metrics = {k: v / nb for k, v in agg.items()}
        val = _evaluate(model, val_loader, device, beta, max_batches=20)
        record = {
            "epoch": epoch,
            "beta": beta,
            "secs": time.time() - t0,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val.items()},
        }
        if grad_var_batches and epoch % 10 == 0:
            gvs = []
            for i, (x, _) in enumerate(val_loader):
                if i >= grad_var_batches:
                    break
                out = grad_variance_decomposition(model, x, device)
                if out is not None:
                    gvs.append(out)
            if gvs:
                g_rep = torch.cat([g[0] for g in gvs])
                g_cor = torch.cat([g[1] for g in gvs])
                record["gradvar_pathwise"] = g_rep.var().item()
                record["gradvar_correction"] = g_cor.var().item()
                record["gradvar_mean_cor_over_rep"] = (
                    (g_cor.abs().mean() / g_rep.abs().mean().clamp_min(1e-20)).item()
                )
        if log_fh:
            log_fh.write(json.dumps(record) + "\n")
            log_fh.flush()

        # Early stopping must not engage during KL warm-up: the val loss
        # rises with beta, so a best-set-during-warmup would fire patience
        # before beta reaches 1.  Track/stop only once warm-up completes.
        if epoch + 1 < warmup_epochs:
            continue
        score = val["loss"]
        if score < best_val - 1e-4:
            best_val = score
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch > patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if log_fh:
        log_fh.close()
    return {"best_val_loss": best_val, "best_epoch": best_epoch}


@torch.no_grad()
def evaluate_iwae(model, dataset, device, n_samples=500, batch_size=64, chunk=100, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    vals = []
    for x, _ in loader:
        x = x.to(device)
        vals.append(model.iwae_loglikelihood(x, n_samples=n_samples, chunk=chunk))
    return torch.cat(vals)
