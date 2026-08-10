"""VAE arms for T1/T2: N-VAE, tied-sigma N-VAE, S-VAE (vMF), Power-Spherical.

Architecture follows Davidson et al. (2018) App. F.1: MLP encoder
[input -> 256 -> 128], decoder [latent -> 128 -> 256 -> input], Bernoulli
decoder for dynamically binarized data.  m = d + 1 ambient coordinates for
spherical arms (S-VAE latent "dimension d" = sphere S^d; brief's Metrics
convention).

Power-Spherical (De Cao & Aziz, 2020): p(z|mu, kappa) ∝ (1 + muᵀz)^kappa on
S^{m-1}; exact reparameterized sampling via a Beta variable and a closed-form
KL against Uniform — no Bessel functions and no rejection loop.  It is a T2
arm because it is the main spherical-latent competitor with a cheap KL.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vmf_kl import log_surface_area, log_vmf_log_norm  # noqa: E402
from vmf_vae import sample_vmf, sample_vmf_with_kl  # noqa: E402


def _log_surface(m: int) -> float:
    return log_surface_area(m).item()


def mlp(sizes: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class BaseVAE(nn.Module):
    """Shared MLP trunk + ELBO/IWAE plumbing. Subclasses implement the latent."""

    latent_family = "base"

    def __init__(self, input_dim: int, latent_dim: int, hidden=(256, 128)):
        super().__init__()
        self.input_dim = input_dim
        self.d = latent_dim
        h1, h2 = hidden
        self.encoder = mlp([input_dim, h1, h2])
        self.enc_head = nn.Linear(h2, self.latent_param_dim)
        self.decoder = mlp([self.decoder_input_dim, h2, h1, input_dim])

    @property
    def latent_param_dim(self) -> int:
        raise NotImplementedError

    @property
    def decoder_input_dim(self) -> int:
        return self.d

    # ---- posterior interface ------------------------------------------------
    def encode_latent(self, h: torch.Tensor):
        """Return (z, kl, aux) for a batch of encoder features."""
        raise NotImplementedError

    def _posterior_params(self, x: torch.Tensor):
        """Deterministic encoder pass for IWAE; returns a tuple of tensors."""
        raise NotImplementedError

    def _iwae_chunk(self, x, post, s: int) -> torch.Tensor:
        """(s, B) log p(x|z)+log p(z)-log q(z|x) for s posterior samples."""
        raise NotImplementedError

    # ---- training -----------------------------------------------------------
    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        z, kl, aux = self.encode_latent(h)
        logits = self.decoder(z)
        return logits, kl, aux

    def elbo_loss(self, x: torch.Tensor, beta: float = 1.0):
        logits, kl, aux = self.forward(x.view(x.shape[0], -1))
        recon = F.binary_cross_entropy_with_logits(
            logits, x.view(x.shape[0], -1), reduction="none"
        ).sum(-1)
        correction = aux.get("correction")
        recon_eff = recon
        if correction is not None:
            recon_eff = recon + recon.detach() * correction
        loss = (recon_eff + beta * kl).mean()
        metrics = {
            "recon": recon.detach().mean().item(),
            "kl": kl.detach().mean().item(),
            # Clean ELBO for logging/early-stopping: the correction term is a
            # pure gradient vehicle and must not pollute the objective value.
            "loss": (recon.detach() + beta * kl.detach()).mean().item(),
        }
        metrics.update(aux.get("metrics", {}))
        return loss, metrics

    # ---- evaluation ---------------------------------------------------------
    @torch.no_grad()
    def iwae_loglikelihood(self, x, n_samples: int = 500, chunk: int = 100):
        """IWAE bound (Burda 2016), chunked over posterior samples."""
        x = x.view(x.shape[0], -1)
        post = self._posterior_params(x)
        pieces = []
        remaining = n_samples
        while remaining > 0:
            s = min(chunk, remaining)
            remaining -= s
            pieces.append(self._iwae_chunk(x, post, s))
        lw = torch.cat(pieces)  # (S, B)
        return lw.logsumexp(0) - math.log(lw.shape[0])

    def _bernoulli_log_px(self, x: torch.Tensor, z: torch.Tensor, b: int, s: int):
        """log p(x|z) with x (B, D) expanded over s samples -> (s, B)."""
        xb = x.repeat_interleave(s, dim=0)
        logits = self.decoder(z)
        log_px = -F.binary_cross_entropy_with_logits(logits, xb, reduction="none").sum(-1)
        return log_px.view(b, s).t()


class GaussianVAE(BaseVAE):
    """Diagonal Gaussian posterior, standard normal prior."""

    latent_family = "gaussian"

    def __init__(self, input_dim: int, latent_dim: int, tied_sigma: bool = False, **kw):
        self.tied_sigma = tied_sigma
        super().__init__(input_dim, latent_dim, **kw)

    @property
    def latent_param_dim(self) -> int:
        return self.d + (1 if self.tied_sigma else self.d)

    def _split(self, out):
        mu = out[..., : self.d]
        if self.tied_sigma:
            logvar = out[..., self.d :].expand(*out.shape[:-1], self.d)
        else:
            logvar = out[..., self.d :]
        return mu, logvar

    def encode_latent(self, h):
        mu, logvar = self._split(self.enc_head(h))
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
        return z, kl, {"metrics": {"mean_sigma": std.detach().mean().item()}}

    def _posterior_params(self, x):
        return self._split(self.enc_head(self.encoder(x)))

    def _iwae_chunk(self, x, post, s):
        mu, logvar = post
        b = mu.shape[0]
        mu_s = mu.repeat_interleave(s, dim=0)
        logvar_s = logvar.repeat_interleave(s, dim=0)
        z = mu_s + torch.exp(0.5 * logvar_s) * torch.randn_like(mu_s)
        log_px = self._bernoulli_log_px(x, z, b, s)
        log_pz = (-0.5 * (z.pow(2).sum(-1) + self.d * math.log(2 * math.pi))).view(b, s).t()
        log_qz = (
            -0.5
            * (
                ((z - mu_s) / torch.exp(0.5 * logvar_s)).pow(2).sum(-1)
                + logvar_s.sum(-1)
                + self.d * math.log(2 * math.pi)
            )
        ).view(b, s).t()
        return log_px + log_pz - log_qz


class _SphericalBase(BaseVAE):
    """Shared posterior plumbing for vMF and Power-Spherical arms."""

    @property
    def m(self) -> int:
        return self.d + 1

    @property
    def latent_param_dim(self) -> int:
        return self.m + 1  # unnormalized mu + kappa

    @property
    def decoder_input_dim(self) -> int:
        return self.m

    def _posterior(self, h):
        out = self.enc_head(h)
        mu = out[..., : self.m]
        mu = mu / mu.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        kappa = F.softplus(out[..., self.m]) + 1e-4
        return mu, kappa

    def _orthogonal_sample(self, mu, omega):
        v = torch.randn_like(mu)
        v_perp = v - (v * mu).sum(dim=-1, keepdim=True) * mu
        w = v_perp / v_perp.norm(dim=-1, keepdim=True).clamp_min(1e-20)
        omega_u = omega.unsqueeze(-1)
        return omega_u * mu + (1 - omega_u * omega_u).clamp_min(0).sqrt() * w


class SphericalVAE(_SphericalBase):
    """S-VAE: vMF posterior on S^{m-1}, m = d+1, Uniform prior."""

    latent_family = "vmf"

    def encode_latent(self, h):
        mu, kappa = self._posterior(h)
        z, correction, kl = sample_vmf_with_kl(mu, kappa)
        aux = {
            "correction": correction,
            "metrics": {"mean_kappa": kappa.detach().mean().item()},
            "mu": mu,
            "kappa": kappa,
        }
        return z, kl, aux

    def _posterior_params(self, x):
        mu, kappa = self._posterior(self.encoder(x))
        log_c = log_vmf_log_norm(self.m, kappa.detach().cpu().double()).to(
            x.device, torch.float32
        )
        return mu, kappa, log_c

    def _iwae_chunk(self, x, post, s):
        mu, kappa, log_c = post
        b = mu.shape[0]
        mu_s = mu.repeat_interleave(s, dim=0)
        kappa_s = kappa.repeat_interleave(s, dim=0)
        z, _ = sample_vmf(mu_s, kappa_s)
        log_px = self._bernoulli_log_px(x, z, b, s)
        log_prior = -_log_surface(self.m)
        log_qz = (log_c.repeat_interleave(s) + kappa_s * (z * mu_s).sum(-1)).view(b, s).t()
        return log_px + log_prior - log_qz


class PowerSphericalVAE(_SphericalBase):
    """Power-Spherical posterior on S^{m-1} (De Cao & Aziz 2020)."""

    latent_family = "power_spherical"

    @staticmethod
    def _wd(kappa) -> torch.dtype:
        # Only digamma/gammaln here (no Bessel), so float32 suffices on MPS.
        return torch.float32 if kappa.device.type == "mps" else torch.float64

    def _log_z(self, kappa):
        """log of the PS normalizer Z = S_{m-2} 2^{k+m-2} B(k+(m-1)/2, (m-1)/2)."""
        from torch.special import gammaln

        m = self.m
        k = kappa.to(self._wd(kappa))
        a = k + (m - 1) / 2.0
        b = torch.as_tensor((m - 1) / 2.0, dtype=k.dtype, device=k.device)
        log_beta = gammaln(a) + gammaln(b) - gammaln(a + b)
        return _log_surface(m - 1) + (k + m - 2) * math.log(2.0) + log_beta

    def _kl(self, kappa):
        """KL(PS || Uniform), closed form with digamma; differentiable."""
        from torch.special import digamma

        m = self.m
        k = kappa.to(self._wd(kappa))
        a = k + (m - 1) / 2.0
        b = (m - 1) / 2.0
        e_log1pw = math.log(2.0) + digamma(a) - digamma(a + b)
        kl = k * e_log1pw + _log_surface(m) - self._log_z(kappa)
        return kl.to(kappa.dtype)

    def _sample_omega(self, kappa):
        m = self.m
        beta = torch.distributions.Beta(kappa + (m - 1) / 2.0, (m - 1) / 2.0)
        return 2.0 * beta.rsample() - 1.0  # reparameterized in kappa

    def encode_latent(self, h):
        mu, kappa = self._posterior(h)
        z = self._orthogonal_sample(mu, self._sample_omega(kappa))
        kl = self._kl(kappa)
        aux = {"metrics": {"mean_kappa": kappa.detach().mean().item()}, "mu": mu, "kappa": kappa}
        return z, kl, aux

    def _posterior_params(self, x):
        mu, kappa = self._posterior(self.encoder(x))
        return mu, kappa, self._log_z(kappa.detach()).to(torch.float32)

    def _iwae_chunk(self, x, post, s):
        mu, kappa, log_z = post
        b = mu.shape[0]
        mu_s = mu.repeat_interleave(s, dim=0)
        kappa_s = kappa.repeat_interleave(s, dim=0)
        z = self._orthogonal_sample(mu_s, self._sample_omega(kappa_s))
        log_px = self._bernoulli_log_px(x, z, b, s)
        log_prior = -_log_surface(self.m)
        log_qz = (
            kappa_s * torch.log1p((z * mu_s).sum(-1).clamp(min=-1 + 1e-7))
            - log_z.repeat_interleave(s)
        ).view(b, s).t()
        return log_px + log_prior - log_qz


def make_vae(family: str, input_dim: int, latent_dim: int, **kw) -> BaseVAE:
    if family == "gaussian":
        return GaussianVAE(input_dim, latent_dim, tied_sigma=False, **kw)
    if family == "gaussian_tied":
        return GaussianVAE(input_dim, latent_dim, tied_sigma=True, **kw)
    if family == "vmf":
        return SphericalVAE(input_dim, latent_dim, **kw)
    if family == "power_spherical":
        return PowerSphericalVAE(input_dim, latent_dim, **kw)
    raise ValueError(f"unknown family {family}")
