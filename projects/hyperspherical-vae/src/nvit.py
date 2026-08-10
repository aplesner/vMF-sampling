"""T3 architectures: conventional ViT and nViT (nGPT-style) VAE backbones.

Both are small (<=20M params) patch transformers for 32x32 images with a
latent head (Gaussian or vMF) and a mirrored decoder.  The nViT follows the
nGPT recipe (Loshchilov et al., ICLR 2025; papers/ngpt.pdf):

  * no LayerNorm/RMSNorm anywhere;
  * every weight matrix with an embedding axis is row-normalized (after each
    optimizer step, and at init);
  * residual updates are spherical retractions with learnable per-dim
    eigen-LRs:  h <- Norm(h + alpha (h_block - h)),  h_block = Norm(block(h));
  * attention: q,k unit-normalized with learnable per-head s_qk, softmax
    scale sqrt(dk); SwiGLU MLP with s_u, s_v scalings (v scaled by sqrt(d));
  * scaling parameters use the (init, scale) effective-LR parameterization
    (stored at `scale`, used at `init`); no weight decay, no LR warmup
    (KL warmup is orthogonal and kept).

The conventional ViT is pre-LN with GELU MLP, matched widths, and identical
latent heads.  RoPE positions in both (no learned positional embeddings, so
the nViT arm has no unnormalized embedding table).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vmf_vae import sample_vmf, sample_vmf_with_kl  # noqa: E402


def _norm(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-20)


class ScaledParam(nn.Parameter):
    """(init, scale) effective-LR parameterization: stored at `scale`, the
    forward pass multiplies by init/scale (nGPT section 2.5)."""

    def __new__(cls, shape, init: float, scale: float):
        p = super().__new__(cls, torch.full(shape, float(scale)))
        p.init_value = float(init)
        p.scale_value = float(scale)
        return p

    def effective(self) -> torch.Tensor:
        return self * (self.init_value / self.scale_value)


def rotary(dim: int, seq_len: int, device, dtype):
    """Standard RoPE cos/sin caches for head-dim `dim` (even)."""
    inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv)  # (L, dim/2)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, L, dk)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


class NLinear(nn.Linear):
    """Linear whose weight rows are unit-normalized (nGPT); normalization is
    applied at init and after each optimizer step via `normalize_weights_`."""

    def normalize_weight_(self):
        with torch.no_grad():
            self.weight.copy_(_norm(self.weight, dim=1))


# ---------------------------------------------------------------------------
# attention / MLP blocks


class ViTAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v)  # scale 1/sqrt(dk)
        return self.out(o.transpose(1, 2).reshape(B, L, D))


class NViTAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.dk = dim // heads
        self.qkv = NLinear(dim, 3 * dim, bias=False)
        self.out = NLinear(dim, dim, bias=False)
        self.s_qk = ScaledParam((heads, self.dk), init=1.0, scale=1.0 / math.sqrt(dim))

    def forward(self, x, cos, sin):
        B, L, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.h, self.dk).transpose(1, 2)
        k = k.view(B, L, self.h, self.dk).transpose(1, 2)
        v = v.view(B, L, self.h, self.dk).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        s = self.s_qk.effective().unsqueeze(0).unsqueeze(2)
        q, k = _norm(q) * s, _norm(k) * s
        o = F.scaled_dot_product_attention(q, k, v, scale=math.sqrt(self.dk))
        return self.out(o.transpose(1, 2).reshape(B, L, D))


class ViTMLP(nn.Module):
    def __init__(self, dim: int, d_mlp: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, d_mlp, bias=False)
        self.fc2 = nn.Linear(d_mlp, dim, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class NViTMLP(nn.Module):
    def __init__(self, dim: int, d_mlp: int):
        super().__init__()
        self.u = NLinear(dim, d_mlp, bias=False)
        self.v = NLinear(dim, d_mlp, bias=False)
        self.out = NLinear(d_mlp, dim, bias=False)
        self.s_u = ScaledParam((d_mlp,), init=1.0, scale=1.0)
        self.s_v = ScaledParam((d_mlp,), init=1.0, scale=1.0)
        self.dim = dim

    def forward(self, x):
        u = self.u(x) * self.s_u.effective()
        v = self.v(x) * self.s_v.effective() * math.sqrt(self.dim)
        return self.out(u * F.silu(v))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, d_mlp: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = ViTAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, d_mlp)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        return x + self.mlp(self.ln2(x))


class NViTBlock(nn.Module):
    def __init__(self, dim: int, heads: int, d_mlp: int):
        super().__init__()
        self.attn = NViTAttention(dim, heads)
        self.mlp = NViTMLP(dim, d_mlp)
        self.alpha_a = ScaledParam((dim,), init=0.05, scale=1.0 / math.sqrt(dim))
        self.alpha_m = ScaledParam((dim,), init=0.05, scale=1.0 / math.sqrt(dim))

    def forward(self, x, cos, sin):
        h_a = _norm(self.attn(x, cos, sin))
        x = _norm(x + self.alpha_a.effective() * (h_a - x))
        h_m = _norm(self.mlp(x))
        return _norm(x + self.alpha_m.effective() * (h_m - x))


# ---------------------------------------------------------------------------
# full encoder/decoder


class _Trunk(nn.Module):
    def __init__(self, dim: int, heads: int, depth: int, seq_len: int, normalized: bool):
        super().__init__()
        self.normalized = normalized
        self.dim = dim
        d_mlp = (8 * dim) // 3 if normalized else 4 * dim  # param-matched SwiGLU
        cls = NViTBlock if normalized else ViTBlock
        self.blocks = nn.ModuleList([cls(dim, heads, d_mlp) for _ in range(depth)])
        self.seq_len = seq_len
        self.heads = heads

    def forward(self, x):
        cos, sin = rotary(self.dim // self.heads, self.seq_len, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return x

    def normalize_weights_(self):
        for mod in self.modules():
            if isinstance(mod, NLinear):
                mod.normalize_weight_()


class ImageEncoder(nn.Module):
    """Image -> sequence of tokens -> latent head features (mean-pooled)."""

    def __init__(self, image_size: int, patch: int, dim: int, heads: int, depth: int, normalized: bool):
        super().__init__()
        self.patch = patch
        self.dim = dim
        n_patches = (image_size // patch) ** 2
        self.patch_embed = nn.Linear(3 * patch * patch, dim, bias=False)
        self.trunk = _Trunk(dim, heads, depth, n_patches, normalized)

    def forward(self, x):
        B = x.shape[0]
        p = self.patch
        x = x.unfold(2, p, p).unfold(3, p, p)  # B, C, H/p, W/p, p, p
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, 3 * p * p)
        x = self.patch_embed(x)
        if self.trunk.normalized:
            x = _norm(x)
        h = self.trunk(x)
        return h.mean(dim=1)  # pooled features for the latent head


class ImageDecoder(nn.Module):
    """Latent -> token sequence -> pixels."""

    def __init__(self, image_size: int, patch: int, dim: int, heads: int, depth: int, normalized: bool):
        super().__init__()
        self.patch = patch
        self.image_size = image_size
        self.normalized = normalized
        self.n_patches = (image_size // patch) ** 2
        self.dim = dim
        self.latent_in = None  # set by the VAE wrapper
        self.trunk = _Trunk(dim, heads, depth, self.n_patches, normalized)
        self.pixel_head = nn.Linear(dim, 3 * patch * patch, bias=True)

    def forward(self, z):
        B = z.shape[0]
        tok = self.latent_in(z).unsqueeze(1).expand(B, self.n_patches, self.dim)
        if self.normalized:
            tok = _norm(tok)
        h = self.trunk(tok)
        px = self.pixel_head(h)
        p = self.patch
        px = px.view(B, self.image_size // p, self.image_size // p, 3, p, p)
        px = px.permute(0, 3, 1, 4, 2, 5).reshape(B, 3, self.image_size, self.image_size)
        return px


# ---------------------------------------------------------------------------
# VAE wrapper: architecture (vit|nvit) x latent (gaussian|vmf) [+sqrt-d rescale]


class ViTVAE(nn.Module):
    """T3 arm. `rescale_sqrt_d` multiplies the (unit-norm) decoder input by
    sqrt(d) — arm 4, the cheap control that decides whether the story is
    input scaling."""

    def __init__(
        self,
        latent: str = "vmf",            # gaussian | vmf
        arch: str = "nvit",             # vit | nvit
        latent_dim: int = 256,
        image_size: int = 32,
        patch: int = 4,
        dim: int = 384,
        heads: int = 6,
        depth: int = 4,
        rescale_sqrt_d: bool = False,
    ):
        super().__init__()
        assert latent in ("gaussian", "vmf")
        assert arch in ("vit", "nvit")
        self.latent = latent
        self.arch = arch
        self.d = latent_dim
        self.rescale_sqrt_d = rescale_sqrt_d
        normalized = arch == "nvit"

        self.encoder_net = ImageEncoder(image_size, patch, dim, heads, depth, normalized)
        self.decoder_net = ImageDecoder(image_size, patch, dim, heads, depth, normalized)

        m = latent_dim + 1 if latent == "vmf" else latent_dim
        self.m = m
        self.latent_head = nn.Linear(dim, 2 * m if latent == "gaussian" else m + 1)
        self.decoder_in = nn.Linear(m, dim, bias=False)
        self.decoder_net.latent_in = self.decoder_in
        # fixed-scale Gaussian decoder (shared across arms)
        self.log_sigma = nn.Parameter(torch.tensor(-1.0))

    # -- posterior ------------------------------------------------------------
    def _posterior(self, feats):
        out = self.latent_head(feats)
        if self.latent == "gaussian":
            mu, logvar = out.chunk(2, dim=-1)
            return mu, logvar
        mu = _norm(out[..., : self.m])
        kappa = F.softplus(out[..., self.m]) + 1e-4
        return mu, kappa

    def encode_latent(self, feats):
        if self.latent == "gaussian":
            mu, logvar = self._posterior(feats)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)
            return z, kl, {"mean_kappa": float("nan"), "mu": mu}
        mu, kappa = self._posterior(feats)
        z, correction, kl = sample_vmf_with_kl(mu, kappa)
        return z, kl, {"correction": correction, "mean_kappa": kappa.detach().mean().item(),
                       "mu": mu, "kappa": kappa}

    def _decoder_input(self, z):
        if self.rescale_sqrt_d:
            z = z * math.sqrt(self.d)
        return z

    def forward(self, x):
        feats = self.encoder_net(x)
        z, kl, aux = self.encode_latent(feats)
        xhat = self.decoder_net(self._decoder_input(z))
        return xhat, kl, aux, z

    def elbo_loss(self, x, beta: float = 1.0):
        xhat, kl, aux, _ = self.forward(x)
        sigma2 = torch.exp(2 * self.log_sigma)
        # Gaussian decoder log-likelihood (summed over pixels), constant dropped
        recon = 0.5 * ((x - xhat).pow(2) / sigma2).sum(dim=(1, 2, 3))
        correction = aux.get("correction")
        recon_eff = recon if correction is None else recon + recon.detach() * correction
        loss = (recon_eff + beta * kl).mean()
        metrics = {
            "recon": recon.detach().mean().item(),
            "kl": kl.detach().mean().item(),
            "loss": (recon.detach() + beta * kl.detach()).mean().item(),
            "mean_kappa": aux.get("mean_kappa", float("nan")),
        }
        return loss, metrics

    @torch.no_grad()
    def iwae_loglikelihood(self, x, n_samples: int = 500, chunk: int = 50):
        """IWAE bound with the Gaussian decoder (shared across arms)."""
        from vmf_kl import log_surface_area, log_vmf_log_norm

        feats = self.encoder_net(x)
        B = x.shape[0]
        npix = x[0].numel()
        sigma2 = torch.exp(2 * self.log_sigma)
        if self.latent == "gaussian":
            mu, logvar = self._posterior(feats)
            post = (mu, logvar)
        else:
            mu, kappa = self._posterior(feats)
            log_c = log_vmf_log_norm(self.m, kappa.detach().cpu().double()).to(
                x.device, torch.float32
            )
            post = (mu, kappa, log_c)
        pieces = []
        remaining = n_samples
        while remaining > 0:
            s = min(chunk, remaining)
            remaining -= s
            if self.latent == "gaussian":
                mu, logvar = post
                mu_s = mu.repeat_interleave(s, dim=0)
                lv_s = logvar.repeat_interleave(s, dim=0)
                z = mu_s + torch.exp(0.5 * lv_s) * torch.randn_like(mu_s)
                log_pz = -0.5 * (z.pow(2).sum(-1) + self.d * math.log(2 * math.pi))
                log_qz = -0.5 * (
                    ((z - mu_s) / torch.exp(0.5 * lv_s)).pow(2).sum(-1)
                    + lv_s.sum(-1)
                    + self.d * math.log(2 * math.pi)
                )
            else:
                mu, kappa, log_c = post
                mu_s = mu.repeat_interleave(s, dim=0)
                kappa_s = kappa.repeat_interleave(s, dim=0)
                z, _ = sample_vmf(mu_s, kappa_s)
                log_pz = torch.full(
                    (z.shape[0],), -log_surface_area(self.m).item(), device=z.device
                )
                log_qz = log_c.repeat_interleave(s) + kappa_s * (z * mu_s).sum(-1)
            xhat = self.decoder_net(self._decoder_input(z))
            x_s = x.repeat_interleave(s, dim=0)
            log_px = -0.5 * (
                (x_s - xhat).pow(2) / sigma2 + math.log(2 * math.pi) + 2 * self.log_sigma
            ).sum(dim=(1, 2, 3))
            pieces.append((log_px + log_pz - log_qz).view(B, s).t())
        lw = torch.cat(pieces)
        return lw.logsumexp(0) - math.log(lw.shape[0])

    def normalize_weights_(self):
        """nGPT post-step normalization of all embedding-axis matrices."""
        if self.arch != "nvit":
            return
        with torch.no_grad():
            self.encoder_net.patch_embed.weight.copy_(
                _norm(self.encoder_net.patch_embed.weight, dim=1)
            )
            self.decoder_net.pixel_head.weight.copy_(
                _norm(self.decoder_net.pixel_head.weight, dim=1)
            )
        self.encoder_net.trunk.normalize_weights_()
        self.decoder_net.trunk.normalize_weights_()


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
