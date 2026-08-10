"""Model arms for Project 1 Tier 0/1.

BaselineMLP: tuned plain MLP (Linear-ReLU stack, standard init, biases).
NormedMLP: nGPT-style normalized MLP — scale-invariant (functionally
normalized) weight rows, learnable per-row eigen-learning-rates,
unit-norm representations after every hidden layer, cosine classifier
with a learnable scalar logit scale.

Heads are extended between phases (5 -> 10 classes) without touching
existing rows.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineMLP(nn.Module):
    def __init__(self, dims=(3072, 1024, 1024, 1024), num_classes=5):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], num_classes)

    def forward_features(self, x):
        return self.body(x)

    def forward(self, x):
        return self.head(self.body(x))

    def extend_head(self, new_total):
        old = self.head
        new = nn.Linear(old.in_features, new_total, bias=True,
                        device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[: old.out_features].copy_(old.weight)
            new.bias[: old.out_features].copy_(old.bias)
        self.head = new

    def weight_matrices(self):
        mats = [m.weight for m in self.body if isinstance(m, nn.Linear)]
        mats.append(self.head.weight)
        return mats


class NormedLinear(nn.Module):
    """Linear layer with functionally normalized rows and a per-row
    learnable eigen-learning-rate alpha:  y = x @ (alpha * w/||w||)^T.

    The raw parameter is scale-invariant; weight decay on it controls the
    equilibrium norm and hence the effective learning rate (van Laarhoven
    2017; Li & Arora 2019). This is the Tier-1 dose-response knob.
    """

    def __init__(self, in_f, out_f, alpha_init=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        self.alpha = nn.Parameter(torch.full((out_f,), float(alpha_init)))

    def forward(self, x):
        w = F.normalize(self.weight, dim=1) * self.alpha[:, None]
        return F.linear(x, w)


class NormedMLP(nn.Module):
    def __init__(self, dims=(3072, 1024, 1024, 1024), num_classes=5,
                 alpha_init=1.0, logit_scale_init=10.0):
        super().__init__()
        blocks = []
        for a, b in zip(dims[:-1], dims[1:]):
            blocks.append(NormedLinear(a, b, alpha_init=alpha_init))
        self.blocks = nn.ModuleList(blocks)
        self.head_weight = nn.Parameter(torch.randn(num_classes, dims[-1]))
        self.logit_scale = nn.Parameter(torch.tensor(float(logit_scale_init)))

    def forward_features(self, x):
        h = F.normalize(x, dim=1)
        for blk in self.blocks:
            h = F.relu(blk(h))
            h = F.normalize(h, dim=1)
        return h

    def forward(self, x):
        h = self.forward_features(x)
        w = F.normalize(self.head_weight, dim=1)
        return self.logit_scale * (h @ w.T)

    def extend_head(self, new_total):
        old = self.head_weight
        new = torch.randn(new_total, old.shape[1], device=old.device, dtype=old.dtype)
        with torch.no_grad():
            new[: old.shape[0]].copy_(old)
        self.head_weight = nn.Parameter(new)

    def weight_matrices(self):
        return [b.weight for b in self.blocks] + [self.head_weight]


def build_model(arm, num_classes=5, **kw):
    if arm == "mlp":
        return BaselineMLP(num_classes=num_classes, **kw)
    if arm == "nmlp":
        return NormedMLP(num_classes=num_classes, **kw)
    raise ValueError(arm)
