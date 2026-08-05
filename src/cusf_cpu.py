"""Lazy CPU-only PyTorch binding for the neighboring CUSF checkout."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def _extension():
    repository = Path(__file__).resolve().parents[2] / "cusf"
    cusf_source = repository / "cusf" / "src"
    if not cusf_source.exists():
        raise FileNotFoundError(
            f"CUSF checkout not found at {repository}; expected the sibling ../cusf repository"
        )
    return load(
        name="vmf_cusf_cpu",
        sources=[
            str(Path(__file__).with_name("cusf_cpu_extension.cpp")),
            str(cusf_source / "bessel" / "iv_log.cpp"),
        ],
        extra_include_paths=[str(cusf_source)],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=False,
    )


def iv_log(order: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Evaluate CUSF's compiled CPU ``log(I_order(x))`` implementation."""

    order, x = torch.broadcast_tensors(order, x)
    if order.device.type != "cpu" or x.device.type != "cpu":
        raise ValueError("the CPU-only CUSF extension requires CPU tensors")
    if order.dtype != x.dtype:
        raise ValueError("order and x must have the same dtype")
    return _extension().iv_log(order.contiguous(), x.contiguous())
