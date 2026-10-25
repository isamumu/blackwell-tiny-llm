# src/ops.py
#
# Every math op the custom forward pass uses goes through an "ops" table.
#   - make_ops("torch")  -> pure PyTorch reference implementations below
#   - make_ops("custom") -> same table, but any op you bind in src/cuda/bindings.cpp
#                           replaces the PyTorch version (the rest fall back)
#
# The reference implementations define "correct": a custom kernel must take the
# same arguments and return the same shapes/dtypes.
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

CUDA_DIR = Path(__file__).resolve().parent / "cuda"

OP_NAMES = ["rms_norm", "linear", "silu_mul", "rope", "attention"]


# ---------------------------------------------------------------------------
# Reference implementations (PyTorch eager)
# ---------------------------------------------------------------------------

def rms_norm(x, weight, eps):
    """x: [..., hidden] fp16, weight: [hidden] fp16 -> [..., hidden] fp16.
    Variance is computed in fp32, matching HF's LlamaRMSNorm."""
    x32 = x.float()
    x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x32.to(x.dtype)


def linear(x, weight):
    """x: [..., in] fp16, weight: [out, in] fp16 -> [..., out] fp16 (no bias in Llama)."""
    return F.linear(x, weight)


def silu_mul(gate, up):
    """SwiGLU: silu(gate) * up. gate/up: [..., intermediate] fp16."""
    return F.silu(gate) * up


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope(q, k, cos, sin):
    """q: [B, n_heads, T, head_dim], k: [B, n_kv_heads, T, head_dim],
    cos/sin: [T, head_dim] fp16 -> (q_rot, k_rot) with the same shapes."""
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


def attention(q, k, v):
    """q: [B, n_heads, Tq, head_dim], k/v: [B, n_kv_heads, Tk, head_dim] (GQA)
    -> [B, n_heads, Tq, head_dim]. Causal when Tq > 1 (prefill from an empty cache)."""
    assert q.shape[2] == 1 or q.shape[2] == k.shape[2], "chunked prefill is not supported"
    return F.scaled_dot_product_attention(q, k, v, is_causal=q.shape[2] > 1, enable_gqa=True)


TORCH_OPS = {name: globals()[name] for name in OP_NAMES}


# ---------------------------------------------------------------------------
# Custom kernel loading
# ---------------------------------------------------------------------------

_ext = None


def load_custom_ext(verbose=False):
    """JIT-compile src/cuda/bindings.cpp + every src/cuda/*.cu into one extension.
    Rebuilds automatically when a source file changes; cached otherwise."""
    global _ext
    if _ext is None:
        import torch.utils.cpp_extension as cpp_ext

        # Use the CUDA 13 toolkit even if an older nvcc (e.g. apt's 11.5 in /usr/bin,
        # which can't target Blackwell) comes first on PATH. torch caches CUDA_HOME at
        # import time, so set the module attribute, not just the env var.
        cuda_home = "/usr/local/cuda" if Path("/usr/local/cuda/bin/nvcc").exists() else cpp_ext.CUDA_HOME
        cpp_ext.CUDA_HOME = os.environ["CUDA_HOME"] = cuda_home
        os.environ["PATH"] = cuda_home + "/bin:" + os.environ.get("PATH", "")
        # Build for the GPU actually present (RTX 5060 = sm_120). A mismatched arch
        # compiles fine but every launch fails with "no kernel image".
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
        sources = [CUDA_DIR / "bindings.cpp", *sorted(CUDA_DIR.glob("*.cu"))]
        _ext = cpp_ext.load(
            name="blackwell_kernels",
            sources=[str(s) for s in sources],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            verbose=verbose,
        )
    return _ext


def make_ops(backend="torch", verbose=False):
    """Returns (ops, overridden) where ops.<name> is callable and `overridden`
    lists the ops served by custom kernels."""
    table = dict(TORCH_OPS)
    overridden = []
    if backend == "custom":
        ext = load_custom_ext(verbose)
        for name in OP_NAMES:
            if hasattr(ext, name):
                table[name] = getattr(ext, name)
                overridden.append(name)
    elif backend != "torch":
        raise ValueError(f"unknown ops backend: {backend}")
    return SimpleNamespace(**table), overridden
