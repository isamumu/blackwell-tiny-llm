# benchmarks/test_ops.py
#
# Per-op check for custom kernels: compares each op bound in src/cuda/bindings.cpp
# against the PyTorch reference in src/ops.py on Llama-3.2-1B shapes (decode and
# prefill), and times both. Faster to iterate on than the full-model benchmark.
#
# Usage: python benchmarks/test_ops.py
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ops import TORCH_OPS, make_ops

HIDDEN, INTER, N_HEADS, N_KV, HEAD_DIM, VOCAB = 2048, 8192, 32, 8, 64, 128256
ATOL, RTOL = 1e-2, 1e-2


def h(*shape):
    return torch.randn(*shape, device="cuda", dtype=torch.float16)


def make_inputs(op, T):
    """Inputs for `op` at sequence length T (T=1 is decode)."""
    if op == "rms_norm":
        return [(h(1, T, HIDDEN), h(HIDDEN), 1e-5)]
    if op == "linear":
        return [(h(1, T, HIDDEN), h(N, HIDDEN) * 0.02) for N in (HIDDEN, INTER)] + [(h(1, T, INTER), h(HIDDEN, INTER) * 0.02)]
    if op == "silu_mul":
        return [(h(1, T, INTER), h(1, T, INTER))]
    if op == "rope":
        return [(h(1, N_HEADS, T, HEAD_DIM), h(1, N_KV, T, HEAD_DIM), h(T, HEAD_DIM).cos(), h(T, HEAD_DIM).sin())]
    if op == "attention":
        Tk = 512 if T == 1 else T
        return [(h(1, N_HEADS, T, HEAD_DIM), h(1, N_KV, Tk, HEAD_DIM), h(1, N_KV, Tk, HEAD_DIM))]
    raise KeyError(op)


def time_ms(fn, args, iters=100):
    for _ in range(10):
        fn(*args)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def max_err(out, ref):
    assert out.shape == ref.shape and out.dtype == ref.dtype, f"got {out.shape}/{out.dtype}, want {ref.shape}/{ref.dtype}"
    ok = torch.allclose(out.float(), ref.float(), atol=ATOL, rtol=RTOL)
    return (out.float() - ref.float()).abs().max().item(), ok


def main():
    torch.manual_seed(0)
    ops, overridden = make_ops("custom", verbose=False)
    if not overridden:
        print("No custom kernels bound in src/cuda/bindings.cpp yet -- nothing to test.")
        return

    failed = False
    print(f"{'op':<10} {'shape':<8} {'max|Δ|':>9} {'torch us':>9} {'custom us':>10} {'speedup':>8}  status")
    for op in overridden:
        for T, label in ((1, "decode"), (128, "prefill")):
            for args in make_inputs(op, T):
                ref = TORCH_OPS[op](*args)
                out = getattr(ops, op)(*args)
                torch.cuda.synchronize()
                errs = [max_err(o, r) for o, r in zip(out, ref)] if isinstance(ref, tuple) else [max_err(out, ref)]
                err, ok = max(e for e, _ in errs), all(o for _, o in errs)
                failed |= not ok
                t_ref = time_ms(TORCH_OPS[op], args) * 1000
                t_cus = time_ms(getattr(ops, op), args) * 1000
                print(f"{op:<10} {label:<8} {err:>9.5f} {t_ref:>9.1f} {t_cus:>10.1f} {t_ref / t_cus:>7.2f}x  "
                      f"{'PASS' if ok else 'FAIL'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
