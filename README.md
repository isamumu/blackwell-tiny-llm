# blackwell-tiny-llm

A from-scratch inference engine for Llama-3.2-1B-Instruct on an RTX 5060 (Blackwell, sm_120, 8GB GDDR7),
built to beat stock HuggingFace by replacing PyTorch ops with custom CUDA kernels.

## The Target

Decoding one token streams every weight from VRAM, so decode speed is capped by memory bandwidth
(~448 GB/s), not by compute. That gives hard ceilings to aim at:

| Configuration | Bytes read per token | Ceiling | Goal |
|---|---|---|---|
| HF baseline (fp16) | ~2.5 GB | ~180 tok/s | 110 tok/s (measured) |
| Optimized fp16 | ~2.5 GB | ~180 tok/s | **≥ 150 tok/s** (1.4x HF) |
| int8 weights, fp16 lm_head | ~1.5 GB | ~300 tok/s | **≥ 230 tok/s** (2x HF) |
| int4 weights, fp16 lm_head | ~1.05 GB | ~430 tok/s | **≥ 300 tok/s** (2.7x HF) |

Every phase is measured against HF for both speed and correctness. A speedup only counts if the
benchmark status is PASS.

## Project Layout

```
src/
  hf_reference.py   load_hf_model(): stock HF Llama, the ground truth
  model.py          BlackwellOptimizedLLM: our forward pass + KV cache; all math goes through `ops`
  ops.py            PyTorch reference ops (the spec) + loader that swaps in custom kernels
  inference.py      text generation demo: python src/inference.py [torch|custom]
  cuda/
    bindings.cpp    bind kernels here under an op name (rms_norm, linear, ...) to activate them
    *.cu            kernels; every .cu file is compiled automatically
benchmarks/
  test_ops.py       per-kernel correctness + µs vs the PyTorch reference (fast iteration loop)
  test_baseline.py  full model: TTFT, decode tok/s, peak memory, logits vs HF for hf/torch/custom
```

Workflow for every kernel: write the `.cu` file, bind it in `bindings.cpp`, then run
`python benchmarks/test_ops.py && python benchmarks/test_baseline.py`.

## Phase 1: Python Baseline ✅

- Hand-written Llama forward pass (`BlackwellOptimizedLLM`) with a preallocated KV cache, borrowing HF's weights.
- Matches HF logits bit-for-bit. Baseline: **110 tok/s decode, 10 ms TTFT (49-token prompt), 2.6 GB peak**.

## Phase 2: C++/CUDA Extension Harness ✅

- JIT-built extension (`torch.utils.cpp_extension`) with per-op fallback to PyTorch.
- Toolchain gotchas: build with CUDA 13 (`/usr/local/cuda`), not apt's CUDA 11.5 in `/usr/bin`, and target
  sm_120, not sm_100. The loader in `src/ops.py` handles both. End every kernel wrapper with
  `C10_CUDA_KERNEL_LAUNCH_CHECK()`.

## Phase 3: Profile and Remove Launch Overhead

At 110 tok/s we are ~60% of the bandwidth ceiling. The rest is largely the CPU issuing several hundred
small kernel launches per token, with the GPU idling between them. Fix this first so later kernel gains
show up in tok/s.

- **Profile:** `nsys profile` one decode step. Count launches per token and look for GPU idle gaps.
  On WSL, first enable GPU performance counters (NVIDIA Control Panel → Developer → Manage GPU Performance
  Counters → allow all users), then `wsl --shutdown`.
- **Stronger bar:** add a `torch.compile` row to the benchmark, so we know whether we beat a good baseline or
  only a naive one.
- **CUDA graphs:** capture one decode step and replay it. This needs static shapes: attend over the full
  preallocated KV cache with a length mask instead of a growing slice.
- **Goal:** ≥ 140 tok/s with plain PyTorch ops, and still PASS.

## Phase 4: Fused Elementwise Kernels

PyTorch runs RMSNorm as ~6 separate kernels, each reading and writing VRAM. Fusing keeps intermediates in
registers.

- **Kernels:** `rms_norm` (one block per row, warp-shuffle reduction, fp32 accumulation), `silu_mul`,
  `rope`, then a new fused op `add_rms_norm` (residual add + norm in one pass). Llama has no norm bias.
- **Goal:** each kernel ≥ 3x faster than PyTorch in `test_ops.py`, and measurable end-to-end gain on top of
  Phase 3.

## Phase 5: Bandwidth-Bound GEMV for Decode

During decode every matmul has M = 1, so each weight is used exactly once and there is no reuse for tiling
to exploit. Speed comes from streaming weights at full bandwidth.

- **Kernels:** a `linear` GEMV with coalesced 128-bit vectorized loads (`half2`/`float4`), warp-level reduction,
  and enough blocks to saturate every SM.
- **Weight layout:** concatenate q/k/v and gate/up at load time (in `BlackwellOptimizedLLM.__init__`) so each
  input is read once and there are fewer launches. Fuse `silu_mul` into the gate/up GEMV output.
- **Goal:** ≥ 80% of 448 GB/s on decode shapes in `test_ops.py`; **≥ 150 tok/s end-to-end**.

## Phase 6: Tiled GEMM for Prefill

With a long prompt, each weight is reused across every prompt token, so prefill is compute-bound and tiling
pays off. This shows up in TTFT.

- **Benchmark:** add a long-prompt mode (e.g. 512–2048 tokens) to `test_baseline.py`. The 49-token prompt is
  too short to stress prefill.
- **Kernels:** shared-memory tiled GEMM → register tiling → tensor cores (`mma` instructions) → double-buffered
  async copies. `linear` dispatches to GEMV when M = 1 and GEMM otherwise.
- **Tuning:** use Nsight Compute to remove bank conflicts and raise tensor-core utilization.
- **Goal:** ≥ 50% of cuBLAS throughput at M = 512. cuBLAS is tuned by NVIDIA, so beating it is a stretch goal.
  The point of this phase is the skills Phase 7 needs.

## Phase 7: On-the-Fly Dequantization

This is where we pass the fp16 ceiling: fewer bytes per weight means more tokens per second, and freed VRAM
allows longer contexts.

- **Quantize offline:** int8 per-channel first, then int4 group-wise (group size 128, fp16 scales). Keep
  `lm_head` in fp16 at first, since it is the most quality-sensitive layer.
- **Reference first:** add a PyTorch dequantize-then-matmul reference in `ops.py`. Then the benchmark
  separates the two error sources: **torch vs hf** = quantization error (tune the scheme),
  **custom vs torch** = kernel error (should be ~0).
- **Kernels:** dequantize inside the GEMV (unpack to fp16 in registers, never write fp16 weights back to VRAM),
  then inside the tiled GEMM for prefill.
- **Quality gate:** top-1 agreement with HF ≥ 95% under teacher forcing, and readable greedy output.
- **Goal:** **≥ 230 tok/s (int8), ≥ 300 tok/s (int4)**, i.e. 2–2.7x faster decode than HF.
