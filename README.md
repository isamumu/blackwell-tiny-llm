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

## Phase 3: Fused Elementwise Kernels ✅

- `benchmarks/profile_decode.py`: step latency, effective bandwidth, NVML readings and a `torch.profiler` table
  for one backend's decode steps. On WSL, GPU performance counters must be enabled first (NVIDIA Control Panel →
  Developer → Manage GPU Performance Counters → allow all users, then `wsl --shutdown`); `nsys` and Nsight
  Compute need this too.
- Kernels: `rms_norm` (one block per row, warp-shuffle reduction, fp32 accumulation), `silu_mul`, `rope`, and
  the fused `add_rms_norm` (residual add + norm in one pass, so each layer's input norm runs at the end of the
  previous layer).
- Speedups vs PyTorch in `test_ops.py`: **rms_norm ~6x, add_rms_norm ~7x, rope ~7–9x, silu_mul ~2x decode /
  1.4x prefill**.

## Phase 4: Bandwidth-Bound GEMV for Decode ✅

- `linear` GEMV: one warp per output row, `float4` (128-bit) loads, fp32 accumulation, warp-shuffle reduction.
  Prefill and unaligned inputs fall back to PyTorch.
- q/k/v weights concatenated at load time into one `wqkv`, so attention does one GEMV instead of three.
- **144 tok/s decode** (from 138), logits PASS vs HF.
- **Measure with a cold L2.** `test_ops.py` reuses one weight, so shapes ≤ 32 MB stay in L2 and report
  impossible numbers (~870 GB/s). Cycling through >256 MB of weight copies gives the real figure.
- **The plain warp-per-row kernel was already bandwidth-bound.** Even with 16-bit loads it hit ~400 GB/s on the
  big layers; lm_head tops out at ~420 GB/s (~94% of 448), which is the practical ceiling on this card.
- **Kept the simple design on purpose.** Staging `x` in shared memory and splitting rows across several warps
  measured within noise (±5%), so they were dropped. `float4` loads stay: they cost a few lines and matter
  once int8/int4 dequant makes the kernel instruction-bound.
- **The win came from fewer launches, not faster GEMVs.** Fusing q/k/v turns three launches into one and gets
  rid of the 512-row k/v GEMVs, which were too small to fill 30 SMs (~190 GB/s).

## Phase 5: On-the-Fly Dequantization

This is where we pass the fp16 ceiling: fewer bytes per weight means more tokens per second, and freed VRAM
allows longer contexts.

- **Quantize offline:** int8 per-channel first, then int4 group-wise (group size 128, fp16 scales). Keep
  `lm_head` in fp16 at first, since it is the most quality-sensitive layer.
- **Reference first:** add a PyTorch dequantize-then-matmul reference in `ops.py`. Then the benchmark
  separates the two error sources: **torch vs hf** = quantization error (tune the scheme),
  **custom vs torch** = kernel error (should be ~0).
- **Kernels:** dequantize inside the GEMV (unpack to fp16 in registers, never write fp16 weights back to VRAM).
  For prefill, dequantize one layer's weights into a reused fp16 buffer (≤ 32 MB) and call cuBLAS: prefill is
  compute-bound, so the extra pass is amortized over the prompt.
- **Quality gate:** top-1 agreement with HF ≥ 95% under teacher forcing, and readable greedy output.
- **Goal:** **≥ 230 tok/s (int8), ≥ 300 tok/s (int4)**, i.e. 2–2.7x faster decode than HF.

## Phase 6: CUDA Graphs and the Final Comparison

With all the kernels written, remove the remaining CPU overhead and measure against strong baselines.

- **CUDA graphs:** capture one decode step and replay it, so hundreds of launches become one. This needs static
  shapes: attend over the full preallocated KV cache with a length mask instead of a growing slice.
- **Stronger bars:** add `torch.compile` (automatic fusion + CUDA graphs) to the benchmark, and compare with
  llama.cpp at matching quantization on the same GPU.
- **Goal:** hit the target table: **≥ 150 tok/s fp16, ≥ 300 tok/s int4**, ahead of `torch.compile`.

## Phase 7 (Optional): Long Context

At the benchmark's ~177 tokens, attention is ~0.2% of decode time. At long contexts it becomes a real share,
and the int4 weights from Phase 5 free enough VRAM to reach the model's full 128K window on an 8GB card.
Do this after Phases 5 and 6: it needs the int4 weights to fit 128K, and builds on Phase 6's attention over the
full preallocated KV cache with a length mask.

| Context | KV cache (fp16, 32 KB/token) | Fits with fp16 weights? | Fits with int4 weights? |
|---|---|---|---|
| 32K | 1 GB | yes | yes |
| 64K | 2 GB | yes | yes |
| 128K | 4 GB | no (~6.85 GB free) | yes; comfortable with int8 KV |

- **Chunked prefill:** feed long prompts in chunks (e.g. 2K tokens) so the MLP's temporary buffers stay small.
  `ops.attention` must then handle a query chunk attending to everything already cached (an offset causal
  mask instead of SDPA's `is_causal`). Raise `max_seq_len` accordingly.
- **Benchmark:** add a long-context mode to `test_baseline.py` that reports decode tok/s at 8K / 32K / 128K.
  Beyond ~32K, check correctness against saved HF reference outputs, since HF's own KV cache adds another copy.
- **Kernels:** flash-decoding (split the KV length across blocks so all SMs work on a single query, then merge
  the partial softmax results), an int8 KV cache (half the bytes and VRAM), and RoPE fused into the KV-cache write.
- **Goal:** decode tok/s at 32K beats PyTorch SDPA, and a 128K context runs end-to-end on the 8GB card.
