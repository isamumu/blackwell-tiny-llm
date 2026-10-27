# benchmarks/profile_decode.py
#
# Profiles decode steps of one backend -- nothing else (no HF reference,
# no correctness checks), so the profile shows only what a decode step costs.
#
#   1. GPU usage: step latency, effective memory bandwidth vs peak, and NVML
#      readings (GPU / memory-controller utilization, power, clocks) sampled
#      while decoding. Needs `pip install nvidia-ml-py`.
#   2. Op/kernel table from torch.profiler.
#
# Usage:
#   python benchmarks/profile_decode.py                     # torch.profiler table, torch backend
#   python benchmarks/profile_decode.py --backend custom    # with your kernels
import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import BlackwellOptimizedLLM

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def bytes_per_step(model, context):
    """Bytes a decode step must read from VRAM: every weight once, plus the KV cache."""
    weights = {id(t): t for w in model.layers for t in vars(w).values()}
    weights[id(model.final_norm)] = model.final_norm
    weights[id(model.lm_head)] = model.lm_head  # embedding lookup reads one row; ignored
    weight_bytes = sum(t.numel() * t.element_size() for t in weights.values())
    kv_bytes = 2 * len(model.layers) * model.n_kv_heads * context * model.head_dim * model.embed.element_size()
    return weight_bytes + kv_bytes


class NvmlSampler(threading.Thread):
    """Polls NVML in the background while decode runs."""

    def __init__(self, interval=0.02):
        super().__init__(daemon=True)
        self.interval, self.samples, self._done = interval, [], threading.Event()

    def run(self):
        while not self._done.is_set():
            self.samples.append((torch.cuda.utilization(), torch.cuda.memory_usage(),
                                 torch.cuda.power_draw() / 1000, torch.cuda.clock_rate()))
            time.sleep(self.interval)

    def stop(self):
        self._done.set()
        self.join()


def gpu_usage(model, decode_step, args):
    try:
        torch.cuda.utilization()
        sampler = NvmlSampler()
    except Exception:
        sampler = None
        print("NVML unavailable (pip install nvidia-ml-py) -- skipping utilization/power/clock readings.\n")

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.measure_steps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.measure_steps)]
    torch.cuda.reset_peak_memory_stats()
    if sampler:
        sampler.start()
    for s, e in zip(starts, ends):
        s.record()
        decode_step()
        e.record()
    torch.cuda.synchronize()
    if sampler:
        sampler.stop()

    step_ms = statistics.median(s.elapsed_time(e) for s, e in zip(starts, ends))
    gb = bytes_per_step(model, args.context) / 1e9
    bw = gb / (step_ms / 1000)
    free, total = torch.cuda.mem_get_info()

    print(f"=== GPU usage: {args.backend} backend, context {args.context}, {args.measure_steps} decode steps ===")
    print(f"Step latency (median):  {step_ms:.2f} ms  ->  {1000 / step_ms:.1f} tok/s")
    print(f"Bytes read per step:    {gb:.3f} GB (weights + KV cache)")
    print(f"Effective bandwidth:    {bw:.0f} GB/s  =  {bw / args.peak_bw:.0%} of {args.peak_bw:.0f} GB/s peak"
          f"  (ceiling at 100%: {args.peak_bw / gb:.0f} tok/s)")
    if sampler and sampler.samples:
        util, mem_util, power, clock = zip(*sampler.samples)
        mean = lambda xs: sum(xs) / len(xs)
        print(f"GPU utilization:        {mean(util):.0f}%  (share of time any kernel was running)")
        print(f"Memory-controller util: {mean(mem_util):.0f}%  (share of time VRAM was being read/written)")
        print(f"Power / SM clock:       {mean(power):.0f} W avg, {max(power):.0f} W max / {mean(clock):.0f} MHz avg")
    print(f"Memory:                 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak allocated, "
          f"{free / 2**30:.2f} of {total / 2**30:.2f} GiB free on device\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default="torch", choices=["torch", "custom"])
    p.add_argument("--context", type=int, default=128, help="tokens in the KV cache (held fixed while profiling)")
    p.add_argument("--steps", type=int, default=5, help="decode steps to profile with torch.profiler")
    p.add_argument("--measure-steps", type=int, default=300, help="decode steps for the GPU usage measurement")
    p.add_argument("--peak-bw", type=float, default=448, help="GPU peak memory bandwidth in GB/s (RTX 5060: 448)")
    p.add_argument("--trace", default=None, help="also write a Chrome/Perfetto trace to this path")
    args = p.parse_args()

    model = BlackwellOptimizedLLM.from_pretrained(MODEL_ID, backend=args.backend)
    cache = model.new_cache()
    tok = torch.zeros(1, 1, dtype=torch.long, device="cuda")

    def decode_step():
        cache.len = args.context  # rewrite the same slot so every step sees the same context length
        return model.forward(tok, cache)

    # Prefill to the requested context length, then warm up (first calls pay one-time setup costs).
    model.forward(torch.randint(0, 1000, (1, args.context), device="cuda"), cache)
    for _ in range(5):
        decode_step()
    torch.cuda.synchronize()

    gpu_usage(model, decode_step, args)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(args.steps):
            decode_step()
        torch.cuda.synchronize()

    events = prof.key_averages()
    has_gpu = any(e.device_time_total > 0 for e in events)
    if not has_gpu:
        print("No GPU kernel timings recorded (on WSL: enable GPU performance counters, see README Phase 3).\n"
              "Showing CPU-side PyTorch ops only.\n")
    sort_by = "device_time_total" if has_gpu else "cpu_time_total"
    print(f"=== Ops: {args.steps} decode steps (divide counts by {args.steps} for per-step) ===")
    print(events.table(sort_by=sort_by, row_limit=25))
    if args.trace:
        prof.export_chrome_trace(args.trace)
        print(f"Trace written to {args.trace} (open in https://ui.perfetto.dev)")


if __name__ == "__main__":
    main()
