# benchmarks/test_baseline.py
#
# Compares performance and numerics of three backends on the same weights:
#   hf     - stock HuggingFace forward (the reference everything is checked against)
#   torch  - src/model.py (BlackwellOptimizedLLM) with pure PyTorch ops (your own baseline)
#   custom - src/model.py (BlackwellOptimizedLLM) with your CUDA kernels (unbound ops fall back to PyTorch)
#
# Usage (from any directory):
#   python benchmarks/test_baseline.py                      # all backends
#   python benchmarks/test_baseline.py --backends torch,custom --runs 5
import argparse
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.hf_reference import load_hf_model
from src.model import BlackwellOptimizedLLM
from src.ops import make_ops

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "Explain why writing custom GPU kernels makes an AI inference engine faster."


class HFRunner:
    def __init__(self, hf_model):
        self.model = hf_model

    def reset(self):
        self.cache = DynamicCache()

    @torch.no_grad()
    def step(self, ids):
        out = self.model(input_ids=ids, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        return out.logits[:, -1]


class CustomRunner:
    def __init__(self, model):
        self.model = model
        self.cache = model.new_cache()

    def reset(self):
        self.cache.reset()

    def step(self, ids):
        return self.model.forward(ids, self.cache)


def generate(runner, prompt_ids, n_tokens):
    """Greedy-generate exactly n_tokens (EOS ignored so every run does the same work).
    Returns (tokens [n_tokens], prefill_ms, decode_ms)."""
    tokens = torch.empty(n_tokens, dtype=torch.long, device="cuda")
    start, prefilled, done = (torch.cuda.Event(enable_timing=True) for _ in range(3))

    runner.reset()
    start.record()
    tok = runner.step(prompt_ids).argmax(-1, keepdim=True)
    prefilled.record()
    tokens[0] = tok[0, 0]
    for i in range(1, n_tokens):
        tok = runner.step(tok).argmax(-1, keepdim=True)
        tokens[i] = tok[0, 0]
    done.record()
    torch.cuda.synchronize()
    return tokens.cpu(), start.elapsed_time(prefilled), prefilled.elapsed_time(done)


def teacher_forced_logits(runner, prompt_ids, ref_tokens):
    """Feed the reference token sequence and collect fp32 logits at every step, so
    one early mismatch doesn't make every later comparison meaningless."""
    runner.reset()
    logits = [runner.step(prompt_ids).float()]
    for t in ref_tokens[:-1].tolist():
        logits.append(runner.step(torch.tensor([[t]], device="cuda")).float())
    return torch.cat(logits)  # [n_tokens, vocab]


def benchmark(runner, prompt_ids, args):
    for _ in range(args.warmup):
        generate(runner, prompt_ids, args.new_tokens)
    torch.cuda.reset_peak_memory_stats()
    runs = [generate(runner, prompt_ids, args.new_tokens) for _ in range(args.runs)]
    prefill_ms = statistics.median(r[1] for r in runs)
    decode_ms = statistics.median(r[2] for r in runs)
    return {
        "tokens": runs[0][0],
        "ttft_ms": prefill_ms,
        "decode_tps": (args.new_tokens - 1) / (decode_ms / 1000),
        "peak_mb": torch.cuda.max_memory_allocated() / 2**20,
    }


def compare(logits, ref_logits, tokens, ref_tokens, args):
    diff = (logits - ref_logits).abs()
    top1 = (logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    mismatch = (tokens != ref_tokens).nonzero()
    first_div = mismatch[0].item() if len(mismatch) else None
    ok = diff.max().item() <= args.atol and top1 >= args.min_top1
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "top1": top1,
        "greedy": "all" if first_div is None else f"first {first_div}",
        "status": "PASS" if ok else "FAIL",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backends", default="hf,torch,custom")
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--atol", type=float, default=0.5, help="max allowed |logit - hf logit|")
    p.add_argument("--min-top1", type=float, default=0.99, help="min teacher-forced argmax agreement with hf")
    p.add_argument("--show-text", action="store_true")
    args = p.parse_args()
    backends = args.backends.split(",")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = load_hf_model(MODEL_ID)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt", return_dict=False
    ).to("cuda")

    runners = {}
    for name in backends:
        if name == "hf":
            runners[name] = HFRunner(hf_model)
        elif name in ("torch", "custom"):
            ops, overridden = make_ops(name)
            if name == "custom":
                print(f"Custom kernels: {', '.join(overridden) or 'none bound yet (all ops use PyTorch)'}")
            runners[name] = CustomRunner(BlackwellOptimizedLLM(hf_model, ops))
        else:
            sys.exit(f"unknown backend: {name}")

    # Reference: HF greedy tokens and the logits along that sequence.
    print("Computing HF reference...")
    hf = HFRunner(hf_model)
    ref_tokens = generate(hf, prompt_ids, args.new_tokens)[0]
    ref_logits = teacher_forced_logits(hf, prompt_ids, ref_tokens)

    results = {}
    for name, runner in runners.items():
        print(f"Running {name}...")
        r = benchmark(runner, prompt_ids, args)
        r.update(compare(teacher_forced_logits(runner, prompt_ids, ref_tokens), ref_logits,
                         r["tokens"], ref_tokens, args))
        results[name] = r
        if args.show_text:
            print(f"  [{name}] {tokenizer.decode(r['tokens'], skip_special_tokens=True)!r}")

    print(f"\nPrompt: {prompt_ids.shape[1]} tokens, generated: {args.new_tokens} tokens, "
          f"median of {args.runs} runs  (tolerance: max|Δ| <= {args.atol}, top1 >= {args.min_top1})")
    header = (f"{'backend':<8} {'TTFT ms':>8} {'decode tok/s':>13} {'vs hf':>6} {'vs torch':>9} "
              f"{'peak MB':>8} {'max|Δ|':>8} {'mean|Δ|':>9} {'top1':>7} {'greedy match':>13}  status")
    print(header + "\n" + "-" * len(header))
    for name, r in results.items():
        vs = lambda b: f"{r['decode_tps'] / results[b]['decode_tps']:.2f}x" if b in results else "-"
        print(f"{name:<8} {r['ttft_ms']:>8.2f} {r['decode_tps']:>13.1f} {vs('hf'):>6} {vs('torch'):>9} "
              f"{r['peak_mb']:>8.0f} {r['max_abs']:>8.4f} {r['mean_abs']:>9.5f} {r['top1']:>7.2%} "
              f"{r['greedy']:>13}  {r['status']}")

    sys.exit(0 if all(r["status"] == "PASS" for r in results.values()) else 1)


if __name__ == "__main__":
    main()
