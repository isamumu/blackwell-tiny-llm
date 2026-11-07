# src/model.py
#
# BlackwellOptimizedLLM: hand-written Llama forward pass with a KV cache. All math goes through `ops`
# (see src/ops.py), so swapping a PyTorch op for your CUDA kernel is a matter of
# binding it in src/cuda/bindings.cpp -- this file doesn't change.
#
# Weights are borrowed from an already-loaded HF model (no copy), so the HF
# reference and this model can live in the same process on an 8GB card.
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.hf_reference import load_hf_model
from src.ops import make_ops


@dataclass
class LayerWeights:
    input_norm: torch.Tensor  # [hidden]
    wq: torch.Tensor          # [n_heads * head_dim, hidden]
    wk: torch.Tensor          # [n_kv_heads * head_dim, hidden]
    wv: torch.Tensor          # [n_kv_heads * head_dim, hidden]
    wo: torch.Tensor          # [hidden, n_heads * head_dim]
    post_norm: torch.Tensor   # [hidden]
    w_gate: torch.Tensor      # [intermediate, hidden]
    w_up: torch.Tensor        # [intermediate, hidden]
    w_down: torch.Tensor      # [hidden, intermediate]


class KVCache:
    """Preallocated per-layer K/V buffers: [layers, B, n_kv_heads, max_seq_len, head_dim]."""

    def __init__(self, n_layers, batch, n_kv_heads, max_seq_len, head_dim, dtype, device):
        shape = (n_layers, batch, n_kv_heads, max_seq_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.len = 0

    def update(self, layer, k, v):
        """Write new k/v ([B, n_kv, T, hd]) at the current position, return everything cached so far."""
        end = self.len + k.shape[2]
        assert end <= self.k.shape[3], "KV cache full; raise max_seq_len"
        self.k[layer, :, :, self.len:end] = k
        self.v[layer, :, :, self.len:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def reset(self):
        self.len = 0


class BlackwellOptimizedLLM:
    @classmethod
    def from_pretrained(cls, model_id, backend="torch", **kwargs):
        """backend: "torch" (PyTorch ops) or "custom" (your CUDA kernels where bound)."""
        ops, _ = make_ops(backend)
        return cls(load_hf_model(model_id), ops, **kwargs)

    def __init__(self, hf_model, ops, max_seq_len=2048):
        cfg = hf_model.config
        m = hf_model.model
        self.ops = ops
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.eps = cfg.rms_norm_eps
        self.max_seq_len = max_seq_len
        self.dtype = m.embed_tokens.weight.dtype
        self.device = m.embed_tokens.weight.device

        self.embed = m.embed_tokens.weight
        self.final_norm = m.norm.weight
        self.lm_head = hf_model.lm_head.weight
        self.layers = [
            LayerWeights(
                input_norm=l.input_layernorm.weight,
                wq=l.self_attn.q_proj.weight,
                wk=l.self_attn.k_proj.weight,
                wv=l.self_attn.v_proj.weight,
                wo=l.self_attn.o_proj.weight,
                post_norm=l.post_attention_layernorm.weight,
                w_gate=l.mlp.gate_proj.weight,
                w_up=l.mlp.up_proj.weight,
                w_down=l.mlp.down_proj.weight,
            )
            for l in m.layers
        ]
        # Llama 3 RoPE uses scaled frequencies; reuse HF's precomputed ones.
        self.inv_freq = m.rotary_emb.inv_freq.float()
        self.rope_scale = m.rotary_emb.attention_scaling

    def new_cache(self, batch=1):
        return KVCache(len(self.layers), batch, self.n_kv_heads, self.max_seq_len,
                       self.head_dim, self.dtype, self.device)

    def _rope_tables(self, start, length):
        pos = torch.arange(start, start + length, device=self.device, dtype=torch.float32)
        freqs = pos[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
        return (emb.cos() * self.rope_scale).to(self.dtype), (emb.sin() * self.rope_scale).to(self.dtype)

    def _attention(self, h, w, cos, sin, cache, layer):
        ops = self.ops
        B, T, _ = h.shape
        q = ops.linear(h, w.wq).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = ops.linear(h, w.wk).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = ops.linear(h, w.wv).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = ops.rope(q, k, cos, sin)
        k, v = cache.update(layer, k, v)
        o = ops.attention(q, k, v)
        o = o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return ops.linear(o, w.wo)

    def _mlp(self, h, w):
        ops = self.ops
        return ops.linear(ops.silu_mul(ops.linear(h, w.w_gate), ops.linear(h, w.w_up)), w.w_down)

    @torch.no_grad()
    def forward(self, input_ids, cache):
        """input_ids: [B, T] -> logits for the last position [B, vocab].
        Pass the whole prompt once (prefill), then one token at a time (decode)."""
        ops = self.ops
        T = input_ids.shape[1]
        cos, sin = self._rope_tables(cache.len, T)

        # Residual stream: each residual add is fused with the norm that follows it,
        # so a layer's input norm happens at the end of the previous layer.
        residual = F.embedding(input_ids, self.embed)
        h = ops.rms_norm(residual, self.layers[0].input_norm, self.eps)  # first norm has no add before it
        for i, w in enumerate(self.layers):
            attn = self._attention(h, w, cos, sin, cache, i)
            h, residual = ops.add_rms_norm(attn, residual, w.post_norm, self.eps)
            mlp = self._mlp(h, w)
            next_norm = self.layers[i + 1].input_norm if i + 1 < len(self.layers) else self.final_norm
            h, residual = ops.add_rms_norm(mlp, residual, next_norm, self.eps)
        cache.len += T

        return ops.linear(h[:, -1], self.lm_head)  # h is already final-normed
