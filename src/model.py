# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class BlackwellOptimizedLLM(nn.Module):
    def __init__(self, model_id: str):
        super().__init__()
        print(f"Loading pretrained weights for {model_id} onto CPU first...")
        
        # FIX: Load onto CPU first, avoiding device_map="cuda" bug on Blackwell
        base_model_cpu = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map=None  # Forces default CPU mapping
        )
        
        print("Explicitly moving weights to Blackwell GPU VRAM...")
        # Forcefully transfer the entire memory graph directly to CUDA
        self.base_model = base_model_cpu.to("cuda")
        print("Model successfully loaded!")
        
    def fused_rmsnorm_placeholder(self, x, weight, eps=1e-6):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + eps) * weight

    def swiglu_activation_placeholder(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(x1) * x2

    def forward(self, input_ids, **kwargs):
        return self.base_model(input_ids, **kwargs)

