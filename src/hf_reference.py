# src/hf_reference.py
#
# The trusted reference: stock HuggingFace Llama. Used as ground truth for
# correctness and as the source of weights for BlackwellOptimizedLLM.
import torch
from transformers import AutoModelForCausalLM


def load_hf_model(model_id: str):
    print(f"Loading pretrained weights for {model_id} onto CPU first...")
    # FIX: Load onto CPU first, avoiding device_map="cuda" bug on Blackwell
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map=None)

    print("Explicitly moving weights to Blackwell GPU VRAM...")
    model = model.to("cuda").eval()
    print("Model successfully loaded!")
    return model
