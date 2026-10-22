import torch
import sys

print("--- Step 1: Small Allocation ---")
try:
    x = torch.randn(1000, 1000, device="cuda", dtype=torch.float16)
    print(f"Successfully allocated a small tensor! Shape: {x.shape}, Memory: {x.element_size() * x.nelement() / 1024:.2f} KB")
except Exception as e:
    print("Small allocation failed:", e)
    sys.exit(1)

print("\n--- Step 2: Model-Sized Allocation (2GB) ---")
try:
    print("Attempting to allocate a raw 2GB tensor directly on your RTX 5060...")
    # 1 Billion elements of float16 = exactly 2GB
    y = torch.empty(1_000_000_000, device="cuda", dtype=torch.float16)
    print("Successfully allocated 2GB of continuous VRAM!")
except Exception as e:
    print("Large allocation failed:", e)
