import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import BlackwellOptimizedLLM

def generate_text(prompt: str, max_new_tokens: int = 64, backend: str = "torch"):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = BlackwellOptimizedLLM.from_pretrained(model_id, backend=backend)
    cache = model.new_cache()

    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=False).to("cuda")

    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)

    # Autoregressive generation: the prompt is processed once (prefill), then only
    # the newest token is fed each step since earlier keys/values live in the cache
    with torch.no_grad():
        for _ in range(max_new_tokens):
            next_token_logits = model.forward(tokens, cache)

            # Greedy decoding (picking the most likely token)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Print the token immediately as it is generated
            word = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(word, end="", flush=True)

            # Stop if the model emits an End-of-Sequence token
            if next_token.item() in tokenizer.all_special_ids:
                break
            tokens = next_token
    print("\n")

if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "torch"
    generate_text("Explain GPU shared memory in one concise sentence.", backend=backend)
