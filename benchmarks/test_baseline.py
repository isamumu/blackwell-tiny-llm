import os
import torch
import time
from transformers import AutoTokenizer
from src.model import BlackwellOptimizedLLM

def main():
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    
    # 1. Initialize tokenizer and custom wrapper model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = BlackwellOptimizedLLM(model_id)
    model.eval()

    # 2. Design sample conversation input
    prompt = "Explain why writing custom GPU kernels makes an AI inference engine faster."
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt", 
        return_dict=False
    ).to("cuda")

    # 3. Warm up the GPU pipeline (forces memory allocation adjustments to clear lag)
    print("Warming up GPU infrastructure...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(inputs[:, :5])

    # 4. Profile token-by-token generation speed
    print("\n--- Running Baseline Performance Test ---")
    max_new_tokens = 128
    current_tokens = inputs
    
    # Track hardware execution boundaries cleanly
    torch.cuda.synchronize()
    start_time = time.time()

    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(current_tokens)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy search sampling (always picks most likely token for consistency)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Stream the generated output word directly to screen
            word = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(word, end="", flush=True)
            
            # Chain the token to inputs for next step
            current_tokens = torch.cat([current_tokens, next_token], dim=-1)
            
            # Break if end of sequence token is generated
            if next_token.item() in tokenizer.all_special_ids:
                break

    torch.cuda.synchronize()
    end_time = time.time()

    # 5. Extract and print metrics
    tokens_generated = current_tokens.shape[1] - inputs.shape[1]
    elapsed_time = end_time - start_time
    tokens_per_second = tokens_generated / elapsed_time

    print("\n" + "="*50)
    print(f"Tokens Generated: {tokens_generated}")
    print(f"Elapsed Time:     {elapsed_time:.4f} seconds")
    print(f"Baseline Speed:   {tokens_per_second:.2f} tokens/sec")
    print("="*50)

if __name__ == "__main__":
    main()

