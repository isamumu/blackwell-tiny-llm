import torch
from transformers import AutoTokenizer
from model import BlackwellOptimizedLLM

def generate_text(prompt: str, max_new_tokens: int = 64):
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = BlackwellOptimizedLLM(model_id)
    model.eval()

    messages = [{"role": "user", "content": prompt}]
    tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")

    print(f"\nPrompt: {prompt}")
    print("Response: ", end="", flush=True)

    # Clean autoregressive generation loop
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(tokens['input_ids'])
            next_token_logits = outputs.logits[:, -1, :]
            
            # Greedy decoding (picking the most likely token)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Print the token immediately as it is generated
            word = tokenizer.decode(next_token[0], skip_special_tokens=True)
            print(word, end="", flush=True)
            
            # Append token to input for the next cycle
            tokens['input_ids'] = torch.cat([tokens['input_ids'], next_token], dim=-1)
            
            # Stop if the model emits an End-of-Sequence token
            if next_token.item() in tokenizer.all_special_ids:
                break
    print("\n")

if __name__ == "__main__":
    generate_text("Explain GPU shared memory in one concise sentence.")

