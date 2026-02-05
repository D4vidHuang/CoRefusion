import os
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from datetime import datetime
import re

# Configuration
DATA_PATH = "data/test_filtered_1024.csv"
RESULTS_DIR = "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model Registry with IDs and descriptions
MODEL_METADATA = {
    "Qwen2.5-Coder-7B-Instruct": {
        "id": "Qwen/Qwen2.5-Coder-7B-Instruct",
    },
    "DeepSeek-Coder-6.7B-Instruct": {
        "id": "deepseek-ai/deepseek-coder-6.7b-instruct",
    },
    "Llama-3.1-8B-Instruct": {
        "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    },
    "StarCoder2-7B": {
        "id": "bigcode/starcoder2-7b",
    }
}

def clean_prediction(text):
    """Extracts a clean identifier from model output."""
    # Remove whitespace and newlines
    text = text.strip().split('\n')[0].strip('`"\' ')
    # Match first valid Java identifier found
    match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
    if match:
        return match.group(0)
    return text

def run_experiment():
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print(f"Loading data from {DATA_PATH}...")
    try:
        # Assuming CSV format: id, masked_code, target
        df = pd.read_csv(DATA_PATH, header=None, names=['id', 'masked_code', 'target'])
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    for model_name, meta in MODEL_METADATA.items():
        print(f"\n{'='*50}")
        print(f"Running Experiment for: {model_name}")
        print(f"Model ID: {meta['id']}")
        print(f"Description: {meta['description']}")
        print(f"{'='*50}")

        try:
            # Load Model and Tokenizer
            tokenizer = AutoTokenizer.from_pretrained(meta['id'], trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                meta['id'], 
                torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
                device_map="auto" if DEVICE == "cuda" else None,
                trust_remote_code=True
            )
            if DEVICE == "cpu":
                model = model.to("cpu")
            model.eval()
        except Exception as e:
            print(f"Failed to load model {model_name}: {e}")
            continue

        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {model_name}"):
            item_id = row['id']
            masked_code = str(row['masked_code'])
            ground_truth = str(row['target']).strip()

            try:
                prediction = ""
                
                if "StarCoder2" in model_name:
                    # FIM format for StarCoder2
                    # Find first mask position
                    parts = masked_code.split("[MASK]", 1)
                    prefix = parts[0]
                    suffix = parts[1] if len(parts) > 1 else ""
                    
                    # FIM Prompt
                    prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id
                        )
                    
                    raw_pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    prediction = clean_prediction(raw_pred)
                
                else:
                    # Instruct format for Qwen, DeepSeek, Llama
                    prompt = (
                        "The following Java code has one or more identifier names replaced by [MASK]. "
                        "Based on the context, what are the original names of these identifiers? "
                        "Provide ONLY the identifier names as your response.\n\n"
                        f"Code:\n{masked_code}\n\n"
                        "Identifier:"
                    )
                    
                    # Apply chat template if available, otherwise use raw prompt
                    if hasattr(tokenizer, "apply_chat_template"):
                        messages = [{"role": "user", "content": prompt}]
                        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        formatted_prompt = prompt

                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else 0
                        )
                    
                    raw_pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    prediction = clean_prediction(raw_pred)

                # Reconstruct full code with prediction
                full_code = masked_code.replace("[MASK]", prediction)
                
                results.append({
                    "id": item_id,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "full_code": full_code,
                    "correct": (prediction == ground_truth)
                })

            except Exception as e:
                print(f"Error on sample {item_id}: {e}")
                results.append({"id": item_id, "error": str(e)})

        # Save results for this model
        output_file = os.path.join(RESULTS_DIR, f"{model_name}_refineID_results.csv")
        pd.DataFrame(results).to_csv(output_file, index=False)
        
        accuracy = sum(1 for r in results if r.get('correct', False)) / len(results) if results else 0
        print(f"Results for {model_name} saved to {output_file}")
        print(f"Accuracy: {accuracy:.2%}")

        # Cleanup memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

if __name__ == "__main__":
    run_experiment()
