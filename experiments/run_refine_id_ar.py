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

# Model Registry with IDs and types
MODEL_METADATA = {
    "StarCoder2-7B": {
        "id": "bigcode/starcoder2-7b",
        "type": "starcoder",
    },
    "DeepSeek-Coder-6.7B-Base": {
        "id": "deepseek-ai/deepseek-coder-6.7b-base",
        "type": "deepseek",
    }
}

def clean_prediction(text, model_type):
    """Extracts a clean identifier from model output."""
    # Remove special tokens that might be in the decoded text
    if model_type == "starcoder":
        for token in ["<fim_prefix>", "<fim_suffix>", "<fim_middle>", "<|endoftext|>", "<file_sep>"]:
            text = text.replace(token, "")
    elif model_type == "deepseek":
        for token in ["<｜fim begin｜>", "<｜fim hole｜>", "<｜fim end｜>", "<｜end of sentence｜>"]:
            text = text.replace(token, "")
            
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
                current_code = masked_code
                predictions = []
                
                # Iterate until all [MASK] tokens are filled
                mask_count = current_code.count("[MASK]")
                
                for i in range(mask_count):
                    # Split at the first [MASK]
                    # Note: we must find the position of the FIRST [MASK]
                    parts = current_code.split("[MASK]", 1)
                    prefix = parts[0]
                    suffix = parts[1] if len(parts) > 1 else ""
                    
                    # Prepare FIM Prompt
                    if meta['type'] == "starcoder":
                        prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
                    elif meta['type'] == "deepseek":
                        # DeepSeek Coder FIM format
                        prompt = f"<｜fim begin｜>{prefix}<｜fim hole｜>{suffix}<｜fim end｜>"
                    else:
                        prompt = f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"
                    
                    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=20,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                        )
                    
                    # Decode only the generated part
                    raw_pred = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    prediction = clean_prediction(raw_pred, meta['type'])
                    predictions.append(prediction)
                    
                    # Replace the first [MASK] with the prediction for the next iteration
                    current_code = prefix + prediction + suffix

                # For evaluation, we typically compare with the ground truth
                # If there are multiple predictions, we store them all but compare the main one or just check if any are wrong.
                # Usually in this dataset, all [MASK] in one row refer to the same identifier.
                # So we take the first prediction for accuracy calculation.
                primary_prediction = predictions[0] if predictions else ""
                
                results.append({
                    "id": item_id,
                    "ground_truth": ground_truth,
                    "prediction": primary_prediction,
                    "all_predictions": "|".join(predictions),
                    "full_code": current_code,
                    "correct": (primary_prediction == ground_truth)
                })

            except Exception as e:
                print(f"Error on sample {item_id}: {e}")
                results.append({"id": item_id, "error": str(e)})

        # Save results for this model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(RESULTS_DIR, f"{model_name}_refineID_results_{timestamp}.csv")
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
