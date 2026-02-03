
import pandas as pd
import os
import torch
import time
from datetime import datetime
from tqdm import tqdm
import csv
import re

import sys
import os

# Add the project root to sys.path so we can import unified_framework
# (project_root is the parent of the current 'experiments' directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import model registry from our framework
from unified_framework import MODEL_REGISTRY

def clean_identifier(text, original_x, ground_truth):
    """
    Cleans the model output to extract the predicted identifier.
    For Diffusion models, we need to find what changed in the [MASK] position.
    For AR models, we usually just take the first word.
    """
    text = text.strip().split('\n')[0].strip('`"\' ')
    # Match only valid java identifiers if possible
    match = re.search(r'[a-zA-Z_][a-zA-Z0-9_]*', text)
    if match:
        return match.group(0)
    return text

def run_comparison_experiment(sample_size=100):
    # 1. Setup
    # Get the project root directory (two levels up from this script in 'experiments/')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, 'data', 'test.csv')
    output_dir = os.path.join(project_root, 'results', 'comparison')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"Reading data from {input_file}...")
    try:
        # Columns: id, X, y
        df = pd.read_csv(input_file, header=None, names=['id', 'X', 'y'])
        # Filter out very long samples if needed, or just sample
        df = df.sample(min(sample_size, len(df)), random_state=42)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Define model pairs for comparison
    model_pairs = [
        ("7B", "diffucoder", "qwen"),
        ("8B", "llada", "llama")
    ]

    all_stats = []

    for size_cat, diff_key, ar_key in model_pairs:
        print(f"\n{'='*40}")
        print(f" COMPARING {size_cat} MODELS: {diff_key.upper()} vs {ar_key.upper()} ")
        print(f"{'='*40}")

        for model_key in [diff_key, ar_key]:
            if model_key not in MODEL_REGISTRY:
                continue
            
            print(f"\n--- Loading {model_key} ---")
            config = MODEL_REGISTRY[model_key]
            try:
                model_instance = config["class"](config["id"])
                model_instance.load()
            except Exception as e:
                print(f"Failed to load {model_key}: {e}")
                continue

            results = []
            correct_count = 0
            total_time = 0

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Evaluating {model_key}"):
                item_id = row['id']
                x_text = str(row['X'])
                y_ground_truth = str(row['y']).strip()

                start_time = time.time()
                try:
                    prediction = ""
                    is_diffusion = model_key in ['diffucoder', 'llada', 'dreamcoder']
                    
                    if is_diffusion:
                        # Use the new is_infill=True logic
                        full_denoised = model_instance.generate(x_text, is_infill=True)
                        # Extract the filled part (this is a heuristic)
                        # For simplicity in this experiment, we look for the identifier that 
                        # replaced the [MASK] if we can, or just use the cleaned full output.
                        # Since we want EM on the identifier:
                        prediction = clean_identifier(full_denoised, x_text, y_ground_truth)
                    else:
                        # AR Models
                        prompt = f"In the following Java code, we replaced one identifier with [MASK]. Please provide ONLY the original identifier name.\n\nCode:\n{x_text}\n\nIdentifier:"
                        raw_pred = model_instance.generate(prompt, max_new_tokens=20)
                        prediction = clean_identifier(raw_pred, x_text, y_ground_truth)

                    elapsed = time.time() - start_time
                    total_time += elapsed
                    
                    # Exact Match (Case Sensitive usually for Java)
                    is_correct = (prediction == y_ground_truth)
                    if is_correct:
                        correct_count += 1

                    results.append({
                        'id': item_id,
                        'ground_truth': y_ground_truth,
                        'prediction': prediction,
                        'correct': is_correct,
                        'latency': elapsed,
                        'full_output_sample': prediction[:50]
                    })

                except Exception as e:
                    results.append({'id': item_id, 'correct': False, 'error': str(e)})

            # Calculate Accuracy
            accuracy = correct_count / len(df)
            avg_latency = total_time / len(df)
            
            print(f"Model: {model_key} | Accuracy: {accuracy:.2%} | Avg Latency: {avg_latency:.4f}s")

            # Save results
            timestamp = datetime.now().strftime("%m%d_%H%M")
            res_file = os.path.join(output_dir, f"{model_key}_{timestamp}.csv")
            pd.DataFrame(results).to_csv(res_file, index=False)
            
            all_stats.append({
                'Size': size_cat,
                'Model': model_key,
                'Accuracy': accuracy,
                'Latency': avg_latency,
                'Type': 'Diffusion' if is_diffusion else 'AR'
            })

            # Cleanup
            del model_instance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

    # Final Summary Report
    summary_df = pd.DataFrame(all_stats)
    summary_file = os.path.join(output_dir, f"summary_{datetime.now().strftime('%m%d_%H%M')}.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print("\n" + "="*50)
    print(" FINAL COMPARISON SUMMARY ")
    print("="*50)
    print(summary_df.to_markdown(index=False))
    print(f"\nDetailed summary saved to: {summary_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to test")
    args = parser.parse_args()
    
    run_comparison_experiment(sample_size=args.samples)
