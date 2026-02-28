"""
Two-Phase Zero-Shot Localization Experiment on Diffusion Models

This script runs the Over-confidence Trap (Phase 1) and Context Sensitivity (Phase 2)
algorithms to identify code smells vs. clean variables. 
Consistency with benchmark_diffusion_models.py is maintained (settings, args, device).
Variable names for evaluation are directly extracted using a Javalang AST parser.
"""

import os
import sys
import csv
import re
import gc
import random
import argparse
import time
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import javalang

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# --- Environment Setting: Mock torchvision ---
class MockModule:
    def __getattr__(self, name): return MockModule()
    def __call__(self, *args, **kwargs): return MockModule()

sys.modules['torchvision'] = MockModule()
sys.modules['torchvision.ops'] = MockModule()
sys.modules['torchvision.transforms'] = MockModule()
if not hasattr(torch.ops, 'torchvision'):
    class DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = DummyOps()

# ---- Configuration ---------------------------------------------------------

DATA_PATH = "data/test.csv"
RESULTS_DIR = "results/localization_experiment"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKS = 512

SMELL_TOKENS = ["data", "res", "temp", "val", "tmp", "x", "i", "foo", "myVar"]

MODEL_REGISTRY = {
    "DiffuCoder-7B":     {"id": "apple/DiffuCoder-7B-Instruct", "type": "diffucoder", "mask_token": "<|mask|>"},
    "DreamCoder-7B":     {"id": "Dream-org/Dream-Coder-v0-Instruct-7B", "type": "dreamcoder", "mask_token": "<|mask|>"},
}

# ---- AST Extraction ---------------------------------------------------------

def extract_identifiers_ast(code):
    """
    Parse the Java code using javalang AST and extract all variable names.
    Since snippets might not be valid class definitions, wrap in a dummy class.
    """
    try:
        tree = javalang.parse.parse(f"class _Dummy {{ {code} }}")
    except Exception:
        try:
             tree = javalang.parse.parse(code)
        except Exception:
             return []
             
    identifiers = set()
    if tree:
        for path, node in tree.filter(javalang.tree.VariableDeclarator):
            if hasattr(node, "name") and node.name: identifiers.add(node.name)
        for path, node in tree.filter(javalang.tree.FormalParameter):
            if hasattr(node, "name") and node.name: identifiers.add(node.name)
        for path, node in tree.filter(javalang.tree.MemberReference):
            if hasattr(node, "member") and node.member: identifiers.add(node.member)
    return list(identifiers)

# ---- Helper Functions -------------------------------------------------------

def get_logits(model, input_ids, attention_mask):
    """Perform forward pass to get unmodified logit distribution."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        
    if hasattr(out, "logits"):
        return out.logits
    # Fallback to taking the first element of output tuple
    if isinstance(out, tuple) and len(out) > 0:
        return out[0]
    return out

def compute_metrics(logits, seq_idx, target_token_id):
    """Compute exact Rank, Probability, and logit Entropy."""
    # Process [batch=1, seq_len, vocab_size] -> target position logits vector
    lp = logits[0, seq_idx, :].float()
    prob = torch.softmax(lp, dim=-1)
    lprob = torch.log(prob + 1e-12)

    entropy = -(prob * lprob).sum().item()
    target_prob = float(prob[target_token_id])

    sorted_ids = torch.argsort(prob, descending=True)
    rank_map = {int(t): r+1 for r, t in enumerate(sorted_ids)}
    target_rank = rank_map.get(int(target_token_id), len(prob))

    return {
        "entropy": entropy,
        "prob": target_prob,
        "rank": target_rank
    }

def apply_context_mask(input_ids, mask_id, target_idx, alpha, rng):
    """Mask out a fraction (alpha) of the context tokens to test Context Sensitivity."""
    ids = input_ids.clone()
    seq = ids[0].tolist()
    n = len(seq)
    # Mask everywhere EXCEPT the start token, end token, or the actual target variable
    eligible = [i for i in range(1, n-1) if i != target_idx]
    k = int(round(alpha * len(eligible)))
    for i in rng.sample(eligible, k) if k else []:
        ids[0, i] = mask_id
    return ids

def approximate_token_index(tokenizer, code, start_char_idx, has_bos):
    """Find token index for a byte-offset occurrence of a word."""
    prefix = code[:start_char_idx]
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    idx = len(prefix_ids)
    if has_bos:
        idx += 1
    return idx

# ---- Main Experiment --------------------------------------------------------

def run_experiment(target_models=None, alphas=[0.5, 0.8], max_samples=None, hf_repo=None, hf_token=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if target_models:
        models_to_run = {n: MODEL_REGISTRY[n] for n in target_models if n in MODEL_REGISTRY}
    else:
        models_to_run = MODEL_REGISTRY

    if not models_to_run:
        print("ERROR: No valid models specified.")
        return

    # 1. Load Data
    data_file = DATA_PATH if os.path.exists(DATA_PATH) else "data/test_filtered_1024.csv"
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file, header=None, names=['id', 'X', 'y'])
    if max_samples:
        df = df.head(max_samples)
    print(f"Loaded {len(df)} samples.")
    
    rng = random.Random(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Iterate Model Registry
    for model_name, meta in models_to_run.items():
        print(f"\n{'='*60}")
        print(f"  Model: {model_name} | Two-Phase Localization")
        print(f"{'='*60}")

        print(f"Loading tokenizer {meta['id']}...")
        tokenizer = AutoTokenizer.from_pretrained(meta["id"], trust_remote_code=True)
        print(f"Loading model {meta['id']}...")
        model = AutoModel.from_pretrained(
            meta["id"], 
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(DEVICE).eval()
        mask_token_id = tokenizer.convert_tokens_to_ids(meta["mask_token"])
        
        results = []

        # 3. Iterate Samples
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Testing {model_name}"):
            sample_id = row['id']
            masked_code = str(row['X'])
            gt_name = str(row['y']).strip()
            
            mask_char_pos = masked_code.find("[MASK]")
            if mask_char_pos == -1:
                continue
                
            full_code = masked_code.replace("[MASK]", gt_name, 1)

            # Step A: Perform AST extraction for target variables
            ast_vars = extract_identifiers_ast(full_code)

            # Step B: Setup Candidates dictionary
            candidates = []
            
            # Ground truth testing (using known MASK offset)
            candidates.append({"type": "ground_truth", "name": gt_name, "char_idx": mask_char_pos, "code": full_code})
            
            # Synthetic Smell injection testing (using known MASK offset)
            sampled_smells = rng.sample(SMELL_TOKENS, 1)
            for s in sampled_smells:
                s_code = masked_code.replace("[MASK]", s, 1)
                candidates.append({"type": "synthetic_smell", "name": s, "char_idx": mask_char_pos, "code": s_code})
                
            # Genuine AST variables from the raw unmasked code snippet
            for ast_v in ast_vars:
                # Find its first occurrence in the original code
                v_char_idx = full_code.find(ast_v)
                if v_char_idx != -1:
                    candidates.append({"type": "ast_extracted", "name": ast_v, "char_idx": v_char_idx, "code": full_code})

            # Step C: Evaluate each Candidate Two-Phase Logic
            for cand in candidates:
                c_code = cand["code"]
                c_name = cand["name"]
                
                # Tokenize
                enc = tokenizer(c_code, return_tensors="pt", truncation=True, max_length=MAX_TOKS)
                input_ids = enc["input_ids"].to(DEVICE)
                attention_mask = enc["attention_mask"].to(DEVICE)
                seq_len = input_ids.shape[1]
                if seq_len < 5: continue
                
                # Target indexing
                has_bos = (input_ids[0, 0].item() == tokenizer.bos_token_id) if hasattr(tokenizer, 'bos_token_id') else False
                target_idx = approximate_token_index(tokenizer, c_code, cand["char_idx"], has_bos)
                target_idx = min(target_idx, seq_len - 2)

                # Target Token Representation
                tok_ids = tokenizer.encode(c_name, add_special_tokens=False)
                # handle None unk_token_id safely
                fallback_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0
                c_tok_id = tok_ids[0] if tok_ids else fallback_id

                # PHASE 1
                try:
                    logits_0 = get_logits(model, input_ids, None)
                    metrics_0 = compute_metrics(logits_0, target_idx, c_tok_id)
                except Exception as e:
                    print(f"Skipping {c_name} due to error: {e}")
                    continue  # Logit extraction error bounds (e.g. truncated idx)

                base_ent, base_rnk, base_prb = metrics_0["entropy"], metrics_0["rank"], metrics_0["prob"]
                
                res_dict = {
                    "model": model_name,
                    "sample_id": sample_id,
                    "type": cand["type"],
                    "variable_name": c_name,
                    "phase1_base_rank": base_rnk,
                    "phase1_base_prob": base_prb,
                    "phase1_base_entropy": base_ent,
                }
                
                # PHASE 2
                for alpha in alphas:
                    masked_ids = apply_context_mask(input_ids, mask_token_id, target_idx, alpha, rng)
                    try:
                        logits_a = get_logits(model, masked_ids, None)
                        m_a = compute_metrics(logits_a, target_idx, c_tok_id)
                        res_dict[f"phase2_entropy_{alpha}"] = m_a["entropy"]
                        res_dict[f"phase2_entropy_diff_{alpha}"] = m_a["entropy"] - base_ent
                    except:
                        res_dict[f"phase2_entropy_{alpha}"] = np.nan
                        res_dict[f"phase2_entropy_diff_{alpha}"] = np.nan

                results.append(res_dict)

        # 4. Save model results
        out_file = os.path.join(RESULTS_DIR, f"{model_name}_two_phase_ast_{timestamp}.csv")
        pd.DataFrame(results).to_csv(out_file, index=False)
        print(f"Results saved to {out_file}")

        # Auto-upload HF
        if hf_repo and HAS_HF_HUB:
            print(f"Uploading {out_file} to Hugging Face...")
            api = HfApi(token=hf_token)
            api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True)
            try:
                api.upload_file(
                    path_or_fileobj=out_file, 
                    path_in_repo=f"localization_experiment/{os.path.basename(out_file)}",
                    repo_id=hf_repo, 
                    repo_type="dataset"
                )
                print("Upload Successful.")
            except Exception as e:
                print(f"HF Upload Failed: {e}")

        # Cleanup
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AST Javalang Two-Phase Zero-Shot Localization")
    parser.add_argument("--model", action="append", default=None, help="Model IDs to test")
    parser.add_argument("--alphas", type=float, nargs='+', default=[0.5, 0.8], help="Masking fractions")
    parser.add_argument("--max-samples", type=int, default=None, help="Num samples")
    parser.add_argument("--hf-repo", type=str, default=None, help="HF upload Repo")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    
    args = parser.parse_args()
    
    # ensure results dir
    if not os.path.exists("results"):
        os.makedirs("results")
        
    run_experiment(
        target_models=args.model,
        alphas=args.alphas,
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token
    )
