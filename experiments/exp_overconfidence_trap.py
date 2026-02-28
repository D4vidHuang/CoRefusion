"""
Experiment 1: Over-confidence Trap (Phase 1)
============================================

HYPOTHESIS
----------
Code Smell identifiers (temp, data, res, …) are high-frequency, generic tokens
widely seen during pretraining. When the model sees a masked position in an
otherwise intact code context, it assigns a disproportionately high probability
to these generic tokens, regardless of whether the surrounding context "needs"
a generic name. Clean, semantically specific names (userAuthToken, encodedURLStr)
are inherently sparser in the training corpus, so the model is much less certain
that the right token is applicable at that position.

KEY DESIGN DECISION
-------------------
Unlike a naive approach that swaps a specific name with a smell name in the same
slot (which confounds context mismatch), this experiment:
  1. Takes the ORIGINAL clean code with the correct identifier already in place.
  2. Masks EXACTLY that one token position with <|mask|>.
  3. Asks the model: "What goes here?"
  4. Records Rank and Probability of the ORIGINAL token at that position.

Then it does the same for tokens manually labelled as "SMELL" vs "SPECIFIC"
(length-based and keyword-based heuristic classifier).

This way context is ALWAYS the natural, correct context for the token being judged.

CLASSIFICATION HEURISTIC
-------------------------
Rather than injecting artificial names, we leverage the benchmark's ground-truth
column (y) to classify the actual developer-written target name:
  - SMELL   : name is in the predefined smell vocabulary (single-char, generic short names)
  - SPECIFIC : name is >= 12 chars (long compound names are inherently more domain-specific)
  - AVERAGE  : 2–11 chars and not in smell vocabulary (middle ground, also included)

METRICS
-------
  base_rank  : rank of the original token in the model's softmax distribution
               (lower = more confident the model is about this token)
  base_prob  : raw probability assigned to the original token
  base_entropy: Shannon entropy of the full softmax at the target position
               (lower = model is more confident overall → over-confidence trap)

OUTPUT
------
  results/overconfidence_trap/overconfidence_{model}_{timestamp}.csv
  results/overconfidence_trap/summary_{timestamp}.csv

Usage:
    python experiments/exp_overconfidence_trap.py
    python experiments/exp_overconfidence_trap.py --model DiffuCoder-7B --max-samples 500
    python experiments/exp_overconfidence_trap.py --list-models
"""

import os
import sys
import csv
import re
import gc
import argparse
import time
import random
import numpy as np
from datetime import datetime

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

# ── torchvision mock (identical pattern to benchmark_diffusion_models.py) ─────
if "torchvision" in sys.modules and (
        getattr(sys.modules["torchvision"], "__spec__", None) is None):
    del sys.modules["torchvision"]

# ── Configuration ─────────────────────────────────────────────────────────────

ROOT_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH   = os.path.join(ROOT_DIR, "data", "test.csv")
RESULTS_DIR = os.path.join(ROOT_DIR, "results", "overconfidence_trap")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_TOKS    = 512

# Code-smell vocabulary — single-char loops, generic short names
SMELL_VOCAB = {
    "x", "y", "z", "a", "b", "c", "n", "k", "v",
    "i", "j", "l", "m",
    "tmp", "temp", "res", "val", "ret", "obj",
    "data", "buf", "str", "idx", "num", "cnt",
    "foo", "bar", "baz", "result", "value", "item",
    "myVar", "myval", "flag",
}

# ── Model Registry ─────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "DiffuCoder-7B-Instruct": {
        "id":         "apple/DiffuCoder-7B-Instruct",
        "type":       "diffucoder",
        "mask_token": "<|mask|>",
    },
    "DiffuCoder-7B-Base": {
        "id":         "apple/DiffuCoder-7B-Base",
        "type":       "diffucoder",
        "mask_token": "<|mask|>",
    },
    "DreamCoder-7B": {
        "id":         "Dream-org/Dream-Coder-v0-Instruct-7B",
        "type":       "dreamcoder",
        "mask_token": "<|mask|>",
    },
}

# ── Identifier classification ──────────────────────────────────────────────────

def classify_identifier(name: str) -> str:
    """
    Classify a developer-written identifier into one of three buckets.

    SMELL   — in the predefined smell vocabulary (generic / over-used names)
    SPECIFIC— long compound name (>=12 chars), inherently domain-specific
    AVERAGE — everything in between (included for completeness)
    """
    if name.lower() in SMELL_VOCAB or name in SMELL_VOCAB:
        return "SMELL"
    if len(name) >= 12:
        return "SPECIFIC"
    return "AVERAGE"

# ── Model utilities ────────────────────────────────────────────────────────────

def load_model(model_id: str, mask_token: str):
    """Load tokenizer and model, return (tokenizer, model, mask_token_id)."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
    ).to(DEVICE).eval()
    mask_id = tokenizer.convert_tokens_to_ids(mask_token)
    return tokenizer, model, mask_id


def single_forward(model, input_ids):
    """Forward pass — returns logits [1, seq, vocab]."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=None)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


def metrics_at_position(logits, position: int, target_token_id: int) -> dict:
    """Extract Rank, Probability and Entropy for target_token_id at `position`."""
    lp      = logits[0, position, :].float()
    probs   = torch.softmax(lp, dim=-1)
    lprobs  = torch.log(probs + 1e-12)
    entropy = -(probs * lprobs).sum().item()

    sorted_ids = torch.argsort(probs, descending=True)
    rank_map   = {int(t): r + 1 for r, t in enumerate(sorted_ids)}
    target_rank = rank_map.get(int(target_token_id), len(rank_map))
    target_prob = float(probs[target_token_id])

    return {
        "base_entropy": entropy,
        "base_rank":    target_rank,
        "base_prob":    target_prob,
    }

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(data_path: str, max_samples: int = None) -> list:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({
                "id":          row[0],
                "masked_code": row[1],
                "target":      row[2].strip(),
            })
    return rows

# ── HF upload helper (identical to benchmark_diffusion_models.py) ──────────────

def upload_to_hf(file_path, repo_id, token):
    if not HAS_HF_HUB or not repo_id:
        return
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=f"overconfidence_trap/{os.path.basename(file_path)}",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"    Uploaded → {repo_id}")
    except Exception as e:
        print(f"    Upload failed: {e}")

# ── Main experiment ────────────────────────────────────────────────────────────

def run_experiment(target_models=None, max_samples=None,
                   hf_repo=None, hf_token=None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    models_to_run = (
        {n: MODEL_REGISTRY[n] for n in target_models if n in MODEL_REGISTRY}
        if target_models else MODEL_REGISTRY
    )
    if not models_to_run:
        print("ERROR: No valid models found.")
        return

    print(f"Loading dataset from {DATA_PATH} …")
    data = load_data(DATA_PATH, max_samples)
    print(f"  {len(data)} samples loaded.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_summaries = []

    for model_name, meta in models_to_run.items():
        print(f"\n{'='*62}")
        print(f"  Experiment: Over-confidence Trap")
        print(f"  Model:  {model_name}  ({meta['id']})")
        print(f"{'='*62}")

        # ── load model
        t0 = time.time()
        try:
            tokenizer, model, mask_id = load_model(meta["id"], meta["mask_token"])
            print(f"  Loaded in {time.time()-t0:.1f}s  |  mask_id={mask_id}  |  device={DEVICE}")
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        results = []

        for row in tqdm(data, desc=f"  {model_name}"):
            masked_code = row["masked_code"]
            gt_name     = row["target"]
            category    = classify_identifier(gt_name)

            # Ground-truth character offset of the identifier
            mask_char = masked_code.find("[MASK]")
            if mask_char == -1:
                continue

            # Reconstruct the full original code (gt name in place of [MASK])
            original_code = masked_code.replace("[MASK]", gt_name, 1)

            # ── Tokenise full code ───────────────────────────────────────────
            enc = tokenizer(
                original_code,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_TOKS,
            )
            input_ids = enc["input_ids"].to(DEVICE)
            seq_len   = input_ids.shape[1]

            # ── Locate the target token position ────────────────────────────
            # Approximate via prefix-length heuristic (same as partial_masking_noise_map.py)
            prefix     = original_code[:mask_char]
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            target_idx = min(len(prefix_ids), seq_len - 2)

            # ── Place <|mask|> ONLY at the target position ─────────────────
            masked_input = input_ids.clone()
            masked_input[0, target_idx] = mask_id

            # ── Target token id ──────────────────────────────────────────────
            gt_tok_ids  = tokenizer.encode(gt_name, add_special_tokens=False)
            gt_token_id = gt_tok_ids[0] if gt_tok_ids else (tokenizer.unk_token_id or 0)

            # ── Forward pass ─────────────────────────────────────────────────
            try:
                logits  = single_forward(model, masked_input)
                metrics = metrics_at_position(logits, target_idx, gt_token_id)
            except Exception as e:
                continue

            results.append({
                "id":            row["id"],
                "gt_name":       gt_name,
                "category":      category,
                "target_idx":    target_idx,
                **metrics,
            })

        # ── Save per-model CSV ───────────────────────────────────────────────
        out_file = os.path.join(
            RESULTS_DIR, f"overconfidence_{model_name}_{timestamp}.csv"
        )
        fieldnames = ["id", "gt_name", "category", "target_idx",
                      "base_entropy", "base_rank", "base_prob"]
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        # ── Per-model summary ────────────────────────────────────────────────
        df = pd.DataFrame(results)
        print(f"\n  --- Over-confidence Trap Summary  ({model_name}) ---")
        print(f"  {'Category':<12} {'N':>6} {'Mean Rank':>10} {'Mean Prob':>10} {'Mean Entropy':>14}")
        print(f"  {'-'*56}")
        for cat, grp in df.groupby("category"):
            print(f"  {cat:<12} {len(grp):>6} "
                  f"{grp['base_rank'].mean():>10.1f} "
                  f"{grp['base_prob'].mean():>10.6f} "
                  f"{grp['base_entropy'].mean():>14.6f}")

        print(f"\n  Results → {out_file}")
        all_summaries.append({
            "model": model_name,
            "n_smell":    len(df[df.category == "SMELL"]),
            "n_specific": len(df[df.category == "SPECIFIC"]),
            "n_average":  len(df[df.category == "AVERAGE"]),
            "mean_rank_smell":    df.loc[df.category=="SMELL",    "base_rank"].mean(),
            "mean_rank_specific": df.loc[df.category=="SPECIFIC", "base_rank"].mean(),
            "mean_prob_smell":    df.loc[df.category=="SMELL",    "base_prob"].mean(),
            "mean_prob_specific": df.loc[df.category=="SPECIFIC", "base_prob"].mean(),
            "mean_ent_smell":     df.loc[df.category=="SMELL",    "base_entropy"].mean(),
            "mean_ent_specific":  df.loc[df.category=="SPECIFIC", "base_entropy"].mean(),
        })

        if hf_repo:
            upload_to_hf(out_file, hf_repo, hf_token)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Global summary ────────────────────────────────────────────────────────
    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(summary_file, index=False)
        print(f"\n  Global summary → {summary_file}")

        print(f"\n{'='*62}")
        print("  OVER-CONFIDENCE TRAP: FINAL SUMMARY")
        print(f"  {'Model':<30} {'Smell Rank':>12} {'Specific Rank':>14}")
        print(f"  {'-'*60}")
        for s in all_summaries:
            print(f"  {s['model']:<30} "
                  f"{s['mean_rank_smell']:>12.1f} "
                  f"{s['mean_rank_specific']:>14.1f}")

        if hf_repo:
            upload_to_hf(summary_file, hf_repo, hf_token)

# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1 — Over-confidence Trap for Code Smell Detection"
    )
    parser.add_argument(
        "--model", action="append", default=None,
        help="Model key(s) from MODEL_REGISTRY. Can repeat. Default: all models."
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples from the dataset (None = full test set)."
    )
    parser.add_argument(
        "--hf-repo", type=str, default=None,
        help="HuggingFace dataset repo ID to upload results."
    )
    parser.add_argument(
        "--hf-token", type=str, default=os.environ.get("HF_TOKEN"),
        help="HF write token (or set HF_TOKEN env var)."
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="Print available models and exit."
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for k, v in MODEL_REGISTRY.items():
            print(f"  {k:<35} {v['id']}")
        sys.exit(0)

    run_experiment(
        target_models=args.model,
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
    )
