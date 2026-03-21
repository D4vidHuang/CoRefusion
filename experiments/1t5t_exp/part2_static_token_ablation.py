"""
Part 2: Static MASK-Token Count Ablation
==========================================

Research question:
  Given that the mask position is known, what is the *optimal number* of
  <|mask|> tokens to place there so that a diffusion language model best
  recovers the original variable name?

Experiment design:
  • Dataset  : data/test.csv  (RefineID, Java variable renaming)
  • Models   : DiffuCoder-7B-Instruct, DreamCoder-7B-Instruct
  • Mask counts tested: [1, 2, 3, 4, 5]
  • Diffusion steps  : 32  (fixed for all runs)
  • Evaluation metrics:
      1. Exact Match (EM)  – prediction == ground_truth (string exact)
      2. LLM-as-Judge (LJ) – Qwen2.5-7B-Instruct judge (binary: 0/1)

Implementation approach (matches benchmark_diffusion_models.py):
  - Replace [MASK] with <|mask|> * k (concatenated, no spaces)
  - Tokenise the whole code snippet directly (NO chat prompt)
  - Run model.diffusion_generate() with max_new_tokens=1 so the model
    denoises *in-place* without generating extra tokens
  - Decode the ENTIRE denoised sequence
  - Extract the filled identifier by anchoring on the surrounding context
    (same extract_all_predictions logic as benchmark_diffusion_models.py)

Usage (from repo root):
    # Full run, both models, mask counts 1-5, with LLM judge
    python experiments/1t5t_exp/part2_static_token_ablation.py \\
        --data data/test.csv \\
        --models both \\
        --mask-counts 1 2 3 4 5 \\
        --steps 32 \\
        --judge-model Qwen/Qwen2.5-7B-Instruct \\
        --max-samples 200

    # EM-only (no judge, faster)
    python experiments/1t5t_exp/part2_static_token_ablation.py --no-judge

    # Single model
    python experiments/1t5t_exp/part2_static_token_ablation.py \\
        --models diffucoder --mask-counts 1 2 3 4 5 --no-judge
"""

import os
import sys
import csv
import re
import gc
import argparse
import time
from datetime import datetime

import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm

# ── torchvision mock ──────────────────────────────────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_REGISTRY = {
    "diffucoder": {
        "name":       "DiffuCoder-7B",
        "id":         "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    "dreamcoder": {
        "name":       "DreamCoder-7B",
        "id":         "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    },
}

RESULTS_DIR = "results/1t5t_exp"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

JUDGE_REGISTRY = {
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct":  "Qwen/Qwen2.5-3B-Instruct",
}

MODEL_COLORS = {
    "DiffuCoder-7B": "#3a86ff",
    "DreamCoder-7B": "#ff6b6b",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path: str, max_samples: int | None = None) -> list[dict]:
    """Load test.csv: columns are id | masked_code | ground_truth (no header)."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({
                "id":           row[0],
                "masked_code":  row[1],
                "ground_truth": row[2].strip(),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Prediction extraction  (copied from benchmark_diffusion_models.py)
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_predictions(full_code: str, masked_code: str) -> list[str]:
    """
    Extract predictions for ALL [MASK] locations in masked_code.

    Aligns the denoised full_code with the original masked_code by anchoring
    on the context surrounding each [MASK] position.

    Returns a list of predicted identifiers (one per [MASK]).
    """
    parts = masked_code.split("[MASK]")
    if len(parts) <= 1:
        return []

    predictions = []
    current_search_start = 0

    for i in range(len(parts) - 1):
        pre  = parts[i]
        post = parts[i + 1]

        # Use short anchors for robustness
        pre_anchor  = pre.strip()[-30:]  if len(pre.strip())  > 30 else pre.strip()
        post_anchor = post.strip()[:30]  if len(post.strip()) > 30 else post.strip()

        # 1. Locate pre_anchor in full_code
        if pre_anchor:
            idx_start = full_code.find(pre_anchor, current_search_start)
            idx_start = (idx_start + len(pre_anchor)) if idx_start != -1 else current_search_start
        else:
            idx_start = current_search_start

        # 2. Locate post_anchor
        if post_anchor:
            idx_end = full_code.find(post_anchor, idx_start)
        else:
            idx_end = -1

        # 3. Extract the gap
        if idx_end != -1:
            gap_content = full_code[idx_start:idx_end].strip()
            current_search_start = idx_end
        else:
            gap_content = full_code[idx_start: idx_start + 60].strip()
            current_search_start = idx_start + 60

        # 4. Extract the first valid Java identifier
        match = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', gap_content)
        predictions.append(match.group(0) if match else gap_content[:20])

    return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Diffusion model inference
# ─────────────────────────────────────────────────────────────────────────────

def run_diffusion_inference(model, tokenizer, masked_code: str,
                            mask_token: str, k: int,
                            steps: int) -> tuple[str, str]:
    """
    Replace [MASK] with k concatenated mask tokens, run diffusion in-place,
    decode the entire denoised sequence, and return (full_code, primary_pred).

    Key design notes (matching benchmark_diffusion_models.py):
      • mask tokens are concatenated WITHOUT spaces: mask_token * k
      • max_new_tokens=1  →  no new tokens generated; model denoises existing ones
      • The WHOLE sequence is decoded (not just new tokens)
      • Prediction is extracted by anchoring on surrounding context
    """
    # Step 1: Replace placeholder
    multi_mask = mask_token * k
    input_code = masked_code.replace("[MASK]", multi_mask)

    # Step 2: Tokenise the full snippet (no chat template)
    inputs      = tokenizer(input_code, return_tensors="pt")
    input_ids   = inputs.input_ids.to(model.device)
    attn_mask   = inputs.attention_mask.to(model.device)

    # Step 3: Diffusion generate — denoise in-place
    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=1,       # no extra tokens; denoising only
            steps=steps,
            temperature=0.3,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.,
        )

    # Step 4: Decode ENTIRE denoised sequence
    gen_ids   = output.sequences[0] if hasattr(output, "sequences") else output[0]
    full_code = tokenizer.decode(gen_ids, skip_special_tokens=True)

    # Step 5: Extract predictions by anchoring on masked_code context
    preds = extract_all_predictions(full_code, masked_code)
    primary_pred = preds[0] if preds else ""

    return full_code, primary_pred


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert Java code reviewer evaluating the quality of variable names "
    "suggested by an AI model.\n\n"
    "Your task: given a code snippet and a ground-truth variable name, decide "
    "whether the predicted variable name is SEMANTICALLY ACCEPTABLE as a replacement.\n\n"
    "Rules:\n"
    "1. ACCEPTABLE if the prediction conveys the same concept as the ground truth, "
    "even if the exact string differs "
    "(e.g. 'bufSize' vs 'bufferSize' are both fine for a buffer-size variable).\n"
    "2. NOT ACCEPTABLE if the prediction clearly describes a different concept.\n"
    "3. Single-letter names are usually NOT ACCEPTABLE unless obviously correct "
    "(e.g. loop counter 'i', 'j').\n"
    "4. Names that are clearly wrong tokens ('0', 'true', 'MASK', 'EOT', etc.) are NOT ACCEPTABLE.\n"
    "5. Abbreviations that preserve the same meaning ARE ACCEPTABLE.\n\n"
    "You MUST respond with EXACTLY one line, either:\n"
    "    VERDICT: 1\n"
    "or\n"
    "    VERDICT: 0\n\n"
    "Do NOT add any other text."
)


def _apply_chat_template(tokenizer, user_text: str) -> str:
    msgs = [{"role": "system",  "content": JUDGE_SYSTEM},
            {"role": "user",    "content": user_text}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        return f"{JUDGE_SYSTEM}\n\nUser: {user_text}\nAssistant:"


def parse_verdict(text: str) -> int:
    m = re.search(r"VERDICT\s*:\s*([01])", text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    for line in text.strip().splitlines():
        line = line.strip()
        if line in ("1", "0"):
            return int(line)
        if line.lower() in ("yes", "acceptable", "correct"):
            return 1
        if line.lower() in ("no", "not acceptable", "incorrect", "wrong"):
            return 0
    return -1  # parse failure


def judge_one(judge_tok, judge_model,
              masked_code: str, prediction: str, ground_truth: str) -> int:
    """Return 1 (acceptable), 0 (not acceptable), or -1 (parse failure)."""
    # Build code context (center ≈2000 chars around the mask position)
    mask_pos = masked_code.find("[MASK]")
    code_with_pred = masked_code.replace("[MASK]", prediction, 1)
    half = 1000
    start = max(0, mask_pos - half)
    end   = min(len(code_with_pred), mask_pos + len(prediction) + half)
    context = ("..." if start > 0 else "") + code_with_pred[start:end] + \
              ("..." if end < len(code_with_pred) else "")

    user_text = (
        f"Code context (the predicted name replaces the masked identifier):\n"
        f"```java\n{context}\n```\n\n"
        f"Ground-truth variable name: `{ground_truth}`\n"
        f"Predicted variable name:    `{prediction}`\n\n"
        f"Is the predicted name semantically acceptable given the code context and "
        f"the ground truth?\n"
        f"Reply with EXACTLY one line: VERDICT: 1   or   VERDICT: 0"
    )
    prompt = _apply_chat_template(judge_tok, user_text)
    inp = judge_tok(prompt, return_tensors="pt", truncation=True,
                    max_length=4096).to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(
            **inp, max_new_tokens=16, do_sample=False,
            pad_token_id=(judge_tok.eos_token_id or judge_tok.pad_token_id or 0),
        )
    new_ids = out[0][inp["input_ids"].shape[1]:]
    raw = judge_tok.decode(new_ids, skip_special_tokens=True).strip()
    return parse_verdict(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment loop: one (model × k) combination
# ─────────────────────────────────────────────────────────────────────────────

def run_one_config(model_key: str, k: int, data: list[dict],
                   steps: int, timestamp: str,
                   judge_tok=None, judge_model=None) -> pd.DataFrame:
    """
    Run inference for one (model, k) configuration and save raw results.
    Returns a DataFrame with per-sample results.
    """
    cfg        = MODELS_REGISTRY[model_key]
    model_name = cfg["name"]
    model_id   = cfg["id"]
    mask_token = cfg["mask_token"]

    print(f"\n{'─'*65}")
    print(f"  {model_name}  |  k={k} mask token(s)  |  steps={steps}")
    print(f"  mask_token: '{mask_token}'  →  repeated {k}x = '{mask_token * k}'")
    print(f"{'─'*65}")

    # ── Load diffusion model ───────────────────────────────────────────────
    t0 = time.time()
    print(f"  Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # ── Inference loop ──────────────────────────────────────────────────────
    rows    = []
    correct = 0
    errors  = 0

    for row in tqdm(data, desc=f"  {model_name} k={k}"):
        item_id      = row["id"]
        masked_code  = row["masked_code"]
        ground_truth = row["ground_truth"]

        try:
            full_code, prediction = run_diffusion_inference(
                model, tokenizer, masked_code, mask_token, k, steps
            )
            exact_match = int(prediction == ground_truth)
            if exact_match:
                correct += 1

            # ── LLM judge (skip if exact match to save compute) ────────────
            if judge_tok is not None and not exact_match:
                # Skip clearly invalid predictions before calling LLM
                if prediction and re.search(r"[a-zA-Z]", prediction):
                    verdict = judge_one(judge_tok, judge_model,
                                        masked_code, prediction, ground_truth)
                else:
                    verdict = 0
            elif exact_match:
                verdict = 1   # exact match → automatically acceptable
            else:
                verdict = -2  # judge disabled

            rows.append({
                "id":           item_id,
                "model":        model_name,
                "mask_count":   k,
                "steps":        steps,
                "ground_truth": ground_truth,
                "prediction":   prediction,
                "exact_match":  exact_match,
                "llm_verdict":  verdict,
            })

        except Exception as e:
            errors += 1
            rows.append({
                "id":           item_id,
                "model":        model_name,
                "mask_count":   k,
                "steps":        steps,
                "ground_truth": ground_truth,
                "prediction":   "",
                "exact_match":  0,
                "llm_verdict":  -1,
                "error":        str(e),
            })
            if errors <= 5:
                print(f"    Error on sample {item_id}: {e}")
            elif errors == 6:
                print("    ... suppressing further error messages")

    # ── Save raw results ────────────────────────────────────────────────────
    raw_df    = pd.DataFrame(rows)
    safe_name = model_name.replace("-", "_")
    raw_path  = os.path.join(RESULTS_DIR,
                             f"part2_raw_{safe_name}_{k}tok_{timestamp}.csv")
    raw_df.to_csv(raw_path, index=False)

    em_mean = raw_df["exact_match"].mean()
    # LLM judge mean excludes "disabled" (-2) and parse failures (-1)
    valid_lj = raw_df[raw_df["llm_verdict"] >= 0]["llm_verdict"]
    lj_mean  = valid_lj.mean() if len(valid_lj) > 0 else float("nan")

    print(f"\n  EM={em_mean:.4f}  ({correct}/{len(data)} correct)")
    if not np.isnan(lj_mean):
        print(f"  LJ={lj_mean:.4f}  ({valid_lj.sum()}/{len(valid_lj)} acceptable)")
    print(f"  Errors: {errors}")
    print(f"  → Saved: {raw_path}")

    # ── Cleanup diffusion model ─────────────────────────────────────────────
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return raw_df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(summary_df: pd.DataFrame, timestamp: str):
    """Bar charts: EM and LLM-Judge accuracy vs number of mask tokens."""
    models    = summary_df["model"].unique()
    k_vals    = sorted(summary_df["mask_count"].unique())
    x         = np.arange(len(k_vals))
    width     = 0.35
    n_models  = len(models)
    offsets   = np.linspace(-width * (n_models - 1) / 2,
                             width * (n_models - 1) / 2, n_models)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, title in [
        ("exact_match", axes[0], "Exact Match (EM)"),
        ("llm_judge",   axes[1], "LLM-as-Judge Acceptance Rate"),
    ]:
        for model_name, offset in zip(models, offsets):
            sub  = summary_df[summary_df["model"] == model_name].set_index("mask_count")
            vals = []
            for k in k_vals:
                v = sub.loc[k, metric] if k in sub.index else 0.
                vals.append(float(v) if not pd.isna(v) else 0.)

            bars = ax.bar(x + offset, vals, width,
                          label=model_name,
                          color=MODEL_COLORS.get(model_name, "grey"),
                          alpha=0.85)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.002,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Number of <|mask|> tokens per variable (k)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in k_vals])
        max_val = summary_df[metric].dropna().max()
        ax.set_ylim(0, min(1.05, max_val * 1.25 + 0.05) if max_val > 0 else 0.2)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle(
        "Effect of Static MASK-Token Count on Variable Renaming Quality\n"
        f"(RefineID · {summary_df['steps'].iloc[0]} diffusion steps · "
        f"{summary_df['n_samples'].iloc[0]} samples)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part2_ablation_{timestamp}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"\n  → Figure saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Part 2 – Static MASK-token count ablation (1-5 tokens).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data",        default="data/test.csv",
                        help="Path to RefineID test.csv (default: data/test.csv)")
    parser.add_argument("--models",      default="both",
                        choices=["diffucoder", "dreamcoder", "both"])
    parser.add_argument("--mask-counts", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5],
                        help="List of k values to test (default: 1 2 3 4 5)")
    parser.add_argument("--steps",       type=int, default=32,
                        help="Number of diffusion steps (default: 32)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Evaluate only first N samples (for quick tests)")
    parser.add_argument("--judge-model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="HF model ID for LLM judge")
    parser.add_argument("--no-judge",    action="store_true",
                        help="Skip LLM judge – evaluate Exact Match only")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  Part 2 – Static MASK-Token Count Ablation")
    print("=" * 65)
    print(f"  Data            : {args.data}")
    print(f"  Diffusion steps : {args.steps}")
    print(f"  Mask counts     : {args.mask_counts}")
    print(f"  Models          : {args.models}")
    print(f"  LLM judge       : {'DISABLED' if args.no_judge else args.judge_model}")
    print(f"  Device          : {DEVICE}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data from {args.data} …")
    data = load_data(args.data, args.max_samples)
    print(f"  {len(data):,} samples loaded.")

    # ── Optional: load LLM judge once (shared across all (model, k) runs) ─
    judge_tok, judge_model = None, None
    if not args.no_judge:
        judge_id = JUDGE_REGISTRY.get(args.judge_model, args.judge_model)
        print(f"\nLoading judge model: {judge_id} …")
        judge_tok = AutoTokenizer.from_pretrained(judge_id, trust_remote_code=True)
        judge_model = AutoModelForCausalLM.from_pretrained(
            judge_id,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            trust_remote_code=True,
        ).eval()
        print("  Judge model loaded.")

    # ── Determine which diffusion models to run ────────────────────────────
    if args.models == "both":
        model_keys = ["diffucoder", "dreamcoder"]
    else:
        model_keys = [args.models]

    all_raw_dfs = []

    for model_key in model_keys:
        for k in args.mask_counts:
            raw_df = run_one_config(
                model_key=model_key,
                k=k,
                data=data,
                steps=args.steps,
                timestamp=timestamp,
                judge_tok=judge_tok,
                judge_model=judge_model,
            )
            all_raw_dfs.append(raw_df)

    # ── Aggregate summary ──────────────────────────────────────────────────
    combined = pd.concat(all_raw_dfs, ignore_index=True)
    summary_rows = []
    for (model_name, k), grp in combined.groupby(["model", "mask_count"]):
        valid_lj = grp[grp["llm_verdict"] >= 0]["llm_verdict"]
        summary_rows.append({
            "model":        model_name,
            "mask_count":   k,
            "steps":        args.steps,
            "n_samples":    len(grp),
            "exact_match":  round(grp["exact_match"].mean(), 4),
            "llm_judge":    round(valid_lj.mean(), 4) if len(valid_lj) > 0 else None,
        })

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, f"part2_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("  PART 2 SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\n  Full summary saved → {summary_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    if len(summary_df) > 0:
        plot_results(summary_df, timestamp)

    # ── Cleanup judge ──────────────────────────────────────────────────────
    if judge_model is not None:
        del judge_model, judge_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── Print best k per model ────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  OPTIMAL MASK COUNT (k*)")
    print("=" * 65)
    for model_name in summary_df["model"].unique():
        sub = summary_df[summary_df["model"] == model_name]
        best_em  = sub.loc[sub["exact_match"].idxmax()]
        print(f"\n  {model_name}:")
        print(f"    Best EM  → k={int(best_em['mask_count'])}  "
              f"EM={best_em['exact_match']:.4f}")
        lj_col = sub["llm_judge"].dropna()
        if len(lj_col) > 0:
            best_lj = sub.loc[lj_col.idxmax()]
            print(f"    Best LLM → k={int(best_lj['mask_count'])}  "
                  f"LJ={best_lj['llm_judge']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
