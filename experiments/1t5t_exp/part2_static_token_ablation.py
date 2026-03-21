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
  • Mask counts tested: [1, 2, 3, 4, 5]        (the "1t–5t" ablation)
  • Diffusion steps  : 32  (fixed for all runs)
  • Evaluation metrics:
      1. Exact Match    (EM)  – prediction == ground_truth (string exact)
      2. LLM-as-Judge  (LJ)  – Qwen2.5-7B-Instruct judge (binary: 0/1)

Output files saved to  results/1t5t_exp/:
  • part2_raw_{model}_{k}tok_{timestamp}.csv   – per-sample predictions & scores
  • part2_summary_{timestamp}.csv              – aggregated EM and LJ per (model, k)
  • figures/part2_*                            – bar charts

Usage (from repo root):
    # Full run, both models, mask counts 1-5, LLM judge enabled
    python experiments/1t5t_exp/part2_static_token_ablation.py \
        --data data/test.csv \
        --models both \
        --mask-counts 1 2 3 4 5 \
        --steps 32 \
        --judge-model Qwen/Qwen2.5-7B-Instruct \
        --max-samples 200

    # Skip LLM judge (faster, EM only)
    python experiments/1t5t_exp/part2_static_token_ablation.py --no-judge

    # Run only DiffuCoder with 1 or 3 masks
    python experiments/1t5t_exp/part2_static_token_ablation.py \
        --models diffucoder --mask-counts 1 3
"""

import os, sys, csv, re, gc, argparse, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# ── torchvision mock ─────────────────────────────────────────────────────────
class _Mock:
    def __getattr__(self, n): return _Mock()
    def __call__(self, *a, **k): return _Mock()
for _m in ["torchvision", "torchvision.ops", "torchvision.transforms"]:
    sys.modules.setdefault(_m, _Mock())
if not hasattr(torch.ops, "torchvision"):
    class _DummyOps:
        def nms(*a, **k): return torch.tensor([])
    torch.ops.torchvision = _DummyOps()
# ─────────────────────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_REGISTRY = {
    "diffucoder": {
        "name": "DiffuCoder-7B",
        "id":   "apple/DiffuCoder-7B-Instruct",
        "mask_token": "<|mask|>",
    },
    "dreamcoder": {
        "name": "DreamCoder-7B",
        "id":   "Dream-org/Dream-Coder-v0-Instruct-7B",
        "mask_token": "<|mask|>",
    },
}

RESULTS_DIR = "results/1t5t_exp"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

JUDGE_REGISTRY = {
    "Qwen2.5-7B-Instruct":  "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct":  "Qwen/Qwen2.5-3B-Instruct",
}

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path: str, max_samples: int | None = None) -> pd.DataFrame:
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                rid = int(row[0])
            except ValueError:
                continue  # skip header if any
            rows.append({"id": rid, "masked_code": row[1], "ground_truth": row[2].strip()})
            if max_samples and len(rows) >= max_samples:
                break
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_prediction(text: str) -> str:
    """Extract a clean Java identifier from model output."""
    text = (text.replace("<|im_end|>", "")
                .replace("<|dlm_pad|>", "")
                .strip())
    first_line = text.split("\n")[0].strip("`\"' ")
    if " " in first_line:
        m = re.search(r"is\s+([a-zA-Z_][a-zA-Z0-9_]*)", first_line, re.I)
        if m:
            return m.group(1)
        last_word = first_line.split()[-1].strip(".,;!?`\"' ")
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", last_word):
            return last_word
    m = re.search(r"[a-zA-Z_][a-zA-Z0-9_]*", first_line)
    return m.group(0) if m else first_line


def build_masked_input_diffucoder(masked_code: str,
                                  mask_token: str, k: int,
                                  tokenizer) -> dict:
    """
    DiffuCoder uses a chat prompt. We put k mask tokens in the assistant prefix
    so the model fills them in via diffusion.
    """
    multi_mask = " ".join([mask_token] * k)
    # Replace the [MASK] placeholder in the code
    input_code = masked_code.replace("[MASK]", multi_mask)
    prompt = (
        f"<|im_start|>system\n"
        f"You are an expert Java developer. "
        f"Fill in the masked variable name in the code below.\n<|im_end|>\n"
        f"<|im_start|>user\n{input_code}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return tokenizer(prompt, return_tensors="pt")


def build_masked_input_dreamcoder(masked_code: str,
                                  mask_token: str, k: int,
                                  tokenizer) -> dict:
    """DreamCoder uses the masked code directly (no chat template)."""
    multi_mask = " ".join([mask_token] * k)
    input_code = masked_code.replace("[MASK]", multi_mask)
    return tokenizer(input_code, return_tensors="pt")


def run_diffusion(model, tokenizer, inputs: dict,
                  steps: int, model_key: str) -> str:
    """Run diffusion_generate and return decoded completion text."""
    input_ids = inputs["input_ids"].to(DEVICE)
    attn_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attn_mask,
            max_new_tokens=20,
            steps=steps,
            output_history=False,
            return_dict_in_generate=True,
            temperature=0.3,
            top_p=0.95,
            alg="entropy",
            alg_temp=0.0,
        )

    gen_ids = output.sequences[0]
    new_ids  = gen_ids[len(input_ids[0]):]
    raw = tokenizer.decode(new_ids.tolist(), skip_special_tokens=False)
    raw = raw.split("<|dlm_pad|>")[0]
    return clean_prediction(raw)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-Judge helpers  (same logic as experiments/llm_judge_variable_naming.py)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert Java code reviewer. "
    "Decide whether the predicted variable name is SEMANTICALLY ACCEPTABLE "
    "given the code context and ground-truth name.\n\n"
    "Rules:\n"
    "1. ACCEPTABLE if the prediction conveys the same concept as ground truth, "
    "even if the exact string differs (e.g. 'bufSize' vs 'bufferSize' are both fine).\n"
    "2. NOT ACCEPTABLE if it describes a different concept.\n"
    "3. Single-letter names (except obvious loop counters i, j, k) are NOT ACCEPTABLE.\n"
    "4. Invalid tokens ('0', 'true', 'MASK') are NOT ACCEPTABLE.\n\n"
    "Respond with EXACTLY one line:\n"
    "    VERDICT: 1\n"
    "or\n"
    "    VERDICT: 0\n"
    "Do NOT add any other text."
)


def _apply_chat_template(tokenizer, user_text: str) -> str:
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_text}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
    except Exception:
        return f"{JUDGE_SYSTEM}\n\nUser: {user_text}\nAssistant:"


def parse_verdict(text: str) -> int:
    m = re.search(r"VERDICT\s*:\s*([01])", text, re.I)
    if m:
        return int(m.group(1))
    for line in text.strip().splitlines():
        line = line.strip()
        if line in ("1", "0"):
            return int(line)
        if line.lower() in ("yes", "acceptable"):
            return 1
        if line.lower() in ("no", "not acceptable"):
            return 0
    return -1


def judge_one(judge_tok, judge_model,
              masked_code: str, prediction: str, ground_truth: str) -> int:
    context = masked_code.replace("[MASK]", prediction)[:2000]
    user_text = (
        f"Code context:\n```java\n{context}\n```\n\n"
        f"Ground-truth variable name: `{ground_truth}`\n"
        f"Predicted variable name:    `{prediction}`\n\n"
        f"Reply with EXACTLY one line: VERDICT: 1   or   VERDICT: 0"
    )
    prompt = _apply_chat_template(judge_tok, user_text)
    inputs = judge_tok(prompt, return_tensors="pt",
                       truncation=True, max_length=4096).to(judge_model.device)
    with torch.no_grad():
        out = judge_model.generate(
            **inputs, max_new_tokens=16, do_sample=False,
            pad_token_id=(judge_tok.eos_token_id or judge_tok.pad_token_id or 0),
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    raw = judge_tok.decode(new_ids, skip_special_tokens=True).strip()
    return parse_verdict(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_one_config(model_key: str, k: int, df: pd.DataFrame,
                   steps: int, timestamp: str,
                   judge_tok=None, judge_model=None) -> pd.DataFrame:
    """Run inference for one (model, k) combination."""
    cfg = MODELS_REGISTRY[model_key]
    model_name = cfg["name"]
    model_id   = cfg["id"]
    mask_token = cfg["mask_token"]

    print(f"\n{'─'*60}")
    print(f"  {model_name}  |  k={k} MASK tokens  |  steps={steps}")
    print(f"{'─'*60}")

    # Load model
    print(f"  Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"{model_name} k={k}"):
        sample_id    = row["id"]
        masked_code  = str(row["masked_code"])
        ground_truth = str(row["ground_truth"]).strip()

        try:
            if model_key == "diffucoder":
                inputs = build_masked_input_diffucoder(
                    masked_code, mask_token, k, tokenizer)
            else:
                inputs = build_masked_input_dreamcoder(
                    masked_code, mask_token, k, tokenizer)

            prediction = run_diffusion(model, tokenizer, inputs, steps, model_key)
            exact_match = int(prediction == ground_truth)

            # LLM judge
            if judge_tok is not None and not exact_match:
                verdict = judge_one(judge_tok, judge_model,
                                    masked_code, prediction, ground_truth)
            else:
                verdict = exact_match  # exact match → automatically acceptable

            rows.append({
                "id":            sample_id,
                "model":         model_name,
                "mask_count":    k,
                "steps":         steps,
                "ground_truth":  ground_truth,
                "prediction":    prediction,
                "exact_match":   exact_match,
                "llm_verdict":   verdict,
            })
        except Exception as e:
            rows.append({
                "id":            sample_id,
                "model":         model_name,
                "mask_count":    k,
                "steps":         steps,
                "ground_truth":  ground_truth,
                "prediction":    "",
                "exact_match":   0,
                "llm_verdict":   -1,
                "error":         str(e),
            })

    # Save raw results
    raw_df = pd.DataFrame(rows)
    safe_name = model_name.replace("-", "_").replace(" ", "_")
    raw_path = os.path.join(RESULTS_DIR,
                            f"part2_raw_{safe_name}_{k}tok_{timestamp}.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"\n  EM={raw_df['exact_match'].mean():.3f}  "
          f"LJ={raw_df[raw_df['llm_verdict']>=0]['llm_verdict'].mean():.3f}")
    print(f"  → Saved: {raw_path}")

    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return raw_df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "DiffuCoder-7B": "#3a86ff",
    "DreamCoder-7B": "#ff6b6b",
}


def plot_results(summary_df: pd.DataFrame, timestamp: str):
    """
    Two side-by-side bar charts:
      Left  – Exact Match vs number of MASK tokens (per model)
      Right – LLM-Judge accuracy vs number of MASK tokens (per model)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    models = summary_df["model"].unique()

    for metric, ax, title in [
        ("exact_match", axes[0], "Exact Match (EM)"),
        ("llm_judge",   axes[1], "LLM-as-Judge Acceptance Rate"),
    ]:
        x    = summary_df["mask_count"].unique()
        x_sorted = sorted(x)
        width = 0.35
        offsets = np.linspace(-width * (len(models) - 1) / 2,
                               width * (len(models) - 1) / 2,
                               len(models))

        for model_name, offset in zip(models, offsets):
            sub = summary_df[summary_df["model"] == model_name].set_index("mask_count")
            vals = [sub.loc[k, metric] if k in sub.index else 0. for k in x_sorted]
            bars = ax.bar(
                np.arange(len(x_sorted)) + offset, vals, width,
                label=model_name, color=MODEL_COLORS.get(model_name, "grey"),
                alpha=0.85
            )
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.003,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Number of MASK tokens (k)")
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(np.arange(len(x_sorted)))
        ax.set_xticklabels([str(k) for k in x_sorted])
        ax.set_ylim(0, min(1.0, summary_df[metric].max() * 1.25 + 0.05))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle(
        "Effect of Static MASK-Token Count on Variable Renaming Quality\n"
        f"(RefineID, {summary_df['steps'].iloc[0]} diffusion steps)",
        fontsize=14, fontweight="bold", y=1.02
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
        description="Part 2 – Static MASK-token count ablation."
    )
    parser.add_argument("--data", default="data/test.csv")
    parser.add_argument("--models", default="both",
                        choices=["diffucoder", "dreamcoder", "both"])
    parser.add_argument("--mask-counts", nargs="+", type=int,
                        default=[1, 2, 3, 4, 5])
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judge (EM only).")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  Part 2 – Static MASK-Token Count Ablation")
    print("=" * 65)
    print(f"  Diffusion steps : {args.steps}")
    print(f"  Mask counts     : {args.mask_counts}")
    print(f"  Models          : {args.models}")
    print(f"  LLM judge       : {'disabled' if args.no_judge else args.judge_model}")

    # Load data
    print(f"\nLoading data from {args.data} …")
    df = load_data(args.data, args.max_samples)
    print(f"  {len(df)} samples loaded.")

    # Optionally load LLM judge
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

    # Determine which diffusion models to run
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
                df=df,
                steps=args.steps,
                timestamp=timestamp,
                judge_tok=judge_tok,
                judge_model=judge_model,
            )
            all_raw_dfs.append(raw_df)

    # Combine and compute summary
    combined = pd.concat(all_raw_dfs, ignore_index=True)

    # Filter valid judge verdicts for LLM metric
    valid_judge = combined[combined["llm_verdict"] >= 0].copy()

    summary_rows = []
    for (model_name, k), grp in combined.groupby(["model", "mask_count"]):
        valid_grp = valid_judge[
            (valid_judge["model"] == model_name) &
            (valid_judge["mask_count"] == k)
        ]
        summary_rows.append({
            "model":        model_name,
            "mask_count":   k,
            "steps":        args.steps,
            "n_samples":    len(grp),
            "exact_match":  grp["exact_match"].mean(),
            "llm_judge":    valid_grp["llm_verdict"].mean() if len(valid_grp) else float("nan"),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, f"part2_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("  PART 2 SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\n  Summary saved → {summary_path}")

    # Plot
    if len(summary_df) > 0:
        plot_results(summary_df, timestamp)

    # Cleanup judge
    if judge_model is not None:
        del judge_model, judge_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Print best k per model per metric
    print("\n" + "=" * 65)
    print("  BEST MASK COUNT (k*)")
    print("=" * 65)
    for model_name in summary_df["model"].unique():
        sub = summary_df[summary_df["model"] == model_name]
        best_em = sub.loc[sub["exact_match"].idxmax()]
        best_lj = sub.loc[sub["llm_judge"].idxmax()] if sub["llm_judge"].notna().any() else None
        print(f"\n  {model_name}:")
        print(f"    Best EM  → k={int(best_em['mask_count'])}  EM={best_em['exact_match']:.4f}")
        if best_lj is not None:
            print(f"    Best LLM → k={int(best_lj['mask_count'])}  LJ={best_lj['llm_judge']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
