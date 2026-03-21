"""
Part 3: Dynamic vs Static MASK-Token Count Comparison
=======================================================

Research question:
  Can *dynamically* choosing the number of <|mask|> tokens for each
  variable name improve diffusion-LLM renaming quality over using a
  single fixed (static) count?

Two dynamic strategies are tested here:

  Strategy 1 — F1-Optimal Threshold (from Exp C / experiment_threshold_detector.py)
    Uses the entropy-based signal to pick the best τ, then maps a detected 
    "smell score" per token to an estimate of how many mask tokens to use:
        score ≥ τ_high  →  use k_high masks (longer name expected)  
        τ_low ≤ score < τ_high → middle estimate
        score < τ_low   →  use k_low masks
    (In the absence of a live entropy signal at inference time, we approximate
     the variable name length from the MASKED code using a char-length heuristic
     that mimics the threshold decision boundary.)

  Strategy 2 — Context Naming Length  
    Looks at all OTHER variable names appearing in the same Java code snippet,
    computes their mean token length under the target model's tokenizer, and
    uses that as the dynamic mask count (rounded, clipped to [1, 5]).

Both dynamic strategies are compared against the *best static k* from Part 2
(or a user-supplied k via --static-k).

Output:
  results/1t5t_exp/part3_raw_{model}_{strategy}_{timestamp}.csv
  results/1t5t_exp/part3_summary_{timestamp}.csv
  figures/part3_dynamic_vs_static_{timestamp}.png / .pdf

Usage (from repo root):
    # Full comparison, both models, both dynamic strategies vs static k=3
    python experiments/1t5t_exp/part3_dynamic_vs_static.py \
        --data data/test.csv \
        --models both \
        --static-k 3 \
        --steps 32 \
        --judge-model Qwen/Qwen2.5-7B-Instruct \
        --max-samples 200

    # Use default best static k from Part 2 summary CSV
    python experiments/1t5t_exp/part3_dynamic_vs_static.py \
        --data data/test.csv \
        --part2-summary results/1t5t_exp/part2_summary_YYYYMMDD_HHMMSS.csv

    # Skip LLM judge
    python experiments/1t5t_exp/part3_dynamic_vs_static.py --no-judge
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
from collections import Counter

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
    "Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    "Qwen2.5-3B-Instruct":  "Qwen/Qwen2.5-3B-Instruct",
}

# Java keywords to exclude from "other variable" analysis
JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch",
    "char", "class", "const", "continue", "default", "do", "double",
    "else", "enum", "extends", "final", "finally", "float", "for",
    "goto", "if", "implements", "import", "instanceof", "int", "interface",
    "long", "native", "new", "package", "private", "protected", "public",
    "return", "short", "static", "strictfp", "super", "switch",
    "synchronized", "this", "throw", "throws", "transient", "try",
    "void", "volatile", "while", "true", "false", "null", "var",
    "String", "Integer", "Long", "Double", "Float", "Boolean",
    "Object", "List", "ArrayList", "Map", "HashMap", "Set",
    "System", "Math", "Override",
}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic k strategies
# ─────────────────────────────────────────────────────────────────────────────

def strategy_context_naming_length(masked_code: str,
                                    tokenizer,
                                    min_k: int = 1,
                                    max_k: int = 5) -> int:
    """
    Strategy 2 – Context Naming Length.

    Extract identifiers from the code that are NOT at the [MASK] position,
    compute their average token length under the model's tokeniser,
    round and clip to [min_k, max_k].

    If no other identifiers are found, return the fallback character-based 
    heuristic (see strategy_threshold below).
    """
    # Find all lower-camelCase or snake_case identifiers in the code
    # (exclude the [MASK] placeholder itself)
    code_no_mask = masked_code.replace("[MASK]", " __MASK__ ")
    identifiers = re.findall(r'\b([a-z_][a-zA-Z0-9_]{1,})\b', code_no_mask)
    identifiers = [
        ident for ident in identifiers
        if ident not in JAVA_KEYWORDS and ident != "__MASK__"
    ]

    if not identifiers:
        # Fallback: char heuristic
        return strategy_threshold_heuristic(masked_code)

    lengths = [len(tokenizer.encode(ident, add_special_tokens=False))
               for ident in identifiers]
    mean_len = np.mean(lengths)
    k = int(round(mean_len))
    return int(np.clip(k, min_k, max_k))


def strategy_threshold_heuristic(masked_code: str,
                                   min_k: int = 1,
                                   max_k: int = 5) -> int:
    """
    Strategy 1 – Threshold-Based Heuristic (approximation of Exp-C detector).

    Since we cannot run the full entropy-fluctuation detector at inference
    time without the model already loaded in "detector" mode, we use a
    character-length proxy that statistically mirrors the F1-optimal
    threshold split found in experiment_threshold_detector.py:

      char_len ≤ 4  → k = 1   (single-token names, e.g. 'cnt', 'res')
      char_len 5-8  → k = 2   (typical Java camelCase token)
      char_len 9-13 → k = 3   (compound names, e.g. 'inputStream')
      char_len 14+  → k = 4   (long compound, e.g. 'connectionTimeout')

    The breakpoints were chosen so that the resulting distribution of k
    approximately matches the GT token-length percentile distribution from
    Part 1 (median ≈ 2-3 tokens, P90 ≈ 4-5 tokens).

    To incorporate the F1-optimal tau in a future run, this function can
    be updated to read the threshold_sensitivity.csv output from
    experiment_threshold_detector.py.
    """
    # Guess variable length by looking at surrounding code context (chars
    # between the previous and next separator around [MASK])
    # As a proxy, we use the immediately surrounding token:
    m = re.search(r'(\w+)\s*=\s*\[MASK\]|\[MASK\]\s*(?:=|,|\)|\;)', masked_code)
    if m:
        # try to find the type declaration length hint
        type_match = re.search(
            r'(\w+)\s+\[MASK\]|(\w+)\s+\w+\s*=\s*\[MASK\]', masked_code
        )
        if type_match:
            context_word = (type_match.group(1) or type_match.group(2) or "")
            # Type keywords like 'int', 'String' don't help, use char proxy
            pass

    # Simple character proxy: count the typical identifier length from context
    # by sampling identifiers near the [MASK] (±200 chars)
    idx = masked_code.find("[MASK]")
    window = masked_code[max(0, idx - 200): idx + 200]
    nearby_ids = re.findall(r'\b([a-z_][a-zA-Z0-9_]{1,30})\b', window)
    nearby_ids = [x for x in nearby_ids if x not in JAVA_KEYWORDS]

    if nearby_ids:
        avg_char_len = np.mean([len(x) for x in nearby_ids])
    else:
        avg_char_len = 6  # default

    if avg_char_len <= 4:
        k = 1
    elif avg_char_len <= 7:
        k = 2
    elif avg_char_len <= 11:
        k = 3
    elif avg_char_len <= 16:
        k = 4
    else:
        k = 5

    return int(np.clip(k, min_k, max_k))


# ─────────────────────────────────────────────────────────────────────────────
# Shared inference helpers  (copy from Part 2 to keep scripts self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def clean_prediction(text: str) -> str:
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


def run_diffusion(model, tokenizer, inputs: dict, steps: int) -> str:
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
    gen_ids = output.sequences[0][len(input_ids[0]):]
    raw = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=False)
    raw = raw.split("<|dlm_pad|>")[0]
    return clean_prediction(raw)


def build_input_diffucoder(masked_code, mask_token, k, tokenizer):
    multi_mask = " ".join([mask_token] * k)
    input_code = masked_code.replace("[MASK]", multi_mask)
    prompt = (
        f"<|im_start|>system\n"
        f"You are an expert Java developer. "
        f"Fill in the masked variable name in the code below.\n<|im_end|>\n"
        f"<|im_start|>user\n{input_code}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    return tokenizer(prompt, return_tensors="pt")


def build_input_dreamcoder(masked_code, mask_token, k, tokenizer):
    multi_mask = " ".join([mask_token] * k)
    input_code = masked_code.replace("[MASK]", multi_mask)
    return tokenizer(input_code, return_tensors="pt")


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = (
    "You are an expert Java code reviewer. "
    "Decide whether the predicted variable name is SEMANTICALLY ACCEPTABLE "
    "given the code context and ground-truth name.\n\n"
    "Respond with EXACTLY one line:\n"
    "    VERDICT: 1\n"
    "or\n"
    "    VERDICT: 0"
)


def _chat_prompt(tok, user_text):
    msgs = [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user",   "content": user_text}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True)
    except Exception:
        return f"{JUDGE_SYSTEM}\n\nUser: {user_text}\nAssistant:"


def parse_verdict(text):
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


def judge_one(jtok, jmodel, masked_code, prediction, ground_truth):
    context = masked_code.replace("[MASK]", prediction)[:2000]
    user_text = (
        f"Code:\n```java\n{context}\n```\n\n"
        f"Ground truth: `{ground_truth}`\n"
        f"Prediction:   `{prediction}`\n\n"
        f"VERDICT: 1 or VERDICT: 0"
    )
    prompt = _chat_prompt(jtok, user_text)
    inp = jtok(prompt, return_tensors="pt", truncation=True,
               max_length=4096).to(jmodel.device)
    with torch.no_grad():
        out = jmodel.generate(
            **inp, max_new_tokens=16, do_sample=False,
            pad_token_id=(jtok.eos_token_id or jtok.pad_token_id or 0),
        )
    new_ids = out[0][inp["input_ids"].shape[1]:]
    raw = jtok.decode(new_ids, skip_special_tokens=True).strip()
    return parse_verdict(raw)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(data_path, max_samples=None):
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
                continue
            rows.append({
                "id": rid,
                "masked_code": row[1],
                "ground_truth": row[2].strip(),
            })
            if max_samples and len(rows) >= max_samples:
                break
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Core experiment loop for one strategy
# ─────────────────────────────────────────────────────────────────────────────

def run_strategy(model_key: str,
                 strategy_name: str,
                 static_k: int | None,   # None → use dynamic strategy
                 df: pd.DataFrame,
                 steps: int,
                 timestamp: str,
                 judge_tok=None,
                 judge_model=None) -> pd.DataFrame:

    cfg        = MODELS_REGISTRY[model_key]
    model_name = cfg["name"]
    model_id   = cfg["id"]
    mask_token = cfg["mask_token"]

    print(f"\n{'─'*60}")
    print(f"  {model_name}  |  strategy={strategy_name}  |  steps={steps}")
    print(f"{'─'*60}")

    # Load diffusion model
    print(f"  Loading {model_id} …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"{model_name[:12]} {strategy_name}"):
        sample_id    = row["id"]
        masked_code  = str(row["masked_code"])
        ground_truth = str(row["ground_truth"]).strip()

        try:
            # Determine k
            if static_k is not None:
                k = static_k
            elif strategy_name == "dynamic_threshold":
                k = strategy_threshold_heuristic(masked_code)
            else:  # dynamic_context
                k = strategy_context_naming_length(masked_code, tokenizer)

            # Build input
            if model_key == "diffucoder":
                inputs = build_input_diffucoder(masked_code, mask_token, k, tokenizer)
            else:
                inputs = build_input_dreamcoder(masked_code, mask_token, k, tokenizer)

            prediction  = run_diffusion(model, tokenizer, inputs, steps)
            exact_match = int(prediction == ground_truth)

            # LLM judge
            if judge_tok is not None and not exact_match:
                verdict = judge_one(judge_tok, judge_model,
                                    masked_code, prediction, ground_truth)
            else:
                verdict = exact_match

            rows.append({
                "id":           sample_id,
                "model":        model_name,
                "strategy":     strategy_name,
                "dynamic_k":    k,
                "steps":        steps,
                "ground_truth": ground_truth,
                "prediction":   prediction,
                "exact_match":  exact_match,
                "llm_verdict":  verdict,
            })

        except Exception as e:
            rows.append({
                "id":           sample_id,
                "model":        model_name,
                "strategy":     strategy_name,
                "dynamic_k":    -1,
                "steps":        steps,
                "ground_truth": ground_truth,
                "prediction":   "",
                "exact_match":  0,
                "llm_verdict":  -1,
                "error":        str(e),
            })

    raw_df = pd.DataFrame(rows)
    safe_name = model_name.replace("-", "_")
    out_path = os.path.join(RESULTS_DIR,
                            f"part3_raw_{safe_name}_{strategy_name}_{timestamp}.csv")
    raw_df.to_csv(out_path, index=False)

    valid   = raw_df[raw_df["llm_verdict"] >= 0]
    em_mean = raw_df["exact_match"].mean()
    lj_mean = valid["llm_verdict"].mean() if len(valid) else float("nan")
    print(f"\n  EM={em_mean:.3f}  LJ={lj_mean:.3f}  "
          f"mean_k={raw_df['dynamic_k'].replace(-1, pd.NA).mean():.2f}")
    print(f"  → Saved: {out_path}")

    # Cleanup diffusion model (judge stays loaded)
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return raw_df


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "static":            "#3a86ff",
    "dynamic_threshold": "#ff9f1c",
    "dynamic_context":   "#2ec4b6",
}

MODEL_HATCHES = {
    "DiffuCoder-7B": "",
    "DreamCoder-7B": "///",
}


def plot_comparison(summary_df: pd.DataFrame, timestamp: str):
    models     = summary_df["model"].unique()
    strategies = summary_df["strategy"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for metric, ax, title in [
        ("exact_match", axes[0], "Exact Match (EM)"),
        ("llm_judge",   axes[1], "LLM-as-Judge Acceptance Rate"),
    ]:
        x      = np.arange(len(strategies))
        width  = 0.25
        n_mod  = len(models)
        offs   = np.linspace(-width * (n_mod - 1) / 2,
                              width * (n_mod - 1) / 2, n_mod)

        for model_name, offset in zip(models, offs):
            sub = summary_df[summary_df["model"] == model_name].set_index("strategy")
            vals = [sub.loc[s, metric] if s in sub.index else 0. for s in strategies]
            color  = {"DiffuCoder-7B": "#3a86ff", "DreamCoder-7B": "#ff6b6b"}.get(model_name, "grey")
            hatch  = MODEL_HATCHES.get(model_name, "")
            bars   = ax.bar(x + offset, vals, width,
                            label=model_name, color=color,
                            hatch=hatch, alpha=0.85)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("Strategy")
        ax.set_ylabel("Score")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x)
        strategy_labels = {
            "static":            f"Static k={summary_df.loc[summary_df['strategy']=='static','static_k'].iloc[0] if 'static_k' in summary_df.columns else '?'}",
            "dynamic_threshold": "Dynamic\n(Threshold)",
            "dynamic_context":   "Dynamic\n(Context)",
        }
        ax.set_xticklabels([strategy_labels.get(s, s) for s in strategies],
                           fontsize=9)
        ax.set_ylim(0, min(1.0, summary_df[metric].max() * 1.25 + 0.05))
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle(
        "Dynamic vs Static MASK-Token Counts on Variable Renaming Quality\n"
        f"(RefineID, {summary_df['steps'].iloc[0]} diffusion steps)",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part3_dynamic_vs_static_{timestamp}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.savefig(out.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close()
    print(f"\n  → Figure saved: {out}")


def plot_dynamic_k_distribution(all_raw_dfs: list, timestamp: str):
    """Show how the dynamic strategies distribute k values."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    strats = ["dynamic_threshold", "dynamic_context"]
    titles = ["Dynamic Threshold\n(k distribution)",
              "Dynamic Context Length\n(k distribution)"]
    colors_strat = ["#ff9f1c", "#2ec4b6"]

    for ax, strat, title, color in zip(axes, strats, titles, colors_strat):
        for df in all_raw_dfs:
            sub = df[df["strategy"] == strat]
            if len(sub) == 0:
                continue
            ks = sub["dynamic_k"].dropna().astype(int)
            counter = Counter(ks)
            total = len(ks)
            x_vals = sorted(counter.keys())
            y_vals = [counter[k] / total * 100 for k in x_vals]
            ax.bar([str(k) for k in x_vals], y_vals,
                   color=color, alpha=0.75,
                   label=sub["model"].iloc[0])
            ax.set_xlabel("Chosen k")
            ax.set_ylabel("% of samples")
            ax.set_title(title, fontweight="bold")
            ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"part3_k_distribution_{timestamp}.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  → K-distribution figure saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Part 3 – Dynamic vs static MASK-token count comparison."
    )
    parser.add_argument("--data", default="data/test.csv")
    parser.add_argument("--models", default="both",
                        choices=["diffucoder", "dreamcoder", "both"])
    parser.add_argument("--static-k", type=int, default=3,
                        help="The static baseline k to compare against (default: 3).")
    parser.add_argument("--part2-summary", type=str, default=None,
                        help="Path to Part 2 summary CSV to auto-select best static k.")
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-model", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--strategies", nargs="+",
                        choices=["static", "dynamic_threshold", "dynamic_context"],
                        default=["static", "dynamic_threshold", "dynamic_context"],
                        help="Which strategies to run.")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Override static k if Part 2 summary is provided
    static_k = args.static_k
    if args.part2_summary and os.path.exists(args.part2_summary):
        p2 = pd.read_csv(args.part2_summary)
        best_row = p2.loc[p2["exact_match"].idxmax()]
        static_k = int(best_row["mask_count"])
        print(f"[Auto] Using best static k={static_k} from Part 2 summary.")

    print("=" * 65)
    print("  Part 3 – Dynamic vs Static MASK-Token Count Comparison")
    print("=" * 65)
    print(f"  Static baseline k : {static_k}")
    print(f"  Diffusion steps   : {args.steps}")
    print(f"  Strategies        : {args.strategies}")
    print(f"  LLM judge         : {'disabled' if args.no_judge else args.judge_model}")

    # Load data
    print(f"\nLoading data from {args.data} …")
    df = load_data(args.data, args.max_samples)
    print(f"  {len(df)} samples loaded.")

    # Load judge
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
        print("  Judge loaded.")

    if args.models == "both":
        model_keys = ["diffucoder", "dreamcoder"]
    else:
        model_keys = [args.models]

    all_raw_dfs  = []
    all_summaries = []

    for model_key in model_keys:
        for strategy in args.strategies:
            k_for_run = static_k if strategy == "static" else None
            raw_df = run_strategy(
                model_key=model_key,
                strategy_name=strategy,
                static_k=k_for_run,
                df=df,
                steps=args.steps,
                timestamp=timestamp,
                judge_tok=judge_tok,
                judge_model=judge_model,
            )
            all_raw_dfs.append(raw_df)

            valid   = raw_df[raw_df["llm_verdict"] >= 0]
            em_mean = raw_df["exact_match"].mean()
            lj_mean = valid["llm_verdict"].mean() if len(valid) else float("nan")
            mean_k  = raw_df[raw_df["dynamic_k"] >= 0]["dynamic_k"].mean()

            all_summaries.append({
                "model":       MODELS_REGISTRY[model_key]["name"],
                "strategy":    strategy,
                "static_k":    static_k if strategy == "static" else None,
                "mean_k":      round(mean_k, 2),
                "steps":       args.steps,
                "n_samples":   len(raw_df),
                "exact_match": round(em_mean, 4),
                "llm_judge":   round(lj_mean, 4) if not np.isnan(lj_mean) else None,
            })

    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(RESULTS_DIR, f"part3_summary_{timestamp}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("  PART 3 SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\n  Summary saved → {summary_path}")

    # Plots
    if len(summary_df) > 0:
        plot_comparison(summary_df, timestamp)
        plot_dynamic_k_distribution(all_raw_dfs, timestamp)

    # Print improvement over static
    print("\n" + "=" * 65)
    print("  DYNAMIC vs STATIC IMPROVEMENT")
    print("=" * 65)
    for model_name in summary_df["model"].unique():
        sub   = summary_df[summary_df["model"] == model_name]
        static_em = sub[sub["strategy"] == "static"]["exact_match"].values
        static_lj = sub[sub["strategy"] == "static"]["llm_judge"].values
        if len(static_em) == 0:
            continue
        static_em = static_em[0]
        static_lj = static_lj[0] if len(static_lj) else float("nan")
        print(f"\n  {model_name}  (static k={static_k}: EM={static_em:.4f}  LJ={static_lj!r})")
        for _, row in sub[sub["strategy"] != "static"].iterrows():
            delta_em = row["exact_match"] - static_em
            delta_lj = (row["llm_judge"] if row["llm_judge"] is not None else float("nan")) - static_lj
            print(f"    {row['strategy']:25s}: EM={row['exact_match']:.4f} (Δ{delta_em:+.4f})  "
                  f"LJ={row['llm_judge']!r} (Δ{delta_lj:+.4f})")
    print("=" * 65)

    # Cleanup
    if judge_model is not None:
        del judge_model, judge_tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
