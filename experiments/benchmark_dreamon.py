"""
Standalone DreamOn benchmark for the refineID (variable-naming refactoring) task.

Strictly follows the DreamOn quickstart from
    https://github.com/DreamLM/DreamOn

Key DreamOn-specific notes (from upstream README):
    * Context window is only 2048 tokens (input + output) -- files in our
      test set go up to ~50K chars / ~15K tokens, so we MUST crop a local
      window around the first [MASK] site.
    * Input format: BOS + prefix + [mask_id]*N + suffix + EOS
    * ``max_new_tokens`` is the MAX canvas size (not "additional"); must be
      >= initial number_of_mask. Quickstart uses 4 / 64.
    * ``number_transfer_tokens=1`` (1 token per denoising step).
    * Recommended sampling: temperature=0.2, top_p=0.9, alg='entropy', alg_temp=0.
    * NO attention_mask, NO steps argument.

Strategy on our refineID data:
    All [MASK] occurrences in a sample are the SAME identifier (e.g.
    sample 3 has 61 masks all referring to ``style``). We:

      1. Crop a CONTEXT_CHARS-sized window around the first [MASK].
      2. Within the window, replace ALL [MASK] with <|mask|>*NUM_MASK_PER_SITE
         (multi-site single-pass infilling, so the model uses each
         occurrence as context for the others).
      3. Run a single diffusion_generate call.
      4. Extract the first identifier by taking the K tokens immediately
         after the prefix and regex-matching a Java identifier. This is
         robust to DreamOn's variable-length output padding.

Usage on Colab:
    !pip install transformers==4.46.2 torch==2.5.1 omegaconf tqdm pandas \
                  huggingface_hub
    !python experiments/benchmark_dreamon.py --max-samples 5  --debug
    !python experiments/benchmark_dreamon.py --max-samples 100
    !python experiments/benchmark_dreamon.py
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
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False


# --- Mock torchvision (some Dream/DreamOn checkpoints try to import it) ---
class _MockModule:
    def __getattr__(self, name): return _MockModule()
    def __call__(self, *args, **kwargs): return _MockModule()

sys.modules.setdefault('torchvision', _MockModule())
sys.modules.setdefault('torchvision.ops', _MockModule())
sys.modules.setdefault('torchvision.transforms', _MockModule())
if not hasattr(torch.ops, 'torchvision'):
    class _DummyOps:
        def nms(*args, **kwargs): return torch.tensor([])
    torch.ops.torchvision = _DummyOps()


# ---- Configuration ---------------------------------------------------------

MODEL_ID = "Dream-org/DreamOn-v0-7B"
DATA_PATH = "data/test.csv"
RESULTS_DIR = "results/dreamon_benchmark"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Window cropping around the first [MASK] (chars).
# Roughly 4 chars/token for code -> 3000 chars ~ 750 tokens of context, well
# within DreamOn's 2048 token budget after we add masks + EOS.
CONTEXT_CHARS = 3000

# How many <|mask|> tokens to put at each [MASK] site initially.
NUM_MASK_PER_SITE = 4

# Max canvas size after DreamOn expansion (per call). Quickstart uses 64.
MAX_NEW_TOKENS = 64

# Cap on the number of [MASK] sites we keep in the window. Sample 3 has 61
# masks; keeping all of them blows up canvas / context budget.
MAX_SITES_IN_WINDOW = 8

# Number of tokens to read AFTER the prefix when extracting the first
# predicted identifier. Java identifiers are typically <= 5 BPE tokens.
EXTRACT_TOKENS = 12

GEN_KWARGS = dict(
    temperature=0.2,
    top_p=0.9,
    alg="entropy",
    alg_temp=0,
    number_transfer_tokens=1,
)


# ---- Window cropping -------------------------------------------------------

def crop_window(code, target_chars=CONTEXT_CHARS, max_sites=MAX_SITES_IN_WINDOW):
    """Crop a window around the FIRST [MASK] in code.

    Returns (window_text, first_mask_was_at_position_in_full_code).
    Tries to keep first [MASK] near the middle so DreamOn has both prefix
    and suffix context. Trims further if too many [MASK] sites are inside.
    """
    first = code.find("[MASK]")
    if first == -1:
        return code, -1

    half = target_chars // 2
    start = max(0, first - half)
    end = min(len(code), first + half)
    window = code[start:end]

    # If the window contains too many [MASK] sites, trim from the END
    # (we keep the first mask near the middle of the window).
    while window.count("[MASK]") > max_sites and end > first + 200:
        end -= 200
        window = code[start:end]

    return window, first


# ---- Multi-site DreamOn prompt ---------------------------------------------

def build_multisite_prompt(window_text, tokenizer, num_mask_per_site=NUM_MASK_PER_SITE):
    """Replace each [MASK] in window_text with num_mask_per_site mask tokens
    and frame with BOS/EOS.

    Returns:
        input_ids: list[int]
        prefix_token_count: int  -- number of tokens of the prefix BEFORE
            the first <|mask|> region (excluding BOS). Use this with
            EXTRACT_TOKENS to read the first denoised identifier.
    """
    parts = window_text.split("[MASK]")
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id
    mask_id = tokenizer.mask_token_id

    ids = [bos]
    prefix_len = None  # tokens of prefix excluding BOS, computed at first mask
    for i, segment in enumerate(parts):
        seg_ids = tokenizer.encode(segment, add_special_tokens=False)
        ids.extend(seg_ids)
        if i < len(parts) - 1:
            if prefix_len is None:
                prefix_len = len(ids) - 1   # exclude BOS at index 0
            ids.extend([mask_id] * num_mask_per_site)
    ids.append(eos)
    return ids, prefix_len


# ---- Identifier extraction -------------------------------------------------

_IDENT_RE = re.compile(r'[a-zA-Z_$][a-zA-Z0-9_$]*')

def extract_identifier_from_tokens(seq, tokenizer, prefix_token_count, n_tokens=EXTRACT_TOKENS):
    """Take the n_tokens immediately after the prefix in the generated seq,
    decode them, and pull out the first valid Java identifier.

    This is robust to DreamOn's variable-length output / trailing padding,
    because:
        * BOS is at index 0 (unchanged)
        * prefix_token_count tokens of prefix follow (unchanged -- they
          weren't masked, so the model cannot rewrite them)
        * the FIRST denoised canvas starts at index 1 + prefix_token_count
        * the identifier we want is the very first non-noise token there.
    """
    if torch.is_tensor(seq):
        seq_list = seq.tolist()
    else:
        seq_list = list(seq)

    start = 1 + prefix_token_count        # skip BOS + prefix
    end = min(len(seq_list), start + n_tokens)
    chunk = seq_list[start:end]
    text = tokenizer.decode(chunk, skip_special_tokens=True)

    m = _IDENT_RE.search(text)
    if m:
        return m.group(0), text
    return text.strip()[:20], text


# ---- Single-sample inference -----------------------------------------------

def predict_one(model, tokenizer, masked_code, debug=False,
                context_chars=CONTEXT_CHARS,
                num_mask_per_site=NUM_MASK_PER_SITE,
                max_new_tokens=MAX_NEW_TOKENS,
                max_sites=MAX_SITES_IN_WINDOW):
    """Crop a window around the first [MASK], multi-site infill, return ident."""
    window, first_pos = crop_window(masked_code, context_chars, max_sites)
    if first_pos < 0:
        return "", "", 0

    input_ids, prefix_token_count = build_multisite_prompt(
        window, tokenizer, num_mask_per_site=num_mask_per_site,
    )
    n_sites = window.count("[MASK]")

    # max_new_tokens must be >= initial canvas size; be safe.
    eff_max_new = max(num_mask_per_site, max_new_tokens)

    input_t = torch.LongTensor([input_ids]).to(model.device)
    with torch.no_grad():
        output = model.diffusion_generate(
            input_t,
            max_new_tokens=eff_max_new,
            return_dict_in_generate=True,
            output_history=False,
            **GEN_KWARGS,
        )
    seq = output.sequences[0] if hasattr(output, "sequences") else output[0]

    ident, raw_chunk = extract_identifier_from_tokens(
        seq, tokenizer, prefix_token_count, n_tokens=EXTRACT_TOKENS,
    )

    if debug:
        full = tokenizer.decode(seq, skip_special_tokens=False)
        print(f"  [debug] window_len={len(window)} chars, sites_in_window={n_sites}, "
              f"prefix_tokens={prefix_token_count}")
        print(f"  [debug] raw chunk after prefix: {raw_chunk!r}")
        print(f"  [debug] first 300 chars of full output (with specials):")
        print("    " + full[:300].replace("\n", "\\n"))

    return ident, raw_chunk, n_sites


# ---- Data loading ----------------------------------------------------------

def load_data(data_path, max_samples=None):
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if max_samples is not None and i >= max_samples:
                break
            rows.append({
                "id": row[0],
                "masked_code": row[1],
                "target": row[2].strip(),
            })
    return rows


# ---- HF upload helper ------------------------------------------------------

def upload_to_hf(file_path, repo_id, token, path_in_repo=None):
    if not HAS_HF_HUB or not repo_id:
        return False
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        filename = os.path.basename(file_path)
        if path_in_repo is None:
            path_in_repo = f"dreamon_benchmark/{filename}"
        print(f"    Uploading {filename} to {repo_id}...")
        api.upload_file(path_or_fileobj=file_path, path_in_repo=path_in_repo,
                        repo_id=repo_id, token=token, repo_type="dataset")
        print("    Upload OK.")
        return True
    except Exception as e:
        print(f"    Upload failed: {e}")
        return False


# ---- Main benchmark --------------------------------------------------------

def run(max_samples=None, hf_repo=None, hf_token=None, debug=False,
        context_chars=CONTEXT_CHARS,
        num_mask_per_site=NUM_MASK_PER_SITE,
        max_new_tokens=MAX_NEW_TOKENS,
        max_sites=MAX_SITES_IN_WINDOW):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading data from {DATA_PATH}...")
    data = load_data(DATA_PATH, max_samples=max_samples)
    print(f"Loaded {len(data)} samples.")

    print(f"\nLoading {MODEL_ID} on {DEVICE}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"  bos_token_id  = {tokenizer.bos_token_id}")
    print(f"  eos_token_id  = {tokenizer.eos_token_id}")
    print(f"  mask_token_id = {tokenizer.mask_token_id}  ({tokenizer.mask_token!r})")

    # ---- Sanity check ---------------------------------------------------
    if debug:
        print("\n[sanity] Quickstart-style infill (single-site):")
        sane_window = "public int add(int a, int b) {\n    return a + [MASK];\n}\n"
        ident, chunk, n = predict_one(
            model, tokenizer, sane_window,
            context_chars=10000, num_mask_per_site=num_mask_per_site,
            max_new_tokens=max_new_tokens, max_sites=max_sites, debug=True,
        )
        print(f"[sanity] ident={ident!r} (any token from {{a,b,1,2,...}} is fine)\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"DreamOn-7B_refineID_{timestamp}.csv")

    correct = 0
    errors = 0
    results = []

    for idx, row in enumerate(tqdm(data, desc="DreamOn-7B")):
        item_id = row["id"]
        masked_code = row["masked_code"]
        ground_truth = row["target"]

        try:
            ident, raw_chunk, n_sites_in_window = predict_one(
                model, tokenizer, masked_code,
                context_chars=context_chars,
                num_mask_per_site=num_mask_per_site,
                max_new_tokens=max_new_tokens,
                max_sites=max_sites,
                debug=debug and idx < 2,
            )
            is_correct = (ident == ground_truth)
            if is_correct:
                correct += 1

            results.append({
                "id": item_id,
                "ground_truth": ground_truth,
                "prediction": ident,
                "correct": is_correct,
                "mask_count_total": masked_code.count("[MASK]"),
                "mask_count_in_window": n_sites_in_window,
                "raw_chunk": raw_chunk[:100],
            })

        except Exception as e:
            errors += 1
            results.append({
                "id": item_id,
                "ground_truth": ground_truth,
                "prediction": "",
                "correct": False,
                "error": str(e),
            })
            if errors <= 5:
                print(f"  Error on {item_id}: {e}")

    # ---- Save ----------------------------------------------------------
    fieldnames = ["id", "ground_truth", "prediction", "correct",
                  "mask_count_total", "mask_count_in_window",
                  "raw_chunk", "error"]
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    accuracy = correct / len(data) if data else 0.0
    print(f"\nAccuracy: {correct}/{len(data)} = {accuracy:.2%}")
    print(f"Errors:   {errors}")
    print(f"Results:  {out_file}")

    if hf_repo:
        upload_to_hf(out_file, hf_repo, hf_token)

    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "accuracy", "correct",
                                          "total", "errors", "results_file",
                                          "context_chars", "num_mask_per_site",
                                          "max_new_tokens", "max_sites"])
        w.writeheader()
        w.writerow({"model": "DreamOn-7B", "accuracy": f"{accuracy:.4f}",
                    "correct": correct, "total": len(data),
                    "errors": errors, "results_file": out_file,
                    "context_chars": context_chars,
                    "num_mask_per_site": num_mask_per_site,
                    "max_new_tokens": max_new_tokens,
                    "max_sites": max_sites})
    if hf_repo:
        upload_to_hf(summary_file, hf_repo, hf_token)

    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DreamOn-7B on refineID.")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--context-chars", type=int, default=CONTEXT_CHARS,
                        help="Window size in chars around the first [MASK].")
    parser.add_argument("--num-mask-per-site", type=int, default=NUM_MASK_PER_SITE,
                        help="Initial <|mask|> tokens per [MASK] site (default 4).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help="Max canvas size after expansion (default 64).")
    parser.add_argument("--max-sites", type=int, default=MAX_SITES_IN_WINDOW,
                        help="Cap on number of [MASK] sites kept in the window.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if args.max_new_tokens < args.num_mask_per_site:
        print(f"ERROR: --max-new-tokens ({args.max_new_tokens}) must be "
              f">= --num-mask-per-site ({args.num_mask_per_site}).")
        sys.exit(1)

    run(
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        debug=args.debug,
        context_chars=args.context_chars,
        num_mask_per_site=args.num_mask_per_site,
        max_new_tokens=args.max_new_tokens,
        max_sites=args.max_sites,
    )
