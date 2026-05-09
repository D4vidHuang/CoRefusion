"""
Standalone DreamOn benchmark for the refineID (variable-naming refactoring) task.

Strictly follows the DreamOn quickstart from
    https://github.com/DreamLM/DreamOn

Key DreamOn-specific notes (from upstream README):
    * Input format: BOS + prefix + [mask_id]*N + suffix + EOS
        (built by ``process_infilling_prompt``)
    * ``max_new_tokens`` is the MAX canvas size (not "additional"); must be
        >= initial number_of_mask.  Quickstart uses 4 / 64.
    * ``number_transfer_tokens=1`` (1 token per denoising step).  Other
        values are not fully tested upstream.
    * Recommended sampling: temperature=0.2, top_p=0.9, alg='entropy', alg_temp=0.
    * NO attention_mask, NO steps argument.

Why a separate script (not wired into ``benchmark_diffusion_models.py``):
    DreamOn requires per-mask infilling with BOS/EOS framing and a much
    larger canvas budget than the single-pass <|mask|>*N replacement that
    DiffuCoder/DreamCoder use.  Mixing them was producing all-MASK outputs
    because the canvas was too tight.

Usage on Colab:
    !pip install transformers==4.46.2 torch==2.5.1 omegaconf tqdm pandas \
                  huggingface_hub
    !python experiments/benchmark_dreamon.py --max-samples 50 --debug
    !python experiments/benchmark_dreamon.py                       # full run

Output:
    results/dreamon_benchmark/DreamOn-7B_refineID_<timestamp>.csv
    results/dreamon_benchmark/summary_<timestamp>.csv
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

# --- Mock torchvision (some Dream/DreamOn checkpoints import it) ---
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
DATA_PATH = "CoRefusion/data/test.csv"
RESULTS_DIR = "results/dreamon_benchmark"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DreamOn quickstart defaults
NUMBER_OF_MASK = 4       # initial canvas size at the [MASK] site
MAX_NEW_TOKENS = 64      # upper bound on canvas size after expansion

GEN_KWARGS = dict(
    temperature=0.2,
    top_p=0.9,
    alg="entropy",
    alg_temp=0,
    number_transfer_tokens=1,
)


# ---- DreamOn input construction (verbatim from quickstart) -----------------

def process_infilling_prompt(prefix, suffix, tokenizer, number_of_mask):
    """Verbatim copy of the helper in the DreamOn README."""
    prefix_ids = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    middle_ids = [tokenizer.mask_token_id] * number_of_mask
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    return prefix_ids + middle_ids + suffix_ids


# ---- Output decoding -------------------------------------------------------

def decode_infill(generated_ids, tokenizer, n_prefix_ids, n_suffix_ids):
    """
    Slice the infilled span out of the generated sequence by ID-position
    rather than text anchors.  Avoids brittle string matching.

    Layout:
        [BOS] + prefix_ids + <denoised_canvas> + suffix_ids + [EOS]
        ^----- n_prefix_ids+1 -----^         ^---- n_suffix_ids+1 ----^

    The denoised_canvas length is variable (DreamOn expands/contracts), so
    we slice from index ``n_prefix_ids+1`` up to ``len-(n_suffix_ids+1)``.
    """
    seq = generated_ids.tolist() if torch.is_tensor(generated_ids) else list(generated_ids)
    start = n_prefix_ids + 1                # skip BOS + prefix tokens
    end = len(seq) - (n_suffix_ids + 1)     # exclude suffix tokens + EOS
    if end <= start:
        return ""
    canvas_ids = seq[start:end]
    return tokenizer.decode(canvas_ids, skip_special_tokens=True)


def extract_identifier(text, fallback_len=20):
    """Pull the first valid Java identifier out of a span of text."""
    m = re.search(r'[a-zA-Z_$][a-zA-Z0-9_$]*', text)
    return m.group(0) if m else text.strip()[:fallback_len]


# ---- Per-sample inference --------------------------------------------------

def infill_one_site(model, tokenizer, prefix, suffix,
                    number_of_mask=NUMBER_OF_MASK,
                    max_new_tokens=MAX_NEW_TOKENS,
                    gen_kwargs=None,
                    debug=False):
    """Run a single DreamOn infilling pass at a single [MASK] site."""
    gen_kwargs = gen_kwargs or GEN_KWARGS

    # Build prompt exactly as the quickstart does
    prefix_ids_full = [tokenizer.bos_token_id] + tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids_full = tokenizer.encode(suffix, add_special_tokens=False) + [tokenizer.eos_token_id]
    n_prefix = len(prefix_ids_full) - 1     # ids excluding BOS for slicing math
    n_suffix = len(suffix_ids_full) - 1     # ids excluding EOS

    input_ids = prefix_ids_full + [tokenizer.mask_token_id] * number_of_mask + suffix_ids_full
    input_ids = torch.LongTensor([input_ids]).to(model.device)

    with torch.no_grad():
        output = model.diffusion_generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_history=debug,        # only collect history when debugging
            **gen_kwargs,
        )

    seq = output.sequences[0] if hasattr(output, "sequences") else output[0]
    canvas_text = decode_infill(seq, tokenizer, n_prefix, n_suffix)

    if debug:
        full_dec = tokenizer.decode(seq, skip_special_tokens=False)
        print("  [debug] canvas span:", repr(canvas_text)[:120])
        print("  [debug] full output (first 400 chars, with specials):")
        print("    " + full_dec[:400].replace("\n", "\\n"))

    return extract_identifier(canvas_text), canvas_text


def predict_sample(model, tokenizer, masked_code, debug=False, **infill_kwargs):
    """For each [MASK] site in masked_code, run a separate infilling pass.

    Walks left-to-right; each step's prefix has previous predictions
    spliced in so the model sees consistent context.

    Returns:
        preds (list[str]): predicted identifier per [MASK] site
        spans (list[str]): the raw infilled canvas per site (for inspection)
    """
    parts = masked_code.split("[MASK]")
    if len(parts) <= 1:
        return [], []

    preds, spans = [], []
    resolved_prefix = parts[0]
    for i in range(len(parts) - 1):
        prefix = resolved_prefix
        # Suffix carries the rest of the code.  Remaining [MASK]s are kept as
        # literal text -- they're just normal tokens in the suffix context.
        suffix = "[MASK]".join(parts[i + 1:])

        ident, canvas = infill_one_site(
            model, tokenizer, prefix, suffix,
            debug=debug, **infill_kwargs,
        )
        preds.append(ident)
        spans.append(canvas)

        resolved_prefix = prefix + ident + parts[i + 1]
        if debug:
            print(f"  [site {i}] -> {ident!r}")

    return preds, spans


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
        number_of_mask=NUMBER_OF_MASK, max_new_tokens=MAX_NEW_TOKENS,
        first_mask_only=False):
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

    # ---- Sanity check: replicate the quickstart prompt once ---------------
    if debug:
        print("\n[sanity] Running quickstart-style infill once...")
        sane_pre = "public int add(int a, int b) {\n    return a + "
        sane_suf = ";\n}\n"
        ident, canvas = infill_one_site(
            model, tokenizer, sane_pre, sane_suf,
            number_of_mask=number_of_mask, max_new_tokens=max_new_tokens,
            debug=True,
        )
        print(f"[sanity] ident={ident!r}, canvas={canvas!r}\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(RESULTS_DIR, f"DreamOn-7B_refineID_{timestamp}.csv")

    correct = 0
    errors = 0
    results = []

    for row in tqdm(data, desc="DreamOn-7B"):
        item_id = row["id"]
        masked_code = row["masked_code"]
        ground_truth = row["target"]

        try:
            if first_mask_only:
                # Faster path: only fill the first [MASK]
                head, _, tail = masked_code.partition("[MASK]")
                # Drop later masks from suffix? -> keep them as literal text
                ident, canvas = infill_one_site(
                    model, tokenizer, head, tail,
                    number_of_mask=number_of_mask,
                    max_new_tokens=max_new_tokens,
                    debug=debug and len(results) < 2,
                )
                preds, spans = [ident], [canvas]
            else:
                preds, spans = predict_sample(
                    model, tokenizer, masked_code,
                    debug=debug and len(results) < 2,
                    number_of_mask=number_of_mask,
                    max_new_tokens=max_new_tokens,
                )

            primary = preds[0] if preds else ""
            is_correct = (primary == ground_truth)
            if is_correct:
                correct += 1

            results.append({
                "id": item_id,
                "ground_truth": ground_truth,
                "prediction": primary,
                "correct": is_correct,
                "mask_count": masked_code.count("[MASK]"),
                "all_predictions": "|".join(preds),
                "first_canvas": (spans[0] if spans else "")[:200],
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

    # ---- Save -------------------------------------------------------------
    fieldnames = ["id", "ground_truth", "prediction", "correct",
                  "mask_count", "all_predictions", "first_canvas", "error"]
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

    # Tiny summary file
    summary_file = os.path.join(RESULTS_DIR, f"summary_{timestamp}.csv")
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "accuracy", "correct",
                                          "total", "errors", "results_file"])
        w.writeheader()
        w.writerow({"model": "DreamOn-7B", "accuracy": f"{accuracy:.4f}",
                    "correct": correct, "total": len(data),
                    "errors": errors, "results_file": out_file})
    if hf_repo:
        upload_to_hf(summary_file, hf_repo, hf_token)

    # Cleanup
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---- CLI -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark DreamOn-7B on refineID.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of test samples (quick smoke run).")
    parser.add_argument("--number-of-mask", type=int, default=NUMBER_OF_MASK,
                        help=f"Initial canvas mask count (default {NUMBER_OF_MASK}).")
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS,
                        help=f"Max canvas size after DreamOn expansion "
                             f"(default {MAX_NEW_TOKENS}). Must be >= --number-of-mask.")
    parser.add_argument("--first-mask-only", action="store_true",
                        help="Only infill the first [MASK] per sample (much faster, "
                             "and is what the EM metric actually scores).")
    parser.add_argument("--debug", action="store_true",
                        help="Print sanity-check + first 2 samples' raw outputs.")
    parser.add_argument("--hf-repo", type=str, default=None)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    if args.max_new_tokens < args.number_of_mask:
        print(f"ERROR: --max-new-tokens ({args.max_new_tokens}) must be "
              f">= --number-of-mask ({args.number_of_mask}).")
        sys.exit(1)

    run(
        max_samples=args.max_samples,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        debug=args.debug,
        number_of_mask=args.number_of_mask,
        max_new_tokens=args.max_new_tokens,
        first_mask_only=args.first_mask_only,
    )
