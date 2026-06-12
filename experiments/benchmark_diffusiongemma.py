"""
Benchmark google/diffusiongemma-26B-A4B-it (block-diffusion dLLM) on refineID.

DiffusionGemma denoises a generation canvas appended AFTER the prompt
(block-autoregressive); it cannot infill <|mask|> in-place like Dream/DreamOn.
There is also no FIM-tokenised base checkpoint -- only the instruct model.
So this engine mirrors the iterative per-site protocol of
benchmark_ar_models_fim.run_fim_on_sample, but through a chat prompt:

    1. take the FIRST remaining [MASK], mark it <FILL_HERE> (later sites stay
       as [MASK], earlier sites already hold their predictions),
    2. ask the model to answer with ONLY the identifier for <FILL_HERE>,
    3. substitute the prediction back and continue with the next site.

Per-site predictions feed the all-sites consistency gate exactly like every
other model in run_all_refineID_unified.py. PROTOCOL CAVEAT for the paper:
this is *prompted* identifier naming, not FIM/infill -- footnote it when
comparing against the base-model rows.

Requires a transformers release that ships the `diffusion_gemma` architecture
(NEWER than the 4.57.1 pinned for Dream/DreamOn -- on DAIC use the separate
$UMBRELLA/pylibs_dgemma tree, see server/setup_dgemma_pylibs.sh). Sampling
uses the model's own generation_config (Entropy-Bounded sampler, <=48 steps),
same "model-faithful defaults" philosophy as the DreamOn quickstart runs.

Standalone smoke test (96GB GPU: bf16 weights are ~52GB, A40 will OOM):
    python experiments/benchmark_diffusiongemma.py --max-samples 5 --debug
Full runs go through the unified runner:
    python experiments/run_all_refineID_unified.py --only DiffusionGemma-26B-A4B
"""

import os
import sys
import csv
import re
import argparse
import time

# ---- Configuration ---------------------------------------------------------

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
DATA_PATH = "data/test.csv"
RESULTS_DIR = "results/diffusiongemma_smoke"
MAX_NEW_TOKENS = 64        # one identifier + (empty) thought-channel tags
MAX_INPUT_TOKENS = 16384   # model does 256K; cap for speed, FIM-style window

FILL_MARK = "<FILL_HERE>"

PROMPT_TEMPLATE = (
    "The Java code below has occurrences of ONE identifier masked.\n"
    "Sites marked [MASK] are other occurrences of the same identifier; the "
    "site marked " + FILL_MARK + " is the one to name now.\n"
    "Reply with ONLY the Java identifier for " + FILL_MARK + " -- a single "
    "name, no explanation, no code, no quotes.\n\n"
    "```java\n{code}\n```"
)

# ---- Loading ----------------------------------------------------------------


def load_diffusiongemma(hf_id=MODEL_ID, hf_token=None):
    """Load (processor, model). Needs transformers with `diffusion_gemma`."""
    from transformers import AutoProcessor
    try:
        from transformers import DiffusionGemmaForBlockDiffusion as Cls
    except ImportError:
        try:
            from transformers import AutoModelForMultimodalLM as Cls
        except ImportError:
            import transformers
            raise ImportError(
                "transformers " + transformers.__version__ + " has neither "
                "DiffusionGemmaForBlockDiffusion nor AutoModelForMultimodalLM. "
                "Install a newer transformers into a SEPARATE dir (do not "
                "touch the Dream/DreamOn-pinned pylibs): "
                "bash server/setup_dgemma_pylibs.sh")
    try:
        processor = AutoProcessor.from_pretrained(hf_id, token=hf_token)
    except ImportError as ex:
        # Gemma4Processor's image processor hard-imports torchvision (tfm 5.x).
        # refineID is text-only: the tokenizer carries the same chat template,
        # so fall back instead of dragging torchvision into the env.
        from transformers import AutoTokenizer
        print("AutoProcessor unavailable (" + str(ex)[:120]
              + ") -> text-only AutoTokenizer fallback")
        processor = AutoTokenizer.from_pretrained(hf_id, token=hf_token)
    model = Cls.from_pretrained(
        hf_id, dtype="auto", device_map="auto", token=hf_token)
    model.eval()
    return processor, model


# ---- Prompt construction ----------------------------------------------------


def _tokenizer_of(processor):
    return getattr(processor, "tokenizer", processor)


def truncate_around_mark(code, tokenizer, max_tokens):
    """Token-truncate code around FILL_MARK, 60/40 prefix/suffix like FIM."""
    if FILL_MARK not in code:
        return code
    prefix, suffix = code.split(FILL_MARK, 1)
    prefix_budget = int(max_tokens * 0.6)
    suffix_budget = max_tokens - prefix_budget
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if len(prefix_ids) > prefix_budget:
        prefix = tokenizer.decode(prefix_ids[-prefix_budget:], skip_special_tokens=True)
    if len(suffix_ids) > suffix_budget:
        suffix = tokenizer.decode(suffix_ids[:suffix_budget], skip_special_tokens=True)
    return prefix + FILL_MARK + suffix


def build_messages(code_with_mark):
    prompt = PROMPT_TEMPLATE.format(code=code_with_mark)
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]


# ---- Prediction cleaning ----------------------------------------------------

# DiffusionGemma channel format: <|channel>thought\n[reasoning]<channel|>final
_THOUGHT_RE = re.compile(r"<\|channel>thought\n.*?<channel\|>", re.DOTALL)
_IDENT_RE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_QUOTED_RE = re.compile(r"[`\"']([A-Za-z_$][A-Za-z0-9_$]*)[`\"']")


def clean_prediction(raw):
    """Extract one Java identifier from raw decoded output."""
    text = _THOUGHT_RE.sub("", raw)
    if "<channel|>" in text:                 # unmatched closing tag
        text = text.split("<channel|>")[-1]
    text = text.replace("<|channel>thought", " ")
    text = re.sub(r"<[^>\n]{0,40}>", " ", text)   # drop remaining tag-likes
    # drop code-fence marker lines (```java etc.) so we don't grab "java"
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
    text = "\n".join(lines)
    # instruct models often answer 'the identifier is "x"' -- prefer quoted
    m = _QUOTED_RE.search(text)
    if m:
        return m.group(1)
    for line in lines:
        line = line.strip().strip("`\"'. ")
        if not line:
            continue
        m = _IDENT_RE.search(line)
        if m:
            return m.group(0)
    m = _IDENT_RE.search(text)
    return m.group(0) if m else ""


# ---- Single-sample inference (signature mirrors run_fim_on_sample) ----------


def run_dgemma_on_sample(masked_code, model, processor, max_input_tokens=MAX_INPUT_TOKENS):
    """Fill all [MASK] sites via iterative per-site chat prompts.

    Returns (predictions, raw_predictions, prompts, final_code) like
    benchmark_ar_models_fim.run_fim_on_sample.
    """
    import torch
    tokenizer = _tokenizer_of(processor)
    current_code = masked_code
    predictions, raw_predictions, prompts = [], [], []
    mask_count = current_code.count("[MASK]")

    for _ in range(mask_count):
        parts = current_code.split("[MASK]", 1)
        prefix = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""

        code_with_mark = truncate_around_mark(
            prefix + FILL_MARK + suffix, tokenizer, max_input_tokens)
        messages = build_messages(code_with_mark)
        prompts.append(code_with_mark)

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        new_ids = out[0][inputs["input_ids"].shape[-1]:]
        raw = processor.decode(new_ids, skip_special_tokens=False)
        raw_predictions.append(raw)

        pred = clean_prediction(raw)
        predictions.append(pred)
        current_code = prefix + pred + suffix

    return predictions, raw_predictions, prompts, current_code


# ---- Standalone smoke test ---------------------------------------------------


def load_data(data_path, max_samples=None):
    csv.field_size_limit(2**31 - 1)
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({"id": row[0], "masked_code": row[1], "target": row[2].strip()})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override MAX_NEW_TOKENS (canvas budget per site)")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    global MAX_NEW_TOKENS
    if args.max_new_tokens:
        MAX_NEW_TOKENS = args.max_new_tokens

    data = load_data(args.data, args.max_samples)
    print(f"Loaded {len(data)} samples from {args.data}")

    t0 = time.time()
    processor, model = load_diffusiongemma(args.model_id, hf_token=args.hf_token)
    print(f"Loaded {args.model_id} in {time.time()-t0:.1f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "DiffusionGemma-26B-A4B.csv")
    fields = ["id", "ground_truth", "n_total_masks", "predictions",
              "first_pred", "first_correct", "error"]
    n_ok = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in data:
            gt = row["target"]
            n_masks = row["masked_code"].count("[MASK]")
            try:
                preds, raws, _, _ = run_dgemma_on_sample(
                    row["masked_code"], model, processor)
                first = preds[0] if preds else ""
                n_ok += int(first == gt)
                w.writerow({"id": row["id"], "ground_truth": gt,
                            "n_total_masks": n_masks,
                            "predictions": "|".join(preds), "first_pred": first,
                            "first_correct": (first == gt), "error": ""})
                if args.debug:
                    print(f"  [{row['id']}] gt={gt!r} preds={preds}")
                    if raws:
                        print(f"    raw[0]={raws[0]!r}")
            except Exception as ex:
                w.writerow({"id": row["id"], "ground_truth": gt,
                            "n_total_masks": n_masks, "predictions": "",
                            "first_pred": "", "first_correct": False,
                            "error": str(ex)[:200]})
                print(f"  error on {row['id']}: {ex}")
            f.flush()
    print(f"first-site EM {n_ok}/{len(data)}  ->  {out_path}")


if __name__ == "__main__":
    main()
