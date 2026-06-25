"""
DreamOn mask-token count ablation (companion to part2_static_token_ablation.py).
=================================================================================

Research question (same as part2): given the mask position is known, how many
mask tokens should we seed so the model best recovers the original name?

part2_static_token_ablation.py answers this for the FIXED-CANVAS dLLMs
(DiffuCoder-7B, DreamCoder-7B) via ``model.diffusion_generate`` with k copies of
``<|mask|>``. DreamOn is a VARIABLE-CANVAS dLLM: k is the *initial* canvas it may
then expand/contract, so it needs its own inference path. We reuse
``benchmark_dreamon.predict_one(..., num_mask_per_site=k)`` — the exact code the
RQ1 unified runner uses for DreamOn — and sweep k in {1..5}.

NOTE on DiffusionGemma: it is block-AR (prompted per-site naming, no in-place
mask canvas), so "number of mask tokens" is undefined for it. It is therefore
excluded from this ablation by design (RQ1-only).

Metric: to be directly comparable with the part2 figure (Fig 6a), the headline
is FIRST-SITE Exact Match (first masked occurrence == ground truth). We also log
all-sites EM (strict) and majority-vote EM for completeness.

Outputs -> results/1t5t_exp/
    dreamon_mask_ablation_raw_<k>tok_<ts>.csv      per-sample, per k
    dreamon_mask_ablation_summary.csv              model,k,em,em_allsites,em_majority,n
                                                   (STABLE name; read by fig06)

Usage (DAIC / GPU):
    python experiments/1t5t_exp/part2_dreamon_mask_ablation.py
    python experiments/1t5t_exp/part2_dreamon_mask_ablation.py --max-samples 20   # smoke
    python experiments/1t5t_exp/part2_dreamon_mask_ablation.py --mask-counts 1 2 3 4 5
"""
import os
import sys
import csv
import gc
import time
import argparse
from collections import Counter
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# repo root on path so we can import the DreamOn benchmark machinery
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
for p in (_REPO, os.path.join(_REPO, "experiments")):
    if p not in sys.path:
        sys.path.insert(0, p)

import benchmark_dreamon as bdo   # predict_one, tile_windows, mocks torchvision

MODEL_ID = "Dream-org/DreamOn-v0-7B"
DATA_PATH = os.path.join(_REPO, "data", "test.csv")
RESULTS_DIR = os.path.join(_REPO, "results", "1t5t_exp")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "dreamon_mask_ablation_summary.csv")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "DreamOn-7B"
CACHE_CLEAR_INTERVAL = 20


def load_data(path, max_samples=None):
    csv.field_size_limit(2**31 - 1)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f)):
            if max_samples is not None and i >= max_samples:
                break
            if len(row) < 3:
                continue
            rows.append({"id": row[0], "masked_code": row[1],
                         "ground_truth": row[2].strip()})
    return rows


def score_site_preds(site_preds, gt):
    """Return (first_correct, allsites_correct, majority_correct)."""
    if not site_preds:
        return 0, 0, 0
    first_c = int(site_preds[0] == gt)
    allsites_c = int(all(p == gt and p != "" for p in site_preds))
    mode_pred = Counter(p for p in site_preds if p).most_common(1)
    maj_c = int(bool(mode_pred) and mode_pred[0][0] == gt)
    return first_c, allsites_c, maj_c


def merge_write_summary(summary_rows):
    """(Re)write the stable summary CSV = other-model prior rows + this model's
    rows so far. Called after EVERY k so a 24h timeout never loses completed k's."""
    prior = []
    if os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, newline="", encoding="utf-8") as f:
            prior = [r for r in csv.DictReader(f) if r.get("model") != MODEL_NAME]
    fields = ["model", "k", "em", "em_allsites", "em_majority", "n"]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in prior:
            w.writerow({c: r.get(c, "") for c in fields})
        for r in summary_rows:
            w.writerow({c: r.get(c, "") for c in fields})


def main():
    ap = argparse.ArgumentParser(description="DreamOn mask-token count ablation (k sweep).")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--mask-counts", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--context-chars", type=int, default=bdo.CONTEXT_CHARS)
    ap.add_argument("--max-new-tokens", type=int, default=bdo.MAX_NEW_TOKENS)
    ap.add_argument("--max-sites", type=int, default=bdo.MAX_SITES_IN_WINDOW)
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 65)
    print("  DreamOn mask-token count ablation")
    print(f"  model={MODEL_ID}  k={args.mask_counts}  device={DEVICE}")
    print("=" * 65)

    data = load_data(args.data, args.max_samples)
    print(f"  {len(data)} samples")

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(DEVICE).eval()
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)

    summary = []
    try:
        for k in args.mask_counts:
            if args.max_new_tokens < k:
                print(f"  [skip k={k}] max-new-tokens {args.max_new_tokens} < k")
                continue
            print(f"\n  ── k={k} {'─'*48}", flush=True)
            rows = []
            fc = ac = mc = errors = 0
            pbar = tqdm(data, desc=f"k={k}", ncols=90, mininterval=5.0)
            for i, row in enumerate(pbar):
                gt = row["ground_truth"]
                try:
                    site_preds, n_win = bdo.predict_one(
                        model, tok, row["masked_code"],
                        context_chars=args.context_chars,
                        num_mask_per_site=k,
                        max_new_tokens=args.max_new_tokens,
                        max_sites=args.max_sites,
                    )
                    f_c, a_c, m_c = score_site_preds(site_preds, gt)
                    fc += f_c; ac += a_c; mc += m_c
                    rows.append({
                        "id": row["id"], "model": MODEL_NAME, "mask_count": k,
                        "ground_truth": gt,
                        "first_pred": site_preds[0] if site_preds else "",
                        "predictions": "|".join(site_preds),
                        "first_correct": f_c, "allsites_correct": a_c,
                        "majority_correct": m_c, "n_sites": len(site_preds),
                    })
                except Exception as e:
                    errors += 1
                    rows.append({"id": row["id"], "model": MODEL_NAME, "mask_count": k,
                                 "ground_truth": gt, "first_pred": "", "predictions": "",
                                 "first_correct": 0, "allsites_correct": 0,
                                 "majority_correct": 0, "n_sites": 0, "error": str(e)})
                    if errors <= 5:
                        print(f"    err {row['id']}: {e}")
                if (i + 1) % CACHE_CLEAR_INTERVAL == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # live running EM on the bar + a flushed milestone every 100 to .out
                done = i + 1
                pbar.set_postfix_str(f"EM={100*fc/done:.1f}% err={errors}")
                if done % 100 == 0:
                    print(f"    k={k}: {done}/{len(data)}  EM_so_far={100*fc/done:.1f}%  "
                          f"errors={errors}", flush=True)
            pbar.close()

            n = len(data)
            em_first = 100.0 * fc / n if n else 0.0
            em_all = 100.0 * ac / n if n else 0.0
            em_maj = 100.0 * mc / n if n else 0.0
            print(f"    EM(first-site)={em_first:.2f}%  EM(all-sites)={em_all:.2f}%  "
                  f"EM(majority)={em_maj:.2f}%  errors={errors}")

            raw_path = os.path.join(RESULTS_DIR, f"dreamon_mask_ablation_raw_{k}tok_{ts}.csv")
            fields = ["id", "model", "mask_count", "ground_truth", "first_pred",
                      "predictions", "first_correct", "allsites_correct",
                      "majority_correct", "n_sites", "error"]
            with open(raw_path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow({c: r.get(c, "") for c in fields})
            summary.append({"model": MODEL_NAME, "k": k,
                            "em": round(em_first, 2), "em_allsites": round(em_all, 2),
                            "em_majority": round(em_maj, 2), "n": n})
            # Persist after each k so a walltime timeout keeps completed k's.
            merge_write_summary(summary)
            print(f"    -> summary updated ({len(summary)} k done): {SUMMARY_CSV}")
    finally:
        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    merge_write_summary(summary)   # final consistent write

    print("\n" + "=" * 65)
    print("  SUMMARY (headline = first-site EM, comparable to Fig 6a)")
    for r in summary:
        print(f"    k={r['k']}  EM={r['em']:.2f}%  (all-sites {r['em_allsites']:.2f}%, "
              f"majority {r['em_majority']:.2f}%)")
    best = max(summary, key=lambda r: r["em"]) if summary else None
    if best:
        print(f"  optimal k* = {best['k']}  (EM {best['em']:.2f}%)")
    print(f"  summary -> {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
