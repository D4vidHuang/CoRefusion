#!/usr/bin/env python3
"""
RQ3 distribution-based smell rank (teacher's revision). RUN ON DAIC (GPU).

Instead of injecting a curated smell token (x/tmp/foo) and reading its rank, we
read the model's FULL output distribution at each RefineID target position and
report the rank of the smell names the model itself surfaces, where "smell" is
decided by the objective rule in smell_rule.py (NO hand-picked vocabulary).

Per position we record:
  gt_rank        : rank of the developer's name  (regime derived from this)
  smell_med_rank : MEDIAN rank over ALL smell-classified vocab tokens   [HEADLINE]
  smell_min_rank : best (lowest) rank among smell tokens                [secondary]
stratified into HighConfident / Uncertain / RareConfident
(gt_rank <= 200 / <= 1000 / > 1000), matching the existing RQ3 regimes.

Reuses the model / window / regime logic of exp_overconfidence_stratified.py.

Outputs (results/distribution_smell_rank/):
  <model>_<ts>.csv               per-position rows
  <model>_<ts>_summary.csv       per-regime medians  (-> feeds the figure)
  <model>_<ts>_smelltokens.tsv   the resolved smell-token set (audit)

Usage:
  python experiments/exp_distribution_smell_rank.py --model DiffuCoder-7B-Base
  python experiments/exp_distribution_smell_rank.py --model DreamCoder-7B --max-samples 200
"""
import os
import sys
import csv
import argparse
import statistics as st
from datetime import datetime

import torch
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import exp_overconfidence_stratified as base          # noqa: E402
from smell_rule import build_smell_token_ids, load_english_words  # noqa: E402

OUT_DIR = os.path.join(base.ROOT_DIR, "results", "distribution_smell_rank")
REGIME_ORDER = ["OVERCONFIDENT", "UNCERTAIN", "CONFIDENT_RARE"]


def rank_of_all(logits, position):
    """LongTensor `ranks` [V] with ranks[t] = 1-based rank of token t in the
    descending-probability order at `position` (lower = model prefers it more)."""
    lp = logits[0, position, :].float()
    order = torch.argsort(lp, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, order.numel() + 1, device=order.device)
    return ranks


def run(model_name, max_samples, thresh_low, thresh_high):
    os.makedirs(OUT_DIR, exist_ok=True)
    if model_name not in base.MODEL_REGISTRY:
        sys.exit(f"unknown model {model_name}; choices: {list(base.MODEL_REGISTRY)}")
    meta = base.MODEL_REGISTRY[model_name]
    tokenizer, model, mask_id = base.load_model(meta["id"], meta["mask_token"])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dump = os.path.join(OUT_DIR, f"{model_name}_{ts}_smelltokens.tsv")
    ew = load_english_words()
    smell_ids = build_smell_token_ids(tokenizer, ew, dump_path=dump)
    smell_t = torch.tensor(sorted(smell_ids), dtype=torch.long, device=base.DEVICE)
    print(f"smell-token set: {len(smell_ids)} tokens  (audit -> {dump})")

    data = base.load_data(base.DATA_PATH, max_samples)
    out_file = os.path.join(OUT_DIR, f"{model_name}_{ts}.csv")
    fields = ["id", "gt_name", "gt_rank", "regime",
              "smell_med_rank", "smell_min_rank", "n_smell", "target_idx"]
    fout = open(out_file, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=fields)
    writer.writeheader()

    clamped = False
    counts = {k: 0 for k in REGIME_ORDER}
    for row in tqdm(data, desc=model_name):
        masked = row["masked_code"]
        gt = row["target"]
        mc = masked.find("[MASK]")
        if mc == -1:
            continue
        original = masked.replace("[MASK]", gt, 1)
        try:
            input_ids, tgt = base.build_centered_window(tokenizer, original, mc, gt, mask_id)
            gt_ids = tokenizer.encode(gt, add_special_tokens=False)
            gt_tid = gt_ids[0] if gt_ids else (tokenizer.unk_token_id or 0)
            logits = base.single_forward(model, input_ids)
            ranks = rank_of_all(logits, tgt)
        except Exception:
            continue
        if not clamped:                       # drop any smell ids beyond the output dim
            smell_t = smell_t[smell_t < ranks.numel()]
            clamped = True
        gt_rank = int(ranks[int(gt_tid)])
        regime = base.classify_regime(gt_rank, thresh_low, thresh_high)
        counts[regime] += 1
        sr = ranks[smell_t]
        writer.writerow({
            "id": row["id"], "gt_name": gt, "gt_rank": gt_rank, "regime": regime,
            "smell_med_rank": float(sr.float().median().item()),
            "smell_min_rank": int(sr.min().item()),
            "n_smell": int(smell_t.numel()), "target_idx": tgt,
        })
    fout.close()

    rows = list(csv.DictReader(open(out_file)))
    summ = os.path.join(OUT_DIR, f"{model_name}_{ts}_summary.csv")
    with open(summ, "w", newline="") as f:
        sw = csv.writer(f)
        sw.writerow(["regime", "n", "median_gt_rank",
                     "median_smell_med_rank", "median_smell_min_rank"])
        for rg in REGIME_ORDER:
            sub = [r for r in rows if r["regime"] == rg]
            if not sub:
                continue
            sw.writerow([rg, len(sub),
                         round(st.median([int(r["gt_rank"]) for r in sub]), 1),
                         round(st.median([float(r["smell_med_rank"]) for r in sub]), 1),
                         round(st.median([float(r["smell_min_rank"]) for r in sub]), 1)])
    print("wrote", out_file)
    print("wrote", summ)
    print("regime counts:", counts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="DiffuCoder-7B-Base",
                    help=f"one of {list(base.MODEL_REGISTRY)}")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--thresh-low", type=int, default=base.THRESH_LOW_DEFAULT)
    ap.add_argument("--thresh-high", type=int, default=base.THRESH_HIGH_DEFAULT)
    a = ap.parse_args()
    run(a.model, a.max_samples, a.thresh_low, a.thresh_high)
