"""
All-sites Exact Match stratified by rename-set cardinality |S|, recomputed
from the authoritative unified RQ1 predictions (NOT hard-coded Table values).

For RefineID RQ1 every sample has ONE target identifier masked at all of its
occurrence sites; the cardinality |S| of a sample is therefore the number of
[MASK] sites (= n_total_masks). A sample is "all-sites correct" iff *every*
site emits the SAME non-empty name and that name equals the ground truth --
exactly the strict ``em_gated`` metric the leaderboard reports (an inconsistent
sample cannot be applied as a refactoring, so it is a failure). Because all-
sites-correct collapses to "all per-site predictions == ground_truth", we
compute it directly from the |-joined ``predictions`` column.

Reads : results/unified_refineID/predictions/<Model>.csv
        (id, ground_truth, n_total_masks, predictions, first_pred,
         first_correct, error)
        results/unified_refineID/leaderboard*.csv   (for arch/engine/params)
Writes: results/unified_refineID/em_by_cardinality.csv
        (model, arch, family, bucket, n_samples, em_pct, overall_em_pct, n)

Usage:
    python analysis/em_by_cardinality.py
    python analysis/em_by_cardinality.py --check     # print vs leaderboard em_gated
"""
import os
import re
import csv
import sys
import argparse

csv.field_size_limit(2**31 - 1)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
UNI = os.path.join(REPO, "results", "unified_refineID")
PRED_DIR = os.path.join(UNI, "predictions")
OUT_CSV = os.path.join(UNI, "em_by_cardinality.csv")

# Cardinality buckets (lo, hi inclusive, label). |S| = number of [MASK] sites.
BUCKETS = [(1, 1, "|S|=1"), (2, 2, "|S|=2"), (3, 5, "|S|=3-5"),
           (6, 10, "|S|=6-10"), (11, 10**9, "|S|>=11")]

# Skip the colab side-runs / duplicates -- keep one canonical CSV per model.
SKIP_FILES = {"DiffuCoder-7B_colab.csv", "DreamCoder-7B_colab.csv"}

# Map architecture string (from leaderboard) to a plotting family.
def family_of(arch: str) -> str:
    a = (arch or "").lower()
    if "dllm" in a or "diffusion" in a or "dreamon" in a or "block-ar" in a:
        return "dLLM"
    if "encoder" in a or "seq2seq" in a or "t5" in a:
        return "Encoder-decoder"
    return "Decoder-only"


def bucket_label(n: int) -> str:
    for lo, hi, lab in BUCKETS:
        if lo <= n <= hi:
            return lab
    return BUCKETS[-1][2]


def load_arch_map():
    """model -> (arch, params, engine) from any leaderboard CSV present."""
    out = {}
    # daic leaderboard is the most recent / authoritative; let it win.
    for fn in ("leaderboard_daic.csv", "leaderboard.csv"):
        p = os.path.join(UNI, fn)
        if not os.path.exists(p):
            continue
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                m = r.get("model")
                if m and m not in out:
                    out[m] = (r.get("arch", ""), r.get("params", ""),
                              r.get("engine", ""))
    return out


def all_sites_correct(predictions: str, ground_truth: str) -> bool:
    """Strict all-sites EM: every site non-empty and == ground_truth."""
    gt = (ground_truth or "").strip()
    if not gt:
        return False
    preds = [p for p in (predictions or "").split("|")]
    if not preds or any(p.strip() == "" for p in preds):
        return False
    return all(p.strip() == gt for p in preds)


def aggregate_model(path: str):
    """Return (bucket_em: dict[label]->(correct,total), overall_correct, n)."""
    agg = {lab: [0, 0] for _, _, lab in BUCKETS}
    overall_c = n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                k = int(r.get("n_total_masks") or 0)
            except (TypeError, ValueError):
                continue
            if k <= 0:
                continue
            n += 1
            ok = all_sites_correct(r.get("predictions", ""), r.get("ground_truth", ""))
            overall_c += int(ok)
            cell = agg[bucket_label(k)]
            cell[1] += 1
            cell[0] += int(ok)
    return agg, overall_c, n


def main():
    ap = argparse.ArgumentParser(description="EM by rename-set cardinality from unified predictions.")
    ap.add_argument("--check", action="store_true",
                    help="Print recomputed overall EM next to leaderboard em_gated.")
    args = ap.parse_args()

    if not os.path.isdir(PRED_DIR):
        sys.exit(f"missing predictions dir: {PRED_DIR}")

    arch_map = load_arch_map()
    files = sorted(fn for fn in os.listdir(PRED_DIR)
                   if fn.endswith(".csv") and fn not in SKIP_FILES)

    rows = []
    check_rows = []
    for fn in files:
        model = fn[:-4]
        path = os.path.join(PRED_DIR, fn)
        agg, overall_c, n = aggregate_model(path)
        if n == 0:
            print(f"  [skip] {model}: no scorable rows")
            continue
        arch, params, engine = arch_map.get(model, ("", "", ""))
        fam = family_of(arch or engine or model)
        overall_em = 100.0 * overall_c / n if n else float("nan")
        for _, _, lab in BUCKETS:
            c, t = agg[lab]
            rows.append({
                "model": model, "arch": arch, "family": fam, "params": params,
                "bucket": lab, "n_samples": t,
                "em_pct": round(100.0 * c / t, 2) if t else "",
                "overall_em_pct": round(overall_em, 2), "n": n,
            })
        check_rows.append((model, overall_em, n))

    os.makedirs(UNI, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "arch", "family", "params",
                                          "bucket", "n_samples", "em_pct",
                                          "overall_em_pct", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_CSV}  ({len(rows)} rows, {len(check_rows)} models)")

    if args.check:
        # Compare recomputed overall all-sites EM vs leaderboard em_gated.
        lb = {}
        for fn in ("leaderboard_daic.csv", "leaderboard.csv"):
            p = os.path.join(UNI, fn)
            if os.path.exists(p):
                with open(p, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        if r.get("model") and r.get("model") not in lb:
                            try:
                                lb[r["model"]] = 100.0 * float(r.get("em_gated") or "nan")
                            except ValueError:
                                pass
        print(f"\n  {'model':<26}{'recomputed':>12}{'em_gated(lb)':>14}{'Δ':>8}")
        for model, em, n in sorted(check_rows, key=lambda x: -x[1]):
            ref = lb.get(model)
            d = f"{em-ref:+.2f}" if ref is not None else "  n/a"
            refs = f"{ref:.2f}" if ref is not None else "  n/a"
            print(f"  {model:<26}{em:>11.2f} {refs:>13}{d:>8}")


if __name__ == "__main__":
    main()
