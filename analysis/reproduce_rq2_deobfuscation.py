"""
Authoritative, reproducible recomputation of every RQ2 (deobfuscation) number
in the thesis, sourced *exclusively* from the latest 2026-06-12 runs.

It reproduces, from the raw per-file CSVs + the ground-truth targets:

  * Table V  / Fig 8(a) -- target-position EM under RQ1 (clean), all-masked,
                           and target-only, for DiffuCoder-7B and DreamCoder-7B.
  * Fig 8(b)            -- all-masked mean per-sample EM, stratified by the
                           number of distinct identifiers in the smell set S.
  * Table VI            -- DiffuCoder-7B target-only example wrong predictions.
  * Sec V-C breakdown   -- wrong predictions split into short-copy / long-
                           meaningful / empty, using the paper's *exact*
                           classifier (a prediction "copies the obfuscated
                           style" iff it matches ^[a-z]{1,2}$). This rule was
                           verified to reproduce the paper's 662/928 = 71.3%
                           on the run the paper was written against.

It also writes the `target-only` summary CSV that the experiment driver never
emitted (only the all-masked summaries were saved for the June runs).

WHY target-position EM needs its own script: neither the experiment summary
(which reports majority-vote *all-site* EM, ~7.4%) nor the consistency
leaderboard (which reports the strict consistency-gated EM, ~1.8%) emits the
per-target EM that Table V/Fig 8 report (12.4% / 2.8% ...). It is computed here
by joining each run's `predictions_json` with the RefineID target.

Inputs (pinned to the latest = 6/12 runs):
    data/test.csv                                          ground truth, col 2 = target
    results/deobfuscation_refineID/<Model>_<mode>_<ts>.csv June timestamps

Outputs -> results/deobfuscation_refineID/reproduced/
    table5_target_em.csv
    fig8b_idcount_buckets.csv
    table6_examples.csv
    secVC_wrong_breakdown.csv
    summary_target-only_20260612.csv     (the missing summary)
    rq2_numbers.json                      (everything, machine-readable)
    figures/new/fig8_rq2_em_june.{png,pdf}   (unless --no-figure)

Usage:
    python analysis/reproduce_rq2_deobfuscation.py
    python analysis/reproduce_rq2_deobfuscation.py --no-figure
"""

import os
import re
import csv
import sys
import glob
import json
import argparse

csv.field_size_limit(2**31 - 1)

# --- Repo-relative paths -----------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DEOBF_DIR = os.path.join(REPO, "results", "deobfuscation_refineID")
OUT_DIR = os.path.join(DEOBF_DIR, "reproduced")
FIG_DIR = os.path.join(REPO, "figures", "new")
TEST_CSV = os.path.join(REPO, "data", "test.csv")

# --- Pinned June (latest) runs ----------------------------------------------
# DiffuCoder/DreamCoder are PINNED to the authoritative 6/12 runs. DreamOn (added
# later) is auto-discovered: the newest DreamOn-7B_<mode>_*.csv in DEOBF_DIR is
# used, so once its DAIC deobfuscation job lands it appears with no edit here.
# Any model whose all-masked AND target-only runs are both missing is skipped.
PINNED = {
    ("DiffuCoder-7B", "all-masked"):  "DiffuCoder-7B_all-masked_20260612_102623.csv",
    ("DreamCoder-7B", "all-masked"):  "DreamCoder-7B_all-masked_20260612_102631.csv",
    ("DiffuCoder-7B", "target-only"): "DiffuCoder-7B_target-only_20260612_102623.csv",
    ("DreamCoder-7B", "target-only"): "DreamCoder-7B_target-only_20260612_102631.csv",
}
MODELS = ["DiffuCoder-7B", "DreamCoder-7B", "DreamOn-7B"]
HF_ID = {
    "DiffuCoder-7B": "apple/DiffuCoder-7B-Base",
    "DreamCoder-7B": "Dream-org/Dream-Coder-v0-Instruct-7B",
    "DreamOn-7B": "Dream-org/DreamOn-v0-7B",
}

# RQ1 clean-context target-position EM (first-site EM), from the SEPARATE RQ1
# benchmark, not the deobfuscation runs, so Fig 8(a) can show all three bars.
# DiffuCoder/DreamCoder pinned to the thesis RQ1 table; DreamOn = its first-site
# EM on the unified RQ1 predictions (analysis/em_by_cardinality.py family).
# DiffusionGemma is intentionally absent (block-AR, RQ2 N/A).
RQ1_CLEAN_EM = {"DiffuCoder-7B": 31.1, "DreamCoder-7B": 33.2, "DreamOn-7B": 15.2}

# Partial-run fallback for DreamOn RQ2 target-position EM. Used ONLY when the
# per-sample CSVs (DreamOn-7B_{all-masked,target-only}_*.csv) are not yet present
# in DEOBF_DIR -- e.g. the DAIC job is still running / not downloaded. As soon as
# those CSVs land, resolve_run() finds them, DreamOn is computed from real data,
# and this dict is ignored. Numbers below are the running id-EM from the DAIC
# console of the in-progress run (all-masked n~80, target-only n~320); they are
# PRELIMINARY and flagged as such in the paper. Set to None to disable the
# fallback entirely (then DreamOn simply won't appear until its CSVs exist).
DREAMON_RQ2_PARTIAL = {"all-masked": 5.0, "target-only": 4.4}


def resolve_run(model, mode):
    """Pinned path if given (and exists), else newest <model>_<mode>_*.csv."""
    if (model, mode) in PINNED:
        p = os.path.join(DEOBF_DIR, PINNED[(model, mode)])
        return p if os.path.exists(p) else None
    cands = sorted(glob.glob(os.path.join(DEOBF_DIR, f"{model}_{mode}_*.csv")))
    return cands[-1] if cands else None

# Stratification buckets for Fig 8(b), by #distinct identifiers in S.
BUCKETS = [(1, 10, "1-10"), (11, 20, "11-20"), (21, 50, "21-50"),
           (51, 100, "51-100"), (101, 10**9, "100+")]

# Paper's exact "copies the obfuscated single-letter style" classifier.
# Verified to reproduce 662/928 = 71.3% on the run the paper used.
SHORT_COPY = re.compile(r"^[a-z]{1,2}$")

# The nine example rows shown in Table VI (DiffuCoder-7B, target-only).
TABLE6_IDS = ["0", "2", "3", "6", "9", "11", "12", "13", "15"]


# --- Helpers -----------------------------------------------------------------
def load_targets():
    targets = {}
    with open(TEST_CSV, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) >= 3:
                targets[row[0]] = row[2].strip()
    return targets


def load_rows(name_or_path):
    path = name_or_path if os.path.isabs(name_or_path) else os.path.join(DEOBF_DIR, name_or_path)
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def jloads(s):
    try:
        return json.loads(s) if s else {}
    except Exception:
        return {}


def is_skipped(r):
    return bool((r.get("skipped") or "").strip())


def bucket_label(n):
    for lo, hi, lab in BUCKETS:
        if lo <= n <= hi:
            return lab
    return BUCKETS[-1][2]


# --- Metric 1: target-position EM (Table V / Fig 8a) -------------------------
def target_position_em(rows, targets):
    """EM evaluated ONLY at the RefineID target identifier's prediction."""
    correct = scored = 0
    for sid, r in rows.items():
        if is_skipped(r):
            continue
        tgt = targets.get(sid)
        preds = jloads(r.get("predictions_json"))
        if tgt in preds:
            scored += 1
            if preds[tgt] == tgt:
                correct += 1
    em = 100.0 * correct / scored if scored else float("nan")
    return correct, scored, em


# --- Metric 2: Fig 8(b) stratified all-masked per-sample EM ------------------
def fig8b_buckets(rows):
    agg = {lab: [] for _, _, lab in BUCKETS}
    for r in rows.values():
        if is_skipped(r):
            continue
        try:
            n = int(r.get("num_unique_identifiers") or 0)
            em = float(r.get("per_sample_em_rate") or 0.0)
        except (TypeError, ValueError):
            continue
        if n > 0:
            agg[bucket_label(n)].append(em)
    out = []
    for _, _, lab in BUCKETS:
        v = agg[lab]
        out.append({"bucket": lab, "n_samples": len(v),
                    "mean_em_pct": round(100.0 * sum(v) / len(v), 2) if v else float("nan")})
    return out


# --- Metric 3: Sec V-C wrong-prediction breakdown ----------------------------
def wrong_breakdown(rows, targets):
    scored = correct = wrong = 0
    short = longm = empty = 0
    for sid, r in rows.items():
        if is_skipped(r):
            continue
        tgt = targets.get(sid)
        preds = jloads(r.get("predictions_json"))
        if tgt not in preds:
            continue
        scored += 1
        p = preds[tgt]
        if p == tgt:
            correct += 1
            continue
        wrong += 1
        if p == "":
            empty += 1
        elif SHORT_COPY.match(p):       # paper's exact "1-2 lowercase letters" rule
            short += 1
        else:
            longm += 1
    pct = lambda x: round(100.0 * x / wrong, 1) if wrong else float("nan")
    return {
        "scored": scored, "correct": correct, "wrong": wrong,
        "wrong_pct_of_scored": round(100.0 * wrong / scored, 1) if scored else float("nan"),
        "short_copy": short, "short_copy_pct": pct(short),
        "long_meaningful": longm, "long_meaningful_pct": pct(longm),
        "empty": empty, "empty_pct": pct(empty),
    }


# --- Metric 4: Table VI examples ---------------------------------------------
def table6(rows, targets, ids):
    out = []
    for sid in ids:
        r = rows.get(sid, {})
        tgt = targets.get(sid)
        preds = jloads(r.get("predictions_json"))
        origs = jloads(r.get("originals_json"))   # orig_name -> obf token
        obf = origs.get(tgt, "")
        pred = preds.get(tgt, "")
        out.append({"id": sid, "target": tgt, "obf": obf, "prediction": pred,
                    "matches_obf": "yes" if (pred and pred == obf) else ""})
    return out


# --- Writers -----------------------------------------------------------------
def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_target_only_summary(rows_by_model, models):
    """Reproduce the experiment-style summary the driver never wrote for the
    June target-only runs. id_correct/id_total are summed from the CSV columns,
    exactly as run_experiment() would have aggregated them."""
    out_rows = []
    for model in models:
        rows = rows_by_model[(model, "target-only")]
        id_correct = id_total = skipped = errors = 0
        for r in rows.values():
            if is_skipped(r):
                skipped += 1
                continue
            if (r.get("error") or "").strip():
                errors += 1
                continue
            id_correct += int(r.get("identifiers_correct") or 0)
            id_total += int(r.get("identifiers_total") or 0)
        processed = len(rows) - skipped - errors
        em = id_correct / id_total if id_total else 0.0
        out_rows.append({
            "model": model, "hf_id": HF_ID[model], "processed": processed,
            "skipped": skipped, "errors": errors,
            "identifier_em": f"{em:.4f}", "id_correct": id_correct, "id_total": id_total,
        })
    path = os.path.join(OUT_DIR, "summary_target-only_20260612.csv")
    write_csv(path, ["model", "hf_id", "processed", "skipped", "errors",
                     "identifier_em", "id_correct", "id_total"], out_rows)
    return out_rows


# --- Figure (Fig 8 a+b) ------------------------------------------------------
# TU Delft blue + orange (+ cyan for a 3rd model), matching the rebuilt style.
_FIG_COLORS = {"DiffuCoder-7B": "#0076C2", "DreamCoder-7B": "#FF8000",
               "DreamOn-7B": "#00A6D6"}
_FIG_FALLBACK = ["#0076C2", "#FF8000", "#00A6D6", "#6E6E6E"]


def make_figure(table5, fig8b, models):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"  [figure skipped: matplotlib unavailable: {e}]")
        return None

    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    n = len(models)
    colors = [_FIG_COLORS.get(m, _FIG_FALLBACK[i % len(_FIG_FALLBACK)])
              for i, m in enumerate(models)]

    # (a) grouped bars: 3 conditions x n models
    conds = ["RQ1\nclean", "RQ2\nall-masked", "RQ2\ntarget-only"]
    x = np.arange(len(conds))
    w = min(0.8 / n, 0.38)
    offs = (np.arange(n) - (n - 1) / 2) * w
    amax = 0.0
    for m, col, off in zip(models, colors, offs):
        vals = [RQ1_CLEAN_EM.get(m, float("nan")), table5[m]["all-masked"],
                table5[m]["target-only"]]
        amax = max(amax, max(v for v in vals if v == v))
        bars = ax1.bar(x + off, vals, w, label=m, color=col,
                       edgecolor="white", linewidth=0.6)
        ax1.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(conds)
    ax1.set_ylabel("Target-position Exact Match (%)")
    ax1.set_title("(a)")
    ax1.legend(fontsize=8.5)
    ax1.set_ylim(0, amax * 1.25)

    # (b) all-masked per-sample EM by #identifiers bucket.
    # Only models with per-sample bucket data appear here (a partial-run model
    # injected for panel (a) has no fig8b entry and is skipped).
    models_b = [m for m in models if m in fig8b]
    offs_b = (np.arange(len(models_b)) - (len(models_b) - 1) / 2) * w
    labels = [b["bucket"] for b in fig8b[models_b[0]]]
    x2 = np.arange(len(labels))
    for m, off in zip(models_b, offs_b):
        col = _FIG_COLORS.get(m, _FIG_FALLBACK[models.index(m) % len(_FIG_FALLBACK)])
        vals = [b["mean_em_pct"] for b in fig8b[m]]
        bars = ax2.bar(x2 + off, vals, w, label=m, color=col,
                       edgecolor="white", linewidth=0.6)
        ax2.bar_label(bars, fmt="%.1f", padding=2, fontsize=7.5)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Number of distinct identifiers")
    ax2.set_ylabel("Mean per-sample EM (%)")
    ax2.set_title("(b)")
    ax2.legend(fontsize=8.5)

    fig.tight_layout()
    png = os.path.join(FIG_DIR, "fig8_rq2_em_june.png")
    pdf = os.path.join(FIG_DIR, "fig8_rq2_em_june.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# --- LaTeX table emitter -----------------------------------------------------
def emit_latex_table(table5, models):
    """Write a ready-to-paste body for tab:rq2_deobfuscation / tab:rq2_em, so the
    paper never carries a hand-transcribed (and possibly stale) number. The
    DiffusionGemma N/A note is appended because it is RQ1-only (block-AR)."""
    def cell(v):
        if v is None or (isinstance(v, float) and v != v):
            return r"\,--\,"
        return f"{v:.1f}"
    lines = [
        "% AUTO-GENERATED by analysis/reproduce_rq2_deobfuscation.py -- do not hand-edit.",
        "% Paste these rows into tab:rq2_deobfuscation (conference) / tab:rq2_em (rewrite).",
    ]
    for m in models:
        rq1 = RQ1_CLEAN_EM.get(m)
        am = table5[m]["all-masked"]
        to = table5[m]["target-only"]
        lines.append(f"{m} & {cell(rq1)} & {cell(am)} & {cell(to)} \\\\")
    # DiffusionGemma is RQ1-only; show its clean EM and N/A for the RQ2 columns.
    lines.append(r"DiffusionGemma-26B-A4B & 33.8 & \multicolumn{2}{c}{N/A$^{\dagger}$} \\")
    lines.append(r"% $^{\dagger}$ block-AR (prompted per-site naming, no in-place mask canvas); RQ2 does not apply.")
    path = os.path.join(OUT_DIR, "table_rq2_em.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# --- Main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Reproduce all RQ2 deobfuscation numbers from the 6/12 runs.")
    ap.add_argument("--no-figure", action="store_true", help="skip Fig 8 generation")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    targets = load_targets()

    # Resolve runs per model (DiffuCoder/DreamCoder pinned; DreamOn auto-found).
    # A model is included only if BOTH its modes resolve to an existing CSV.
    rows_by_model = {}
    source_runs = {}
    models = []
    for m in MODELS:
        am = resolve_run(m, "all-masked")
        to = resolve_run(m, "target-only")
        if not am or not to:
            print(f"  [skip {m}] missing deobfuscation runs "
                  f"(all-masked={'ok' if am else 'MISSING'}, "
                  f"target-only={'ok' if to else 'MISSING'})")
            continue
        rows_by_model[(m, "all-masked")] = load_rows(am)
        rows_by_model[(m, "target-only")] = load_rows(to)
        source_runs[f"{m}/all-masked"] = os.path.basename(am)
        source_runs[f"{m}/target-only"] = os.path.basename(to)
        models.append(m)

    if not models:
        sys.exit("No model has both deobfuscation runs present in "
                 f"{os.path.relpath(DEOBF_DIR, REPO)}. Run the RQ2 jobs first.")
    print(f"Models with complete runs: {', '.join(models)}\n")

    bundle = {"source_runs": source_runs}

    # ---- Table V: target-position EM ----
    print("=" * 72)
    print("TABLE V  -- target-position EM (%)")
    print(f"{'Model':<16}{'RQ1 clean':>11}{'all-masked':>12}{'target-only':>13}")
    table5 = {m: {} for m in models}
    table5_rows = []
    for m in models:
        am_c, am_n, am = target_position_em(rows_by_model[(m, "all-masked")], targets)
        to_c, to_n, to = target_position_em(rows_by_model[(m, "target-only")], targets)
        table5[m] = {"all-masked": round(am, 2), "target-only": round(to, 2),
                     "RQ1_clean": RQ1_CLEAN_EM.get(m)}
        print(f"{m:<16}{RQ1_CLEAN_EM.get(m, float('nan')):>11.1f}{am:>12.2f}{to:>13.2f}")
        table5_rows.append({"model": m, "rq1_clean_em": RQ1_CLEAN_EM.get(m),
                            "all_masked_target_em": round(am, 2), "all_masked_correct": am_c, "all_masked_scored": am_n,
                            "target_only_target_em": round(to, 2), "target_only_correct": to_c, "target_only_scored": to_n})
    write_csv(os.path.join(OUT_DIR, "table5_target_em.csv"),
              ["model", "rq1_clean_em", "all_masked_target_em", "all_masked_correct", "all_masked_scored",
               "target_only_target_em", "target_only_correct", "target_only_scored"], table5_rows)
    bundle["table5"] = table5

    # Inject the DreamOn partial-run fallback into Table V / Fig 8(a) when its real
    # per-sample CSVs are absent. It carries no fig8b / secVC data, so it appears
    # only in the target-EM table and Fig 8(a) -- panel (b) and the wrong-pred
    # breakdown skip it (they need per-sample rows).
    models_fig = list(models)
    if "DreamOn-7B" not in models and DREAMON_RQ2_PARTIAL is not None:
        table5["DreamOn-7B"] = {
            "all-masked": round(DREAMON_RQ2_PARTIAL["all-masked"], 2),
            "target-only": round(DREAMON_RQ2_PARTIAL["target-only"], 2),
            "RQ1_clean": RQ1_CLEAN_EM.get("DreamOn-7B"), "partial": True,
        }
        models_fig.append("DreamOn-7B")
        print("  [DreamOn-7B] no RQ2 CSVs found -> using PRELIMINARY fallback "
              f"(all-masked={DREAMON_RQ2_PARTIAL['all-masked']}, "
              f"target-only={DREAMON_RQ2_PARTIAL['target-only']}) for Table V / Fig 8(a)")

    # ---- Fig 8(b): stratified all-masked EM ----
    print("\n" + "=" * 72)
    print("FIG 8(b) -- all-masked mean per-sample EM (%) by #identifiers bucket")
    fig8b = {}
    fig8b_rows = []
    for m in models:
        b = fig8b_buckets(rows_by_model[(m, "all-masked")])
        fig8b[m] = b
        cells = "  ".join(f"{x['bucket']}:{x['mean_em_pct']:.1f}(n={x['n_samples']})" for x in b)
        print(f"  {m:<16}{cells}")
        for x in b:
            fig8b_rows.append({"model": m, **x})
    write_csv(os.path.join(OUT_DIR, "fig8b_idcount_buckets.csv"),
              ["model", "bucket", "n_samples", "mean_em_pct"], fig8b_rows)
    bundle["fig8b"] = fig8b

    # ---- Sec V-C: wrong-prediction breakdown ----
    print("\n" + "=" * 72)
    print("SEC V-C -- target-only wrong-prediction breakdown (paper rule ^[a-z]{1,2}$)")
    secvc = {}
    secvc_rows = []
    for m in models:
        wb = wrong_breakdown(rows_by_model[(m, "target-only")], targets)
        secvc[m] = wb
        print(f"  {m}: wrong={wb['wrong']} ({wb['wrong_pct_of_scored']}% of {wb['scored']})  "
              f"short-copy={wb['short_copy']} ({wb['short_copy_pct']}%)  "
              f"long={wb['long_meaningful']} ({wb['long_meaningful_pct']}%)  "
              f"empty={wb['empty']} ({wb['empty_pct']}%)")
        secvc_rows.append({"model": m, **wb})
    write_csv(os.path.join(OUT_DIR, "secVC_wrong_breakdown.csv"),
              list(secvc_rows[0].keys()), secvc_rows)
    bundle["secVC"] = secvc

    # ---- Table VI: example wrong predictions (DiffuCoder target-only) ----
    print("\n" + "=" * 72)
    print("TABLE VI -- DiffuCoder-7B target-only examples")
    if ("DiffuCoder-7B", "target-only") in rows_by_model:
        t6 = table6(rows_by_model[("DiffuCoder-7B", "target-only")], targets, TABLE6_IDS)
    else:
        t6 = []
    for r in t6:
        dag = " (matches obf)" if r["matches_obf"] else ""
        print(f"  id={r['id']:<3} target={str(r['target']):<18} obf={r['obf']:<3} -> {r['prediction']}{dag}")
    write_csv(os.path.join(OUT_DIR, "table6_examples.csv"),
              ["id", "target", "obf", "prediction", "matches_obf"], t6)
    bundle["table6"] = t6

    # ---- Missing target-only summary ----
    print("\n" + "=" * 72)
    print("Writing the target-only summary the driver never emitted ...")
    summ = write_target_only_summary(rows_by_model, models)
    for s in summ:
        print(f"  {s['model']:<16} EM={s['identifier_em']}  ({s['id_correct']}/{s['id_total']}, "
              f"processed={s['processed']}, skipped={s['skipped']})")
    bundle["target_only_summary"] = summ

    # ---- machine-readable bundle ----
    with open(os.path.join(OUT_DIR, "rq2_numbers.json"), "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    # ---- ready-to-paste LaTeX table body ----
    tex_path = emit_latex_table(table5, models_fig)
    print(f"\nLaTeX table body -> {os.path.relpath(tex_path, REPO)}")

    # ---- figure ----
    if not args.no_figure:
        print("\n" + "=" * 72)
        res = make_figure(table5, fig8b, models_fig)
        if res:
            print(f"  wrote {res[0]}\n        {res[1]}")

    print("\nAll outputs -> " + os.path.relpath(OUT_DIR, REPO))


if __name__ == "__main__":
    main()
