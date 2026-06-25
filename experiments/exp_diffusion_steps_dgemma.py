"""
Diffusion-step sweep for DiffusionGemma-26B-A4B (Fig 6 companion).
==================================================================

DiffuCoder/DreamCoder expose ``diffusion_generate(steps=T)``; DiffusionGemma is
block-AR and runs an internal entropy-bounded sampler whose denoising-step count
is read from ``model.generation_config`` (default: max 48 steps WITH adaptive
early stopping that halts at ~12-16 steps). To put DiffusionGemma on the same
"does extra denoising help?" axis we:

  1. DISABLE adaptive early stopping, so the sampler always runs all its steps;
  2. set the max-denoising-steps to T;
  3. sweep T over the same grid and record EM + per-sample latency.

Field names in generation_config are model-version dependent, so on startup we
DUMP the full generation_config and auto-detect the relevant fields (override
with --max-steps-field / --early-stop-fields if the auto-detect misses).

Metric: first-site EM (prediction at the first [MASK] == ground truth), matching
exp_diffusion_steps_benchmark.py so the three models share one y-axis. Per-sample
latency is the wall-clock for the whole sample (DiffusionGemma names each site
with its own generate() call -- noted in the caption).

COST: forcing all steps over the full grid is ~8x the RQ1 dgemma cost, so this
defaults to a 100-sample subset and a grid capped at 48 (its recommended max).
Bump --max-samples / --steps once you see the per-step rate.

Output (same schema as exp_diffusion_steps_benchmark.py so fig07 picks it up):
    results/diffusion_steps_benchmark/DiffusionGemma-26B-A4B_<ts>.csv   (detail)
    results/diffusion_steps_benchmark/summary_dgemma_<ts>.csv           (summary)

Usage (DAIC, 96GB card + pylibs_dgemma):
    PYTHONPATH=$UMBRELLA/pylibs_dgemma:$PYTHONPATH \
      python experiments/exp_diffusion_steps_dgemma.py --max-samples 20 --debug   # smoke + dump config
    PYTHONPATH=... python experiments/exp_diffusion_steps_dgemma.py                # default sweep
"""
import os
import sys
import csv
import gc
import re
import time
import argparse
from datetime import datetime

import torch
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import benchmark_diffusiongemma as bdg   # load_diffusiongemma, run_dgemma_on_sample, load_data

DATA_PATH = os.path.join(_REPO, "data", "test.csv")
RESULTS_DIR = os.path.join(_REPO, "results", "diffusion_steps_benchmark")
MODEL_NAME = "DiffusionGemma-26B-A4B"
DEFAULT_GRID = [1, 2, 4, 8, 16, 32, 48]   # 48 = recommended max denoising steps

# candidate generation_config field names (auto-detected; override via CLI)
MAX_STEP_CANDIDATES = ["num_diffusion_steps", "max_denoising_steps", "denoising_steps",
                       "num_denoising_steps", "diffusion_steps", "num_inference_steps",
                       "max_steps", "num_steps"]
EARLY_STOP_BOOL_CANDIDATES = ["adaptive_early_stopping", "early_stopping",
                              "use_adaptive_stopping", "adaptive_stopping",
                              "use_early_stopping", "do_early_stopping"]
EARLY_STOP_THRESH_CANDIDATES = ["early_stopping_entropy_threshold",
                                "adaptive_stopping_entropy_threshold",
                                "stopping_entropy_threshold", "entropy_stop_threshold",
                                "early_stopping_threshold"]


def _gc_dict(model):
    gc_obj = model.generation_config
    try:
        return gc_obj.to_dict()
    except Exception:
        return {k: getattr(gc_obj, k) for k in dir(gc_obj) if not k.startswith("_")}


def _detect(d, candidates, regex):
    for c in candidates:
        if c in d:
            return c
    for k in d:
        if re.search(regex, k, re.I):
            return k
    return None


def configure(model, steps, max_field, stop_bool_fields, stop_thresh_fields, verbose=False):
    """Set max-denoising-steps=steps and disable adaptive early stopping."""
    gc_obj = model.generation_config
    if max_field:
        setattr(gc_obj, max_field, int(steps))
        # also pin a min if such a field exists, so it cannot stop before T
        for mn in ("min_denoising_steps", "min_steps", "num_min_steps"):
            if hasattr(gc_obj, mn):
                setattr(gc_obj, mn, int(steps))
    for f in stop_bool_fields:
        if hasattr(gc_obj, f):
            setattr(gc_obj, f, False)
    for f in stop_thresh_fields:
        if hasattr(gc_obj, f):
            setattr(gc_obj, f, -1.0)   # avg entropy is >=0, so < -1 never triggers
    if verbose:
        shown = {max_field: getattr(gc_obj, max_field, None)} if max_field else {}
        for f in stop_bool_fields + stop_thresh_fields:
            if hasattr(gc_obj, f):
                shown[f] = getattr(gc_obj, f)
        print(f"    [config] {shown}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="DiffusionGemma diffusion-step sweep.")
    ap.add_argument("--data", default=DATA_PATH)
    ap.add_argument("--max-samples", type=int, default=100)
    ap.add_argument("--steps", nargs="+", type=int, default=DEFAULT_GRID)
    ap.add_argument("--max-steps-field", default=None, help="override max-steps gen-config field")
    ap.add_argument("--early-stop-fields", default=None,
                    help="comma list of bool fields to set False (override auto-detect)")
    ap.add_argument("--model-id", default=bdg.MODEL_ID)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    grid = sorted(set(int(s) for s in args.steps))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 68)
    print(f"  DiffusionGemma diffusion-step sweep  grid={grid}  device={device}")
    print("=" * 68)

    data = bdg.load_data(args.data, args.max_samples)
    print(f"  {len(data)} samples")

    t0 = time.time()
    processor, model = bdg.load_diffusiongemma(args.model_id)
    print(f"  loaded in {time.time()-t0:.1f}s")

    # ---- DUMP generation_config so the real field names are visible in the log ----
    gcd = _gc_dict(model)
    print("\n  ===== model.generation_config =====")
    for k in sorted(gcd):
        print(f"    {k} = {gcd[k]!r}")
    print("  ===================================\n", flush=True)

    # ---- detect / override the knobs ----
    max_field = args.max_steps_field or _detect(gcd, MAX_STEP_CANDIDATES,
                                                 r"(max.*step|step.*max|num.*step|denois.*step|diffusion.*step|inference.*step)")
    if args.early_stop_fields:
        stop_bool = [s.strip() for s in args.early_stop_fields.split(",") if s.strip()]
    else:
        stop_bool = [c for c in EARLY_STOP_BOOL_CANDIDATES if c in gcd]
        auto = _detect(gcd, [], r"(early.*stop|adaptive.*stop)")
        if auto and auto not in stop_bool:
            stop_bool.append(auto)
    stop_thresh = [c for c in EARLY_STOP_THRESH_CANDIDATES if c in gcd]
    auto_t = _detect(gcd, [], r"(stop.*entropy|entropy.*thresh|stop.*thresh)")
    if auto_t and auto_t not in stop_thresh:
        stop_thresh.append(auto_t)

    print(f"  max-steps field   : {max_field or 'NOT FOUND'}")
    print(f"  early-stop bools  : {stop_bool or 'none'}")
    print(f"  early-stop thresh : {stop_thresh or 'none'}")
    if not max_field:
        print("  [FATAL] no max-denoising-steps field found in generation_config.\n"
              "          Inspect the dump above and re-run with --max-steps-field <name>.",
              file=sys.stderr)
        sys.exit(2)
    print(flush=True)

    detail_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_{ts}.csv")
    df = open(detail_path, "w", newline="", encoding="utf-8")
    dw = csv.DictWriter(df, fieldnames=["model", "steps", "sample_id", "ground_truth",
                                        "prediction", "correct", "time_per_sample"])
    dw.writeheader()

    summary = []
    try:
        for steps in grid:
            print(f"\n  ── steps={steps} {'─'*46}", flush=True)
            configure(model, steps, max_field, stop_bool, stop_thresh, verbose=True)
            correct = errors = 0
            times = []
            for i, row in enumerate(data):
                gt = row["target"]
                try:
                    t1 = time.perf_counter()
                    preds, _raw, _pr, _final = bdg.run_dgemma_on_sample(
                        row["masked_code"], model, processor)
                    dt = time.perf_counter() - t1
                    pred = preds[0] if preds else ""
                except Exception as e:
                    errors += 1
                    pred, dt = "", float("nan")
                    if errors <= 3:
                        print(f"    err {row['id']}: {e}", flush=True)
                ok = int(pred == gt)
                correct += ok
                if not np.isnan(dt):
                    times.append(dt)
                dw.writerow({"model": MODEL_NAME, "steps": steps, "sample_id": row["id"],
                             "ground_truth": gt, "prediction": pred, "correct": ok,
                             "time_per_sample": f"{dt:.4f}"})
                df.flush()
                if (i + 1) % 20 == 0:
                    print(f"    steps={steps}: {i+1}/{len(data)}  EM_so_far={100*correct/(i+1):.1f}%",
                          flush=True)
            n = len(data)
            em = correct / n if n else 0.0
            mean_t = float(np.mean(times)) if times else float("nan")
            print(f"    EM={100*em:.2f}%  mean_time={mean_t:.3f}s  errors={errors}", flush=True)
            summary.append({"model": MODEL_NAME, "steps": steps,
                            "exact_match_rate": round(em, 6),
                            "mean_time_per_sample": round(mean_t, 4),
                            "correct": correct, "total": n, "errors": errors})
            # incremental summary write (timeout-safe)
            sp = os.path.join(RESULTS_DIR, f"summary_dgemma_{ts}.csv")
            with open(sp, "w", newline="", encoding="utf-8") as sf:
                sw = csv.DictWriter(sf, fieldnames=["model", "steps", "exact_match_rate",
                                                    "mean_time_per_sample", "correct", "total", "errors"])
                sw.writeheader()
                sw.writerows(summary)
    finally:
        df.close()
        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n  detail  -> {detail_path}")
    print(f"  summary -> {os.path.join(RESULTS_DIR, f'summary_dgemma_{ts}.csv')}")
    print("\n  STEP SENSITIVITY (DiffusionGemma)")
    for r in summary:
        print(f"    steps={r['steps']:>3}  EM={100*r['exact_match_rate']:.2f}%  "
              f"time={r['mean_time_per_sample']:.3f}s")


if __name__ == "__main__":
    main()
