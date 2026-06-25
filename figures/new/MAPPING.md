# New figures ↔ thesis originals

Generated 2026-06-13 from the DAIC results in `results/`. Unified colour palette:
**blue `#0076C2` + orange `#FF8000`** (TU Delft). In the two-model RQ2 figures
DiffuCoder = blue, DreamCoder = orange; in the scatter blue = AR/Seq2Seq,
orange = dLLM. Every new figure re-uses the **all-sites-consistency** evaluation
(`analysis/identifier_similarity_metrics.eval_sample`): an identifier is *usable*
only if EVERY one of its sites emits the SAME non-empty name; only then is the
single agreed name scored.

Pipeline verified two ways: (a) our all-masked EM among consistent groups
(DiffuCoder 11.7% / DreamCoder 12.3%) reproduces the thesis all-masked
target-EM (12.2% / 13.9%); (b) an independent recompute straight from the raw
experiment CSV reproduces the headline numbers exactly.

| New figure | Thesis original | What is new |
|---|---|---|
| `fig1_rq2_consistency.png` | **Table V** + **Fig. 8(a)** (RQ2 target-position EM: RQ1-clean / all-masked / target-only) | Panel (a): strict all-sites EM instead of per-target EM — the collapse is sharper (all-masked 1.8/1.9%, target-only 1.5/2.6% vs thesis 12.2/13.9 & 3.1/4.1). Panel (b): the NEW consistency-rate lens — only 15% of identifiers agree under all-masked, 55–58% under target-only. RQ1-clean bars are the thesis reference (31.1/33.2). |
| `fig2_rq2_by_idcount.png` | **Fig. 8(b)** (all-masked mean per-sample EM by # distinct identifiers) | Same x-axis buckets, but strict all-sites EM. Monotone decline 12.5%→0% confirms the thesis trend under the stricter metric. |
| `fig3_rq2_copybias.png` | **§V-C / Fig. 8** "71.3% of wrong predictions copy the obfuscated single-letter style" | The copy-bias re-measured on the consistent *agreed* name: 42–46% among wrong groups. Lower than 71.3% because the thesis counted per-position 1–2-letter predictions among all wrong, while this requires a consistent agreed name. |
| `fig4_dreamon_java.png` | **§VIII Future Work #1** (fine-tune DreamOn at identifier scale) + the DreamOn-7B rows | NEW result: identifier-scale fine-tune (7B-Java, canvas mask layout) gives only a marginal bump over base (strict EM 8.7%→10.2%); the placeholder layout is degenerate (1.0% EM, 86.5% consistency — it consistently copies the literal `__MASKED_VAR__`). |
| `fig5_lj_vs_em_scatter.png` | **Table II** (EM + LJ columns) / the EM-vs-LJ relationship | LLM-as-Judge acceptance vs strict all-sites EM, one point per model, judge = Qwen2.5-7B-Instruct (thesis default). EM & LJ taken from the SAME judge CSV (self-consistent). dLLMs (orange) dominate both axes; all points sit above y=x (LJ ≥ EM = the semantically-acceptable-but-not-exact mass). Companion numbers in `lj_vs_em_table.csv`. |

## LLM-as-Judge coverage (judge = Qwen2.5-7B-Instruct)
22 of 23 registered models have LJ. **To re-run:**
- **Missing entirely:** `DiffusionGemma-26B-A4B` (predictions are all-empty → nothing to judge; fix the benchmark run first, then judge).
- **Partial (predictions truncated → re-run benchmark then re-judge):** `DiffuCoder-7B` (n=692), `DreamCoder-7B` (n=673), `CodeT5p-2B` (n=909), `CodeT5p-6B` (n=193, too small to trust). These show as hollow markers in the scatter.

## Data caveats (important)
- **RQ1 leaderboard (Table II / Fig. 5) NOT regenerated.** `results/unified_refineID/`
  is a mid-debug re-run: DiffuCoder/DreamCoder predictions are truncated
  (692/673 rows) with zeroed metrics, DiffusionGemma is all-zero, DreamOn-7B has
  1142 rows. The thesis RQ1 numbers should stand; re-run the clean benchmark
  before regenerating these.
- **DreamOn-7B-Java canvas = 800 samples** (chunks for sites 200–999; chunk 0,
  samples 0–199, is missing). Re-run `--start 0 --max-samples 200` to complete.
- **DreamOn-0.5B-Java** and **7B quickstart** are 20-sample smokes only (0.5B
  output is degenerate); excluded from the figures.
- The base DreamOn-7B bar in fig4 uses the (suspect) unified row — treat as
  indicative; re-run base cleanly for a publication number.

## Reproduce
```
results/deobfuscation_refineID/consistency_leaderboard_*.csv   # scorer output
figures/new/analysis_data.json                                 # all numbers
figures/new/deob_per_group_scored.csv                          # per identifier group
# regenerate: python experiments/score_deobfuscation_consistency.py  (numbers)
#             then the plotting in /tmp/make_figs.py
```
