# Rebuilt thesis figures — TU Delft blue + orange

All 13 figures from *CoReFusion: Refactoring Identifier Names with Diffusion
Language Models* rebuilt in one unified palette. Each figure is written as
`figNN_<name>.{pdf,png}` (vector PDF for LaTeX + PNG preview).

**Palette** (shared module [`corefusion_style.py`](corefusion_style.py)):
`BLUE = #0076C2`, `ORANGE = #FF8000`, plus dark/mid/light/pale of each and
neutral grays. Recurring semantics: **blue = DiffuCoder / AR-Seq2Seq /
clean / resolved-correct**, **orange = DreamCoder / dLLM / smelly / changed**,
**gray = mask / placeholder**.

To regenerate a figure: `cd <repo root> && python3 figures/rebuilt/figNN_<name>.py`

| Rebuilt file | Thesis fig | Source script / data | Reproduction |
|---|---|---|---|
| `fig01_dllm_denoising` | **Fig. 1** | `analysis/make_dllm_demo_figure.py` (self-contained) | exact — color constants remapped only |
| `fig02_dreamon_canvas` | **Fig. 2** | `analysis/make_dreamon_canvas_figure.py` (self-contained) | exact — color constants remapped only |
| `fig03_fixed_canvas` | **Fig. 3** | `analysis/make_fixed_canvas_figure.py` (self-contained) | exact — color constants remapped only |
| `fig04_rq2_example` | **Fig. 4** | `analysis/make_rq2_example_figure.py` (self-contained) | exact — color constants remapped only |
| `fig05_em_by_cardinality` | **Fig. 5** | Table III values (raw CSVs in `data/benchmark_ReFineID_*`) | exact published numbers |
| `fig06_mask_count_ablation` | **Fig. 6** | embedded numbers (raw in `results/1t5t_exp/`) | exact published numbers |
| `fig07_diffusion_steps` | **Fig. 7** | Table IV values | recreated from table — raw timings not on disk |
| `fig08_rq2_results` | **Fig. 8** | `figures/new/analysis_data.json` (`thesis.table5`, `deob_buckets`) | thesis numbers (matches paper, not June re-run) |
| `fig09_context_sensitivity` | **Fig. 9** | `results/overconfidence_localization/DiffuCoder-7B-Base_20260303_100019.csv` | reproduced from data |
| `fig10_cosine_similarity` | **Fig. 10** | `results/results_20260405_2056/edit_signals_{layer_wise,diffusion_steps}.csv` | reproduced from data |
| `fig11_umap` | **Fig. 11** | recolored from `results/results_20260405_2056/umap_{layer_wise,diffusion_steps}.png` | **recolored raster** — raw hidden-state vectors not saved, UMAP can't be recomputed without a GPU re-run |
| `fig12_entropy_change` | **Fig. 12** | `results/abc_exp/abc/token_ranking_20260312_230225.csv` (`mean_entropy_change`) | reproduced from data |
| `fig13_commitment_step` | **Fig. 13** | `results/abc_exp/abc/unmasking_order_20260312_225458.csv` | reproduced from data |

## Notes
- **Fig. 11** is the only figure that is a recolor rather than a replot: the
  per-point UMAP coordinates were never persisted (only the rendered images +
  cosine/L2 summary CSVs survive). Red→orange and blue→TU-Delft-blue were
  remapped by hue on the saved images; the PDF therefore embeds a raster, not
  vector text. Re-run `experiments/experiment_a_internal_rep.py` on a GPU and
  save `features_umap` to get a fully vector version.
- **Fig. 7** timings are recreated from Table IV (the raw per-T latency log is gone).
- **Fig. 8** uses the thesis numbers so it matches the printed paper; the June
  re-run in `figures/new/fig8_rq2_em_june.*` has different (stricter) values.
- Redundant in-figure titles were dropped from the multi-panel data plots so the
  LaTeX `\caption` is the single source of the title.
