# Running CoReFusion on DAIC (TU Delft AI cluster)

`$HOME` quota is only ~5 MB, so **every** cache is redirected onto the umbrella
share `/tudelft.net/staff-umbrella/CoReFusion`. Always `source env_daic.sh`
before doing anything.

Environment strategy (DAIC-native): use the cluster's prebuilt
`py-torch/2.5.1` module (built for `cuda/12.9`) and layer a **venv on the
umbrella** with `--system-site-packages` that only adds `transformers==4.57.1`
and the other small deps. No conda (DAIC has no conda module), no 2 GB torch
download.

## 0. Get these files onto the server
From the repo at `/tudelft.net/staff-umbrella/CoReFusion/CoRefusion`:
```bash
git pull          # if the clone tracks the same remote
```
(or scp / copy the `server/` folder over.)

## 1. Verify the environment (login node)
```bash
bash server/verify_env.sh
```
Confirms: home quota, umbrella space, `data/test.csv` present, your SLURM
**account** (required), partitions + GPU types, the `py-torch` module.

## 2. One-time setup (creates the venv on the umbrella)
```bash
kinit                              # refresh Kerberos so the umbrella stays mounted
bash server/setup_daic.sh
```
Creates `/tudelft.net/staff-umbrella/CoReFusion/venvs/corefusion` and installs
`transformers==4.57.1` + extras on top of the module torch.

SLURM facts (confirmed 2026-06-08): `--account=testusers`,
`--partition=ewi-st` (group nodes gpu[12,46-52]), GPUs `nvidia_a40` (48 GB) and
`nvidia_rtx_pro_6000` (~96 GB). **Use A40** — the module `py-torch/2.5.1` is too
old for the Blackwell RTX PRO 6000 (sm_120) and our models fit in 48 GB anyway.

## 3. Smoke test (5 samples, smallest model) — directly runnable
```bash
cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
sbatch server/smoke_test.slurm
squeue -u $USER
tail -f logs/smoke_<jobid>.out
```
Expect: model materialises into `hf_models/`, debug prints per-window
predictions, then a summary table; CSVs land in `results/dreamon_java/`.

> ⚠️ transformers 4.57.1 is much newer than the 4.46.2 the Dream/DreamOn custom
> modeling code was originally written for. The `--debug` smoke test exists to
> catch any incompatibility on 5 samples. If loading/generation breaks, the
> known-good fallback is `transformers==4.46.2` (edit `requirements.txt`,
> `rm -rf venvs/corefusion`, re-run setup).

## 4. Full run
Use `run_benchmark.slurm` (A40, 64 GB, 24 h). Args after the script name pass
straight through to the benchmark:
```bash
sbatch server/run_benchmark.slurm                          # both models, full 1000
sbatch server/run_benchmark.slurm --model dreamon-7b-Java  # just 7B
sbatch server/run_benchmark.slurm --max-samples 100        # quick partial
sbatch server/run_benchmark.slurm --hf-repo D4vidHuang/benchmark_ReFineID_DreamOn_Java
```
7B is ~15 GB bf16 → fits A40 (48 GB) comfortably. For HF upload set `HF_TOKEN`
in `env_daic.sh` first.
