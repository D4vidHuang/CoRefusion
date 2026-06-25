# DreamOn + DiffusionGemma — RQ1/RQ2 + Fig 4/5 runbook

老师的三个要求落地：**Fig 4 (`fig05_em_by_cardinality`) 和 Fig 5 (`fig06_mask_count_ablation`)
加所有模型**、**给 DiffusionGemma/DreamOn 重跑 mask-token 实验**、**RQ1/RQ2 跑这两个模型**，
外加 **RQ2 表 `tab:rq2_deobfuscation` + `fig8_rq2_em_june` 加新模型结果**。

关键架构事实：**DiffusionGemma-26B-A4B 是 block-AR**（prompt 后追加 canvas、逐 site 提示命名），
**无法在原位放 `[MASK]`**。所以它**只进 RQ1**；mask-token ablation 和 RQ2 去混淆对它 N/A（图/表里标脚注）。
DreamOn-7B 是 variable-canvas，三个实验全跑。

---

## 0. 现状（哪些已完成）

| 实验 | DreamOn-7B | DiffusionGemma-26B-A4B | 状态 |
|---|---|---|---|
| **RQ1**（unified all-sites EM） | ✅ 已有 | ✅ 已有 | **完成**，预测在 `results/unified_refineID/predictions/`，EM 在 `leaderboard_daic.csv`（DreamOn em_gated=8.7%，DiffusionGemma=14.7%） |
| **Fig 4** em_by_cardinality | ✅ | ✅ | **本地已重建**（数据驱动，含全部模型） |
| **mask ablation (Fig 5)** | ⏳ 待 DAIC 跑 | N/A（block-AR） | 代码+图已就绪，等 GPU |
| **RQ2 去混淆（表+fig8）** | ⏳ 待 DAIC 跑 | N/A（block-AR） | driver+表+图已就绪，等 GPU |

→ **只剩两个 GPU 作业要在 DAIC 上跑：DreamOn 的 mask ablation 和 DreamOn 的 RQ2 去混淆。**

---

## 1. 把最新代码同步到 DAIC（raw 拉取，DAIC 无 git）

本次改动的文件：
- `experiments/experiment_deobfuscation_refineID.py`（RQ2 加 DreamOn 引擎）
- `experiments/1t5t_exp/part2_dreamon_mask_ablation.py`（新，DreamOn mask ablation）
- `analysis/em_by_cardinality.py`（新，Fig4 聚合）
- `analysis/reproduce_rq2_deobfuscation.py`（加 DreamOn，自动发现 + LaTeX emitter）
- `server/jobs/DreamOn-mask-ablation.slurm`、`server/jobs/DreamOn-RQ2-deobf.slurm`（新）

**第一步（Mac）：commit + push** —— raw 拉的是 GitHub 上已 push 的文件，所以**必须先推**：
```bash
cd ~/Desktop/CoRefusion
git add -A && git commit -m "DreamOn RQ2/ablation + Fig4/5 all-models" && git push origin main
```
> 不在 `main` 分支？把下面 raw 一行里的 `main` 换成你的分支名，或 `REF=<分支> bash ...`。
> push 后等 ~30s（raw CDN 缓存；fetch 脚本已带 `?t=` 时间戳绕过）。

**第二步（DAIC，任意目录直接粘）：一行 raw 拉取**
```bash
curl -fsSL https://raw.githubusercontent.com/D4vidHuang/CoRefusion/main/server/jobs/fetch_dreamon_fig45_from_raw.sh | bash
```
这会把上面所有文件（+ 它们依赖的 `benchmark_dreamon.py`）拉到 `…/CoRefusion` 对应路径，
并 grep 校验关键改动是否到位（命中 WARN 就等几分钟重跑）。指定分支：`REF=feat/xxx` 时改用
`curl -fsSL "$RAW/server/jobs/fetch_dreamon_fig45_from_raw.sh" -o /tmp/f.sh && REF=feat/xxx bash /tmp/f.sh`。

---

## 2. 在 DAIC 上跑（直接粘贴）

```bash
cd /tudelft.net/staff-umbrella/CoReFusion/CoRefusion
mkdir -p logs

# --- 先冒烟（各 ~20 条，确认能跑、不报错）---
sbatch server/jobs/DreamOn-mask-ablation.slurm --max-samples 20
sbatch server/jobs/DreamOn-RQ2-deobf.slurm    --max-samples 20

# 看日志确认非空预测后，再提全量：
# --- 全量 ---
sbatch server/jobs/DreamOn-mask-ablation.slurm     # k=1..5 × 1000 样本  -> Fig 5
sbatch server/jobs/DreamOn-RQ2-deobf.slurm         # all-masked + target-only × 1000 -> 表 + Fig 8

# 排队/运行情况
squeue --me
```

提示：
- mask ablation 是 5 次全量 pass，较慢。要更快可拆成 5 个单 k 作业：
  `for k in 1 2 3 4 5; do sbatch server/jobs/DreamOn-mask-ablation.slurm --mask-counts $k; done`
  （脚本会把每个模型的行合并进同一个 summary）。
- 两个作业都用**主环境**（transformers 4.57.1），**不要**带 dgemma 的 PYTHONPATH override。
- DreamOn 7B bf16 在 A40(48G) 够用，无需 96G 卡。

产物：
- `results/1t5t_exp/dreamon_mask_ablation_summary.csv`
- `results/deobfuscation_refineID/DreamOn-7B_all-masked_<ts>.csv`、`..._target-only_<ts>.csv`

---

## 3. 把结果拉回 Mac

```bash
# 在 Mac 上:
cd ~/Desktop/CoRefusion
rsync -av daic:/tudelft.net/staff-umbrella/CoReFusion/CoRefusion/results/1t5t_exp/dreamon_mask_ablation_summary.csv \
          results/1t5t_exp/
rsync -av "daic:/tudelft.net/staff-umbrella/CoReFusion/CoRefusion/results/deobfuscation_refineID/DreamOn-7B_*" \
          results/deobfuscation_refineID/
```

---

## 4. 本地重建图表（拉回结果后一条龙）

```bash
cd ~/Desktop/CoRefusion
PY=./.venv-fig/bin/python      # 任意带 numpy/matplotlib 的 python 都行

# Fig 4（已可重建，无需等 DAIC）— 数据驱动，含全部模型
$PY analysis/em_by_cardinality.py            # 重算 results/unified_refineID/em_by_cardinality.csv
$PY figures/rebuilt/fig05_em_by_cardinality.py

# Fig 5（等 DreamOn ablation summary 落地后）
$PY figures/rebuilt/fig06_mask_count_ablation.py

# RQ2 表 + Fig 8（等 DreamOn 去混淆 CSV 落地后；DreamOn 自动并入）
$PY analysis/reproduce_rq2_deobfuscation.py
#   -> results/deobfuscation_refineID/reproduced/table_rq2_em.tex   (可直接粘进论文的表体)
#   -> figures/new/fig8_rq2_em_june.{pdf,png}

# 把更新后的图拷进论文仓库
bash figures/sync_to_paper.sh
```

---

## 5. 论文侧（已改好的地方）

- `conference_101719.tex`
  - 表 `tab:rq2_deobfuscation`：加了 DreamOn-7B 行（RQ1=15.2，RQ2 两格先占位 `--`，跑完用
    `table_rq2_em.tex` 替换）+ DiffusionGemma-26B-A4B 行（RQ1=33.8，RQ2 N/A + `\dagger` 脚注）。
  - Fig 4 (`fig:single_multi`) / Fig 5 (`fig:mask_ablation`) / Fig 8 caption 已更新成"含所有模型"
    并说明 DiffusionGemma 为何 N/A。
- `docs/rq2_deobfuscation_rewrite.tex` 表 `tab:rq2_em` 同步更新。

**DreamOn 的 RQ2 两个数字**是当前唯一的占位项：DAIC 跑完 + 第 4 步重生成后，
`reproduce_rq2_deobfuscation.py` 会打印并写出 `table_rq2_em.tex`，把对应两格替换即可。

---

## 6. 一句话总结改了什么

- **RQ1**：DreamOn + DiffusionGemma 早已跑完，Fig 4 现在从 `unified_refineID/predictions` **重算全部模型**
  的 per-|S| 全site EM（已验证与 leaderboard `em_gated` 完全一致）。
- **mask ablation / RQ2**：DreamOn 走 variable-canvas 路径接进了去混淆 driver 和新 ablation 脚本；
  DiffusionGemma 因 block-AR 架构在这两处标 N/A（老师确认的口径）。
