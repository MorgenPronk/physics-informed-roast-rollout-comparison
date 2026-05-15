# Physics-Informed Repair of Mechanistic ODEs: Industrial Coffee Roasting

This repository accompanies the manuscript
*Where the MLP plugs in: a position-spectrum comparison of physics-informed
repair strategies for industrial coffee roasting* (Scientific Reports
submission).

It contains the model implementations, hyperparameter-sweep driver,
final-retrain scripts, manuscript-figure builders, and LaTeX sources used to
reproduce the reported tables and figures.

## Repository layout

```
manuscript/scientific_reports/submission_latex/    LaTeX submission (main.tex, references.bib, figures/)
scripts/                                           HPO sweep, retrains, probes, figure builders
src/roaster_piml/                                  Model implementations and data pipeline
reports/manuscript_hpo/                            HPO sweep results, final retrains, priors probes, spectrum_stats
data/                                              Data placeholders (see "Data" below)
```

## Environment setup

Use a repository-local virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
```

For GPU training on an NVIDIA card, replace the default CPU torch wheel with
the official CUDA build:

```powershell
.\.venv\Scripts\python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0
```

Verify CUDA availability:

```powershell
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Data

The manuscript analyses use a 221-roast production cohort loaded from
`data/processed/roast_timeseries_p2_only.csv`. That CSV is a de-identified
export of routine production telemetry from a partner roasting facility, with
partner- and plant-identifying labels removed before release.

The cohort CSV is not committed to this repository by default
(see `.gitignore`). See the manuscript's *Data availability* section for how
to obtain it, or contact the corresponding author.

Once the cohort CSV is in `data/processed/`, all scripts below will pick it
up automatically.

## Reproducing the manuscript

The pipeline is a hyperparameter sweep, a three-seed final retrain for each
model class, optional probes for the positive-control and the `bs=8`
limitation note, and figure builders that consume the resulting JSONs.

### 1. Hyperparameter sweep (40 trials per class)

```powershell
.\.venv\Scripts\python scripts/tune_manuscript_models.py --phase sweep_blackbox
.\.venv\Scripts\python scripts/tune_manuscript_models.py --phase sweep_whitebox
.\.venv\Scripts\python scripts/tune_manuscript_models.py --phase sweep_greybox
.\.venv\Scripts\python scripts/tune_manuscript_models.py --phase sweep_multi_closure
.\.venv\Scripts\python scripts/tune_manuscript_models.py --phase sweep_residual_ff
.\.venv\Scripts\python scripts/tune_manuscript_models.py --phase sweep_residual_ff_unbounded
```

Each phase samples 40 hyperparameter configurations at seed 11, selects the
configuration with the best pooled validation rollout `R^2`, and writes
`reports/manuscript_hpo/<class>/best_config.json` plus per-trial logs.

### 2. Three-seed final retrain (seeds 11/23/37)

```powershell
.\.venv\Scripts\python scripts/tune_manuscript_models.py --phase final_eval
.\.venv\Scripts\python scripts/retrain_multi_closure_final.py
.\.venv\Scripts\python scripts/retrain_residual_ff_final.py --best-config reports/manuscript_hpo/residual_ff/best_config.json --output reports/manuscript_hpo/residual_ff_final.json
.\.venv\Scripts\python scripts/retrain_residual_ff_final.py --best-config reports/manuscript_hpo/residual_ff_unbounded/best_config.json --output reports/manuscript_hpo/residual_ff_unbounded_final.json
```

Outputs land in `reports/manuscript_hpo/` as `final_test_metrics.json`,
`multi_closure_final.json`, `residual_ff_final.json`, and
`residual_ff_unbounded_final.json`.

### 3. Probes referenced in the manuscript

```powershell
.\.venv\Scripts\python scripts/probe_priors.py        # positive-control variants (R^2 ~= 0.93)
.\.venv\Scripts\python scripts/probe_greybox_bs8.py   # bs=8 limitation note
```

### 4. Seed-11 trajectories for the representative-rollouts figure

```powershell
.\.venv\Scripts\python scripts/regenerate_seed11_trajectories.py
```

Writes `reports/manuscript_hpo/seed11_rollouts.json` containing per-roast
test trajectories for all six model classes at seed 11.

### 5. Cross-class statistics (paired Wilcoxon, TOST, per-roast Pearson)

```powershell
.\.venv\Scripts\python scripts/analyze_spectrum_statistics.py
```

Writes `reports/manuscript_hpo/spectrum_stats/spectrum_stats.{json,md}`.

### 6. Manuscript figures

```powershell
.\.venv\Scripts\python scripts/build_figure_position_spectrum.py
.\.venv\Scripts\python scripts/build_figure_per_roast_correlation.py
.\.venv\Scripts\python scripts/build_figure_representative_rollouts.py
.\.venv\Scripts\python scripts/build_roast_profile_preprocessing_schematic.py
.\.venv\Scripts\python scripts/plot_hpo_diagnostics.py
```

Outputs go to `manuscript/scientific_reports/submission_latex/figures/`.

### 7. Compile the LaTeX submission

```powershell
cd manuscript/scientific_reports/submission_latex
latexmk -pdf main.tex
```

## Key result files

After running the pipeline, the manuscript's headline numbers are in:

- `reports/manuscript_hpo/final_test_metrics.json` &mdash; mechanistic baseline, single-closure PI, residual LSTM (historical), neural baseline
- `reports/manuscript_hpo/multi_closure_final.json` &mdash; multi-closure PI
- `reports/manuscript_hpo/residual_ff_final.json` &mdash; bounded FF residual
- `reports/manuscript_hpo/residual_ff_unbounded_final.json` &mdash; unbounded FF residual
- `reports/manuscript_hpo/seed11_rollouts.json` &mdash; per-roast test trajectories for Figure 3
- `reports/manuscript_hpo/spectrum_stats/spectrum_stats.json` &mdash; pairwise Wilcoxon, TOST, and per-roast Pearson correlations

## Citation and license

- Citation metadata: `CITATION.cff`
- License: `LICENSE`
