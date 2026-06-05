# Optimizer Reproducibility on NLP Text Classification

A small reproducibility study inspired by:

> Descending through a Crowded Valley: Benchmarking Deep Learning Optimizers (Schneider et al., 2021)

The original paper shows that optimizer rankings are sensitive to tuning budget, learning-rate schedule, and random seed. This project transfers the core mechanism to NLP text classification.

## Research Question

Do optimizer rankings remain stable on NLP text classification, or do they shift when moving from default hyperparameters to small-budget random search and repeated random seeds?

## Study Design

| Setting | Value |
|---------|-------|
| Datasets | IMDB (sentiment, 2-class) and AG News (topic, 4-class) |
| Model | TextCNN (Kim 2014): Embedding → Conv1d × 3 → MaxPool → Linear |
| Optimizers | SGD, Adam, RMSProp, AdamW |
| Phase 1 | Default hyperparameters, 5 seeds |
| Phase 2 | Small-budget random search (10 trials, 1 tuning seed), **dropout fixed at 0.5** |
| Phase 3 | Best config from Phase 2 rerun across 5 seeds |
| Metric | Test accuracy at the epoch with best validation accuracy |

Dropout is deliberately held fixed across all search spaces so that only optimizer-specific hyperparameters (lr, weight\_decay, momentum) are compared.

## Key Results

All accuracy values are `best_epoch_test_accuracy` — test accuracy at the epoch with the best validation accuracy. Mean over 5 seeds.

### IMDB (max\_len=512, ~40 s/run on RTX 4060)

| Budget | Adam | AdamW | RMSProp | SGD |
|--------|------|-------|---------|-----|
| Default | 87.65% ② | **87.66% ①** | 86.37% ③ | 50.00% ④ |
| Tuned   | **87.84% ①** | 87.35% ② | 86.61% ③ | 82.54% ④ |

### AG News (max\_len=128, ~65 s/run on RTX 4060)

| Budget | Adam | AdamW | RMSProp | SGD |
|--------|------|-------|---------|-----|
| Default | 90.92% ③ | 90.95% ② | **91.05% ①** | 25.00% ④ |
| Tuned   | 90.51% ③ | **91.03% ①** | 90.96% ② | 88.93% ④ |

**Findings:**
- SGD with its NLP-standard default (lr=0.1) collapses to chance on both tasks. Small-budget tuning rescues it substantially, but with noticeably higher seed variance than Adam/AdamW.
- On IMDB, Adam and AdamW are nearly tied at default. After tuning, Adam edges ahead while AdamW drops — the ranking reverses.
- On AG News, RMSProp ranks first at default settings but drops to second after tuning; AdamW takes first. Rankings are dataset-dependent.
- One SGD tuned run (IMDB seed 103) drops from 81.2% at epoch 4 to 62.5% at epoch 5, illustrating why best-val-epoch accuracy matters over final-epoch.
- Variance analysis (eta-squared): IMDB — optimizer 0.554, budget 0.112, seed 0.001. AG News — optimizer 0.458, budget 0.134, seed ≈0.000. Optimizer choice dominates; seed variation is negligible for non-SGD optimizers.

## Repository Layout

```text
configs/
  imdb_textcnn.json         # IMDB experiment config (max_len=512)
  ag_news_textcnn.json      # AG News experiment config (max_len=128)
scripts/
  train_textcnn.py          # single training run
  run_default_baselines.py  # Phase 1: default configs × 5 seeds
  run_random_search.py      # Phase 2: random search (10 trials)
  run_best_seed_reruns.py   # Phase 3: best config × 5 seeds
  collect_results.py        # aggregate summaries → CSV
  analyze_variance.py       # eta-squared variance analysis, optional LMEM
  make_plots.py             # accuracy bar chart + rank-change plot
  run_all_imdb.py           # one-command pipeline for IMDB
  run_all_ag_news.py        # one-command pipeline for AG News
  patch_summaries.py        # back-fill best_epoch_test_accuracy in old runs
  verify_results.py         # sanity-check all result files
results/
  imdb/                     # summaries, CSVs, plots (runs/ excluded from git)
  ag_news/
```

## Setup

Create a conda environment and install dependencies:

```bash
conda create -n imdb_nlp python=3.11 -y
conda activate imdb_nlp
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install datasets tqdm pandas matplotlib
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## Running the Full Pipeline

### One command (recommended)

```bash
# IMDB
python scripts/run_all_imdb.py

# AG News
python scripts/run_all_ag_news.py
```

Both scripts run all four phases (baselines → random search → reruns → collect+analyze+plot) and support `--skip-to {baselines,random_search,reruns,plots}` to resume from any phase. All run scripts skip existing completed runs automatically.

### Step by step

```bash
python scripts/run_default_baselines.py --config configs/imdb_textcnn.json
python scripts/run_random_search.py     --config configs/imdb_textcnn.json
python scripts/run_best_seed_reruns.py  --config configs/imdb_textcnn.json \
    --best-configs results/imdb/best_configs_small_tuned.json
python scripts/collect_results.py  --results-dir results/imdb
python scripts/analyze_variance.py --all-runs results/imdb/all_runs.csv \
    --output results/imdb/variance_analysis.md
python scripts/make_plots.py       --results-dir results/imdb
```

### Verify results

```bash
python scripts/verify_results.py
```

## Output Files

| File | Description |
|------|-------------|
| `default_summaries.csv` | Phase 1 results (20 runs per dataset) |
| `random_search_trials.csv` | Phase 2 trial results (40 runs) |
| `small_tuned_rerun_summaries.csv` | Phase 3 results (20 runs) |
| `best_configs_small_tuned.json` | Selected best hyperparameters per optimizer |
| `all_runs.csv` | Combined Phase 1 + Phase 3 for analysis |
| `optimizer_summary.csv` | Mean/std/rank per optimizer and budget |
| `default_vs_tuned_pivot.csv` | Default vs tuned comparison table |
| `variance_analysis.md` | Eta-squared analysis, optional LMEM |
| `plots/*.png` | Accuracy bar chart and rank-change plot |

## Design Decisions

**Why fix dropout?** Dropout is a model regularization parameter, not an optimizer hyperparameter. Tuning it per optimizer would confound the comparison: a "better" result might reflect a lucky dropout value rather than a better optimizer. All runs use dropout=0.5.

**Why best-val-epoch test accuracy?** Model selection is done on the validation set. Reporting test accuracy at the best-val epoch is consistent with that selection criterion. Reporting the final epoch instead would penalize optimizers whose models happen to degrade slightly after the best checkpoint — most visible with SGD.

**Why two seeds (hp\_sampling\_seed and tuning\_seed)?** `tuning_seed` controls which training seed is used during the random search phase. `hp_sampling_seed` (default 2026) controls the random draw of hyperparameter values. Both are recorded in the config for full reproducibility.

## Notes

This is intentionally a smaller study than the original benchmark (4 optimizers, 2 datasets, 5 seeds). The goal is to demonstrate the core reproducibility mechanism — ranking instability across tuning budgets and seeds — in an NLP setting.
