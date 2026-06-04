# Optimizer Reproducibility on NLP Text Classification

This repository contains a small reproducibility study inspired by:

> Descending through a Crowded Valley: Benchmarking Deep Learning Optimizers

The original paper studies how optimizer comparisons can change under different workloads, tuning budgets, learning-rate schedules, and random seeds. This project transfers the core idea to NLP text classification.

## Research Question

Do optimizer rankings remain stable on an NLP text classification task, or do they change with default hyperparameters, small-budget random search, and repeated random seeds?

## Study Design

Main setup:

- datasets: AG News or IMDB
- model: TextCNN
- optimizers: SGD, Adam, RMSProp, AdamW
- conditions: default settings and small random-search tuning budget
- final evaluation: best tuned configurations rerun across multiple seeds

This is intentionally smaller than the original benchmark. The goal is not to reproduce all 15 optimizers and all 8 tasks, but to test the same reproducibility mechanism in an NLP setting.

## Repository Layout

```text
configs/
  ag_news_textcnn.json      # full AG News experiment config
  imdb_textcnn.json         # full IMDB experiment config
scripts/
  train_textcnn.py          # single TextCNN training run
  run_default_baselines.py  # default optimizer baselines across seeds
  run_random_search.py      # small-budget random search
  run_best_seed_reruns.py   # multi-seed reruns of selected best configs
  collect_results.py        # aggregate run summaries
  analyze_variance.py       # simple variance analysis, optional LMEM
  make_plots.py             # result plots
project_plan.md             # project planning notes
run_experiment_workflow.md  # full workflow
run_full_runtime_check.md   # runtime check instructions
```

## Setup

Create or activate a Python environment with PyTorch installed.

```bash
python -m pip install -r requirements.txt
```

Check whether CUDA is available:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

On Windows with an RTX 4060, make sure the installed PyTorch build supports CUDA. If `torch.cuda.is_available()` prints `False`, install the CUDA-enabled PyTorch package from the official PyTorch instructions.

## Runtime Check

Before running the full study, measure one complete IMDB run:

```bash
python scripts/train_textcnn.py --dataset imdb --optimizer adam --epochs 5 --batch-size 64 --max-len 256 --device auto --run-name imdb_textcnn_adam_full_runtime
```

Outputs are written to:

```text
results/runs/imdb_textcnn_adam_full_runtime/
```

Use the runtime to decide the final scope:

- under 20 minutes: IMDB is feasible as a main dataset
- 20-40 minutes: IMDB is feasible, but use a reduced study if needed
- over 40 minutes: use AG News as the main dataset and IMDB as a smaller confirmation experiment

## Full AG News Workflow

```bash
python scripts/run_default_baselines.py --config configs/ag_news_textcnn.json
python scripts/run_random_search.py --config configs/ag_news_textcnn.json
python scripts/run_best_seed_reruns.py --config configs/ag_news_textcnn.json --best-configs results/ag_news/best_configs_small_tuned.json
python scripts/collect_results.py --results-dir results/ag_news
python scripts/analyze_variance.py --all-runs results/ag_news/all_runs.csv --output results/ag_news/variance_analysis.md
python scripts/make_plots.py --results-dir results/ag_news
```

For IMDB, replace the config and results directory:

```bash
python scripts/run_default_baselines.py --config configs/imdb_textcnn.json
python scripts/run_random_search.py --config configs/imdb_textcnn.json
python scripts/run_best_seed_reruns.py --config configs/imdb_textcnn.json --best-configs results/imdb/best_configs_small_tuned.json
python scripts/collect_results.py --results-dir results/imdb
python scripts/analyze_variance.py --all-runs results/imdb/all_runs.csv --output results/imdb/variance_analysis.md
python scripts/make_plots.py --results-dir results/imdb
```

## Expected Outputs

The workflow creates:

- `all_runs.csv`: all default and rerun results
- `optimizer_summary.csv`: mean/std/rank per optimizer and budget
- `default_vs_tuned_pivot.csv`: default versus tuned comparison
- `variance_analysis.md`: descriptive variance analysis and optional LMEM
- `plots/accuracy_by_optimizer_budget.png`
- `plots/rank_changes.png`

## Notes

The tuning phase intentionally uses one tuning seed and then reruns selected best configurations across multiple seeds. This mirrors a practical setting and allows the study to discuss lucky tuning, seed variance, and conditional optimizer rankings.
