# Experiment Workflow

This is the full workflow for the reproducibility study.

## 0. Environment

Use the `sci_comp` conda environment:

```bash
conda activate sci_comp
python -m pip install datasets tqdm pandas matplotlib
```

On Windows with the RTX 4060, also install a CUDA-enabled PyTorch build if the environment does not already have one. Check with:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 1. Choose Dataset in Config

Edit `experiment_config.json`.

Alternatively, use one of the ready-made configs:

```text
configs/ag_news_textcnn.json
configs/imdb_textcnn.json
```

For AG News:

```json
"dataset": "ag_news",
"max_len": 128
```

For IMDB:

```json
"dataset": "imdb",
"max_len": 256
```

If runtime is too high, temporarily reduce:

```json
"epochs": 3,
"tuning_trials_per_optimizer": 5,
"seeds": [101, 102, 103]
```

## 2. Dry Run Commands

Dry run prints commands without training:

```bash
python scripts/run_default_baselines.py --config configs/ag_news_textcnn.json --dry-run
python scripts/run_random_search.py --config configs/ag_news_textcnn.json --dry-run
```

## 3. Run Default Baselines

```bash
python scripts/run_default_baselines.py --config configs/ag_news_textcnn.json
```

This runs each optimizer with default hyperparameters across the configured seeds.

Outputs:

```text
results/ag_news/default_summaries.csv
results/ag_news/default_summaries.json
```

## 4. Run Small-Budget Random Search

```bash
python scripts/run_random_search.py --config configs/ag_news_textcnn.json
```

This uses one tuning seed and samples hyperparameters for each optimizer.

Outputs:

```text
results/ag_news/random_search_trials.csv
results/ag_news/best_configs_small_tuned.json
```

## 5. Rerun Best Tuned Configurations Across Seeds

```bash
python scripts/run_best_seed_reruns.py --config configs/ag_news_textcnn.json --best-configs results/ag_news/best_configs_small_tuned.json
```

This takes the selected best configuration per optimizer and reruns it across the configured seeds.

Outputs:

```text
results/ag_news/small_tuned_rerun_summaries.csv
results/ag_news/small_tuned_rerun_summaries.json
```

## 6. Collect Results

```bash
python scripts/collect_results.py --results-dir results/ag_news
```

Outputs:

```text
results/ag_news/all_runs.csv
results/ag_news/optimizer_summary.csv
results/ag_news/default_vs_tuned_pivot.csv
```

## 7. Variance Analysis

```bash
python scripts/analyze_variance.py --all-runs results/ag_news/all_runs.csv --output results/ag_news/variance_analysis.md
```

Optional LMEM attempt:

```bash
python -m pip install statsmodels tabulate
python scripts/analyze_variance.py --all-runs results/ag_news/all_runs.csv --output results/ag_news/variance_analysis.md --try-lmem
```

## 8. Make Plots

```bash
python scripts/make_plots.py --results-dir results/ag_news
```

Output:

```text
results/ag_news/plots/accuracy_by_optimizer_budget.png
results/ag_news/plots/rank_changes.png
```

## 9. Scientific Story

The final analysis should answer:

- Does the default optimizer ranking differ from the tuned ranking?
- How large is seed variance for each optimizer?
- Does tuning improvement exceed seed-level variation?
- Does Adam/AdamW remain a strong baseline on this NLP task?
- Are any conclusions conditional on tuning budget?

This is intentionally a smaller version of the original paper's benchmark, not a full reproduction of all 15 optimizers and 8 tasks.
