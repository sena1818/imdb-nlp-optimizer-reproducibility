# Experiment Workflow

Step-by-step guide for the full reproducibility study.

## 0. Environment

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

---

## Quick Start (one command)

```bash
python scripts/run_all_imdb.py       # ~55 min on RTX 4060
python scripts/run_all_ag_news.py    # ~130 min on RTX 4060
```

Both scripts run all four phases and skip completed runs on restart.
Use `--skip-to {baselines,random_search,reruns,plots}` to resume from any phase.
Use `--dry-run` to print commands without executing.

---

## Step-by-Step

### Phase 1 — Default Baselines (4 optimizers × 5 seeds = 20 runs)

```bash
python scripts/run_default_baselines.py --config configs/imdb_textcnn.json
```

Output: `results/imdb/default_summaries.{csv,json}`

### Phase 2 — Random Search (4 optimizers × 10 trials = 40 runs)

Dropout is fixed at 0.5. Only lr, weight_decay, and momentum are searched.
HP sampling is controlled by `hp_sampling_seed` (2026, recorded in config).

```bash
python scripts/run_random_search.py --config configs/imdb_textcnn.json
```

Output: `results/imdb/random_search_trials.{csv,json}`, `best_configs_small_tuned.json`

### Phase 3 — Multi-Seed Reruns (4 optimizers × 5 seeds = 20 runs)

```bash
python scripts/run_best_seed_reruns.py \
    --config configs/imdb_textcnn.json \
    --best-configs results/imdb/best_configs_small_tuned.json
```

Output: `results/imdb/small_tuned_rerun_summaries.{csv,json}`

### Phase 4 — Analysis and Plots

```bash
python scripts/collect_results.py  --results-dir results/imdb
python scripts/analyze_variance.py --all-runs results/imdb/all_runs.csv \
    --output results/imdb/variance_analysis.md
python scripts/make_plots.py       --results-dir results/imdb
```

Optional LMEM:
```bash
pip install statsmodels
python scripts/analyze_variance.py --all-runs results/imdb/all_runs.csv \
    --output results/imdb/variance_analysis.md --try-lmem
```

### Verify

```bash
python scripts/verify_results.py
```

---

## For AG News

Substitute `configs/imdb_textcnn.json` → `configs/ag_news_textcnn.json`
and `results/imdb` → `results/ag_news`, or simply:

```bash
python scripts/run_all_ag_news.py
```

---

## Config Parameters

| Parameter | IMDB | AG News | Notes |
|-----------|------|---------|-------|
| `max_len` | 512 | 128 | 512 covers ~92% of IMDB reviews |
| `epochs` | 5 | 5 | |
| `batch_size` | 64 | 64 | |
| `seeds` | [101..105] | [101..105] | Phase 1 and Phase 3 |
| `tuning_seed` | 101 | 101 | Training seed used in Phase 2 |
| `hp_sampling_seed` | 2026 | 2026 | Controls HP draws in Phase 2 |
| `tuning_trials_per_optimizer` | 10 | 10 | |

---

## Metric Definitions

| Field | Meaning |
|-------|---------|
| `best_val_accuracy` | Best validation accuracy across all epochs |
| `best_epoch_by_val_accuracy` | Epoch index where val was highest |
| `best_epoch_test_accuracy` | **Primary metric**: test acc at the best-val epoch |
| `final_test_accuracy` | Test acc at last epoch (kept for reference) |

`best_epoch_test_accuracy` is used because model selection is based on
validation performance, so test accuracy should be evaluated at the same
checkpoint. Reporting the final epoch would penalise optimisers whose
models degrade slightly after their peak — most visible with SGD.

---

## Scientific Questions

1. Does the default optimizer ranking differ from the tuned ranking?
2. How large is seed variance for each optimizer and condition?
3. Does the tuning gain exceed seed-level variation?
4. Is the ranking dataset-dependent (IMDB vs AG News)?
5. Are conclusions conditional on tuning budget?
