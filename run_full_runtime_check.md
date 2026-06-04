# Full Runtime Check: AG News vs IMDB

This file is for deciding whether the full project should run locally, on the RTX 4060 8GB, or on a rented GPU.

## What To Measure First

Run one complete TextCNN training on IMDB with Adam. This gives a realistic upper bound for the slower dataset.

Recommended first full run:

```bash
conda activate sci_comp
cd /Users/sena/Desktop/Heidelberg_SciComp/26SS/Reproducible_ML/reproducibility_study
python scripts/train_textcnn.py --dataset imdb --optimizer adam --epochs 5 --batch-size 64 --max-len 256 --device auto --run-name imdb_textcnn_adam_full_runtime
```

On Windows, use the same command after `cd` into the copied project folder:

```powershell
conda activate sci_comp
python scripts/train_textcnn.py --dataset imdb --optimizer adam --epochs 5 --batch-size 64 --max-len 256 --device auto --run-name imdb_textcnn_adam_full_runtime
```

The output summary is saved to:

```text
results/runs/imdb_textcnn_adam_full_runtime/summary.json
```

Per-epoch metrics are saved to:

```text
results/runs/imdb_textcnn_adam_full_runtime/metrics.csv
```

## Optional AG News Runtime Check

Run the same pipeline on AG News:

```bash
python scripts/train_textcnn.py --dataset ag_news --optimizer adam --epochs 5 --batch-size 64 --max-len 128 --device auto --run-name ag_news_textcnn_adam_full_runtime
```

AG News should normally be faster because the texts are shorter.

## Expected Runtime Before Running

These are rough planning estimates, not measured results.

| machine | IMDB + TextCNN + Adam, 5 epochs | comment |
|---|---:|---|
| RTX 4060 8GB | about 10-40 minutes | likely enough for the project |
| Apple Silicon MPS | about 20-90 minutes | possible, but less predictable |
| CPU only | about 1-4 hours | not ideal for many repeated runs |

The full reproducibility study multiplies this cost by many runs. If one IMDB Adam run takes 20 minutes, then 60-80 runs would be too expensive unless we reduce epochs, use AG News as the main dataset, or rent/borrow a GPU.

## Decision Rule

Use IMDB as a main dataset only if one full Adam run is reasonably fast:

- good: under 20 minutes
- acceptable: 20-40 minutes
- risky: over 40 minutes

If IMDB is slow, use:

```text
AG News = full experiment
IMDB = smaller confirmation experiment
```

This still gives a nice story about task dependence without making the project too heavy.
