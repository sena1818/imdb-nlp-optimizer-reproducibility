# Pilot Runtime Check

This pilot compares the runtime of the same lightweight TextCNN pipeline on AG News and IMDB.

The goal is not yet to produce final experimental results. The goal is to estimate:

- whether the pipeline trains correctly
- runtime per epoch
- total runtime per run
- whether AG News or IMDB is feasible for the full reproducibility study

## Environment

Use the `sci_comp` conda environment.

```bash
conda run -n sci_comp python -m pip install datasets tqdm
```

The script uses HuggingFace `datasets`, so the first run needs internet access to download AG News or IMDB unless the dataset is already cached.

## Recommended First Commands

Run AG News pilot:

```bash
conda run -n sci_comp python scripts/train_textcnn.py --dataset ag_news --epochs 1 --max-train 8000 --max-test 2000 --batch-size 64 --max-len 128 --run-name ag_news_pilot
```

Run IMDB pilot:

```bash
conda run -n sci_comp python scripts/train_textcnn.py --dataset imdb --epochs 1 --max-train 8000 --max-test 2000 --batch-size 64 --max-len 256 --run-name imdb_pilot
```

## Suggested Interpretation

AG News is expected to be faster because texts are shorter. IMDB is expected to be slower because reviews are longer and normally need a larger `max-len`.

If IMDB is less than about 3x slower than AG News, it is still realistic for the final study. If it is much slower, keep AG News as the main task and use IMDB only as a smaller optional extension.

## Runtime Results

Measured on RTX 4060, full training split, 5 epochs.

| dataset | device | train size | max\_len | batch size | epochs | total time | sec/epoch | test accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AG News | RTX 4060 (CUDA) | 108,000 | 128 | 64 | 5 | ~65 s | ~11 s | ~90.7% |
| IMDB | RTX 4060 (CUDA) | 22,500 | 512 | 64 | 5 | ~40 s | ~4 s | ~87.7% |

IMDB is fast enough to be the main dataset. Both datasets were used in the full study.
max\_len was set to 512 for IMDB (covers 91.7% of reviews) and 128 for AG News (texts are short).
