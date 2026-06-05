#!/usr/bin/env python3
"""
Post-hoc patch: add best_epoch_test_accuracy to summary.json files that
were created before this field existed. Reads metrics.csv for each run,
finds the epoch with the highest val_accuracy, and records the test_accuracy
at that epoch. Safe to re-run: skips runs that already have the field.
"""
import json
import pathlib

import pandas as pd

from experiment_utils import project_path


def patch_run(run_dir: pathlib.Path) -> bool:
    summary_path = run_dir / "summary.json"
    metrics_path = run_dir / "metrics.csv"
    if not summary_path.exists() or not metrics_path.exists():
        return False

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if "best_epoch_test_accuracy" in summary:
        return False  # already patched

    metrics = pd.read_csv(metrics_path)
    best_row = metrics.loc[metrics["val_accuracy"].idxmax()]
    summary["best_epoch_test_accuracy"] = float(best_row["test_accuracy"])
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return True


def main():
    for dataset in ("imdb", "ag_news"):
        runs_dir = project_path(f"results/{dataset}/runs")
        if not runs_dir.exists():
            continue
        patched = 0
        skipped = 0
        for run_dir in sorted(runs_dir.iterdir()):
            if run_dir.is_dir():
                if patch_run(run_dir):
                    patched += 1
                else:
                    skipped += 1
        print(f"{dataset}: patched={patched}  already_ok={skipped}")


if __name__ == "__main__":
    main()
