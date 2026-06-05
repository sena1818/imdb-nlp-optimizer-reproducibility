#!/usr/bin/env python3
"""
Collect default and tuned rerun results into summary tables.

Primary source: individual runs/*/summary.json files (always up-to-date,
contain best_epoch_test_accuracy). Fallback: pre-aggregated JSON files
(used when runs/ directory is absent, e.g. fresh clone from GitHub).
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from experiment_utils import project_path


def load_rows_from_runs(runs_dir: Path, name_prefix: str, budget: str, phase: str):
    """Read individual summary.json files for runs matching name_prefix."""
    rows = []
    if not runs_dir.exists():
        return rows
    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir() and run_dir.name.startswith(name_prefix):
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                row = json.loads(summary_path.read_text(encoding="utf-8"))
                row.setdefault("budget", budget)
                row.setdefault("phase", phase)
                rows.append(row)
    return rows


def load_rows_from_json(path: Path, budget: str, phase: str):
    """Fallback: read pre-aggregated JSON file."""
    if not path.exists():
        return []
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row.setdefault("budget", budget)
        row.setdefault("phase", phase)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect default and tuned rerun results into summary tables.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = project_path(args.results_dir)
    runs_dir = results_dir / "runs"
    rows = []

    if runs_dir.exists():
        # Primary: read directly from individual run summaries (always complete)
        dataset = results_dir.name  # e.g. "imdb" or "ag_news"
        rows.extend(load_rows_from_runs(runs_dir, f"default_{dataset}_", "default", "default"))
        rows.extend(load_rows_from_runs(runs_dir, f"rerun_small_tuned_{dataset}_", "small_tuned", "rerun"))
    else:
        # Fallback: use pre-aggregated JSON (fresh clone, runs/ excluded from git)
        rows.extend(load_rows_from_json(results_dir / "default_summaries.json", "default", "default"))
        rows.extend(load_rows_from_json(results_dir / "small_tuned_rerun_summaries.json", "small_tuned", "rerun"))

    if not rows:
        raise SystemExit("No result summaries found. Run default baselines and tuned reruns first.")

    df = pd.DataFrame(rows)
    # Prefer best_epoch_test_accuracy (test acc at best val epoch) over
    # final_test_accuracy (test acc at last epoch). Fall back for old runs.
    if "best_epoch_test_accuracy" in df.columns:
        df["test_accuracy"] = df["best_epoch_test_accuracy"].fillna(df["final_test_accuracy"])
    else:
        df["test_accuracy"] = df["final_test_accuracy"]
    keep = [
        "dataset",
        "optimizer",
        "budget",
        "phase",
        "seed",
        "test_accuracy",
        "best_epoch_test_accuracy",
        "final_test_accuracy",
        "best_val_accuracy",
        "best_epoch_by_val_accuracy",
        "final_test_loss",
        "total_seconds",
        "seconds_per_epoch_mean",
        "run_name",
        "run_dir",
    ]
    keep = [col for col in keep if col in df.columns]
    df = df[keep].sort_values(["budget", "optimizer", "seed"])
    df.to_csv(results_dir / "all_runs.csv", index=False)

    grouped = (
        df.groupby(["dataset", "budget", "optimizer"], as_index=False)
        .agg(
            mean_test_accuracy=("test_accuracy", "mean"),
            std_test_accuracy=("test_accuracy", "std"),
            min_test_accuracy=("test_accuracy", "min"),
            max_test_accuracy=("test_accuracy", "max"),
            n=("test_accuracy", "count"),
            mean_seconds=("total_seconds", "mean"),
        )
        .sort_values(["dataset", "budget", "mean_test_accuracy"], ascending=[True, True, False])
    )
    grouped["rank_within_budget"] = grouped.groupby(["dataset", "budget"])["mean_test_accuracy"].rank(
        ascending=False, method="min"
    )
    grouped.to_csv(results_dir / "optimizer_summary.csv", index=False)

    pivot = grouped.pivot_table(
        index=["dataset", "optimizer"],
        columns="budget",
        values=["mean_test_accuracy", "std_test_accuracy", "rank_within_budget"],
    )
    pivot.to_csv(results_dir / "default_vs_tuned_pivot.csv")

    print(f"saved={results_dir / 'all_runs.csv'}")
    print(f"saved={results_dir / 'optimizer_summary.csv'}")
    print(f"saved={results_dir / 'default_vs_tuned_pivot.csv'}")


if __name__ == "__main__":
    main()
