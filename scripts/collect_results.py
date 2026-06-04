#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd

from experiment_utils import project_path


def load_rows(path, budget, phase):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    for row in rows:
        row.setdefault("budget", budget)
        row.setdefault("phase", phase)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect default and tuned rerun results into summary tables.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = project_path(args.results_dir)
    rows = []
    rows.extend(load_rows(results_dir / "default_summaries.json", "default", "default"))
    rows.extend(load_rows(results_dir / "small_tuned_rerun_summaries.json", "small_tuned", "rerun"))

    if not rows:
        raise SystemExit("No result summaries found. Run default baselines and tuned reruns first.")

    df = pd.DataFrame(rows)
    keep = [
        "dataset",
        "optimizer",
        "budget",
        "phase",
        "seed",
        "final_test_accuracy",
        "best_val_accuracy",
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
            mean_test_accuracy=("final_test_accuracy", "mean"),
            std_test_accuracy=("final_test_accuracy", "std"),
            min_test_accuracy=("final_test_accuracy", "min"),
            max_test_accuracy=("final_test_accuracy", "max"),
            n=("final_test_accuracy", "count"),
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
