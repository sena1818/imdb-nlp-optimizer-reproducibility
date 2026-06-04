#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from experiment_utils import PROJECT_ROOT

cache_dir = PROJECT_ROOT / ".cache"
mpl_cache_dir = PROJECT_ROOT / ".matplotlib_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

import matplotlib.pyplot as plt
import pandas as pd

from experiment_utils import project_path


def plot_accuracy_bars(summary, output_dir):
    budgets = list(summary["budget"].drop_duplicates())
    optimizers = list(summary["optimizer"].drop_duplicates())
    x = range(len(optimizers))
    width = 0.8 / max(1, len(budgets))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, budget in enumerate(budgets):
        subset = summary[summary["budget"] == budget].set_index("optimizer").reindex(optimizers)
        positions = [value + (i - (len(budgets) - 1) / 2) * width for value in x]
        ax.bar(
            positions,
            subset["mean_test_accuracy"],
            width=width,
            yerr=subset["std_test_accuracy"],
            capsize=4,
            label=budget,
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(optimizers)
    ax.set_ylabel("Mean test accuracy")
    ax.set_title("Optimizer performance with seed variance")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_by_optimizer_budget.png", dpi=200)
    plt.close(fig)


def plot_rank_changes(summary, output_dir):
    pivot = summary.pivot(index="optimizer", columns="budget", values="rank_within_budget")
    if pivot.shape[1] < 2:
        return

    budgets = list(pivot.columns)
    fig, ax = plt.subplots(figsize=(7, 5))
    for optimizer, row in pivot.iterrows():
        ax.plot(range(len(budgets)), row[budgets], marker="o", label=optimizer)
        ax.text(len(budgets) - 1 + 0.03, row[budgets[-1]], optimizer, va="center")

    ax.set_xticks(range(len(budgets)))
    ax.set_xticklabels(budgets)
    ax.invert_yaxis()
    ax.set_ylabel("Rank, lower is better")
    ax.set_title("Ranking changes across tuning budgets")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "rank_changes.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Create plots from collected optimizer summaries.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    results_dir = project_path(args.results_dir)
    summary_path = results_dir / "optimizer_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}. Run scripts/collect_results.py first.")

    output_dir = results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    plot_accuracy_bars(summary, output_dir)
    plot_rank_changes(summary, output_dir)
    print(f"saved_plots={output_dir}")


if __name__ == "__main__":
    main()
