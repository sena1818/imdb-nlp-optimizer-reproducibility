#!/usr/bin/env python3
"""Sanity-check all experiment results before final submission."""
import json
import pathlib
import sys

import pandas as pd

from experiment_utils import project_path

ISSUES = []


def check(condition, msg):
    if not condition:
        ISSUES.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  ok:   {msg}")


def verify_dataset(dataset):
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset}")
    print(f"{'='*50}")

    results_dir = project_path(f"results/{dataset}")
    runs_dir = results_dir / "runs"

    # --- run counts ---
    runs = list(runs_dir.iterdir())
    default_runs = [r for r in runs if r.name.startswith("default_")]
    tune_runs    = [r for r in runs if r.name.startswith("tune_")]
    rerun_runs   = [r for r in runs if r.name.startswith("rerun_")]

    check(len(default_runs) == 20, f"default runs: {len(default_runs)} == 20")
    check(len(tune_runs)    == 40, f"tune runs:    {len(tune_runs)} == 40")
    check(len(rerun_runs)   == 20, f"rerun runs:   {len(rerun_runs)} == 20")

    # --- all summary.json have best_epoch_test_accuracy ---
    missing_field = [r.name for r in runs
                     if "best_epoch_test_accuracy" not in
                     json.load(open(r / "summary.json"))]
    check(len(missing_field) == 0,
          f"best_epoch_test_accuracy present in all summaries "
          f"(missing: {missing_field[:3] if missing_field else []})")

    # --- dropout fixed at 0.5 for tune and rerun ---
    bad_dropout = []
    for r in tune_runs + rerun_runs:
        s = json.load(open(r / "summary.json"))
        if abs(s.get("dropout", 0.5) - 0.5) > 1e-6:
            bad_dropout.append((r.name, s.get("dropout")))
    check(len(bad_dropout) == 0,
          f"dropout==0.5 in all tune/rerun runs (bad: {bad_dropout[:3]})")

    # --- no NaN in key metrics ---
    all_runs = pd.read_csv(results_dir / "all_runs.csv")
    check(all_runs["test_accuracy"].isna().sum() == 0,
          f"no NaN in test_accuracy ({len(all_runs)} rows)")
    check(all_runs["best_val_accuracy"].isna().sum() == 0,
          f"no NaN in best_val_accuracy")

    # --- all 4 optimizers × 2 budgets present ---
    for opt in ["sgd", "adam", "rmsprop", "adamw"]:
        for budget in ["default", "small_tuned"]:
            n = len(all_runs[(all_runs["optimizer"] == opt) &
                             (all_runs["budget"] == budget)])
            check(n == 5, f"{opt}/{budget}: {n} seeds == 5")

    # --- For non-SGD optimizers, best_epoch and final should be close.
    #     SGD can degrade sharply after the best epoch (e.g. seed 103 IMDB:
    #     best_epoch=81.2% but epoch5=62.5%), which is itself a key finding. ---
    merged = all_runs[all_runs["optimizer"] != "sgd"].dropna(
        subset=["best_epoch_test_accuracy", "final_test_accuracy"])
    gap = (merged["best_epoch_test_accuracy"] - merged["final_test_accuracy"]).abs()
    check(gap.max() < 0.05,
          f"non-SGD: max |best_epoch - final| < 5% (actual max={gap.max():.4f})")

    # --- SGD default should be ~chance ---
    sgd_def = all_runs[(all_runs["optimizer"]=="sgd") & (all_runs["budget"]=="default")]
    expected_chance = {"imdb": 0.50, "ag_news": 0.25}[dataset]
    check(abs(sgd_def["test_accuracy"].mean() - expected_chance) < 0.01,
          f"SGD default ~chance ({sgd_def['test_accuracy'].mean():.3f})")

    # --- plots exist ---
    for plot in ["accuracy_by_optimizer_budget.png", "rank_changes.png"]:
        check((results_dir / "plots" / plot).exists(), f"plot exists: {plot}")

    # --- variance analysis exists ---
    check((results_dir / "variance_analysis.md").exists(), "variance_analysis.md exists")


def main():
    for ds in ("imdb", "ag_news"):
        verify_dataset(ds)

    print(f"\n{'='*50}")
    if ISSUES:
        print(f"ISSUES FOUND ({len(ISSUES)}):")
        for i in ISSUES:
            print(f"  - {i}")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
