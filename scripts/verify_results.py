#!/usr/bin/env python3
"""
Sanity-check all experiment results.

Works in two modes:
  Full mode   — when results/{dataset}/runs/ exists (local machine after training)
                Checks run counts, individual summary.json fields, dropout values.
  Light mode  — when runs/ is absent (fresh GitHub clone, only CSV/JSON committed)
                Checks CSV row counts, column completeness, plots, variance analysis.
"""
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
    full_mode = runs_dir.exists()
    print(f"  mode: {'full (runs/ present)' if full_mode else 'light (fresh clone)'}")

    if full_mode:
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
                         json.loads((r / "summary.json").read_text(encoding="utf-8"))]
        check(len(missing_field) == 0,
              f"best_epoch_test_accuracy in all summaries "
              f"(missing: {missing_field[:3] if missing_field else []})")

        # --- dropout fixed at 0.5 for tune and rerun ---
        bad_dropout = []
        for r in tune_runs + rerun_runs:
            s = json.loads((r / "summary.json").read_text(encoding="utf-8"))
            if abs(s.get("dropout", 0.5) - 0.5) > 1e-6:
                bad_dropout.append((r.name, s.get("dropout")))
        check(len(bad_dropout) == 0,
              f"dropout==0.5 in all tune/rerun runs (bad: {bad_dropout[:3]})")

    # --- CSV-level checks (both modes) ---
    all_runs_path = results_dir / "all_runs.csv"
    check(all_runs_path.exists(), "all_runs.csv exists")
    if not all_runs_path.exists():
        return  # can't do further checks without CSV

    all_runs = pd.read_csv(all_runs_path)

    check(len(all_runs) == 40,
          f"all_runs.csv has 40 rows (20 default + 20 rerun), got {len(all_runs)}")
    check("test_accuracy" in all_runs.columns,
          "test_accuracy column present (= best_epoch_test_accuracy)")
    check("best_epoch_test_accuracy" in all_runs.columns,
          "best_epoch_test_accuracy column present")
    check(all_runs["test_accuracy"].isna().sum() == 0,
          f"no NaN in test_accuracy")
    check(all_runs["best_epoch_test_accuracy"].isna().sum() == 0,
          f"no NaN in best_epoch_test_accuracy")
    check(all_runs["best_val_accuracy"].isna().sum() == 0,
          f"no NaN in best_val_accuracy")

    # test_accuracy must equal best_epoch_test_accuracy for all rows
    mismatch = (all_runs["test_accuracy"] != all_runs["best_epoch_test_accuracy"]).sum()
    check(mismatch == 0,
          f"test_accuracy == best_epoch_test_accuracy for all rows (mismatch: {mismatch})")

    # --- all 4 optimizers × 2 budgets × 5 seeds ---
    for opt in ["sgd", "adam", "rmsprop", "adamw"]:
        for budget in ["default", "small_tuned"]:
            n = len(all_runs[(all_runs["optimizer"] == opt) &
                             (all_runs["budget"] == budget)])
            check(n == 5, f"{opt}/{budget}: {n} seeds == 5")

    # --- non-SGD best_epoch vs final gap ---
    merged = all_runs[all_runs["optimizer"] != "sgd"].dropna(
        subset=["best_epoch_test_accuracy", "final_test_accuracy"])
    gap = (merged["best_epoch_test_accuracy"] - merged["final_test_accuracy"]).abs()
    check(gap.max() < 0.05,
          f"non-SGD: max |best_epoch - final| < 5% (actual max={gap.max():.4f})")

    # --- SGD default ~chance ---
    sgd_def = all_runs[(all_runs["optimizer"] == "sgd") & (all_runs["budget"] == "default")]
    expected_chance = {"imdb": 0.50, "ag_news": 0.25}[dataset]
    check(abs(sgd_def["test_accuracy"].mean() - expected_chance) < 0.01,
          f"SGD default ~chance ({sgd_def['test_accuracy'].mean():.3f})")

    # --- output files exist ---
    for plot in ["accuracy_by_optimizer_budget.png", "rank_changes.png"]:
        check((results_dir / "plots" / plot).exists(), f"plot exists: {plot}")
    check((results_dir / "variance_analysis.md").exists(), "variance_analysis.md exists")
    check((results_dir / "optimizer_summary.csv").exists(), "optimizer_summary.csv exists")


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
