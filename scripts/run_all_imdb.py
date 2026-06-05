#!/usr/bin/env python3
"""
Run the full three-phase experiment pipeline on IMDB.

Phases:
  1. Default baselines  — 4 optimizers x 5 seeds (20 runs)
  2. Random search      — 4 optimizers x 10 trials (40 runs)
  3. Best-config reruns — 4 optimizers x 5 seeds   (20 runs)
  4. Collect results and generate plots

Usage:
    python scripts/run_all_imdb.py
    python scripts/run_all_imdb.py --dry-run
    python scripts/run_all_imdb.py --skip-to random_search
    python scripts/run_all_imdb.py --skip-to reruns
    python scripts/run_all_imdb.py --skip-to plots
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "imdb_textcnn.json"
PYTHON = sys.executable

PHASES = ["baselines", "random_search", "reruns", "plots"]


def run(cmd, dry_run=False):
    printable = " ".join(str(p) for p in cmd)
    print(f"\n>>> {printable}\n")
    if dry_run:
        return
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Command failed with exit code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="Full IMDB experiment pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument(
        "--skip-to",
        choices=PHASES,
        default="baselines",
        help="Skip earlier phases and start from this phase.",
    )
    args = parser.parse_args()

    start_idx = PHASES.index(args.skip_to)
    results_dir = PROJECT_ROOT / "results" / "imdb"
    best_configs = results_dir / "best_configs_small_tuned.json"

    if start_idx <= PHASES.index("baselines"):
        print("=" * 60)
        print("PHASE 1: Default baselines (4 optimizers x 5 seeds)")
        print("=" * 60)
        run([PYTHON, "scripts/run_default_baselines.py", "--config", str(CONFIG)], args.dry_run)

    if start_idx <= PHASES.index("random_search"):
        print("=" * 60)
        print("PHASE 2: Random search (4 optimizers x 10 trials)")
        print("=" * 60)
        run([PYTHON, "scripts/run_random_search.py", "--config", str(CONFIG)], args.dry_run)

    if start_idx <= PHASES.index("reruns"):
        print("=" * 60)
        print("PHASE 3: Best-config multi-seed reruns (4 optimizers x 5 seeds)")
        print("=" * 60)
        run(
            [
                PYTHON,
                "scripts/run_best_seed_reruns.py",
                "--config", str(CONFIG),
                "--best-configs", str(best_configs),
            ],
            args.dry_run,
        )

    if start_idx <= PHASES.index("plots"):
        print("=" * 60)
        print("PHASE 4: Collect results, variance analysis, and plots")
        print("=" * 60)
        run([PYTHON, "scripts/collect_results.py", "--results-dir", str(results_dir)], args.dry_run)
        run([PYTHON, "scripts/analyze_variance.py",
             "--all-runs", str(results_dir / "all_runs.csv"),
             "--output", str(results_dir / "variance_analysis.md")], args.dry_run)
        run([PYTHON, "scripts/make_plots.py", "--results-dir", str(results_dir)], args.dry_run)

    print("\nDone. Results in:", results_dir)


if __name__ == "__main__":
    main()
