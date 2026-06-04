#!/usr/bin/env python3
import argparse
import random

from experiment_utils import (
    load_json,
    metric_for_selection,
    project_path,
    read_summary,
    run_command,
    sample_params,
    train_command,
    write_csv,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Run small-budget random search for each optimizer.")
    parser.add_argument("--config", default="experiment_config.json")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(project_path(args.config))
    trials = args.trials or int(config.get("tuning_trials_per_optimizer", 10))
    tuning_seed = int(config.get("tuning_seed", config["seeds"][0]))
    output_root = project_path(config.get("output_root", "results"))
    output_dir = output_root / "runs"
    rng = random.Random(args.random_seed)

    all_trials = []
    best_configs = {}

    for optimizer in config["optimizers"]:
        best_trial = None
        for trial_idx in range(1, trials + 1):
            params = sample_params(config["search_spaces"][optimizer], rng)
            run_name = f"tune_{config['dataset']}_{optimizer}_trial{trial_idx:03d}_seed{tuning_seed}"
            run_dir = output_dir / run_name
            cmd = train_command(config, optimizer, tuning_seed, params, run_name, output_dir)
            code = run_command(cmd, dry_run=args.dry_run)
            if code != 0:
                raise SystemExit(code)

            trial_row = {
                "dataset": config["dataset"],
                "optimizer": optimizer,
                "budget": "small_tuned",
                "phase": "tuning",
                "trial": trial_idx,
                "seed": tuning_seed,
                "run_name": run_name,
                **params,
            }
            if not args.dry_run:
                summary = read_summary(run_dir)
                trial_row.update(
                    {
                        "best_val_accuracy": summary["best_val_accuracy"],
                        "final_test_accuracy": summary["final_test_accuracy"],
                        "total_seconds": summary["total_seconds"],
                        "run_dir": summary["run_dir"],
                    }
                )
                if best_trial is None or metric_for_selection(summary) > best_trial["best_val_accuracy"]:
                    best_trial = trial_row.copy()
            all_trials.append(trial_row)

        if best_trial is not None:
            best_configs[optimizer] = {
                "optimizer": optimizer,
                "selected_by": "best_val_accuracy_on_single_tuning_seed",
                "budget": "small_tuned",
                "tuning_seed": tuning_seed,
                "trial": best_trial["trial"],
                "best_val_accuracy": best_trial["best_val_accuracy"],
                "final_test_accuracy_on_tuning_seed": best_trial["final_test_accuracy"],
                "params": {
                    "lr": best_trial["lr"],
                    "weight_decay": best_trial["weight_decay"],
                    "momentum": best_trial["momentum"],
                    "dropout": best_trial["dropout"],
                },
            }

    if not args.dry_run:
        write_json(output_root / "random_search_trials.json", all_trials)
        write_csv(output_root / "random_search_trials.csv", all_trials)
        write_json(output_root / "best_configs_small_tuned.json", best_configs)
        print(f"saved={output_root / 'best_configs_small_tuned.json'}")


if __name__ == "__main__":
    main()
