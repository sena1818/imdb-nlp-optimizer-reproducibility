#!/usr/bin/env python3
import argparse

from experiment_utils import (
    load_json,
    project_path,
    read_summary,
    run_command,
    train_command,
    write_csv,
    write_json,
)


def main():
    parser = argparse.ArgumentParser(description="Rerun selected tuned configurations across seeds.")
    parser.add_argument("--config", default="experiment_config.json")
    parser.add_argument("--best-configs", default="results/best_configs_small_tuned.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_json(project_path(args.config))
    best_configs = load_json(project_path(args.best_configs))
    output_root = project_path(config.get("output_root", "results"))
    output_dir = output_root / "runs"
    summaries = []

    for optimizer in config["optimizers"]:
        selected = best_configs[optimizer]
        params = selected["params"]
        for seed in config["seeds"]:
            run_name = f"rerun_small_tuned_{config['dataset']}_{optimizer}_seed{seed}"
            run_dir = output_dir / run_name
            if not args.dry_run and (run_dir / "summary.json").exists():
                print(f"skip (already done): {run_name}")
                summary = read_summary(run_dir)
                summary.update({"budget": "small_tuned", "phase": "rerun",
                                 "optimizer": optimizer, "selected_trial": selected["trial"]})
                summaries.append(summary)
                continue
            cmd = train_command(config, optimizer, seed, params, run_name, output_dir)
            code = run_command(cmd, dry_run=args.dry_run)
            if code != 0:
                raise SystemExit(code)
            if not args.dry_run:
                summary = read_summary(run_dir)
                summary.update(
                    {
                        "budget": "small_tuned",
                        "phase": "rerun",
                        "optimizer": optimizer,
                        "selected_trial": selected["trial"],
                    }
                )
                summaries.append(summary)

    if summaries:
        write_json(output_root / "small_tuned_rerun_summaries.json", summaries)
        write_csv(output_root / "small_tuned_rerun_summaries.csv", summaries)
        print(f"saved={output_root / 'small_tuned_rerun_summaries.csv'}")


if __name__ == "__main__":
    main()
