#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

from experiment_utils import project_path


def eta_squared_anova(df, factor, response):
    grand_mean = df[response].mean()
    ss_total = ((df[response] - grand_mean) ** 2).sum()
    grouped = df.groupby(factor)[response]
    ss_factor = sum(len(values) * (values.mean() - grand_mean) ** 2 for _, values in grouped)
    if ss_total == 0:
        return 0.0, ss_factor, ss_total
    return ss_factor / ss_total, ss_factor, ss_total


def main():
    parser = argparse.ArgumentParser(
        description="Compute simple variance analysis and optional mixed-effects model."
    )
    parser.add_argument("--all-runs", default="results/all_runs.csv")
    parser.add_argument("--response", default="test_accuracy",
                        help="Column to analyse. Default: test_accuracy "
                             "(= best_epoch_test_accuracy, the test acc at the "
                             "best-val-epoch checkpoint). Use final_test_accuracy "
                             "only for comparison.")
    parser.add_argument("--output", default="results/variance_analysis.md")
    parser.add_argument("--try-lmem", action="store_true")
    args = parser.parse_args()

    all_runs = project_path(args.all_runs)
    output = project_path(args.output)
    df = pd.read_csv(all_runs)
    df = df.dropna(subset=[args.response, "optimizer", "budget", "seed"])

    lines = []
    lines.append("# Variance Analysis")
    lines.append("")
    lines.append("This is a lightweight analysis for the reproducibility study.")
    lines.append("")
    lines.append("## Descriptive Variation")
    lines.append("")

    for factor in ["optimizer", "budget", "seed"]:
        if factor in df.columns:
            eta, ss_factor, ss_total = eta_squared_anova(df, factor, args.response)
            lines.append(f"- `{factor}` eta-squared: `{eta:.4f}`")

    lines.append("")
    lines.append("## Group Summary")
    lines.append("")
    summary = (
        df.groupby(["budget", "optimizer"], as_index=False)[args.response]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    lines.append("```text")
    lines.append(summary.to_string(index=False))
    lines.append("```")

    if args.try_lmem:
        lines.append("")
        lines.append("## Optional LMEM")
        lines.append("")
        try:
            import statsmodels.formula.api as smf

            model = smf.mixedlm(
                f"{args.response} ~ C(optimizer) + C(budget) + C(optimizer):C(budget)",
                data=df,
                groups=df["seed"],
            )
            fit = model.fit(reml=True)
            lines.append("```text")
            lines.append(str(fit.summary()))
            lines.append("```")
        except Exception as exc:
            lines.append(
                "LMEM could not be fitted. This is okay for the first project version; "
                "the descriptive variance analysis above is still usable."
            )
            lines.append("")
            lines.append(f"Error: `{type(exc).__name__}: {exc}`")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
