import csv
import json
import math
import random
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train_textcnn.py"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def project_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def base_train_args(config):
    args = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--dataset",
        config["dataset"],
        "--epochs",
        str(config["epochs"]),
        "--batch-size",
        str(config["batch_size"]),
        "--max-len",
        str(config["max_len"]),
        "--max-train",
        str(config.get("max_train", -1)),
        "--max-test",
        str(config.get("max_test", -1)),
        "--val-fraction",
        str(config.get("val_fraction", 0.1)),
        "--data-seed",
        str(config.get("data_seed", 13)),
        "--device",
        config.get("device", "auto"),
        "--num-workers",
        str(config.get("num_workers", 0)),
    ]
    return args


def train_command(config, optimizer, seed, params, run_name, output_dir):
    cmd = base_train_args(config)
    cmd.extend(
        [
            "--optimizer",
            optimizer,
            "--seed",
            str(seed),
            "--lr",
            str(params.get("lr", 1e-3)),
            "--weight-decay",
            str(params.get("weight_decay", 0.0)),
            "--momentum",
            str(params.get("momentum", 0.0)),
            "--dropout",
            str(params.get("dropout", 0.5)),
            "--output-dir",
            str(output_dir),
            "--run-name",
            run_name,
        ]
    )
    for optional in ("embed_dim", "filters", "max_vocab", "min_freq", "grad_clip"):
        if optional in config:
            flag = "--" + optional.replace("_", "-")
            cmd.extend([flag, str(config[optional])])
    return cmd


def run_command(cmd, dry_run=False):
    printable = " ".join(str(part) for part in cmd)
    print(printable)
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return completed.returncode


def read_summary(run_dir):
    return load_json(Path(run_dir) / "summary.json")


def sample_from_spec(spec, rng):
    kind = spec[0]
    if kind == "fixed":
        return spec[1]
    if kind == "choice":
        return rng.choice(spec[1])
    if kind == "uniform":
        return rng.uniform(float(spec[1]), float(spec[2]))
    if kind == "loguniform":
        low = math.log(float(spec[1]))
        high = math.log(float(spec[2]))
        return math.exp(rng.uniform(low, high))
    if kind == "zero_or_loguniform":
        low = math.log(float(spec[1]))
        high = math.log(float(spec[2]))
        zero_probability = float(spec[3])
        if rng.random() < zero_probability:
            return 0.0
        return math.exp(rng.uniform(low, high))
    raise ValueError(f"Unknown search-space spec: {spec}")


def sample_params(search_space, rng):
    return {name: sample_from_spec(spec, rng) for name, spec in search_space.items()}


def metric_for_selection(summary):
    return float(summary["best_val_accuracy"])
