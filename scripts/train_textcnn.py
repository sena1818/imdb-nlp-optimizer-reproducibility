#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
DATASET_IDS = {
    "ag_news": "fancyzhx/ag_news",
    "imdb": "stanfordnlp/imdb",
}
NUM_CLASSES = {
    "ag_news": 4,
    "imdb": 2,
}


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def set_seed(seed, deterministic=False):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset_splits(dataset_name):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it in your conda environment with "
            "`python -m pip install datasets tqdm`."
        ) from exc

    dataset_id = DATASET_IDS[dataset_name]
    ds = load_dataset(dataset_id)
    return ds["train"], ds["test"], NUM_CLASSES[dataset_name], dataset_id


def to_examples(split):
    examples = []
    for ex in split:
        examples.append({"text": ex["text"], "label": int(ex["label"])})
    return examples


def maybe_subset(examples, max_items, seed):
    if max_items is None or max_items < 0 or max_items >= len(examples):
        return list(examples)
    indices = list(range(len(examples)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    return [examples[i] for i in indices[:max_items]]


def split_train_validation(examples, val_fraction, seed):
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("--val-fraction must be between 0 and 0.5")
    indices = list(range(len(examples)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    val_size = max(1, int(round(len(indices) * val_fraction)))
    val_idx = set(indices[:val_size])
    train = [ex for i, ex in enumerate(examples) if i not in val_idx]
    val = [ex for i, ex in enumerate(examples) if i in val_idx]
    return train, val


def build_vocab(examples, max_vocab, min_freq):
    counts = Counter()
    for ex in examples:
        counts.update(tokenize(ex["text"]))

    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counts.most_common(max_vocab - len(vocab)):
        if freq < min_freq:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_examples(examples, vocab, max_len):
    encoded = []
    labels = []
    unk_id = vocab["<unk>"]
    pad_id = vocab["<pad>"]

    for ex in examples:
        ids = [vocab.get(token, unk_id) for token in tokenize(ex["text"])[:max_len]]
        if len(ids) < max_len:
            ids.extend([pad_id] * (max_len - len(ids)))
        encoded.append(ids)
        labels.append(ex["label"])

    return torch.tensor(encoded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        embed_dim=128,
        filters=128,
        kernel_sizes=(3, 4, 5),
        dropout=0.5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, filters, kernel_size=k) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernel_sizes), num_classes)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            hidden = F.relu(conv(embedded))
            pooled.append(F.max_pool1d(hidden, kernel_size=hidden.size(2)).squeeze(2))
        features = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(features))


def make_optimizer(name, parameters, lr, weight_decay, momentum):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=momentum)
    raise ValueError(f"Unsupported optimizer: {name}")


def make_loader(x, y, batch_size, shuffle, device, num_workers):
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_metrics_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_accuracy",
        "test_loss",
        "test_accuracy",
        "epoch_seconds",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train(args):
    set_seed(args.seed, deterministic=args.deterministic)
    device = pick_device(args.device)

    start_total = time.perf_counter()
    train_split, test_split, num_classes, dataset_id = load_dataset_splits(args.dataset)
    all_train_examples = maybe_subset(to_examples(train_split), args.max_train, args.data_seed)
    test_examples = maybe_subset(to_examples(test_split), args.max_test, args.data_seed + 1)
    train_examples, val_examples = split_train_validation(
        all_train_examples, args.val_fraction, args.data_seed + 2
    )

    vocab = build_vocab(train_examples, args.max_vocab, args.min_freq)
    x_train, y_train = encode_examples(train_examples, vocab, args.max_len)
    x_val, y_val = encode_examples(val_examples, vocab, args.max_len)
    x_test, y_test = encode_examples(test_examples, vocab, args.max_len)

    train_loader = make_loader(x_train, y_train, args.batch_size, True, device, args.num_workers)
    val_loader = make_loader(x_val, y_val, args.batch_size, False, device, args.num_workers)
    test_loader = make_loader(x_test, y_test, args.batch_size, False, device, args.num_workers)

    model = TextCNN(
        vocab_size=len(vocab),
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        filters=args.filters,
        dropout=args.dropout,
    ).to(device)
    optimizer = make_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or (
        f"{args.dataset}_textcnn_{args.optimizer}_seed{args.seed}_{timestamp}"
    )
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(
        {
            "dataset_id": dataset_id,
            "device_resolved": str(device),
            "num_classes": num_classes,
            "train_size": len(train_examples),
            "validation_size": len(val_examples),
            "test_size": len(test_examples),
            "vocab_size": len(vocab),
            "train_batches": len(train_loader),
        }
    )
    write_json(run_dir / "config.json", config)

    print(f"run_dir={run_dir}")
    print(f"dataset={args.dataset} dataset_id={dataset_id}")
    print(f"device={device} optimizer={args.optimizer} seed={args.seed}")
    print(
        f"train={len(train_examples)} val={len(val_examples)} test={len(test_examples)} "
        f"vocab={len(vocab)} batches/epoch={len(train_loader)}"
    )

    metrics_rows = []
    best_val_accuracy = -math.inf
    best_epoch = None
    best_epoch_test_accuracy = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            seen += y.size(0)
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(train_loss=f"{loss_sum / seen:.4f}")

        epoch_seconds = time.perf_counter() - epoch_start
        train_loss = loss_sum / seen
        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        elapsed = time.perf_counter() - start_total

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "epoch_seconds": epoch_seconds,
            "elapsed_seconds": elapsed,
        }
        metrics_rows.append(row)
        write_metrics_csv(run_dir / "metrics.csv", metrics_rows)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_epoch_test_accuracy = test_metrics["accuracy"]
            if args.save_model:
                torch.save(model.state_dict(), run_dir / "best_model.pt")

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"epoch_seconds={epoch_seconds:.2f} "
            f"elapsed_seconds={elapsed:.2f}"
        )

    total_seconds = time.perf_counter() - start_total
    final = metrics_rows[-1]
    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "dataset": args.dataset,
        "dataset_id": dataset_id,
        "optimizer": args.optimizer,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "max_len": args.max_len,
        "train_size": len(train_examples),
        "validation_size": len(val_examples),
        "test_size": len(test_examples),
        "vocab_size": len(vocab),
        "total_seconds": total_seconds,
        "seconds_per_epoch_mean": sum(r["epoch_seconds"] for r in metrics_rows) / len(metrics_rows),
        "best_epoch_by_val_accuracy": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch_test_accuracy": best_epoch_test_accuracy,
        "final_test_accuracy": final["test_accuracy"],
        "final_test_loss": final["test_loss"],
    }
    write_json(run_dir / "summary.json", summary)
    print("\nSUMMARY")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a lightweight TextCNN for optimizer reproducibility experiments."
    )
    parser.add_argument("--dataset", choices=["ag_news", "imdb"], required=True)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw", "rmsprop"], default="adam")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--max-train", type=int, default=-1, help="-1 means full training split")
    parser.add_argument("--max-test", type=int, default=-1, help="-1 means full test split")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-vocab", type=int, default=50000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-seed",
        type=int,
        default=13,
        help="Controls data subset and validation split. Keep fixed across training seeds.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument("--output-dir", default="results/runs")
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
