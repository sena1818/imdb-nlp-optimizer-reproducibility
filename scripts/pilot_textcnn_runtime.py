#!/usr/bin/env python3
import argparse
import json
import random
import re
import time
from collections import Counter
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


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_hf_dataset(name):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with `python3 -m pip install -r requirements.txt`."
        ) from exc

    if name == "ag_news":
        ds = load_dataset("fancyzhx/ag_news")
        return ds["train"], ds["test"], 4
    if name == "imdb":
        ds = load_dataset("stanfordnlp/imdb")
        return ds["train"], ds["test"], 2
    raise ValueError(f"Unsupported dataset: {name}")


def take_subset(split, n, seed):
    n = min(n, len(split))
    indices = list(range(len(split)))
    rnd = random.Random(seed)
    rnd.shuffle(indices)
    indices = indices[:n]
    return [split[i] for i in indices]


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
    unk = vocab["<unk>"]
    pad = vocab["<pad>"]

    for ex in examples:
        ids = [vocab.get(tok, unk) for tok in tokenize(ex["text"])[:max_len]]
        if len(ids) < max_len:
            ids.extend([pad] * (max_len - len(ids)))
        encoded.append(ids)
        labels.append(int(ex["label"]))

    return torch.tensor(encoded, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes, embed_dim=100, filters=96, kernel_sizes=(3, 4, 5), dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            nn.Conv1d(embed_dim, filters, kernel_size=k) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        features = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, kernel_size=h.size(2)).squeeze(2)
            features.append(h)
        x = torch.cat(features, dim=1)
        x = self.dropout(x)
        return self.fc(x)


def make_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss_sum += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return {"loss": loss_sum / total, "accuracy": correct / total}


def train(args):
    set_seed(args.seed)
    device = pick_device(args.device)

    start_total = time.perf_counter()
    train_split, eval_split, num_classes = load_hf_dataset(args.dataset)
    train_examples = take_subset(train_split, args.max_train, args.seed)
    eval_examples = take_subset(eval_split, args.max_eval, args.seed + 1)

    vocab = build_vocab(train_examples, args.max_vocab, args.min_freq)
    x_train, y_train = encode_examples(train_examples, vocab, args.max_len)
    x_eval, y_eval = encode_examples(eval_examples, vocab, args.max_len)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    eval_loader = DataLoader(
        TensorDataset(x_eval, y_eval),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = TextCNN(
        vocab_size=len(vocab),
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        filters=args.filters,
        dropout=args.dropout,
    ).to(device)
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    epoch_times = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start_epoch = time.perf_counter()
        loss_sum = 0.0
        seen = 0

        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for x, y in progress:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            seen += y.size(0)
            progress.set_postfix(loss=f"{loss_sum / seen:.4f}")

        epoch_time = time.perf_counter() - start_epoch
        epoch_times.append(epoch_time)
        metrics = evaluate(model, eval_loader, device)
        print(
            f"epoch={epoch} train_loss={loss_sum / seen:.4f} "
            f"eval_loss={metrics['loss']:.4f} eval_acc={metrics['accuracy']:.4f} "
            f"epoch_seconds={epoch_time:.2f}"
        )

    total_time = time.perf_counter() - start_total
    final_metrics = evaluate(model, eval_loader, device)
    summary = {
        "dataset": args.dataset,
        "device": str(device),
        "optimizer": args.optimizer,
        "seed": args.seed,
        "epochs": args.epochs,
        "max_train": len(train_examples),
        "max_eval": len(eval_examples),
        "max_len": args.max_len,
        "batch_size": args.batch_size,
        "vocab_size": len(vocab),
        "total_seconds": total_time,
        "epoch_seconds": epoch_times,
        "final_eval_loss": final_metrics["loss"],
        "final_eval_accuracy": final_metrics["accuracy"],
    }

    print("\nRUNTIME_SUMMARY")
    print(json.dumps(summary, indent=2))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"saved_summary={out}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pilot runtime check for TextCNN on AG News or IMDB.")
    parser.add_argument("--dataset", choices=["ag_news", "imdb"], required=True)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw", "rmsprop"], default="adam")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train", type=int, default=8000)
    parser.add_argument("--max-eval", type=int, default=2000)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-vocab", type=int, default=30000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--filters", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
