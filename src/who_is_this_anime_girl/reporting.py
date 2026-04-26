from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import write_json


def flatten_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history:
        row = {"epoch": item["epoch"]}
        row.update(item.get("metrics", {}))
        rows.append(row)
    return rows


def write_history_csv(path: str | Path, history: list[dict[str, Any]]) -> None:
    rows = flatten_history(history)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    if "epoch" in fieldnames:
        fieldnames.remove("epoch")
        fieldnames.insert(0, "epoch")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_group(rows: list[dict[str, Any]], output_path: Path, keys: list[str], title: str, ylabel: str) -> bool:
    available = [key for key in keys if any(key in row for row in rows)]
    if not available:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(8, 5))
    for key in available:
        values = [row.get(key) for row in rows]
        plt.plot(epochs, values, marker="o", linewidth=1.8, label=key)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return True


def save_training_artifacts(output_dir: str | Path, history: list[dict[str, Any]]) -> None:
    output_dir = Path(output_dir)
    rows = flatten_history(history)
    write_json(output_dir / "history.json", history)
    write_history_csv(output_dir / "metrics.csv", history)
    plot_metric_group(
        rows,
        output_dir / "curves" / "loss.png",
        ["train/loss", "train/contrastive_loss", "train/classifier_loss"],
        "Training Loss",
        "Loss",
    )
    plot_metric_group(
        rows,
        output_dir / "curves" / "retrieval.png",
        ["val/recall@1", "val/recall@5", "val/recall@10", "val/mrr"],
        "Validation Retrieval Metrics",
        "Score",
    )
    plot_metric_group(
        rows,
        output_dir / "curves" / "accuracy.png",
        ["train/accuracy", "val/recall@1"],
        "Accuracy and Recall",
        "Score",
    )
    plot_metric_group(rows, output_dir / "curves" / "lr.png", ["lr"], "Learning Rate", "LR")
