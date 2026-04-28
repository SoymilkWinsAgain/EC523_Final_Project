#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import matplotlib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_is_this_anime_girl.data import PKBatchSampler, make_dataset
from who_is_this_anime_girl.evaluate import evaluate_model, plot_comparison_bars, write_csv
from who_is_this_anime_girl.losses import supervised_contrastive_loss
from who_is_this_anime_girl.model import create_model
from who_is_this_anime_girl.train import DEFAULTS, run_training
from who_is_this_anime_girl.utils import resolve_device, write_json


MODEL_NAME = "facebook/deit-small-patch16-224"
MANIFEST_DIR = ROOT / "data" / "manifests" / "danbooru2018_top500"
REPORT_DIR = ROOT / "artifacts" / "deit_top500_experiment_v2"

V1_SCRATCH_RUN = ROOT / "runs" / "deit_small_scratch_top500"
V1_FINETUNE_RUN = ROOT / "runs" / "deit_small_hf_finetune_top500"
SCRATCH_RUN = ROOT / "runs" / "deit_small_scratch_top500_v2_steps"
FINETUNE_RUN = ROOT / "runs" / "deit_small_hf_finetune_top500_v2_steps"
MINI_RUN = REPORT_DIR / "mini_train_check"

MEMORY_LIMIT_GB = 6.8
TARGET_MAX_MINUTES = 25.0
SEED = 523


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def assert_manifest(path: Path, expected_rows: int, expected_identities: int) -> dict[str, Any]:
    rows = 0
    identities: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows += 1
            identities.add(str(row["identity"]))
            if not Path(row["path"]).exists():
                raise FileNotFoundError(row["path"])
    if rows != expected_rows:
        raise AssertionError(f"{path} has {rows} rows, expected {expected_rows}.")
    if len(identities) != expected_identities:
        raise AssertionError(f"{path} has {len(identities)} identities, expected {expected_identities}.")
    return {"path": str(path), "rows": rows, "identities": len(identities)}


def load_dataset_sample(name: str, train: bool) -> dict[str, Any]:
    dataset = make_dataset(
        None,
        MANIFEST_DIR / f"{name}.jsonl",
        image_size=224,
        train=train,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    image, label = dataset[0]
    return {
        "name": name,
        "rows": len(dataset),
        "classes": len(dataset.classes),
        "sample_shape": list(image.shape),
        "sample_label": int(label),
        "sample_identity": dataset.classes[int(label)],
    }


def data_checks() -> dict[str, Any]:
    checks = {
        "manifests": [
            assert_manifest(MANIFEST_DIR / "train.jsonl", 40000, 500),
            assert_manifest(MANIFEST_DIR / "val.jsonl", 10000, 500),
            assert_manifest(MANIFEST_DIR / "gallery.jsonl", 2500, 500),
            assert_manifest(MANIFEST_DIR / "query.jsonl", 7500, 500),
        ],
        "dataset_samples": [
            load_dataset_sample("train", train=True),
            load_dataset_sample("val", train=False),
            load_dataset_sample("gallery", train=False),
            load_dataset_sample("query", train=False),
        ],
    }
    write_json(REPORT_DIR / "data_checks.json", checks)
    return checks


def make_train_dataset():
    return make_dataset(
        None,
        MANIFEST_DIR / "train.jsonl",
        image_size=224,
        train=True,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )


def smoke_model(pretrained: bool, identities_per_batch: int, samples_per_identity: int, seed: int, batches: int = 40) -> dict[str, Any]:
    device = resolve_device("auto")
    dataset = make_train_dataset()
    sampler = PKBatchSampler(
        dataset.targets,
        identities_per_batch=identities_per_batch,
        samples_per_identity=samples_per_identity,
        batches_per_epoch=batches,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4, pin_memory=device.type == "cuda")
    model = create_model(
        backbone_backend="hf-transformers",
        model_name=MODEL_NAME,
        pretrained=pretrained,
        embedding_dim=256,
        projection_hidden_dim=512,
        num_classes=len(dataset.classes),
        trust_remote_code=False,
        finetune_mode="full",
    ).to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model.train()
    start = time.perf_counter()
    completed = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)
            loss = supervised_contrastive_loss(outputs["embedding"], labels, temperature=0.07)
            loss = loss + 0.1 * F.cross_entropy(outputs["logits"], labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        completed += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3 if device.type == "cuda" else 0.0

    eval_metrics = capped_eval(model, device)
    del model, optimizer, loader, sampler, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "pretrained": pretrained,
        "identities_per_batch": identities_per_batch,
        "samples_per_identity": samples_per_identity,
        "batch_size": identities_per_batch * samples_per_identity,
        "batches": completed,
        "elapsed_sec": elapsed,
        "sec_per_batch": elapsed / max(1, completed),
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
        "capped_eval": eval_metrics,
    }


def capped_eval(model: torch.nn.Module, device: torch.device) -> dict[str, float]:
    val_dataset = make_dataset(
        None,
        MANIFEST_DIR / "val.jsonl",
        image_size=224,
        train=False,
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )
    subset = Subset(val_dataset, list(range(min(512, len(val_dataset)))))
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                outputs = model(images)
            if "logits" in outputs:
                correct += int((outputs["logits"].argmax(dim=1) == labels).sum().detach().cpu())
                total += int(labels.numel())
    return {"samples": float(total), "classifier_accuracy": correct / total if total else 0.0}


def choose_training_shape(smoke: list[dict[str, Any]]) -> dict[str, Any]:
    peak = max(item["peak_allocated_gb"] for item in smoke)
    sec_per_batch = max(item["sec_per_batch"] for item in smoke)
    identities_per_batch = 16
    samples_per_identity = 6
    scratch_batches = 200
    finetune_batches = 300
    scratch_epochs = 16
    finetune_epochs = 10

    scratch_minutes = sec_per_batch * scratch_epochs * scratch_batches / 60
    finetune_minutes = sec_per_batch * finetune_epochs * finetune_batches / 60
    time_fallback = scratch_minutes > TARGET_MAX_MINUTES or finetune_minutes > TARGET_MAX_MINUTES
    memory_fallback = peak > MEMORY_LIMIT_GB
    if time_fallback:
        scratch_batches = 175
        finetune_batches = 260
        scratch_minutes = sec_per_batch * scratch_epochs * scratch_batches / 60
        finetune_minutes = sec_per_batch * finetune_epochs * finetune_batches / 60
    if memory_fallback:
        identities_per_batch = 16
        samples_per_identity = 4

    plan = {
        "identities_per_batch": identities_per_batch,
        "samples_per_identity": samples_per_identity,
        "batch_size": identities_per_batch * samples_per_identity,
        "scratch_epochs": scratch_epochs,
        "scratch_batches_per_epoch": scratch_batches,
        "scratch_updates": scratch_epochs * scratch_batches,
        "scratch_sample_presentations": scratch_epochs * scratch_batches * identities_per_batch * samples_per_identity,
        "scratch_estimated_minutes": scratch_minutes,
        "finetune_epochs": finetune_epochs,
        "finetune_batches_per_epoch": finetune_batches,
        "finetune_updates": finetune_epochs * finetune_batches,
        "finetune_sample_presentations": finetune_epochs * finetune_batches * identities_per_batch * samples_per_identity,
        "finetune_estimated_minutes": finetune_minutes,
        "time_fallback": time_fallback,
        "memory_fallback": memory_fallback,
        "peak_allocated_gb": peak,
        "sec_per_batch_for_projection": sec_per_batch,
    }
    write_json(REPORT_DIR / "training_plan.json", plan)
    return plan


def training_args(
    output_dir: Path,
    pretrained: bool,
    lr: float,
    min_lr: float,
    epochs: int,
    batches_per_epoch: int,
    identities_per_batch: int,
    samples_per_identity: int,
    warmup_epochs: int,
) -> Namespace:
    values = dict(DEFAULTS)
    values.update(
        {
            "train_dir": None,
            "train_manifest": str(MANIFEST_DIR / "train.jsonl"),
            "val_dir": None,
            "val_manifest": str(MANIFEST_DIR / "val.jsonl"),
            "output_dir": str(output_dir),
            "backbone_backend": "hf-transformers",
            "model_name": MODEL_NAME,
            "pretrained": pretrained,
            "trust_remote_code": False,
            "hf_token": os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            "timm_kwargs": {},
            "finetune_mode": "full",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_target_modules": None,
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "embedding_dim": 256,
            "projection_hidden_dim": 512,
            "epochs": epochs,
            "identities_per_batch": identities_per_batch,
            "samples_per_identity": samples_per_identity,
            "batches_per_epoch": batches_per_epoch,
            "lr": lr,
            "weight_decay": 1e-4,
            "scheduler": "cosine-warmup",
            "min_lr": min_lr,
            "warmup_epochs": warmup_epochs,
            "step_size": 5,
            "gamma": 0.1,
            "milestones": None,
            "temperature": 0.07,
            "classification_weight": 0.1,
            "freeze_backbone_epochs": 0,
            "workers": 4,
            "device": "auto",
            "amp": True,
            "seed": SEED,
        }
    )
    return Namespace(**values)


def run_training_job(
    name: str,
    output_dir: Path,
    pretrained: bool,
    lr: float,
    min_lr: float,
    epochs: int,
    batches_per_epoch: int,
    identities_per_batch: int,
    samples_per_identity: int,
    warmup_epochs: int,
) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    run_training(
        training_args(
            output_dir=output_dir,
            pretrained=pretrained,
            lr=lr,
            min_lr=min_lr,
            epochs=epochs,
            batches_per_epoch=batches_per_epoch,
            identities_per_batch=identities_per_batch,
            samples_per_identity=samples_per_identity,
            warmup_epochs=warmup_epochs,
        )
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_allocated_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
    return {
        "name": name,
        "output_dir": str(output_dir),
        "pretrained": pretrained,
        "lr": lr,
        "min_lr": min_lr,
        "epochs": epochs,
        "batches_per_epoch": batches_per_epoch,
        "identities_per_batch": identities_per_batch,
        "samples_per_identity": samples_per_identity,
        "batch_size": identities_per_batch * samples_per_identity,
        "updates": epochs * batches_per_epoch,
        "sample_presentations": epochs * batches_per_epoch * identities_per_batch * samples_per_identity,
        "elapsed_sec": elapsed,
        "elapsed_min": elapsed / 60,
        "peak_allocated_gb": peak_allocated_gb,
        "peak_reserved_gb": peak_reserved_gb,
    }


def run_mini_check(plan: dict[str, Any]) -> dict[str, Any]:
    result = run_training_job(
        name="mini_train_check",
        output_dir=MINI_RUN,
        pretrained=True,
        lr=1e-4,
        min_lr=1e-6,
        epochs=1,
        batches_per_epoch=5,
        identities_per_batch=int(plan["identities_per_batch"]),
        samples_per_identity=int(plan["samples_per_identity"]),
        warmup_epochs=1,
    )
    write_json(REPORT_DIR / "mini_train_check.json", result)
    return result


def model_specs_for_comparison() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "deit_small_raw_hf",
            "type": "raw",
            "backbone_backend": "hf-transformers",
            "model_name": MODEL_NAME,
            "pretrained": True,
            "image_size": 224,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
        }
    ]
    optional_checkpoints = [
        ("deit_small_scratch_top500_v1", V1_SCRATCH_RUN / "best.pt"),
        ("deit_small_hf_finetune_top500_v1", V1_FINETUNE_RUN / "best.pt"),
        ("deit_small_scratch_top500_v2_steps", SCRATCH_RUN / "best.pt"),
        ("deit_small_hf_finetune_top500_v2_steps", FINETUNE_RUN / "best.pt"),
    ]
    for name, checkpoint in optional_checkpoints:
        if checkpoint.exists():
            specs.append({"name": name, "type": "checkpoint", "checkpoint": str(checkpoint)})
    return specs


def evaluate_experiment() -> list[dict[str, Any]]:
    device = resolve_device("auto")
    dataset_spec = {
        "query_manifest": str(MANIFEST_DIR / "query.jsonl"),
        "gallery_manifest": str(MANIFEST_DIR / "gallery.jsonl"),
    }
    rows = [evaluate_model(spec, dataset_spec, top_k=[1, 5, 10], batch_size=64, workers=4, device=device) for spec in model_specs_for_comparison()]
    write_json(REPORT_DIR / "comparison_metrics.json", rows)
    write_csv(REPORT_DIR / "comparison_metrics.csv", rows)
    plot_comparison_bars(REPORT_DIR, rows)
    plot_v2_comparison_bars(rows)
    plot_v1_v2_delta(rows)
    return rows


def plot_v2_comparison_bars(rows: list[dict[str, Any]]) -> None:
    wanted = {
        "deit_small_raw_hf",
        "deit_small_scratch_top500_v2_steps",
        "deit_small_hf_finetune_top500_v2_steps",
    }
    selected = [row for row in rows if row["name"] in wanted]
    if len(selected) < 3:
        return

    names = [row["name"].replace("deit_small_", "").replace("_top500", "") for row in selected]
    recall_keys = ["recall@1", "recall@5", "recall@10"]
    x_positions = list(range(len(names)))
    width = 0.22
    plt.figure(figsize=(9, 5.4))
    for offset, key in enumerate(recall_keys):
        values = [float(row.get(key, 0.0)) for row in selected]
        positions = [x + (offset - 1) * width for x in x_positions]
        plt.bar(positions, values, width=width, label=key)
    plt.xticks(x_positions, names, rotation=15, ha="right")
    plt.ylim(0.0, 1.0)
    plt.ylabel("Recall")
    plt.title("V2 Retrieval Recall Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "comparison_v2_recall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5.2))
    plt.bar(names, [float(row.get("mrr", 0.0)) for row in selected], color="#1d4ed8")
    plt.ylim(0.0, 1.0)
    plt.ylabel("MRR")
    plt.title("V2 Retrieval MRR Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "comparison_v2_mrr.png", dpi=180)
    plt.close()


def plot_v1_v2_delta(rows: list[dict[str, Any]]) -> None:
    by_name = {row["name"]: row for row in rows}
    pairs = [
        ("scratch", "deit_small_scratch_top500_v1", "deit_small_scratch_top500_v2_steps"),
        ("fine-tune", "deit_small_hf_finetune_top500_v1", "deit_small_hf_finetune_top500_v2_steps"),
    ]
    metrics = ["recall@1", "recall@5", "recall@10", "mrr"]
    available_pairs = [(label, old, new) for label, old, new in pairs if old in by_name and new in by_name]
    if not available_pairs:
        return
    x_positions = list(range(len(metrics)))
    width = 0.35
    plt.figure(figsize=(9, 5.2))
    for offset, (label, old_name, new_name) in enumerate(available_pairs):
        deltas = [float(by_name[new_name].get(metric, 0.0)) - float(by_name[old_name].get(metric, 0.0)) for metric in metrics]
        positions = [x + (offset - (len(available_pairs) - 1) / 2) * width for x in x_positions]
        plt.bar(positions, deltas, width=width, label=label)
    plt.axhline(0.0, color="#111827", linewidth=1.0)
    plt.xticks(x_positions, metrics)
    plt.ylabel("V2 - V1")
    plt.title("V2 Retrieval Metric Delta")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "comparison_v1_v2_delta.png", dpi=180)
    plt.close()


def command_text(
    output_dir: Path,
    pretrained: bool,
    lr: float,
    min_lr: float,
    epochs: int,
    batches_per_epoch: int,
    identities_per_batch: int,
    samples_per_identity: int,
    warmup_epochs: int,
) -> str:
    pretrained_flag = "--pretrained" if pretrained else "--no-pretrained"
    return (
        "conda run -n jigsaw python scripts/train.py "
        f"--train-manifest {MANIFEST_DIR / 'train.jsonl'} "
        f"--val-manifest {MANIFEST_DIR / 'val.jsonl'} "
        f"--output-dir {output_dir} --backbone-backend hf-transformers "
        f"--model-name {MODEL_NAME} {pretrained_flag} --finetune-mode full "
        "--image-mean 0.5,0.5,0.5 --image-std 0.5,0.5,0.5 "
        f"--epochs {epochs} --batches-per-epoch {batches_per_epoch} "
        f"--identities-per-batch {identities_per_batch} --samples-per-identity {samples_per_identity} "
        f"--lr {lr} --min-lr {min_lr} --scheduler cosine-warmup --warmup-epochs {warmup_epochs} --amp"
    )


def best_epoch_summary(output_dir: Path) -> dict[str, Any]:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        return {}
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    best = max(rows, key=lambda row: float(row.get("val/recall@1") or 0.0))
    return {
        "epoch": int(best["epoch"]),
        "val_recall@1": float(best.get("val/recall@1") or 0.0),
        "val_recall@5": float(best.get("val/recall@5") or 0.0),
        "val_recall@10": float(best.get("val/recall@10") or 0.0),
        "val_mrr": float(best.get("val/mrr") or 0.0),
    }


def write_report(
    checks: dict[str, Any],
    smoke: list[dict[str, Any]],
    plan: dict[str, Any],
    mini: dict[str, Any],
    runs: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [
        "# DeiT-Small Top-500 V2 Step-Scale Experiment",
        "",
        "## Hardware",
        "",
        f"- GPU: {gpu.stdout.strip() or 'unknown'}",
        f"- CUDA available: {torch.cuda.is_available()}",
        "",
        "## Data Checks",
        "",
        f"- Train / val / gallery / query rows: {checks['manifests'][0]['rows']} / {checks['manifests'][1]['rows']} / {checks['manifests'][2]['rows']} / {checks['manifests'][3]['rows']}",
        f"- Identities per manifest: {checks['manifests'][0]['identities']}",
        f"- Dataset sample tensor shape: {checks['dataset_samples'][0]['sample_shape']}",
        "",
        "## Smoke Gate",
        "",
        "| model | batch | batches | sec/batch | peak allocated GB | peak reserved GB | capped eval acc |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in smoke:
        label = "hf_pretrained" if item["pretrained"] else "scratch"
        lines.append(
            f"| {label} | {item['batch_size']} | {item['batches']} | {item['sec_per_batch']:.3f} | "
            f"{item['peak_allocated_gb']:.2f} | {item['peak_reserved_gb']:.2f} | {item['capped_eval']['classifier_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Chosen Training Plan",
            "",
            f"- Batch shape: {plan['identities_per_batch']} identities x {plan['samples_per_identity']} samples = {plan['batch_size']}",
            f"- Scratch: {plan['scratch_epochs']} epochs x {plan['scratch_batches_per_epoch']} batches = {plan['scratch_updates']} updates, {plan['scratch_sample_presentations']} sample presentations",
            f"- HF fine-tune: {plan['finetune_epochs']} epochs x {plan['finetune_batches_per_epoch']} batches = {plan['finetune_updates']} updates, {plan['finetune_sample_presentations']} sample presentations",
            f"- Time fallback used: {plan['time_fallback']}; memory fallback used: {plan['memory_fallback']}",
            "",
            "## Mini Train Check",
            "",
            f"- Output: `{mini['output_dir']}`",
            f"- Runtime: {mini['elapsed_min']:.2f} min",
            f"- Peak allocated/reserved: {mini['peak_allocated_gb']:.2f} / {mini['peak_reserved_gb']:.2f} GB",
            "",
            "## Training Runs",
            "",
        ]
    )
    for run in runs:
        best = best_epoch_summary(Path(run["output_dir"]))
        lines.extend(
            [
                f"### {run['name']}",
                "",
                f"- Output: `{run['output_dir']}`",
                f"- Runtime: {run['elapsed_min']:.1f} min",
                f"- Peak allocated/reserved: {run['peak_allocated_gb']:.2f} / {run['peak_reserved_gb']:.2f} GB",
                f"- Updates / sample presentations: {run['updates']} / {run['sample_presentations']}",
                f"- Best validation epoch: {best.get('epoch', 'n/a')}, Recall@1={best.get('val_recall@1', 0):.4f}, MRR={best.get('val_mrr', 0):.4f}",
                f"- Command: `{command_text(Path(run['output_dir']), run['pretrained'], run['lr'], run['min_lr'], run['epochs'], run['batches_per_epoch'], run['identities_per_batch'], run['samples_per_identity'], 2 if not run['pretrained'] else 1)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Retrieval Comparison",
            "",
            "| model | recall@1 | recall@5 | recall@10 | mrr | valid queries |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {row['name']} | {row.get('recall@1', 0):.4f} | {row.get('recall@5', 0):.4f} | "
            f"{row.get('recall@10', 0):.4f} | {row.get('mrr', 0):.4f} | {row.get('valid_queries', 0):.0f} |"
        )
    lines.extend(
        [
            "",
            "## Slide Figures",
            "",
            "- `comparison_recall.png`",
            "- `comparison_mrr.png`",
            "- `comparison_v2_recall.png`",
            "- `comparison_v2_mrr.png`",
            "- `comparison_v1_v2_delta.png`",
            "- each run: `curves/loss.png`, `curves/retrieval.png`, `curves/accuracy.png`, `curves/lr.png`",
            "",
        ]
    )
    (REPORT_DIR / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    checks = data_checks()
    smoke = [
        smoke_model(pretrained=False, identities_per_batch=16, samples_per_identity=6, seed=SEED, batches=40),
        smoke_model(pretrained=True, identities_per_batch=16, samples_per_identity=6, seed=SEED + 1, batches=40),
    ]
    write_json(REPORT_DIR / "smoke_test.json", smoke)
    plan = choose_training_shape(smoke)
    mini = run_mini_check(plan)
    runs = [
        run_training_job(
            name="deit_small_scratch_top500_v2_steps",
            output_dir=SCRATCH_RUN,
            pretrained=False,
            lr=5e-4,
            min_lr=5e-6,
            epochs=int(plan["scratch_epochs"]),
            batches_per_epoch=int(plan["scratch_batches_per_epoch"]),
            identities_per_batch=int(plan["identities_per_batch"]),
            samples_per_identity=int(plan["samples_per_identity"]),
            warmup_epochs=2,
        ),
        run_training_job(
            name="deit_small_hf_finetune_top500_v2_steps",
            output_dir=FINETUNE_RUN,
            pretrained=True,
            lr=1e-4,
            min_lr=1e-6,
            epochs=int(plan["finetune_epochs"]),
            batches_per_epoch=int(plan["finetune_batches_per_epoch"]),
            identities_per_batch=int(plan["identities_per_batch"]),
            samples_per_identity=int(plan["samples_per_identity"]),
            warmup_epochs=1,
        ),
    ]
    write_json(REPORT_DIR / "training_runs.json", runs)
    metrics = evaluate_experiment()
    write_report(checks, smoke, plan, mini, runs, metrics)


if __name__ == "__main__":
    main()
