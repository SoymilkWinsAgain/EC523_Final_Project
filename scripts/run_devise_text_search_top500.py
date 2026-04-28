#!/usr/bin/env python
from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from argparse import Namespace
from pathlib import Path
from typing import Any

import faiss


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from who_is_this_anime_girl.devise import (
    DeVISEImageModel,
    DeVISETransformation,
    TEXT_MODEL_NAME,
    EmbeddingPairDataset,
    TextEmbeddingEncoder,
    build_devise_gallery_index,
    evaluate_text_embeddings_against_index,
    keyword_match_metrics,
    precompute_image_embeddings,
    precompute_text_embeddings,
    run_devise_training,
)
from who_is_this_anime_girl.infer import CachedGallerySearcher
from who_is_this_anime_girl.index import build_gallery_index
from who_is_this_anime_girl.losses import symmetric_image_text_contrastive_loss
from who_is_this_anime_girl.metrics import retrieval_metrics
from who_is_this_anime_girl.model import load_checkpoint
from who_is_this_anime_girl.utils import resolve_device, set_seed, write_json


MANIFEST_DIR = ROOT / "data" / "manifests" / "danbooru_clip_top500"
TRAIN_MANIFEST = MANIFEST_DIR / "train.jsonl"
VAL_MANIFEST = MANIFEST_DIR / "val.jsonl"
REPORT_DIR = ROOT / "artifacts" / "devise_transform_top500"
TEXT_EMBED_DIR = REPORT_DIR / "text_embeddings"
IMAGE_EMBED_DIR = REPORT_DIR / "image_embeddings"
INDEX_DIR = REPORT_DIR / "index"
IMAGE_INDEX_DIR = REPORT_DIR / "image_index"
RUN_DIR = ROOT / "runs" / "devise_transform_hf_finetune_v2_qwen3_256_top500"
CHECKPOINT = RUN_DIR / "best.pt"
GALLERY_DIR = ROOT / "data" / "danbooru_clip_top500" / "images"
BASE_IMAGE_CHECKPOINT = ROOT / "runs" / "deit_small_hf_finetune_top500_v2_steps" / "best.pt"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_manifests() -> dict[str, Any]:
    result = {}
    for name, expected in [("train", 40000), ("val", 10000)]:
        path = MANIFEST_DIR / f"{name}.jsonl"
        rows = read_jsonl(path)
        missing = [row["path"] for row in rows[:1000] if not Path(row["path"]).exists()]
        identities = {row["identity"] for row in rows}
        if len(rows) != expected:
            raise AssertionError(f"{path} has {len(rows)} rows; expected {expected}")
        if missing:
            raise FileNotFoundError(f"{path} has missing image paths, first: {missing[0]}")
        result[name] = {"rows": len(rows), "identities": len(identities)}
    write_json(REPORT_DIR / "manifest_checks.json", result)
    return result


def environment_checks() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    existing_path = REPORT_DIR / "environment_checks.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text(encoding="utf-8"))
        if (
            existing.get("qwen_embedding_shape") == [64, 256]
            and existing.get("base_image_checkpoint_load_ok")
            and existing.get("base_image_trainable_parameters") == 0
            and existing.get("transformation_trainable_parameters") == 525568
        ):
            existing["reused"] = True
            write_json(existing_path, existing)
            return existing
    import sentence_transformers
    import transformers

    rows = read_jsonl(TRAIN_MANIFEST)[:64]
    texts = [row["text"] for row in rows]
    started = time.time()
    encoder = TextEmbeddingEncoder(model_name=TEXT_MODEL_NAME, embedding_dim=256, device="auto")
    embeddings = encoder.encode(texts, batch_size=16, query=False)
    norms = np.linalg.norm(embeddings, axis=1)
    base_model, _ = load_checkpoint(BASE_IMAGE_CHECKPOINT, map_location="cpu")
    transformation = DeVISETransformation(image_embedding_dim=256, text_embedding_dim=256, hidden_dim=512)
    wrapped = DeVISEImageModel(base_model, transformation)
    base_trainable = sum(parameter.numel() for parameter in wrapped.image_model.parameters() if parameter.requires_grad)
    transform_trainable = sum(parameter.numel() for parameter in transformation.parameters() if parameter.requires_grad)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    result = {
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "transformers": transformers.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "qwen_model": TEXT_MODEL_NAME,
        "qwen_smoke_rows": len(texts),
        "qwen_embedding_shape": list(embeddings.shape),
        "qwen_norm_min": float(norms.min()),
        "qwen_norm_max": float(norms.max()),
        "qwen_smoke_sec": time.time() - started,
        "base_image_checkpoint": str(BASE_IMAGE_CHECKPOINT.resolve()),
        "base_image_checkpoint_load_ok": True,
        "base_image_trainable_parameters": int(base_trainable),
        "transformation_trainable_parameters": int(transform_trainable),
        "gpu": gpu.stdout.strip(),
    }
    if embeddings.shape != (64, 256):
        raise AssertionError(f"Unexpected Qwen embedding shape: {embeddings.shape}")
    if base_trainable != 0:
        raise AssertionError(f"Base image model should be frozen; got {base_trainable} trainable parameters")
    if transform_trainable != 525568:
        raise AssertionError(f"Unexpected transformation trainable parameters: {transform_trainable}")
    write_json(REPORT_DIR / "environment_checks.json", result)
    return result


def ensure_text_embeddings(force: bool = False) -> dict[str, Any]:
    train = precompute_text_embeddings(
        TRAIN_MANIFEST,
        TEXT_EMBED_DIR,
        split="train",
        model_name=TEXT_MODEL_NAME,
        embedding_dim=256,
        batch_size=32,
        device="auto",
        include_character=False,
        force=force,
    )
    val = precompute_text_embeddings(
        VAL_MANIFEST,
        TEXT_EMBED_DIR,
        split="val",
        model_name=TEXT_MODEL_NAME,
        embedding_dim=256,
        batch_size=32,
        device="auto",
        include_character=False,
        force=force,
    )
    result = {"train": train, "val": val}
    write_json(REPORT_DIR / "text_embedding_summary.json", result)
    return result


def ensure_image_embeddings(force: bool = False) -> dict[str, Any]:
    train = precompute_image_embeddings(
        image_checkpoint=BASE_IMAGE_CHECKPOINT,
        manifest_path=TEXT_EMBED_DIR / "train.jsonl",
        output_dir=IMAGE_EMBED_DIR,
        split="train",
        image_size=224,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        batch_size=128,
        workers=4,
        device="auto",
        force=force,
    )
    val = precompute_image_embeddings(
        image_checkpoint=BASE_IMAGE_CHECKPOINT,
        manifest_path=TEXT_EMBED_DIR / "val.jsonl",
        output_dir=IMAGE_EMBED_DIR,
        split="val",
        image_size=224,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        batch_size=128,
        workers=4,
        device="auto",
        force=force,
    )
    result = {"train": train, "val": val, "base_image_checkpoint": str(BASE_IMAGE_CHECKPOINT.resolve())}
    write_json(REPORT_DIR / "image_embedding_summary.json", result)
    return result


def smoke_train(batch_candidates: list[int]) -> dict[str, Any]:
    set_seed(523)
    device = resolve_device("auto")
    results: list[dict[str, Any]] = []
    for batch_size in batch_candidates:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
            dataset = EmbeddingPairDataset(
                TEXT_EMBED_DIR / "train.jsonl",
                IMAGE_EMBED_DIR / "train_image_embeddings.npy",
                TEXT_EMBED_DIR / "train_text_embeddings.npy",
            )
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, pin_memory=device.type == "cuda")
            from who_is_this_anime_girl.devise import DeVISETransformation

            transformation = DeVISETransformation(image_embedding_dim=256, text_embedding_dim=256, hidden_dim=512).to(device)
            logit_scale = torch.nn.Parameter(torch.tensor(np.log(1 / 0.07), device=device, dtype=torch.float32))
            optimizer = torch.optim.AdamW(list(transformation.parameters()) + [logit_scale], lr=1e-4, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
            started = time.time()
            batches = 0
            transformation.train()
            for image_embeddings, text_embeddings, _, _ in loader:
                if batches >= 20:
                    break
                image_embeddings = image_embeddings.to(device, non_blocking=True)
                text_embeddings = text_embeddings.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    loss = symmetric_image_text_contrastive_loss(transformation(image_embeddings), text_embeddings, logit_scale)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                batches += 1
            elapsed = time.time() - started
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            row = {
                "batch_size": batch_size,
                "batches": batches,
                "sec_per_batch": elapsed / max(1, batches),
                "peak_allocated_gb": peak,
                "ok": peak < 6.8 and batches == 20,
            }
            results.append(row)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            results.append({"batch_size": batch_size, "ok": False, "error": str(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    safe = [row for row in results if row.get("ok")]
    selected = max(safe, key=lambda row: int(row["batch_size"])) if safe else None
    if selected is None:
        raise RuntimeError(f"No smoke-test batch size passed: {results}")
    result = {"candidates": results, "selected": selected}
    write_json(REPORT_DIR / "smoke_test.json", result)
    return result


def training_args(output_dir: Path, epochs: int, batch_size: int, batches_per_epoch: int | None = None) -> Namespace:
    return Namespace(
        output_dir=str(output_dir),
        image_checkpoint=str(BASE_IMAGE_CHECKPOINT),
        train_manifest=str(TEXT_EMBED_DIR / "train.jsonl"),
        train_image_embeddings=str(IMAGE_EMBED_DIR / "train_image_embeddings.npy"),
        train_text_embeddings=str(TEXT_EMBED_DIR / "train_text_embeddings.npy"),
        val_manifest=str(TEXT_EMBED_DIR / "val.jsonl"),
        val_image_embeddings=str(IMAGE_EMBED_DIR / "val_image_embeddings.npy"),
        val_text_embeddings=str(TEXT_EMBED_DIR / "val_text_embeddings.npy"),
        image_size=224,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        batch_size=batch_size,
        eval_batch_size=128,
        embedding_dim=256,
        projection_hidden_dim=512,
        epochs=epochs,
        lr=1e-4,
        weight_decay=1e-4,
        min_lr=1e-6,
        warmup_epochs=1,
        temperature=0.07,
        workers=4,
        device="auto",
        amp=True,
        hf_token=None,
        seed=523,
        batches_per_epoch=batches_per_epoch,
    )


def choose_epoch_plan(batch_size: int, sec_per_batch: float, elapsed_so_far: float, budget_sec: float = 30 * 60) -> dict[str, Any]:
    full_batches_per_epoch = math.floor(40000 / batch_size)
    projected_train_sec_10 = full_batches_per_epoch * 10 * sec_per_batch
    reserve_sec = 4 * 60
    selected_epochs = 10 if elapsed_so_far + projected_train_sec_10 + reserve_sec <= budget_sec else 6
    result = {
        "batch_size": batch_size,
        "full_batches_per_epoch": full_batches_per_epoch,
        "sec_per_batch": sec_per_batch,
        "elapsed_before_training_sec": elapsed_so_far,
        "budget_sec": budget_sec,
        "reserve_sec": reserve_sec,
        "projected_train_sec_10_epochs": projected_train_sec_10,
        "selected_epochs": selected_epochs,
        "reason": "within_30_min_budget" if selected_epochs == 10 else "projected_total_exceeded_30_min_budget",
    }
    write_json(REPORT_DIR / "epoch_plan.json", result)
    return result


def train_all(batch_size: int, epochs: int) -> dict[str, Any]:
    mini_dir = REPORT_DIR / "mini_train_check"
    if (mini_dir / "best.pt").exists() and (mini_dir / "training_summary.json").exists():
        mini = json.loads((mini_dir / "training_summary.json").read_text(encoding="utf-8"))
        mini["output_dir"] = str(mini_dir.resolve())
        mini["reused"] = True
    else:
        mini = run_devise_training(training_args(mini_dir, epochs=1, batch_size=batch_size, batches_per_epoch=5))

    if CHECKPOINT.exists() and (RUN_DIR / "training_summary.json").exists():
        full = json.loads((RUN_DIR / "training_summary.json").read_text(encoding="utf-8"))
        full["output_dir"] = str(RUN_DIR.resolve())
        full["reused"] = True
    else:
        full = run_devise_training(training_args(RUN_DIR, epochs=epochs, batch_size=batch_size, batches_per_epoch=None))
    result = {"mini": mini, "full": full}
    write_json(REPORT_DIR / "training_runs.json", result)
    return result


def build_index_and_evaluate() -> dict[str, Any]:
    image_metadata_path = IMAGE_INDEX_DIR / "metadata.json"
    image_index_path = IMAGE_INDEX_DIR / "gallery.faiss"
    reused_image_index = False
    if image_metadata_path.exists() and image_index_path.exists():
        image_metadata = json.loads(image_metadata_path.read_text(encoding="utf-8"))
        image_index = faiss.read_index(str(image_index_path))
        reused_image_index = (
            image_index.ntotal == 10000
            and image_index.d == 256
            and image_metadata.get("embedding_space") == "image"
            and len(image_metadata.get("items", [])) == 10000
        )
    if not reused_image_index:
        image_metadata = build_gallery_index(
            checkpoint_path=CHECKPOINT,
            gallery_dir=None,
            gallery_manifest=TEXT_EMBED_DIR / "val.jsonl",
            output_dir=IMAGE_INDEX_DIR,
            batch_size=128,
            workers=4,
            device="auto",
        )
    if image_metadata["embedding_dim"] != 256 or len(image_metadata["items"]) != 10000:
        raise AssertionError(f"Unexpected image index metadata: dim={image_metadata['embedding_dim']} items={len(image_metadata['items'])}")

    started = time.time()
    metadata_path = INDEX_DIR / "metadata.json"
    index_path = INDEX_DIR / "gallery.faiss"
    reused_index = False
    if metadata_path.exists() and index_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_path))
        if index.ntotal == 10000 and index.d == 256 and len(metadata.get("items", [])) == 10000:
            reused_index = True
        else:
            metadata = {}
    else:
        metadata = {}

    if not reused_index:
        metadata = build_devise_gallery_index(
            checkpoint_path=CHECKPOINT,
            gallery_dir=None,
            gallery_manifest=TEXT_EMBED_DIR / "val.jsonl",
            output_dir=INDEX_DIR,
            batch_size=128,
            workers=4,
            device="auto",
        )
    build_sec = 0.0 if reused_index else time.time() - started
    if metadata["embedding_dim"] != 256 or len(metadata["items"]) != 10000:
        raise AssertionError(f"Unexpected index metadata: dim={metadata['embedding_dim']} items={len(metadata['items'])}")

    metrics = evaluate_text_embeddings_against_index(
        text_embeddings_path=TEXT_EMBED_DIR / "val_text_embeddings.npy",
        records_path=TEXT_EMBED_DIR / "val.jsonl",
        index_dir=INDEX_DIR,
    )
    metrics["index_build_sec"] = build_sec
    metrics["index_reused"] = reused_index
    metrics["index_ntotal"] = len(metadata["items"])
    metrics["embedding_dim"] = metadata["embedding_dim"]
    keyword = keyword_match_metrics(metadata["items"], top_k=5)
    image_space = compute_image_space_metrics()
    write_json(REPORT_DIR / "text_to_image_metrics.json", metrics)
    write_json(REPORT_DIR / "keyword_match_metrics.json", keyword)
    write_csv(REPORT_DIR / "text_to_image_metrics.csv", [metrics])
    plot_metrics(metrics)
    return {
        "index": metadata,
        "image_index": image_metadata,
        "image_index_reused": reused_image_index,
        "image_space_metrics": image_space,
        "text_metrics": metrics,
        "keyword_metrics": keyword,
    }


def plot_metrics(metrics: dict[str, Any]) -> None:
    figures = REPORT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    keys = ["image_recall@1", "image_recall@5", "image_recall@10", "identity_recall@1", "identity_recall@5", "identity_recall@10"]
    plt.figure(figsize=(9, 5))
    plt.bar(keys, [float(metrics.get(key, 0.0)) for key in keys])
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Recall")
    plt.title("Text-to-Image Retrieval")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "text_recall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(["image_mrr", "identity_mrr"], [float(metrics.get("image_mrr", 0.0)), float(metrics.get("identity_mrr", 0.0))])
    plt.ylim(0, 1.0)
    plt.ylabel("MRR")
    plt.title("Text-to-Image MRR")
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "text_mrr.png", dpi=180)
    plt.close()


def compute_image_space_metrics() -> dict[str, Any]:
    rows = read_jsonl(TEXT_EMBED_DIR / "val.jsonl")
    embeddings = np.load(IMAGE_EMBED_DIR / "val_image_embeddings.npy").astype("float32")
    identities = sorted({row["identity"] for row in rows})
    class_to_idx = {identity: index for index, identity in enumerate(identities)}
    labels = np.array([class_to_idx[row["identity"]] for row in rows], dtype=np.int64)
    metrics = retrieval_metrics(embeddings, labels)
    metrics["model"] = "hf_finetune_v2_original_image_embedding"
    metrics["embedding_space"] = "image"
    metrics["valid_queries"] = int(metrics["valid_queries"])
    write_json(REPORT_DIR / "image_space_metrics.json", metrics)
    return metrics


def sample_queries() -> dict[str, Any]:
    searcher = CachedGallerySearcher(
        checkpoint_path=CHECKPOINT,
        index_dir=IMAGE_INDEX_DIR,
        device="auto",
        text_index_dir=INDEX_DIR,
        text_model_name=TEXT_MODEL_NAME,
        text_embedding_dim=256,
        text_device="auto",
    )
    queries = [
        "hatsune miku",
        "hatsune miku blue hair",
        "blue hair school uniform",
        "touhou red bow",
        "silver hair armor wings",
        "love live brown hair sweater",
    ]
    results = {query: searcher.search_text(query, top_k=5) for query in queries}
    write_json(REPORT_DIR / "sample_text_matches.json", results)
    return results


def post_form(url: str, fields: dict[str, str]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(url, data=encoded, headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def ui_fallback_png(path: Path, title: str, lines: list[str]) -> None:
    image = Image.new("RGB", (1280, 720), "#f6f7f9")
    draw = ImageDraw.Draw(image)
    draw.text((48, 44), "Who Is This Anime Girl", fill="#17202a")
    draw.text((48, 86), title, fill="#5f6b7a")
    y = 140
    for line in lines[:12]:
        draw.text((64, y), line[:160], fill="#17202a")
        y += 42
    image.save(path)


def ui_smoke() -> dict[str, Any]:
    port = 8012
    command = [
        sys.executable,
        str(ROOT / "scripts" / "serve.py"),
        "--checkpoint",
        str(CHECKPOINT),
        "--gallery-dir",
        str(GALLERY_DIR),
        "--index-dir",
        str(IMAGE_INDEX_DIR),
        "--text-index-dir",
        str(INDEX_DIR),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--device",
        "auto",
        "--text-device",
        "auto",
        "--workers",
        "4",
    ]
    log_path = REPORT_DIR / "ui_server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        base = f"http://127.0.0.1:{port}"
        status = None
        for _ in range(90):
            try:
                with urllib.request.urlopen(base + "/api/status", timeout=2) as response:
                    status = json.loads(response.read().decode("utf-8"))
                break
            except Exception:
                time.sleep(1)
        if status is None:
            raise RuntimeError("UI server did not become ready.")
        text_result = post_form(base + "/api/query_text", {"query": "hatsune miku blue hair", "top_k": "5"})
        first_url_ok = False
        if text_result.get("matches") and text_result["matches"][0].get("gallery_url"):
            with urllib.request.urlopen(base + text_result["matches"][0]["gallery_url"], timeout=30) as response:
                first_url_ok = response.status == 200

        ui_fallback_png(REPORT_DIR / "ui_home.png", "model ready; index ready; text query enabled", [f"identities: {len(status.get('identities', {}))}"])
        lines = [
            f"query: {text_result.get('query')}",
            f"mode: {text_result.get('mode')}",
        ]
        for match in text_result.get("matches", [])[:5]:
            lines.append(f"{match.get('identity')} score={float(match.get('score', 0.0)):.4f} path={match.get('path')}")
        ui_fallback_png(REPORT_DIR / "ui_text_query_result.png", "Text query result", lines)

        result = {
            "ok": bool(text_result.get("ok") and text_result.get("matches")),
            "status": status,
            "query_response": text_result,
            "gallery_url_ok": first_url_ok,
            "screenshots": {
                "home": str((REPORT_DIR / "ui_home.png").resolve()),
                "text_query": str((REPORT_DIR / "ui_text_query_result.png").resolve()),
                "fallback_rendered": True,
            },
        }
        write_json(REPORT_DIR / "ui_smoke.json", result)
        return result
    finally:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=20)


def write_report(
    env: dict[str, Any],
    text_summary: dict[str, Any],
    image_summary: dict[str, Any],
    smoke: dict[str, Any],
    epoch_plan: dict[str, Any],
    training: dict[str, Any],
    evals: dict[str, Any],
    samples: dict[str, Any],
    ui: dict[str, Any],
) -> None:
    metrics = evals["text_metrics"]
    image_space = evals["image_space_metrics"]
    keyword = evals["keyword_metrics"]
    lines = [
        "# DeViSE Text-to-Image Top-500",
        "",
        "## Configuration",
        f"- Text model: `{TEXT_MODEL_NAME}` frozen, 256-dim normalized embeddings.",
        f"- Image embedding model: `{BASE_IMAGE_CHECKPOINT}` frozen.",
        "- Trainable part: image-embedding to text-embedding transformation block only.",
        f"- Transformation block: `256 -> 512 -> 512 -> 256`, SiLU, L2 normalize; trainable parameters `{env['transformation_trainable_parameters']}`.",
        "- Training data: `data/manifests/danbooru_clip_top500/train.jsonl` with character text removed and tag underscores converted to spaces.",
        f"- GPU: `{env.get('gpu', '')}`",
        "",
        "## Data And Smoke",
        f"- Text embeddings: train `{text_summary['train']['rows']}`, val `{text_summary['val']['rows']}`.",
        f"- Frozen image embeddings: train `{image_summary['train']['rows']}`, val `{image_summary['val']['rows']}`.",
        f"- Image queries use original image-space index `{IMAGE_INDEX_DIR}`; text queries use DeViSE text-space index `{INDEX_DIR}`.",
        f"- Base image trainable parameters during DeViSE training: `{env['base_image_trainable_parameters']}`.",
        f"- Selected batch: `{smoke['selected']['batch_size']}`, sec/batch `{smoke['selected']['sec_per_batch']:.3f}`, peak allocated `{smoke['selected']['peak_allocated_gb']:.2f}GB`.",
        f"- Selected epochs: `{epoch_plan['selected_epochs']}` ({epoch_plan['reason']}); projected 10-epoch transform training `{epoch_plan['projected_train_sec_10_epochs']:.1f}s`.",
        "",
        "## Text-to-Image Metrics",
        "These are text-query metrics in the transformed Qwen text space; `image_recall` means exact target image for a text query, not uploaded-image retrieval.",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in ["image_recall@1", "image_recall@5", "image_recall@10", "image_mrr", "identity_recall@1", "identity_recall@5", "identity_recall@10", "identity_mrr"]:
        lines.append(f"| {key} | {float(metrics.get(key, 0.0)):.4f} |")
    lines.extend(
        [
            "",
            "## Image-to-Image Metrics",
            "Original frozen image embeddings on the validation set, excluding the query image itself.",
            f"- Recall@1/5/10: `{image_space['recall@1']:.4f}` / `{image_space['recall@5']:.4f}` / `{image_space['recall@10']:.4f}`",
            f"- MRR: `{image_space['mrr']:.4f}`",
            "",
            "## Keyword Name Matching",
            f"- Identity queries: `{keyword['queries']}`",
            f"- Hard-match Recall@1: `{keyword['recall@1']:.4f}`",
            "",
            "## Runtime Outputs",
            f"- Training run: `{training['full']['output_dir']}`",
            f"- Image-space FAISS index: `{IMAGE_INDEX_DIR}`",
            f"- Text-space FAISS index: `{INDEX_DIR}`",
            f"- UI smoke: `ok={ui['ok']}`, gallery URL ok `{ui['gallery_url_ok']}`",
            "- Figures: `artifacts/devise_transform_top500/figures/text_recall.png`, `text_mrr.png`",
            "",
            "## Sample Queries",
        ]
    )
    for query, result in samples.items():
        top = result.get("matches", [{}])[0]
        lines.append(f"- `{query}` -> `{top.get('identity')}` ({result.get('mode')}, score `{float(top.get('score', 0.0)):.4f}`)")
    (REPORT_DIR / "experiment_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.time()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    validate_manifests()
    env = environment_checks()
    text_summary = ensure_text_embeddings(force=False)
    image_summary = ensure_image_embeddings(force=False)
    smoke = smoke_train([128, 256, 512, 1024])
    epoch_plan = choose_epoch_plan(
        batch_size=int(smoke["selected"]["batch_size"]),
        sec_per_batch=float(smoke["selected"]["sec_per_batch"]),
        elapsed_so_far=time.time() - started,
    )
    training = train_all(batch_size=int(smoke["selected"]["batch_size"]), epochs=int(epoch_plan["selected_epochs"]))
    evals = build_index_and_evaluate()
    samples = sample_queries()
    ui = ui_smoke()
    write_report(env, text_summary, image_summary, smoke, epoch_plan, training, evals, samples, ui)
    print(json.dumps({"report": str((REPORT_DIR / "experiment_report.md").resolve()), "metrics": evals["text_metrics"], "ui_ok": ui["ok"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
