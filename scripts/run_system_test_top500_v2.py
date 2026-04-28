#!/usr/bin/env python
from __future__ import annotations

import csv
import gc
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import faiss
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_is_this_anime_girl.data import make_dataset
from who_is_this_anime_girl.index import build_gallery_index
from who_is_this_anime_girl.infer import CachedGallerySearcher
from who_is_this_anime_girl.metrics import extract_embeddings
from who_is_this_anime_girl.model import load_checkpoint
from who_is_this_anime_girl.utils import resolve_device, write_json


MANIFEST_DIR = ROOT / "data" / "manifests" / "danbooru2018_top500"
GALLERY_MANIFEST = MANIFEST_DIR / "gallery.jsonl"
QUERY_MANIFEST = MANIFEST_DIR / "query.jsonl"
REPORT_DIR = ROOT / "artifacts" / "system_test_top500_v2"
INDEX_ROOT = REPORT_DIR / "indexes"
FIGURE_DIR = REPORT_DIR / "figures"
UI_GALLERY_DIR = ROOT / "data" / "ui_gallery_top500_v2"
UI_INDEX_DIR = REPORT_DIR / "ui_index"
UI_CHECKPOINT = ROOT / "runs" / "deit_small_hf_finetune_top500_v2_steps" / "best.pt"

MODELS = [
    {
        "name": "scratch_v2",
        "checkpoint": ROOT / "runs" / "deit_small_scratch_top500_v2_steps" / "best.pt",
        "threshold_identity_recall@1": 0.06,
    },
    {
        "name": "hf_finetune_v2",
        "checkpoint": ROOT / "runs" / "deit_small_hf_finetune_top500_v2_steps" / "best.pt",
        "threshold_identity_recall@1": 0.65,
    },
]
TOP_K = [1, 5, 10]
SEED = 523


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "model",
        "index_ntotal",
        "metadata_items",
        "embedding_dim",
        "build_index_sec",
        "query_embedding_sec",
        "faiss_search_sec",
        "image/recall@1",
        "image/recall@5",
        "image/recall@10",
        "image/mrr",
        "identity/recall@1",
        "identity/recall@5",
        "identity/recall@10",
        "identity/mrr",
        "valid_queries",
        "threshold_passed",
    ]
    fieldnames = sorted({key for row in rows for key in row})
    ordered = [key for key in preferred if key in fieldnames] + [key for key in fieldnames if key not in preferred]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def validate_manifests() -> dict[str, Any]:
    gallery = read_manifest(GALLERY_MANIFEST)
    query = read_manifest(QUERY_MANIFEST)
    for name, rows, expected_rows in [("gallery", gallery, 2500), ("query", query, 7500)]:
        missing = [row["path"] for row in rows if not Path(row["path"]).exists()]
        if missing:
            raise FileNotFoundError(f"{name} manifest has missing images, first: {missing[0]}")
        identities = {row["identity"] for row in rows}
        if len(rows) != expected_rows or len(identities) != 500:
            raise AssertionError(f"{name} manifest has {len(rows)} rows and {len(identities)} identities.")
    result = {
        "gallery_rows": len(gallery),
        "query_rows": len(query),
        "gallery_identities": len({row["identity"] for row in gallery}),
        "query_identities": len({row["identity"] for row in query}),
        "missing_images": 0,
    }
    write_json(REPORT_DIR / "manifest_checks.json", result)
    return result


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_index_for_model(model: dict[str, Any]) -> dict[str, Any]:
    output_dir = INDEX_ROOT / model["name"]
    start = time.perf_counter()
    metadata = build_gallery_index(
        checkpoint_path=model["checkpoint"],
        gallery_dir=None,
        gallery_manifest=GALLERY_MANIFEST,
        output_dir=output_dir,
        batch_size=128,
        workers=4,
        device="auto",
    )
    elapsed = time.perf_counter() - start
    faiss_index = faiss.read_index(str(output_dir / "gallery.faiss"))
    if faiss_index.ntotal != 2500:
        raise AssertionError(f"{model['name']} index ntotal={faiss_index.ntotal}, expected 2500.")
    if len(metadata["items"]) != 2500:
        raise AssertionError(f"{model['name']} metadata items={len(metadata['items'])}, expected 2500.")
    if int(metadata["embedding_dim"]) != 256:
        raise AssertionError(f"{model['name']} embedding_dim={metadata['embedding_dim']}, expected 256.")
    release_cuda()
    return {
        "model": model["name"],
        "checkpoint": str(Path(model["checkpoint"]).resolve()),
        "index_dir": str(output_dir.resolve()),
        "build_index_sec": elapsed,
        "index_ntotal": int(faiss_index.ntotal),
        "metadata_items": len(metadata["items"]),
        "embedding_dim": int(metadata["embedding_dim"]),
    }


def metric_from_ranks(ranks: np.ndarray, top_k: list[int]) -> dict[str, float]:
    valid = ranks[ranks > 0]
    if len(valid) == 0:
        return {f"recall@{k}": 0.0 for k in top_k} | {"mrr": 0.0, "valid_queries": 0.0}
    metrics = {f"recall@{k}": float(np.mean(valid <= k)) for k in top_k}
    metrics["mrr"] = float(np.mean(1.0 / valid))
    metrics["valid_queries"] = float(len(valid))
    return metrics


def evaluate_index(model: dict[str, Any], index_info: dict[str, Any]) -> dict[str, Any]:
    index_dir = Path(index_info["index_dir"])
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    faiss_index = faiss.read_index(str(index_dir / "gallery.faiss"))
    device = resolve_device("auto")

    query_dataset = make_dataset(
        None,
        QUERY_MANIFEST,
        image_size=int(metadata["image_size"]),
        train=False,
        mean=metadata.get("image_mean", [0.485, 0.456, 0.406]),
        std=metadata.get("image_std", [0.229, 0.224, 0.225]),
    )
    query_loader = DataLoader(query_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=device.type == "cuda")
    loaded_model, _ = load_checkpoint(model["checkpoint"], map_location=device)
    loaded_model.to(device)
    start_embed = time.perf_counter()
    query_embeddings, query_labels = extract_embeddings(loaded_model, query_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    query_embedding_sec = time.perf_counter() - start_embed
    del loaded_model
    release_cuda()

    query_embeddings = query_embeddings.astype("float32")
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    start_search = time.perf_counter()
    scores, indices = faiss_index.search(query_embeddings, faiss_index.ntotal)
    faiss_search_sec = time.perf_counter() - start_search

    class_to_idx = {str(key): int(value) for key, value in metadata["class_to_idx"].items()}
    gallery_identity_ids = np.array([int(item["label"]) for item in metadata["items"]], dtype=np.int64)
    query_identity_ids = np.array([class_to_idx[query_dataset.classes[int(label)]] for label in query_labels], dtype=np.int64)
    unique_query_ids = set(int(item) for item in query_identity_ids)
    unique_gallery_ids = set(int(item) for item in gallery_identity_ids)
    missing_query_ids = unique_query_ids - unique_gallery_ids
    if missing_query_ids:
        raise AssertionError(f"{model['name']} has query identities missing from gallery: {sorted(missing_query_ids)[:5]}")

    ranked_gallery_ids = gallery_identity_ids[indices]
    image_matches = ranked_gallery_ids == query_identity_ids[:, None]
    image_ranks = first_match_ranks(image_matches)
    image_metrics = metric_from_ranks(image_ranks, TOP_K)

    identity_ranks = np.zeros(len(query_identity_ids), dtype=np.int64)
    num_identities = len(class_to_idx)
    for row_index, correct_identity in enumerate(query_identity_ids):
        identity_scores = np.full(num_identities, -np.inf, dtype=np.float32)
        np.maximum.at(identity_scores, ranked_gallery_ids[row_index], scores[row_index])
        ranked_identity_ids = np.argsort(-identity_scores)
        positions = np.flatnonzero(ranked_identity_ids == correct_identity)
        identity_ranks[row_index] = int(positions[0]) + 1 if len(positions) else 0
    identity_metrics = metric_from_ranks(identity_ranks, TOP_K)
    threshold = float(model["threshold_identity_recall@1"])

    return {
        "model": model["name"],
        "index_ntotal": int(faiss_index.ntotal),
        "metadata_items": len(metadata["items"]),
        "embedding_dim": int(metadata["embedding_dim"]),
        "build_index_sec": float(index_info["build_index_sec"]),
        "query_embedding_sec": query_embedding_sec,
        "faiss_search_sec": faiss_search_sec,
        **{f"image/{key}": value for key, value in image_metrics.items() if key != "valid_queries"},
        **{f"identity/{key}": value for key, value in identity_metrics.items() if key != "valid_queries"},
        "valid_queries": identity_metrics["valid_queries"],
        "threshold_identity_recall@1": threshold,
        "threshold_passed": bool(identity_metrics["recall@1"] >= threshold and identity_metrics["valid_queries"] == 7500),
    }


def first_match_ranks(matches: np.ndarray) -> np.ndarray:
    has_match = matches.any(axis=1)
    first = np.argmax(matches, axis=1) + 1
    return np.where(has_match, first, 0).astype(np.int64)


def make_sample_matches(index_infos: dict[str, dict[str, Any]], sample_count: int = 20) -> list[dict[str, Any]]:
    query_rows = read_manifest(QUERY_MANIFEST)
    rng = random.Random(SEED)
    sampled_rows = rng.sample(query_rows, sample_count)
    samples: list[dict[str, Any]] = []
    for model in MODELS:
        searcher = CachedGallerySearcher(
            checkpoint_path=model["checkpoint"],
            index_dir=index_infos[model["name"]]["index_dir"],
            device="auto",
        )
        model_samples = []
        for row in sampled_rows:
            matches = searcher.search_file(row["path"], top_k=5)
            model_samples.append(
                {
                    "query_path": row["path"],
                    "expected_identity": row["identity"],
                    "top1_identity": matches[0]["identity"] if matches else None,
                    "top1_correct": bool(matches and matches[0]["identity"] == row["identity"]),
                    "matches": matches,
                }
            )
        samples.append({"model": model["name"], "samples": model_samples})
        del searcher
        release_cuda()
    write_json(REPORT_DIR / "sample_matches.json", samples)
    return samples


def prepare_ui_gallery() -> dict[str, Any]:
    rows = read_manifest(GALLERY_MANIFEST)
    if UI_GALLERY_DIR.exists():
        shutil.rmtree(UI_GALLERY_DIR)
    UI_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    linked = 0
    for row in rows:
        source = Path(row["path"]).resolve()
        identity_dir = UI_GALLERY_DIR / str(row["identity"])
        identity_dir.mkdir(parents=True, exist_ok=True)
        target = identity_dir / source.name
        suffix = 1
        while target.exists() or target.is_symlink():
            target = identity_dir / f"{source.stem}_{suffix}{source.suffix}"
            suffix += 1
        target.symlink_to(source)
        linked += 1
    result = {
        "gallery_dir": str(UI_GALLERY_DIR.resolve()),
        "linked_images": linked,
        "identities": len({row["identity"] for row in rows}),
    }
    write_json(REPORT_DIR / "ui_gallery.json", result)
    return result


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def request_json(url: str, data: bytes | None = None, headers: dict[str, str] | None = None, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_multipart(url: str, fields: dict[str, str], files: dict[str, Path], timeout: int = 120) -> dict[str, Any]:
    boundary = f"----codex{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(str(value).encode())
        body.extend(b"\r\n")
    for name, path in files.items():
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode())
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return request_json(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        timeout=timeout,
    )


def wait_for_server(base_url: str, timeout_sec: int = 120) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            return request_json(f"{base_url}/api/status", timeout=5)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Server did not become ready: {last_error}")


def fetch_ok(url: str, timeout: int = 30) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except urllib.error.URLError:
        return False


def find_playwright_node_path() -> str | None:
    candidates = sorted(Path.home().glob(".npm/_npx/*/node_modules"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "playwright").exists():
            return str(candidate)
    subprocess.run(["npx", "playwright", "--version"], cwd=ROOT, check=False, capture_output=True, text=True)
    candidates = sorted(Path.home().glob(".npm/_npx/*/node_modules"), key=lambda path: path.stat().st_mtime, reverse=True)
    for candidate in candidates:
        if (candidate / "playwright").exists():
            return str(candidate)
    return None


def run_playwright_smoke(base_url: str, query_image: Path) -> dict[str, Any]:
    node_path = find_playwright_node_path()
    if not node_path:
        return {"ok": False, "error": "Could not locate Playwright node package from npx cache."}
    script_path = REPORT_DIR / "ui_smoke_playwright.js"
    home_png = REPORT_DIR / "ui_home.png"
    query_png = REPORT_DIR / "ui_query_result.png"
    script_path.write_text(
        "\n".join(
            [
                "const { chromium } = require('playwright');",
                "(async () => {",
                f"  const baseUrl = {json.dumps(base_url)};",
                f"  const queryImage = {json.dumps(str(query_image))};",
                f"  const homePath = {json.dumps(str(home_png))};",
                f"  const queryPath = {json.dumps(str(query_png))};",
                "  const browser = await chromium.launch({ headless: true });",
                "  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } });",
                "  await page.goto(baseUrl, { waitUntil: 'networkidle' });",
                "  await page.waitForSelector('#identityList .identity-pill', { timeout: 120000 });",
                "  await page.screenshot({ path: homePath, fullPage: true });",
                "  await page.setInputFiles('input[name=\"image\"]', queryImage);",
                "  await page.fill('input[name=\"top_k\"]', '5');",
                "  await page.click('#queryForm button[type=\"submit\"]');",
                "  await page.waitForSelector('.match-card', { timeout: 120000 });",
                "  await page.screenshot({ path: queryPath, fullPage: true });",
                "  await browser.close();",
                "})().catch((error) => { console.error(error); process.exit(1); });",
                "",
            ]
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["NODE_PATH"] = node_path
    command = ["node", str(script_path)]
    run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    if run.returncode != 0 and "Executable doesn't exist" in (run.stderr + run.stdout):
        subprocess.run(["npx", "playwright", "install", "chromium"], cwd=ROOT, check=True, timeout=600)
        run = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
    return {
        "ok": run.returncode == 0,
        "stdout": run.stdout[-2000:],
        "stderr": run.stderr[-4000:],
        "home_screenshot": str(home_png) if home_png.exists() else None,
        "query_screenshot": str(query_png) if query_png.exists() else None,
    }


def choose_ui_query() -> tuple[Path, str]:
    sample_path = REPORT_DIR / "sample_matches.json"
    if sample_path.exists():
        sample_blocks = json.loads(sample_path.read_text(encoding="utf-8"))
        for block in sample_blocks:
            if block.get("model") != "hf_finetune_v2":
                continue
            for item in block.get("samples", []):
                if item.get("top1_correct") and Path(item["query_path"]).exists():
                    return Path(item["query_path"]), str(item["expected_identity"])
    row = read_manifest(QUERY_MANIFEST)[0]
    return Path(row["path"]), str(row["identity"])


def render_fallback_ui_pngs(status: dict[str, Any], query_response: dict[str, Any]) -> dict[str, Any]:
    home_png = REPORT_DIR / "ui_home.png"
    query_png = REPORT_DIR / "ui_query_result.png"
    identities = list((status.get("identities") or {}).items())[:28]

    home = Image.new("RGB", (1440, 1000), "#f8fafc")
    draw = ImageDraw.Draw(home)
    draw.text((48, 44), "Who Is This Anime Girl", fill="#0f172a")
    draw.text((48, 86), "model ready; index ready", fill="#334155")
    draw.text((48, 138), "Gallery", fill="#0f172a")
    x, y = 48, 184
    for identity, count in identities:
        text = f"{identity}: {count}"
        draw.rounded_rectangle((x, y, x + 300, y + 34), radius=8, fill="#e2e8f0")
        draw.text((x + 12, y + 9), text[:38], fill="#0f172a")
        y += 46
        if y > 900:
            y = 184
            x += 340
    home.save(home_png)

    query = Image.new("RGB", (1440, 1000), "#f8fafc")
    draw = ImageDraw.Draw(query)
    draw.text((48, 44), "Who Is This Anime Girl", fill="#0f172a")
    draw.text((48, 86), "Query result", fill="#334155")
    draw.text((48, 138), f"Found {len(query_response.get('matches') or [])} match(es).", fill="#0f172a")
    y = 190
    for rank, match in enumerate(query_response.get("matches") or [], start=1):
        draw.rounded_rectangle((48, y, 1392, y + 112), radius=8, fill="#ffffff", outline="#cbd5e1")
        draw.text((72, y + 22), f"#{rank} {match.get('identity')}", fill="#0f172a")
        draw.text((72, y + 52), f"Similarity: {float(match.get('score', 0.0)):.4f}", fill="#334155")
        draw.text((72, y + 80), str(match.get("path", ""))[:170], fill="#64748b")
        y += 132
    query.save(query_png)
    return {
        "ok": True,
        "reason": "Rendered fallback PNGs because Playwright browser dependencies were unavailable.",
        "home_screenshot": str(home_png),
        "query_screenshot": str(query_png),
    }


def run_ui_smoke() -> dict[str, Any]:
    prepare_ui_gallery()
    if UI_INDEX_DIR.exists():
        shutil.rmtree(UI_INDEX_DIR)
    UI_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    query_image, expected_identity = choose_ui_query()
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = REPORT_DIR / "ui_server.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "serve.py"),
                "--checkpoint",
                str(UI_CHECKPOINT),
                "--gallery-dir",
                str(UI_GALLERY_DIR),
                "--index-dir",
                str(UI_INDEX_DIR),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--device",
                "auto",
                "--workers",
                "4",
            ],
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            initial_status = wait_for_server(base_url)
            start_rebuild = time.perf_counter()
            rebuild = request_json(f"{base_url}/api/rebuild", data=b"", timeout=600)
            rebuild_sec = time.perf_counter() - start_rebuild
            rebuilt_status = request_json(f"{base_url}/api/status", timeout=30)
            start_query = time.perf_counter()
            query_response = post_multipart(
                f"{base_url}/api/query",
                fields={"top_k": "5"},
                files={"image": query_image},
                timeout=180,
            )
            api_query_sec = time.perf_counter() - start_query
            matches = query_response.get("matches", [])
            gallery_url_ok = False
            if matches and matches[0].get("gallery_url"):
                gallery_url_ok = fetch_ok(base_url + matches[0]["gallery_url"])
            playwright = run_playwright_smoke(base_url, query_image)
            if not playwright.get("ok"):
                playwright["fallback"] = render_fallback_ui_pngs(rebuilt_status, query_response)
            result = {
                "ok": bool(
                    initial_status.get("ok")
                    and rebuild.get("ok")
                    and int(rebuild.get("indexed_images", 0)) == 2500
                    and query_response.get("ok")
                    and matches
                    and gallery_url_ok
                    and (playwright.get("ok") or playwright.get("fallback", {}).get("ok"))
                ),
                "base_url": base_url,
                "server_log": str(log_path),
                "expected_identity": expected_identity,
                "query_image": str(query_image),
                "initial_status": initial_status,
                "rebuilt_status": rebuilt_status,
                "rebuild": rebuild,
                "rebuild_sec": rebuild_sec,
                "query_response": query_response,
                "api_query_sec": api_query_sec,
                "gallery_url_ok": gallery_url_ok,
                "playwright": playwright,
            }
            write_json(REPORT_DIR / "ui_smoke.json", result)
            return result
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


def plot_results(rows: list[dict[str, Any]]) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    names = [row["model"] for row in rows]
    x_positions = list(range(len(names)))
    width = 0.22
    plt.figure(figsize=(8, 5))
    for offset, key in enumerate(["identity/recall@1", "identity/recall@5", "identity/recall@10"]):
        positions = [x + (offset - 1) * width for x in x_positions]
        plt.bar(positions, [float(row[key]) for row in rows], width=width, label=key.replace("identity/", ""))
    plt.xticks(x_positions, names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Identity-Aggregated Recall")
    plt.title("System Retrieval Recall")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "recall.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    x = np.arange(len(names))
    plt.bar(x - 0.18, [float(row["image/mrr"]) for row in rows], width=0.36, label="image")
    plt.bar(x + 0.18, [float(row["identity/mrr"]) for row in rows], width=0.36, label="identity")
    plt.xticks(x, names)
    plt.ylim(0.0, 1.0)
    plt.ylabel("MRR")
    plt.title("System Retrieval MRR")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "mrr.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    latency_keys = ["build_index_sec", "query_embedding_sec", "faiss_search_sec"]
    bottom = np.zeros(len(rows))
    for key in latency_keys:
        values = np.array([float(row[key]) for row in rows])
        plt.bar(names, values, bottom=bottom, label=key.replace("_sec", ""))
        bottom += values
    plt.ylabel("Seconds")
    plt.title("System Test Latency")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "latency.png", dpi=180)
    plt.close()


def write_report(checks: dict[str, Any], rows: list[dict[str, Any]], ui_smoke: dict[str, Any]) -> None:
    lines = [
        "# Top-500 V2 System Retrieval Test",
        "",
        "## Data",
        "",
        f"- Gallery/query rows: {checks['gallery_rows']} / {checks['query_rows']}",
        f"- Gallery/query identities: {checks['gallery_identities']} / {checks['query_identities']}",
        "",
        "## Scale Retrieval Metrics",
        "",
        "| model | identity R@1 | identity R@5 | identity R@10 | identity MRR | image R@1 | image R@5 | image R@10 | valid | pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['identity/recall@1']:.4f} | {row['identity/recall@5']:.4f} | "
            f"{row['identity/recall@10']:.4f} | {row['identity/mrr']:.4f} | "
            f"{row['image/recall@1']:.4f} | {row['image/recall@5']:.4f} | {row['image/recall@10']:.4f} | "
            f"{row['valid_queries']:.0f} | {row['threshold_passed']} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            "| model | build index sec | query embed sec | FAISS search sec | index ntotal | embedding dim |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['build_index_sec']:.2f} | {row['query_embedding_sec']:.2f} | "
            f"{row['faiss_search_sec']:.2f} | {row['index_ntotal']} | {row['embedding_dim']} |"
        )
    top_match = (ui_smoke.get("query_response", {}).get("matches") or [{}])[0]
    lines.extend(
        [
            "",
            "## UI Smoke",
            "",
            f"- OK: {ui_smoke.get('ok')}",
            f"- Base URL during test: `{ui_smoke.get('base_url')}`",
            f"- Rebuild indexed images: {ui_smoke.get('rebuild', {}).get('indexed_images')}",
            f"- API query seconds: {ui_smoke.get('api_query_sec', 0):.2f}",
            f"- Expected/top-1 identity: `{ui_smoke.get('expected_identity')}` / `{top_match.get('identity')}`",
            f"- Gallery URL fetch OK: {ui_smoke.get('gallery_url_ok')}",
            f"- Playwright screenshots OK: {ui_smoke.get('playwright', {}).get('ok')}",
            f"- Screenshot fallback used: {bool(ui_smoke.get('playwright', {}).get('fallback'))}",
            "",
            "## Outputs",
            "",
            "- `system_metrics.json` / `system_metrics.csv`",
            "- `sample_matches.json`",
            "- `latency_summary.json`",
            "- `figures/recall.png`, `figures/mrr.png`, `figures/latency.png`",
            "- `ui_home.png`, `ui_query_result.png`",
            "",
        ]
    )
    (REPORT_DIR / "system_test_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    checks = validate_manifests()

    index_infos: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    latency: list[dict[str, Any]] = []
    for model in MODELS:
        if not Path(model["checkpoint"]).exists():
            raise FileNotFoundError(model["checkpoint"])
        index_info = build_index_for_model(model)
        index_infos[model["name"]] = index_info
        row = evaluate_index(model, index_info)
        rows.append(row)
        latency.append(
            {
                "model": model["name"],
                "build_index_sec": row["build_index_sec"],
                "query_embedding_sec": row["query_embedding_sec"],
                "faiss_search_sec": row["faiss_search_sec"],
            }
        )

    write_json(REPORT_DIR / "system_metrics.json", rows)
    write_csv(REPORT_DIR / "system_metrics.csv", rows)
    write_json(REPORT_DIR / "latency_summary.json", latency)
    make_sample_matches(index_infos)
    plot_results(rows)

    ui_smoke = run_ui_smoke()
    write_report(checks, rows, ui_smoke)

    if not all(row["threshold_passed"] for row in rows):
        raise SystemExit("At least one retrieval threshold failed.")
    if not ui_smoke.get("ok"):
        raise SystemExit("UI smoke test failed.")


if __name__ == "__main__":
    main()
