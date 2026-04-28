#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_is_this_anime_girl.utils import write_json


IMAGE_RE = re.compile(r"^danbooru-images/danbooru-images/\d+/(\d+)\.jpg$")


@dataclass(frozen=True)
class ImageEntry:
    zip_path: str
    file_size: int
    compress_size: int


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() == "true"


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_dir_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.() -]+", "_", value)
    cleaned = cleaned.strip(" ._")
    return cleaned or "unknown"


def parse_csv(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def collect_tags(tags: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_category = {
        "general": [],
        "artist": [],
        "copyright": [],
        "character": [],
        "meta": [],
    }
    for tag in tags:
        name = tag.get("name")
        if not name:
            continue
        category = str(tag.get("category"))
        if category == "0":
            by_category["general"].append(str(name))
        elif category == "1":
            by_category["artist"].append(str(name))
        elif category == "3":
            by_category["copyright"].append(str(name))
        elif category == "4":
            by_category["character"].append(str(name))
        elif category == "5":
            by_category["meta"].append(str(name))
    return by_category


def truncate(values: list[str], limit: int) -> list[str]:
    if limit <= 0:
        return []
    return values[:limit]


def build_text(tags: dict[str, list[str]], max_general: int, max_copyright: int, max_artist: int, include_meta: bool) -> str:
    parts: list[str] = []
    if tags["character"]:
        parts.append("character: " + ", ".join(tags["character"]))
    if tags["copyright"]:
        parts.append("copyright: " + ", ".join(truncate(tags["copyright"], max_copyright)))
    if tags["artist"]:
        parts.append("artist: " + ", ".join(truncate(tags["artist"], max_artist)))
    if tags["general"]:
        parts.append("tags: " + ", ".join(truncate(tags["general"], max_general)))
    if include_meta and tags["meta"]:
        parts.append("meta: " + ", ".join(tags["meta"]))
    return ". ".join(parts) + "."


def build_image_index(archive: zipfile.ZipFile) -> dict[str, ImageEntry]:
    image_index: dict[str, ImageEntry] = {}
    for info in archive.infolist():
        match = IMAGE_RE.match(info.filename)
        if match:
            image_index[match.group(1)] = ImageEntry(info.filename, info.file_size, info.compress_size)
    return image_index


def iter_metadata_names(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        info.filename
        for info in archive.infolist()
        if info.filename.startswith("danbooru-metadata/") and info.filename.endswith(".json")
    )


def scan_candidates(args: argparse.Namespace, archive: zipfile.ZipFile, image_index: dict[str, ImageEntry]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ratings = parse_csv(args.ratings)
    candidates: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    metadata_names = iter_metadata_names(archive)

    for metadata_index, metadata_name in enumerate(metadata_names, 1):
        started = time.time()
        matched = 0
        with archive.open(metadata_name) as handle:
            for raw_line in handle:
                stats["metadata_rows"] += 1
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError:
                    stats["json_errors"] += 1
                    continue

                post_id = str(record.get("id"))
                image = image_index.get(post_id)
                if image is None:
                    continue
                matched += 1
                stats["metadata_with_image"] += 1

                if as_bool(record.get("is_deleted")):
                    stats["skipped_deleted"] += 1
                    continue
                if args.exclude_banned and as_bool(record.get("is_banned")):
                    stats["skipped_banned"] += 1
                    continue
                if args.exclude_flagged and as_bool(record.get("is_flagged")):
                    stats["skipped_flagged"] += 1
                    continue
                if args.exclude_pending and as_bool(record.get("is_pending")):
                    stats["skipped_pending"] += 1
                    continue

                rating = str(record.get("rating", ""))
                if ratings and rating not in ratings:
                    stats["skipped_rating"] += 1
                    continue

                tags = collect_tags(record.get("tags") or [])
                if not tags["character"]:
                    stats["skipped_no_character"] += 1
                    continue
                if args.single_character_only and len(tags["character"]) != 1:
                    stats["skipped_multi_character"] += 1
                    continue
                if len(tags["general"]) < args.min_general_tags:
                    stats["skipped_few_general_tags"] += 1
                    continue

                identity = tags["character"][0] if args.single_character_only else "+".join(tags["character"])
                target_path = (
                    Path(args.image_output_dir)
                    / safe_dir_name(identity)
                    / f"{post_id}.jpg"
                )
                text = build_text(
                    tags,
                    max_general=args.max_general_tags,
                    max_copyright=args.max_copyright_tags,
                    max_artist=args.max_artist_tags,
                    include_meta=args.include_meta_tags,
                )
                candidates.append(
                    {
                        "post_id": as_int(post_id),
                        "path": str(target_path.resolve()),
                        "zip_path": image.zip_path,
                        "identity": identity,
                        "characters": tags["character"],
                        "rating": rating,
                        "file_ext": str(record.get("file_ext", "jpg")),
                        "file_size": image.file_size,
                        "image_width": as_int(record.get("image_width")),
                        "image_height": as_int(record.get("image_height")),
                        "source_url": str(record.get("source") or ""),
                        "tags": {
                            "character": tags["character"],
                            "copyright": truncate(tags["copyright"], args.max_record_tags),
                            "artist": truncate(tags["artist"], args.max_record_tags),
                            "general": truncate(tags["general"], args.max_record_tags),
                            "meta": truncate(tags["meta"], args.max_record_tags),
                        },
                        "text": text,
                    }
                )
                stats["candidates"] += 1

        print(
            f"scanned {metadata_index:02d}/{len(metadata_names)} {Path(metadata_name).name}: "
            f"matched={matched} candidates={stats['candidates']} sec={time.time() - started:.1f}",
            flush=True,
        )

    stats["images_in_archive"] = len(image_index)
    stats["metadata_files"] = len(metadata_names)
    return candidates, dict(stats)


def round_robin_select(groups: dict[str, list[dict[str, Any]]], target_total: int) -> list[dict[str, Any]]:
    identities = sorted(groups, key=lambda name: (-len(groups[name]), name))
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < target_total:
        added = 0
        for identity in identities:
            records = groups[identity]
            if offset < len(records):
                selected.append(records[offset])
                added += 1
                if len(selected) == target_total:
                    break
        if added == 0:
            break
        offset += 1
    return selected


def select_subset(args: argparse.Namespace, candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        grouped[str(candidate["identity"])].append(candidate)

    ranked_identities = sorted(grouped, key=lambda name: (-len(grouped[name]), name))
    if args.max_identities:
        ranked_identities = ranked_identities[: args.max_identities]

    capped_groups: dict[str, list[dict[str, Any]]] = {}
    identity_counts: dict[str, dict[str, int]] = {}
    for identity in ranked_identities:
        records = list(grouped[identity])
        rng.shuffle(records)
        if len(records) < args.min_images_per_identity:
            continue
        capped = records[: args.max_images_per_identity] if args.max_images_per_identity else records
        capped_groups[identity] = capped
        identity_counts[identity] = {"available": len(records), "selected_cap": len(capped)}

    selected = round_robin_select(capped_groups, args.target_total)
    rng.shuffle(selected)
    for record in selected:
        record["subset"] = args.subset_name

    summary = {
        "candidate_identities": len(grouped),
        "candidate_images": len(candidates),
        "selected_identities": len({record["identity"] for record in selected}),
        "selected_images": len(selected),
        "target_total": args.target_total,
        "max_identities": args.max_identities,
        "min_images_per_identity": args.min_images_per_identity,
        "max_images_per_identity": args.max_images_per_identity,
        "identity_counts": identity_counts,
    }
    return selected, summary


def split_rows(args: argparse.Namespace, selected: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(args.seed + 17)
    by_identity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_identity[str(row["identity"])].append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    desired_val = args.val_count if args.val_count else max(1, round(len(selected) * args.val_ratio))

    for identity in sorted(by_identity):
        rows = list(by_identity[identity])
        rng.shuffle(rows)
        val_count = round(len(rows) * desired_val / max(1, len(selected)))
        if len(rows) >= 5:
            val_count = max(1, val_count)
        val.extend(rows[:val_count])
        train.extend(rows[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)

    if len(val) > desired_val:
        extra = len(val) - desired_val
        moved = val[-extra:]
        val = val[:-extra]
        train.extend(moved)
    elif len(val) < desired_val:
        needed = desired_val - len(val)
        moved = train[-needed:]
        train = train[:-needed]
        val.extend(moved)

    if args.train_count and len(train) > args.train_count:
        overflow = train[args.train_count :]
        train = train[: args.train_count]
        val.extend(overflow)
    if args.val_count and len(val) > args.val_count:
        overflow = val[args.val_count :]
        val = val[: args.val_count]
        train.extend(overflow)

    for row in train:
        row["split"] = "train"
    for row in val:
        row["split"] = "val"
    rng.shuffle(train)
    rng.shuffle(val)
    return {"train": train, "val": val}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def extract_images(archive: zipfile.ZipFile, rows: list[dict[str, Any]]) -> dict[str, Any]:
    copied = 0
    reused = 0
    bytes_written = 0
    started = time.time()
    for index, row in enumerate(rows, 1):
        target = Path(row["path"])
        if target.exists() and target.stat().st_size == int(row["file_size"]):
            reused += 1
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(row["zip_path"]) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            copied += 1
            bytes_written += target.stat().st_size
        if index % 1000 == 0:
            print(f"extracted/reused {index}/{len(rows)} images", flush=True)
    return {
        "copied_images": copied,
        "reused_images": reused,
        "bytes_written": bytes_written,
        "elapsed_sec": time.time() - started,
    }


def validate_rows(rows: list[dict[str, Any]], require_paths: bool) -> dict[str, Any]:
    identities = Counter(str(row["identity"]) for row in rows)
    missing = []
    if require_paths:
        for row in rows:
            if not Path(row["path"]).exists():
                missing.append(row["path"])
                if len(missing) >= 10:
                    break
    return {
        "rows": len(rows),
        "identities": len(identities),
        "min_per_identity": min(identities.values()) if identities else 0,
        "max_per_identity": max(identities.values()) if identities else 0,
        "missing_path_examples": missing,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a CLIP/SigLIP-style Danbooru image-text subset from data/archive.zip without unpacking the full archive."
    )
    parser.add_argument("--archive", default=str(ROOT / "data" / "archive.zip"))
    parser.add_argument("--subset-name", default="danbooru_clip_top500")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "danbooru_clip_top500"))
    parser.add_argument("--manifest-dir", default=str(ROOT / "data" / "manifests" / "danbooru_clip_top500"))
    parser.add_argument("--image-output-dir", default=None)
    parser.add_argument("--ratings", default="s")
    parser.add_argument("--single-character-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-banned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-flagged", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--exclude-pending", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-general-tags", type=int, default=5)
    parser.add_argument("--max-general-tags", type=int, default=32)
    parser.add_argument("--max-copyright-tags", type=int, default=8)
    parser.add_argument("--max-artist-tags", type=int, default=2)
    parser.add_argument("--max-record-tags", type=int, default=64)
    parser.add_argument("--include-meta-tags", action="store_true")
    parser.add_argument("--target-total", type=int, default=50000)
    parser.add_argument("--train-count", type=int, default=40000)
    parser.add_argument("--val-count", type=int, default=10000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-identities", type=int, default=500)
    parser.add_argument("--min-images-per-identity", type=int, default=20)
    parser.add_argument("--max-images-per-identity", type=int, default=150)
    parser.add_argument("--seed", type=int, default=523)
    parser.add_argument("--dry-run", action="store_true", help="Write manifests and summary but do not extract selected images.")
    parser.add_argument("--print-full-summary", action="store_true", help="Print the full summary JSON, including per-identity counts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = str(Path(args.output_dir))
    if args.image_output_dir is None:
        args.image_output_dir = str(Path(args.output_dir) / "images")
    if args.train_count and args.val_count:
        args.target_total = args.train_count + args.val_count

    started = time.time()
    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        image_index = build_image_index(archive)
        candidates, scan_summary = scan_candidates(args, archive, image_index)
        selected, selection_summary = select_subset(args, candidates)
        split = split_rows(args, selected)
        all_rows = split["train"] + split["val"]

        manifest_dir = Path(args.manifest_dir)
        counts = {
            "train": write_jsonl(manifest_dir / "train.jsonl", split["train"]),
            "val": write_jsonl(manifest_dir / "val.jsonl", split["val"]),
            "all": write_jsonl(manifest_dir / "all.jsonl", all_rows),
        }

        extraction_summary = {"dry_run": bool(args.dry_run)}
        if not args.dry_run:
            extraction_summary.update(extract_images(archive, all_rows))

    estimated_image_gb = sum(int(row["file_size"]) for row in all_rows) / 1024**3
    summary = {
        "archive": str(archive_path.resolve()),
        "subset_name": args.subset_name,
        "output_dir": str(Path(args.output_dir).resolve()),
        "manifest_dir": str(Path(args.manifest_dir).resolve()),
        "image_output_dir": str(Path(args.image_output_dir).resolve()),
        "filters": {
            "ratings": sorted(parse_csv(args.ratings)),
            "single_character_only": args.single_character_only,
            "exclude_banned": args.exclude_banned,
            "exclude_flagged": args.exclude_flagged,
            "exclude_pending": args.exclude_pending,
            "min_general_tags": args.min_general_tags,
        },
        "text": {
            "max_general_tags": args.max_general_tags,
            "max_copyright_tags": args.max_copyright_tags,
            "max_artist_tags": args.max_artist_tags,
            "include_meta_tags": args.include_meta_tags,
        },
        "counts": counts,
        "validation": {
            "train": validate_rows(split["train"], require_paths=not args.dry_run),
            "val": validate_rows(split["val"], require_paths=not args.dry_run),
            "all": validate_rows(all_rows, require_paths=not args.dry_run),
        },
        "estimated_selected_image_gb": estimated_image_gb,
        "scan": scan_summary,
        "selection": selection_summary,
        "extraction": extraction_summary,
        "elapsed_sec": time.time() - started,
    }
    write_json(Path(args.manifest_dir) / "summary.json", summary)
    write_json(Path(args.output_dir) / "summary.json", summary)
    if args.print_full_summary:
        printable = summary
    else:
        printable = {
            "archive": summary["archive"],
            "subset_name": summary["subset_name"],
            "manifest_dir": summary["manifest_dir"],
            "image_output_dir": summary["image_output_dir"],
            "counts": summary["counts"],
            "estimated_selected_image_gb": summary["estimated_selected_image_gb"],
            "scan": summary["scan"],
            "selection": {
                key: value
                for key, value in summary["selection"].items()
                if key != "identity_counts"
            },
            "extraction": summary["extraction"],
            "validation": summary["validation"],
            "elapsed_sec": summary["elapsed_sec"],
        }
    print(json.dumps(printable, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
