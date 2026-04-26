from __future__ import annotations

import argparse
import io
import json
import lzma
import random
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from .utils import write_json


COMMON_CHARACTER_FIELDS = ("tag_string_character", "tag_string_characters", "characters", "character_tags")


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [tag for tag in value.split() if tag]
    if isinstance(value, list):
        tags: list[str] = []
        for item in value:
            if isinstance(item, str):
                tags.append(item)
            elif isinstance(item, dict):
                name = item.get("name") or item.get("tag") or item.get("tag_name")
                category = item.get("category") or item.get("category_id")
                if name and category in {4, "4", "character", "characters"}:
                    tags.append(str(name))
        return tags
    return []


def extract_character_tags(record: dict[str, Any], field: str | None = None) -> list[str]:
    if field:
        return normalize_tags(record.get(field))
    for candidate in COMMON_CHARACTER_FIELDS:
        tags = normalize_tags(record.get(candidate))
        if tags:
            return tags
    return normalize_tags(record.get("tags"))


def record_id(record: dict[str, Any]) -> int:
    for key in ("id", "image_id", "post_id"):
        if key in record and record[key] is not None:
            return int(record[key])
    raise KeyError("Metadata record does not contain id, image_id, or post_id.")


def record_extension(record: dict[str, Any], image_subdir: str, use_metadata_extension_for_512px: bool) -> str:
    if image_subdir == "512px" and not use_metadata_extension_for_512px:
        return "jpg"
    extension = record.get("file_ext") or record.get("file_extension") or record.get("ext")
    if extension:
        return str(extension).lower().lstrip(".")
    source = record.get("file_url") or record.get("source") or ""
    suffix = Path(str(source)).suffix.lower().lstrip(".")
    return suffix or "jpg"


def danbooru_image_path(
    image_root: str | Path,
    record: dict[str, Any],
    image_subdir: str,
    use_metadata_extension_for_512px: bool = False,
) -> Path:
    image_id = record_id(record)
    bucket = f"{image_id % 1000:04d}"
    extension = record_extension(record, image_subdir, use_metadata_extension_for_512px)
    return Path(image_root) / image_subdir / bucket / f"{image_id}.{extension}"


def iter_jsonl_bytes(handle: io.BufferedReader | io.BytesIO) -> Iterator[dict[str, Any]]:
    text = io.TextIOWrapper(handle, encoding="utf-8")
    for line in text:
        line = line.strip()
        if line:
            yield json.loads(line)


def iter_metadata_records(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_file() and (child.name.endswith(".jsonl") or child.name.endswith(".jsonl.xz")):
                yield from iter_metadata_records(child)
        return

    name = path.name
    if name.endswith((".tar.xz", ".txz")):
        with tarfile.open(path, mode="r:xz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                yield from iter_jsonl_bytes(extracted)
        return

    if name.endswith(".xz"):
        with lzma.open(path, mode="rb") as handle:
            yield from iter_jsonl_bytes(handle)
        return

    with path.open("rb") as handle:
        yield from iter_jsonl_bytes(handle)


def split_records(records: list[dict[str, Any]], split: tuple[float, float, float], rng: random.Random) -> dict[str, list[dict[str, Any]]]:
    shuffled = list(records)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_ratio, val_ratio, test_ratio = split
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    if n >= 2 and train_ratio > 0:
        n_train = max(1, n_train)
    if n >= 3 and val_ratio > 0:
        n_val = max(1, n_val)
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    if n >= 3 and test_ratio > 0 and n_test == 0:
        if n_val > 1:
            n_val -= 1
        elif n_train > 1:
            n_train -= 1

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def prepare_danbooru_manifests(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    ratings = set(parse_csv(args.ratings))
    split_values = tuple(float(value) for value in parse_csv(args.split))
    if len(split_values) != 3 or abs(sum(split_values) - 1.0) > 1e-6:
        raise ValueError("--split must contain three comma-separated ratios that sum to 1.0")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scanned = 0
    skipped_missing = 0
    skipped_no_identity = 0
    skipped_multi_identity = 0

    for metadata in iter_metadata_records(args.metadata):
        scanned += 1
        if args.limit_records and scanned > args.limit_records:
            break

        rating = str(metadata.get("rating", ""))
        if ratings and rating not in ratings:
            continue

        identities = extract_character_tags(metadata, field=args.identity_field)
        if not identities:
            skipped_no_identity += 1
            continue
        if args.single_character_only and len(identities) != 1:
            skipped_multi_identity += 1
            continue

        path = danbooru_image_path(
            args.image_root,
            metadata,
            image_subdir=args.image_subdir,
            use_metadata_extension_for_512px=args.use_metadata_extension_for_512px,
        )
        if args.require_image_exists and not path.exists():
            skipped_missing += 1
            continue

        image_id = record_id(metadata)
        file_ext = record_extension(metadata, args.image_subdir, args.use_metadata_extension_for_512px)
        for identity in identities:
            if args.max_images_per_identity and len(grouped[identity]) >= args.max_images_per_identity:
                continue
            grouped[identity].append(
                {
                    "path": str(path),
                    "identity": identity,
                    "danbooru_id": image_id,
                    "rating": rating,
                    "file_ext": file_ext,
                    "source": args.image_subdir,
                }
            )

    eligible = {identity: records for identity, records in grouped.items() if len(records) >= args.min_images_per_identity}
    if args.max_identities:
        ranked = sorted(eligible.items(), key=lambda item: (-len(item[1]), item[0]))[: args.max_identities]
        eligible = dict(ranked)

    split_records_by_name = {"train": [], "val": [], "test": []}
    for identity, records in sorted(eligible.items()):
        parts = split_records(records, split_values, rng)
        for split_name, split_records_for_identity in parts.items():
            split_records_by_name[split_name].extend(split_records_for_identity)

    output_dir = Path(args.output_dir)
    counts = {
        split_name: write_jsonl(output_dir / f"{split_name}.jsonl", records)
        for split_name, records in split_records_by_name.items()
    }
    all_records = [record for records in split_records_by_name.values() for record in records]
    counts["all"] = write_jsonl(output_dir / "all.jsonl", all_records)

    summary = {
        "metadata": str(Path(args.metadata)),
        "image_root": str(Path(args.image_root)),
        "image_subdir": args.image_subdir,
        "scanned_records": scanned,
        "eligible_identities": len(eligible),
        "counts": counts,
        "skipped_missing_images": skipped_missing,
        "skipped_no_identity": skipped_no_identity,
        "skipped_multi_identity": skipped_multi_identity,
        "min_images_per_identity": args.min_images_per_identity,
        "max_images_per_identity": args.max_images_per_identity,
        "ratings": sorted(ratings),
        "single_character_only": args.single_character_only,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create training manifests from Danbooru2021 metadata without copying images.")
    parser.add_argument("--metadata", required=True, help="Path to JSONL, JSONL.XZ, metadata directory, or metadata JSON tarball.")
    parser.add_argument("--image-root", required=True, help="Root directory containing original/ or 512px/ buckets.")
    parser.add_argument("--output-dir", required=True, help="Directory for train.jsonl, val.jsonl, test.jsonl, all.jsonl, and summary.json.")
    parser.add_argument("--image-subdir", default="512px", choices=["512px", "original"], help="Danbooru image subtree to reference.")
    parser.add_argument("--ratings", default="s", help="Comma-separated ratings to keep. Use an empty string to keep all ratings.")
    parser.add_argument("--identity-field", default=None, help="Override metadata field containing character tags.")
    parser.add_argument("--single-character-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-images-per-identity", type=int, default=5)
    parser.add_argument("--max-images-per-identity", type=int, default=200, help="Use 0 to keep all images per identity.")
    parser.add_argument("--max-identities", type=int, default=0, help="Use 0 to keep all eligible identities.")
    parser.add_argument("--split", default="0.8,0.1,0.1", help="Train,val,test ratios.")
    parser.add_argument("--limit-records", type=int, default=0, help="Stop after scanning this many metadata records. Use 0 for no limit.")
    parser.add_argument("--require-image-exists", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-metadata-extension-for-512px", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=523)
    return parser.parse_args()


def main() -> None:
    summary = prepare_danbooru_manifests(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
