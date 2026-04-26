from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download


def parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List or download selected Hugging Face model files.")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repository ID.")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output-dir", default="artifacts/hf_models")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token. Prefer HF_TOKEN in the environment.")
    parser.add_argument("--list-files", action="store_true", help="Only list repository files.")
    parser.add_argument("--filename", action="append", default=[], help="Download one exact file. Can be repeated.")
    parser.add_argument("--include", default=None, help="Comma-separated allow patterns for snapshot_download.")
    parser.add_argument("--allow-full-snapshot", action="store_true", help="Permit downloading the whole repository snapshot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_files:
        files = list_repo_files(args.repo_id, repo_type=args.repo_type, revision=args.revision, token=args.token)
        print(json.dumps(files, indent=2))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.filename:
        downloaded = [
            hf_hub_download(
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                revision=args.revision,
                filename=filename,
                local_dir=output_dir,
                token=args.token,
            )
            for filename in args.filename
        ]
        print(json.dumps(downloaded, indent=2))
        return

    allow_patterns = parse_csv(args.include)
    if allow_patterns is None and not args.allow_full_snapshot:
        raise ValueError("Refusing a full snapshot by default. Pass --include, --filename, or --allow-full-snapshot.")

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=output_dir,
        allow_patterns=allow_patterns,
        token=args.token,
    )
    print(path)


if __name__ == "__main__":
    main()
