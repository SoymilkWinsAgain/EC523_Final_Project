#!/usr/bin/env python
from pathlib import Path
import json
import sys
import argparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from who_is_this_anime_girl.devise import TEXT_MODEL_NAME
from who_is_this_anime_girl.infer import CachedGallerySearcher


def main() -> None:
    parser = argparse.ArgumentParser(description="Query a DeViSE gallery index with text.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--text-model", default=TEXT_MODEL_NAME)
    parser.add_argument("--text-embedding-dim", type=int, default=256)
    parser.add_argument("--text-device", default="auto")
    args = parser.parse_args()

    searcher = CachedGallerySearcher(
        checkpoint_path=args.checkpoint,
        index_dir=args.index_dir,
        device=args.device,
        text_model_name=args.text_model,
        text_embedding_dim=args.text_embedding_dim,
        text_device=args.text_device,
    )
    result = searcher.search_text(args.query, top_k=args.top_k)
    print(json.dumps({"ok": True, "query": args.query, **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
