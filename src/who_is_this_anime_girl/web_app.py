from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image

from .data import count_images_by_identity
from .index import build_gallery_index
from .infer import search_image_bytes


STATIC_DIR = Path(__file__).with_name("static")


def safe_identity_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown_identity"


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


class AnimeGirlHandler(BaseHTTPRequestHandler):
    checkpoint_path: Path
    gallery_dir: Path
    index_dir: Path
    device: str
    workers: int

    def log_message(self, format: str, *args) -> None:
        print(f"{self.address_string()} - {format % args}")

    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self.serve_file(STATIC_DIR / "index.html")
            return
        if route.startswith("/static/"):
            static_path = (STATIC_DIR / route.removeprefix("/static/")).resolve()
            if not is_relative_to(static_path, STATIC_DIR):
                self.send_error(403)
                return
            self.serve_file(static_path)
            return
        if route == "/api/status":
            self.handle_status()
            return
        if route.startswith("/gallery/"):
            relative = unquote(route.removeprefix("/gallery/"))
            gallery_path = (self.gallery_dir / relative).resolve()
            if not is_relative_to(gallery_path, self.gallery_dir):
                self.send_error(403)
                return
            self.serve_file(gallery_path)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        try:
            if route == "/api/enroll":
                self.handle_enroll()
            elif route == "/api/rebuild":
                self.handle_rebuild()
            elif route == "/api/query":
                self.handle_query()
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"ok": False, "error": str(exc)}, status=500)

    def parse_form(self) -> cgi.FieldStorage:
        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": self.headers.get("Content-Length", "0"),
            },
        )

    def handle_status(self) -> None:
        self.send_json(
            {
                "ok": True,
                "checkpoint_exists": self.checkpoint_path.exists(),
                "index_exists": (self.index_dir / "gallery.faiss").exists() and (self.index_dir / "metadata.json").exists(),
                "checkpoint": str(self.checkpoint_path),
                "gallery_dir": str(self.gallery_dir),
                "index_dir": str(self.index_dir),
                "identities": count_images_by_identity(self.gallery_dir),
            }
        )

    def handle_enroll(self) -> None:
        form = self.parse_form()
        identity = safe_identity_name(form.getfirst("identity", ""))
        upload_fields = form["images"] if "images" in form else []
        if not isinstance(upload_fields, list):
            upload_fields = [upload_fields]
        if not upload_fields:
            self.send_json({"ok": False, "error": "No image files were uploaded."}, status=400)
            return

        identity_dir = self.gallery_dir / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []
        for field in upload_fields:
            if not getattr(field, "filename", None):
                continue
            image = Image.open(field.file).convert("RGB")
            filename = f"{uuid.uuid4().hex}.jpg"
            output_path = identity_dir / filename
            image.save(output_path, format="JPEG", quality=95)
            saved_paths.append(str(output_path))

        self.send_json({"ok": True, "identity": identity, "saved": saved_paths, "count": len(saved_paths)})

    def handle_rebuild(self) -> None:
        if not self.checkpoint_path.exists():
            self.send_json({"ok": False, "error": "Checkpoint does not exist."}, status=400)
            return
        metadata = build_gallery_index(
            checkpoint_path=self.checkpoint_path,
            gallery_dir=self.gallery_dir,
            output_dir=self.index_dir,
            workers=self.workers,
            device=self.device,
        )
        self.send_json({"ok": True, "indexed_images": len(metadata["items"])})

    def handle_query(self) -> None:
        form = self.parse_form()
        if "image" not in form:
            self.send_json({"ok": False, "error": "No query image was uploaded."}, status=400)
            return
        if not self.checkpoint_path.exists():
            self.send_json({"ok": False, "error": "Checkpoint does not exist."}, status=400)
            return
        if not (self.index_dir / "gallery.faiss").exists():
            self.send_json({"ok": False, "error": "Gallery index does not exist. Rebuild the index first."}, status=400)
            return

        image_field = form["image"]
        image_bytes = image_field.file.read()
        top_k = int(form.getfirst("top_k", "5"))
        matches = search_image_bytes(
            checkpoint_path=self.checkpoint_path,
            index_dir=self.index_dir,
            image_bytes=image_bytes,
            top_k=top_k,
            device=self.device,
        )
        gallery_root = self.gallery_dir.resolve()
        for match in matches:
            match_path = Path(match["path"]).resolve()
            if is_relative_to(match_path, gallery_root):
                match["gallery_url"] = "/gallery/" + str(match_path.relative_to(gallery_root)).replace("\\", "/")
        self.send_json({"ok": True, "matches": matches})


def run_server(
    checkpoint: str | Path,
    gallery_dir: str | Path,
    index_dir: str | Path,
    host: str,
    port: int,
    device: str,
    workers: int,
) -> None:
    AnimeGirlHandler.checkpoint_path = Path(checkpoint)
    AnimeGirlHandler.gallery_dir = Path(gallery_dir)
    AnimeGirlHandler.index_dir = Path(index_dir)
    AnimeGirlHandler.device = device
    AnimeGirlHandler.workers = workers

    AnimeGirlHandler.gallery_dir.mkdir(parents=True, exist_ok=True)
    AnimeGirlHandler.index_dir.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((host, port), AnimeGirlHandler)
    print(f"Serving Who Is This Anime Girl at http://{host}:{port}")
    print("Press Ctrl+C to stop the server.")
    server.serve_forever()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local anime character retrieval web UI.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained checkpoint.")
    parser.add_argument("--gallery-dir", default="data/gallery", help="ImageFolder gallery directory.")
    parser.add_argument("--index-dir", default="artifacts/gallery_index", help="Directory containing the FAISS gallery index.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_server(
        checkpoint=args.checkpoint,
        gallery_dir=args.gallery_dir,
        index_dir=args.index_dir,
        host=args.host,
        port=args.port,
        device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
