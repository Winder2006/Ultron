from __future__ import annotations

import argparse
import os
from pathlib import Path
import requests


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Piper voice (.onnx/.json)")
    parser.add_argument("--onnx", required=True, help="URL to .onnx file on Hugging Face")
    parser.add_argument("--json", required=True, help="URL to .json config file on Hugging Face")
    parser.add_argument("--out", default="voices", help="Destination directory for voice files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    onnx_path = out_dir / "voice.onnx"
    json_path = out_dir / "voice.json"

    download_file(args.onnx, onnx_path)
    download_file(args.json, json_path)
    print(f"Saved: {onnx_path}")
    print(f"Saved: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


