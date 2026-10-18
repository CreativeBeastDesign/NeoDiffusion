#!/usr/bin/env python3
"""ingest.py — copy raw evidence into paper/facts/data/ immutably and record its sha256.

    python3 paper/tools/ingest.py scratch/track1_kv_starvation_k1.jsonl [more files...] [--subdir 2026-10-18]

Refuses to overwrite an existing file (data is immutable — ingest a corrected file under a new
name). Appends `<sha256>  <relative name>` to facts/data/MANIFEST.sha256 and prints a block you
can paste into the run manifest (R-*.md): source path, size, sha256, row count for JSONL.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "facts" / "data"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--subdir", default="")
    a = ap.parse_args()
    dest_dir = DATA / a.subdir if a.subdir else DATA
    dest_dir.mkdir(parents=True, exist_ok=True)
    manifest = DATA / "MANIFEST.sha256"
    rc = 0
    for src in map(Path, a.files):
        if not src.is_file():
            print(f"skip: {src} is not a file", file=sys.stderr); rc = 1; continue
        dest = dest_dir / src.name
        if dest.exists():
            print(f"refuse: {dest} exists (immutable); rename the source to ingest a new version", file=sys.stderr)
            rc = 1; continue
        shutil.copy2(src, dest)
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        if h != hashlib.sha256(src.read_bytes()).hexdigest():
            print(f"copy mismatch for {src}", file=sys.stderr); dest.unlink(); rc = 1; continue
        dest.chmod(0o444)
        name = str(dest.relative_to(DATA))
        with manifest.open("a") as m:
            m.write(f"{h}  {name}\n")
        rows = sum(1 for _ in dest.open()) if dest.suffix == ".jsonl" else None
        print(f"data: facts/data/{name}\n  source: {src}\n  bytes: {dest.stat().st_size}\n  sha256: {h}"
              + (f"\n  jsonl rows: {rows}" if rows is not None else ""))
    return rc


if __name__ == "__main__":
    sys.exit(main())
