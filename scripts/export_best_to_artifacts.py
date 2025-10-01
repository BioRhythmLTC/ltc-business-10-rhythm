#!/usr/bin/env python3
"""
Export best_model to artifacts WITHOUT re-serializing via Transformers.

This performs a byte-preserving copy of all files from best_dir to art_dir,
then preserves/creates metadata files like label_mapping.json and versions.json
if provided via flags. This guarantees tokenizer files remain identical to what
was produced during training.

Usage:
  python3 scripts/export_best_to_artifacts.py \
    --best_dir /path/to/ner_runs/<alias>/<run_id>/best_model \
    --art_dir  /path/to/artifacts/<alias>/<run_id>

Optional:
  --label_mapping /path/to/label_mapping.json  (will be copied into art_dir)
  --versions      /path/to/versions.json       (will be copied into art_dir)
  --clean_art_dir (remove art_dir before copying)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Byte-preserving export of best_model → artifacts")
    p.add_argument("--best_dir", required=True, help="Path to best_model directory")
    p.add_argument("--art_dir", required=True, help="Target artifacts directory")
    p.add_argument("--label_mapping", default="", help="Optional label_mapping.json to place in art_dir")
    p.add_argument("--versions", default="", help="Optional versions.json to place in art_dir")
    p.add_argument("--clean_art_dir", action="store_true", help="Remove art_dir before copying")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def copy_file(src: str, dst: str) -> None:
    ensure_dir(os.path.dirname(dst))
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()

    best_dir = os.path.abspath(args.best_dir)
    art_dir = os.path.abspath(args.art_dir)

    if not os.path.isdir(best_dir):
        print(f"ERROR: best_dir not found: {best_dir}", file=sys.stderr)
        sys.exit(2)

    if args.clean_art_dir and os.path.isdir(art_dir):
        shutil.rmtree(art_dir)

    ensure_dir(art_dir)

    # Byte-preserving directory copy (overwrites existing files)
    for root, dirs, files in os.walk(best_dir):
        rel = os.path.relpath(root, best_dir)
        out_root = art_dir if rel == "." else os.path.join(art_dir, rel)
        ensure_dir(out_root)
        for d in dirs:
            ensure_dir(os.path.join(out_root, d))
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(out_root, f)
            copy_file(src, dst)

    # Optionally add metadata files without changing model/tokenizer files
    if args.label_mapping:
        copy_file(os.path.abspath(args.label_mapping), os.path.join(art_dir, "label_mapping.json"))
    if args.versions:
        copy_file(os.path.abspath(args.versions), os.path.join(art_dir, "versions.json"))

    print("Export completed:")
    print("  best_dir:", best_dir)
    print("  art_dir: ", art_dir)


if __name__ == "__main__":
    main()
