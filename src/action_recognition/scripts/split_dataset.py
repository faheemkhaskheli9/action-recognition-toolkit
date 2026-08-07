"""Add stratified train/val/test splits to a manifest, and write its label_map.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from action_recognition.data.manifest import (
    build_label_map,
    read_manifest,
    save_label_map,
    stratified_split,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="manifest.csv with video_path,label columns")
    parser.add_argument("--output", type=Path, default=None, help="Defaults to overwriting the input manifest")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = read_manifest(args.manifest)
    df = stratified_split(df, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed)

    output = args.output or args.manifest
    write_manifest(df, output)

    label_map = build_label_map(df["label"])
    label_map_path = output.with_name("label_map.json")
    save_label_map(label_map, label_map_path)

    counts = df["split"].value_counts().to_dict()
    print(f"Wrote {output} ({counts}) and {label_map_path} ({len(label_map)} classes)")


if __name__ == "__main__":
    main()
