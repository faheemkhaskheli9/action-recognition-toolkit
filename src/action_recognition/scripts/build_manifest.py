"""Build manifest.csv from a `<root>/<class_name>/<video file>` folder layout.

For datasets labeled via the Django web app instead, this script isn't needed
— the app writes manifest.csv directly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from action_recognition.data.manifest import discover_class_folders, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Folder containing one subfolder per class")
    parser.add_argument("output", type=Path, help="Where to write manifest.csv")
    args = parser.parse_args()

    df = discover_class_folders(args.root)
    write_manifest(df, args.output)
    print(f"Wrote {len(df)} rows across {df['label'].nunique()} classes to {args.output}")


if __name__ == "__main__":
    main()
