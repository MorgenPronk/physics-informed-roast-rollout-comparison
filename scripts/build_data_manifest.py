#!/usr/bin/env python
"""Create a file inventory for all provided thesis assets."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import Iterable, List


def sha256_digest(path: Path, max_bytes: int = 65_536) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 64)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
            if total >= max_bytes:
                break
    return digest.hexdigest()


def iter_files(roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                files.append(path)
    return sorted(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest for thesis assets")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("data/metadata/data_manifest.csv"))
    parser.add_argument("--hash", action="store_true", help="Include partial SHA256 hash")
    args = parser.parse_args()

    project_root = args.root.resolve()
    source_roots = [
        project_root / "Thesis",
        project_root / "Scientific Reports Journal Paper",
    ]

    files = iter_files(source_roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in files:
        rel = path.relative_to(project_root)
        top_folder = rel.parts[0] if rel.parts else ""
        subgroup = rel.parts[1] if len(rel.parts) > 1 else ""
        rows.append(
            {
                "relative_path": rel.as_posix(),
                "top_folder": top_folder,
                "subgroup": subgroup,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                "sha256_head_64kb": sha256_digest(path) if args.hash else "",
            }
        )

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["relative_path", "top_folder", "subgroup", "extension", "size_bytes", "sha256_head_64kb"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_md = Path("data/metadata/data_manifest_summary.md")
    total_size = sum(row["size_bytes"] for row in rows)
    summary_md.write_text(
        "\n".join(
            [
                "# Data Manifest Summary",
                "",
                f"- Files indexed: {len(rows)}",
                f"- Total size (bytes): {total_size}",
                f"- Hashes included: {args.hash}",
                f"- Output CSV: `{args.output.as_posix()}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Indexed {len(rows)} files -> {args.output}")


if __name__ == "__main__":
    main()
