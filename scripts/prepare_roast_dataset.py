#!/usr/bin/env python
"""Normalize raw roast CSVs and create a consolidated modeling dataset."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from roaster_piml.dataset import consolidate_rows, normalize_roast_csv  # noqa: E402


def find_candidate_csvs(raw_root: Path) -> List[Path]:
    candidates: List[Path] = []
    for path in raw_root.rglob("*.csv"):
        name = path.name.lower()
        if "basic_stats" in name:
            continue
        candidates.append(path)
    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare roaster dataset from raw exports")
    parser.add_argument("--raw-root", type=Path, default=Path("Thesis/prior_experiment_files"))
    parser.add_argument("--normalized-root", type=Path, default=Path("data/interim/normalized"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/roast_timeseries.csv"))
    parser.add_argument("--index", type=Path, default=Path("data/metadata/normalized_files.csv"))
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum CSV files to process. Use 0 to process all discovered files.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=5,
        help="Minimum seconds between retained samples within each roast.",
    )
    args = parser.parse_args()

    candidates = find_candidate_csvs(args.raw_root)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    normalized_paths: List[Path] = []
    index_rows = []

    for csv_path in candidates:
        rel = csv_path.relative_to(args.raw_root)
        target = args.normalized_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)

        stats = normalize_roast_csv(csv_path, target)
        if stats is None:
            index_rows.append(
                {
                    "source": csv_path.as_posix(),
                    "normalized": "",
                    "status": "skipped",
                    "input_rows": 0,
                    "kept_rows": 0,
                    "dropped_rows": 0,
                }
            )
            continue

        normalized_paths.append(target)
        index_rows.append(
            {
                "source": csv_path.as_posix(),
                "normalized": target.as_posix(),
                "status": "ok",
                "input_rows": stats.input_rows,
                "kept_rows": stats.kept_rows,
                "dropped_rows": stats.dropped_rows,
            }
        )

    args.index.parent.mkdir(parents=True, exist_ok=True)
    with args.index.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "normalized", "status", "input_rows", "kept_rows", "dropped_rows"],
        )
        writer.writeheader()
        writer.writerows(index_rows)

    total_rows = consolidate_rows(normalized_paths, args.output, step_seconds=args.downsample)
    print(f"CSV files discovered: {len(find_candidate_csvs(args.raw_root))}")
    print(f"CSV files processed: {len(candidates)}")
    print(f"Normalized files: {len(normalized_paths)}")
    print(f"Consolidated rows: {total_rows}")


if __name__ == "__main__":
    main()
