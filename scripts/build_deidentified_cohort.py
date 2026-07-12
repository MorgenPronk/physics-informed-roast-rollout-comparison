#!/usr/bin/env python
"""Build the de-identified public cohort CSV (release option 1).

Produces data/deidentified/roast_timeseries_deidentified_p2_only.csv from the
private production cohort with three changes and NO change to any measured
value:

  1. Roast IDs are replaced with neutral identifiers (``ROAST_###_P##``),
     chosen so that ``stable_hash_bucket`` assigns every roast to the SAME
     train/val/test band as its original ID. The released deterministic split
     logic therefore reproduces the manuscript's exact 150/35/36 cohort
     membership from this file.
  2. Timestamps are re-based per roast to a fictitious year-2000 epoch,
     preserving every intra-roast sample interval exactly (the physics uses
     only the intervals). Production dates and times are removed.
  3. Only the 221 modeled cohort roasts (labels P10/P13/P16) are included;
     non-cohort roasts in the raw export are dropped.

The original->neutral mapping is written to
data/metadata/roast_id_neutralization_map_PRIVATE.json (private, gitignored)
and is shared by scripts/build_release_tree.py so released derived artifacts
use the same neutral names.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from roaster_piml.io_utils import stable_hash_bucket  # noqa: E402
from scripts.tune_manuscript_models import load_manuscript_cohort  # noqa: E402

IN_CSV = ROOT / "data" / "processed" / "roast_timeseries_p2_only.csv"
OUT_CSV = ROOT / "data" / "deidentified" / "roast_timeseries_deidentified_p2_only.csv"
MAP_PATH = ROOT / "data" / "metadata" / "roast_id_neutralization_map_PRIVATE.json"

EPOCH = datetime(2000, 1, 1, 6, 0, 0)


def band(bucket: int) -> str:
    return "train" if bucket < 70 else ("val" if bucket < 85 else "test")


def main() -> None:
    sequences = load_manuscript_cohort(IN_CSV, None, ["P10", "P13", "P16"], ["p2"])
    cohort_ids = sorted(s.roast_id for s in sequences)
    print(f"cohort roasts: {len(cohort_ids)}")

    # Band-preserving neutral assignment: smallest unused index whose neutral
    # ID hashes into the same split band as the original ID.
    id_map: dict[str, str] = {}
    used: set[int] = set()
    for rid in cohort_ids:
        label = rid.rsplit("_", 1)[1]
        target = band(stable_hash_bucket(rid))
        i = 1
        while True:
            if i not in used:
                cand = f"ROAST_{i:03d}_{label}"
                if band(stable_hash_bucket(cand)) == target:
                    id_map[rid] = cand
                    used.add(i)
                    break
            i += 1
            if i > 9999:
                raise RuntimeError(f"no band-preserving neutral ID for {rid}")
    bands = [band(stable_hash_bucket(v)) for v in id_map.values()]
    print("neutral split bands:", {b: bands.count(b) for b in ("train", "val", "test")})

    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(id_map, indent=2), encoding="utf-8")
    print(f"private map (221 roasts, band-preserving) -> {MAP_PATH}")

    # Rewrite the CSV: cohort-only, neutral IDs, per-roast re-based timestamps.
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    roast_offset: dict[str, timedelta] = {}
    roast_t0: dict[str, datetime] = {}
    n_out = 0
    with IN_CSV.open("r", encoding="utf-8", newline="") as fin, \
         OUT_CSV.open("w", encoding="utf-8", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()
        for k, row in enumerate(reader):
            rid = row["roast_id"]
            if rid not in id_map:
                continue  # non-cohort roast: drop
            ts = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            if rid not in roast_t0:
                roast_t0[rid] = ts
                roast_offset[rid] = (EPOCH + timedelta(hours=len(roast_t0))) - ts
            row["timestamp"] = (ts + roast_offset[rid]).strftime("%Y-%m-%d %H:%M:%S")
            row["roast_id"] = id_map[rid]
            writer.writerow(row)
            n_out += 1
    print(f"wrote {OUT_CSV} ({n_out} rows, {len(roast_t0)} roasts)")


if __name__ == "__main__":
    main()
