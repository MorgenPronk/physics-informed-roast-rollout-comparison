"""Utilities for filtering roast sequences into cleaner experimental subsets."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterable, Sequence, TypeVar

P_CODE_PATTERN = re.compile(r"(P\d+)")

T = TypeVar("T")


def parse_p_code(roast_id: str) -> str | None:
    match = P_CODE_PATTERN.search(roast_id)
    if match is None:
        return None
    return match.group(1)


def infer_source_bucket(source_path: str) -> str:
    normalized = source_path.replace("\\", "/").lower()
    if "/p2/" in normalized:
        return "p2"
    if "/p3/" in normalized:
        return "p3"
    if "/p4/" in normalized and "azure" in normalized:
        return "azure_p4"
    if "/p4/" in normalized:
        return "p4"
    if "coffee_data_with_mass" in normalized:
        return "coffee_data_with_mass"
    return "other"


def build_roast_metadata_index(metadata_path: Path) -> Dict[str, Dict[str, str]]:
    if not metadata_path.exists():
        return {}

    index: Dict[str, Dict[str, str]] = {}
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            normalized = row.get("normalized", "").strip()
            status = row.get("status", "").strip().lower()
            source = row.get("source", "").strip()
            if status != "ok" or not normalized:
                continue
            roast_id = Path(normalized).stem
            index[roast_id] = {
                "source_path": source,
                "source_bucket": infer_source_bucket(source),
            }
    return index


def filter_sequences(
    sequences: Sequence[T],
    *,
    metadata_index: Dict[str, Dict[str, str]] | None = None,
    include_p_codes: Iterable[str] | None = None,
    exclude_p_codes: Iterable[str] | None = None,
    include_source_buckets: Iterable[str] | None = None,
    exclude_source_buckets: Iterable[str] | None = None,
) -> list[T]:
    include_p = {value.strip().upper() for value in include_p_codes or [] if value.strip()}
    exclude_p = {value.strip().upper() for value in exclude_p_codes or [] if value.strip()}
    include_sources = {value.strip().lower() for value in include_source_buckets or [] if value.strip()}
    exclude_sources = {value.strip().lower() for value in exclude_source_buckets or [] if value.strip()}
    metadata_index = metadata_index or {}

    filtered: list[T] = []
    for seq in sequences:
        roast_id = getattr(seq, "roast_id", "")
        p_code = parse_p_code(roast_id)
        source_bucket = metadata_index.get(roast_id, {}).get("source_bucket", "")

        if include_p and (p_code is None or p_code.upper() not in include_p):
            continue
        if exclude_p and p_code is not None and p_code.upper() in exclude_p:
            continue
        if include_sources and source_bucket.lower() not in include_sources:
            continue
        if exclude_sources and source_bucket.lower() in exclude_sources:
            continue
        filtered.append(seq)
    return filtered
