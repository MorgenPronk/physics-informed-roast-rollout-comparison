"""I/O helpers with robust parsing for messy industrial exports."""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

ENCODINGS = (
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
    "latin-1",
)
DATETIME_FORMATS = (
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)


def _is_probably_wrong_decode(text: str) -> bool:
    if not text:
        return True
    nul_ratio = text.count("\x00") / max(1, len(text))
    return nul_ratio > 0.05


def read_text_with_fallback(path: Path) -> str:
    last_error: Optional[Exception] = None
    for encoding in ENCODINGS:
        try:
            text = path.read_text(encoding=encoding)
            if _is_probably_wrong_decode(text):
                continue
            return text.replace("\x00", "")
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc

    if last_error is not None:
        # Last resort for malformed files.
        raw = path.read_bytes()
        return raw.decode("latin-1", errors="replace").replace("\x00", "")
    raise RuntimeError(f"Unable to read {path}")


def detect_delimiter(text: str) -> str:
    sample = "\n".join(text.splitlines()[:20])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter
    except Exception:
        semicolons = sample.count(";")
        commas = sample.count(",")
        tabs = sample.count("\t")
        if tabs > semicolons and tabs > commas:
            return "\t"
        return ";" if semicolons > commas else ","


def parse_float(value: str) -> Optional[float]:
    if value is None:
        return None
    raw = value.strip().strip('"')
    if not raw:
        return None

    cleaned = raw.replace("\u00a0", " ").replace(" ", "")

    if "," in cleaned and "." not in cleaned:
        cleaned = cleaned.replace(",", ".")
    elif "," in cleaned and "." in cleaned:
        # Assume comma is thousands separator when both are present.
        cleaned = cleaned.replace(",", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_timestamp(value: str) -> Optional[dt.datetime]:
    if value is None:
        return None
    raw = value.strip().strip('"')
    if not raw:
        return None

    for fmt in DATETIME_FORMATS:
        try:
            return dt.datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def csv_dict_reader(path: Path) -> Iterator[Tuple[Dict[str, str], str]]:
    text = read_text_with_fallback(path)
    delimiter = detect_delimiter(text)
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    for row in reader:
        yield row, delimiter


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_timestamp(value: dt.datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def stable_split_key(text: str) -> int:
    return sum(ord(char) for char in text) % 100


def stable_hash_bucket(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def write_csv(path: Path, fieldnames: Iterable[str], rows: List[Dict[str, object]]) -> None:
    ensure_parent_dir(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
