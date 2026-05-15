"""Dataset preparation utilities for roaster trajectories."""

from __future__ import annotations

import csv
import datetime as dt
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from xml.etree import ElementTree as ET

from .io_utils import detect_delimiter, format_timestamp, parse_float, parse_timestamp, read_text_with_fallback
from .schema import CANONICAL_COLUMNS, canonicalize_header, has_minimum_columns


@dataclass
class NormalizationStats:
    input_rows: int
    kept_rows: int
    dropped_rows: int


EXPERIMENT_XLSX_COLUMNS = {
    "timestamp": 1,
    "tc": 2,
    "t1": 4,
    "t2": 6,
    "t3": 7,
    # Zenia's MATLAB uses column 20 (1-based), which is the BF command signal.
    "flow_gas": 19,
    "vac": 23,
    "drum_speed": 26,
    "vaf_close": 27,
    "vat_open": 29,
    "gas_pressure": 40,
    "air_speed": 71,
    "set_bf": 19,
    "set_vac": 24,
    "set_tt": 25,
    "set_vaf": 30,
    "set_vat": 32,
}
_OOXML_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_EXPERIMENT_NAME_RE = re.compile(
    r"^ROAST_(?P<month>\d+)_(?P<day>\d+)_(?P<year>\d+)_(?P<hour>\d+)_(?P<minute>\d+)_(?P<second>\d+)_(?P<ampm>AM|PM)_P(?P<code>\d+)\.xlsx$",
    re.IGNORECASE,
)


def _iter_rows(path: Path) -> Iterator[Dict[str, str]]:
    text = read_text_with_fallback(path)
    delimiter = detect_delimiter(text)
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    for row in reader:
        yield row


def _excel_col_to_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha()).upper()
    value = 0
    for char in letters:
        value = value * 26 + (ord(char) - ord("A") + 1)
    return value - 1


def _excel_serial_to_datetime(value: float) -> dt.datetime:
    base = dt.datetime(1899, 12, 30)
    return base + dt.timedelta(days=float(value))


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    values: List[str] = []
    for item in root.findall("a:si", _OOXML_NS):
        text = "".join(node.text or "" for node in item.iterfind(".//a:t", _OOXML_NS))
        values.append(text)
    return values


def _xlsx_cell_value(cell: ET.Element, shared_strings: List[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    value_node = cell.find("a:v", _OOXML_NS)
    if value_node is None:
        inline_text = cell.find("a:is/a:t", _OOXML_NS)
        return inline_text.text if inline_text is not None and inline_text.text is not None else ""
    raw = value_node.text or ""
    if cell_type == "s" and raw:
        return shared_strings[int(raw)]
    return raw


def _iter_xlsx_rows(path: Path) -> Iterator[Dict[int, str]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = _xlsx_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
        for row in root.findall(".//a:sheetData/a:row", _OOXML_NS):
            values: Dict[int, str] = {}
            for cell in row.findall("a:c", _OOXML_NS):
                cell_ref = cell.attrib.get("r", "")
                if not cell_ref:
                    continue
                values[_excel_col_to_index(cell_ref)] = _xlsx_cell_value(cell, shared_strings)
            if values:
                yield values


def parse_experiment_filename(path: Path) -> Dict[str, object]:
    match = _EXPERIMENT_NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected experiment filename: {path.name}")
    code = int(match.group("code"))
    if code < 100:
        raise ValueError(f"Unable to decode recipe/mass from experiment filename: {path.name}")
    recipe_number = code // 100
    batch_mass_kg = float(code % 100)
    return {
        "recipe_number": recipe_number,
        "batch_mass_kg": batch_mass_kg,
        "encoded_code": code,
    }


def normalize_roast_csv(input_path: Path, output_path: Path) -> Optional[NormalizationStats]:
    rows = list(_iter_rows(input_path))
    if not rows:
        return None

    header_map = canonicalize_header(rows[0].keys())
    if not has_minimum_columns(header_map.values()):
        return None

    normalized_rows: List[Dict[str, object]] = []
    seen_timestamps = set()

    for raw in rows:
        record: Dict[str, object] = {column: "" for column in CANONICAL_COLUMNS}

        for source_key, target_key in header_map.items():
            value = raw.get(source_key, "")
            if target_key == "timestamp":
                ts = parse_timestamp(value)
                record[target_key] = format_timestamp(ts) if ts else ""
            else:
                parsed = parse_float(value)
                record[target_key] = "" if parsed is None else parsed

        timestamp = record.get("timestamp")
        tc = record.get("tc")
        if not timestamp or tc == "":
            continue

        if timestamp in seen_timestamps:
            continue
        seen_timestamps.add(timestamp)
        normalized_rows.append(record)

    if not normalized_rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_rows)

    kept = len(normalized_rows)
    total = len(rows)
    return NormalizationStats(input_rows=total, kept_rows=kept, dropped_rows=total - kept)


def normalize_roast_xlsx(input_path: Path, output_path: Path) -> Optional[NormalizationStats]:
    rows = list(_iter_xlsx_rows(input_path))
    if len(rows) <= 1:
        return None

    normalized_rows: List[Dict[str, object]] = []
    seen_timestamps = set()

    for raw in rows[1:]:
        timestamp_value = parse_float(raw.get(EXPERIMENT_XLSX_COLUMNS["timestamp"], ""))
        if timestamp_value is None:
            continue
        timestamp = format_timestamp(_excel_serial_to_datetime(timestamp_value))

        record: Dict[str, object] = {column: "" for column in CANONICAL_COLUMNS}
        record["timestamp"] = timestamp
        for column, idx in EXPERIMENT_XLSX_COLUMNS.items():
            if column == "timestamp":
                continue
            parsed = parse_float(raw.get(idx, ""))
            record[column] = "" if parsed is None else parsed

        if not record["timestamp"] or record["tc"] == "":
            continue
        if timestamp in seen_timestamps:
            continue
        seen_timestamps.add(timestamp)
        normalized_rows.append(record)

    if not normalized_rows:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_COLUMNS)
        writer.writeheader()
        writer.writerows(normalized_rows)

    kept = len(normalized_rows)
    total = max(len(rows) - 1, 0)
    return NormalizationStats(input_rows=total, kept_rows=kept, dropped_rows=total - kept)


def load_normalized_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, object]] = []
        for row in reader:
            parsed: Dict[str, object] = {}
            for key in CANONICAL_COLUMNS:
                value = row.get(key, "")
                if key == "timestamp":
                    parsed[key] = parse_timestamp(value)
                else:
                    parsed[key] = parse_float(str(value))
            rows.append(parsed)
    return rows


def downsample_rows(rows: List[Dict[str, object]], step_seconds: int) -> List[Dict[str, object]]:
    if step_seconds <= 1:
        return rows

    filtered: List[Dict[str, object]] = []
    last_ts: Optional[dt.datetime] = None
    for row in rows:
        ts = row.get("timestamp")
        if not isinstance(ts, dt.datetime):
            continue
        if last_ts is None or (ts - last_ts).total_seconds() >= step_seconds:
            filtered.append(row)
            last_ts = ts
    return filtered


def consolidate_rows(
    paths: Iterable[Path],
    output_path: Path,
    step_seconds: int = 5,
    metadata_lookup: Optional[Dict[str, Dict[str, object]]] = None,
) -> int:
    consolidated: List[Dict[str, object]] = []
    extra_fields: List[str] = []
    if metadata_lookup:
        seen = set()
        for metadata in metadata_lookup.values():
            for key in metadata:
                if key not in seen:
                    seen.add(key)
                    extra_fields.append(key)

    for path in paths:
        roast_id = path.stem
        rows = load_normalized_csv(path)
        rows = downsample_rows(rows, step_seconds=step_seconds)
        metadata = metadata_lookup.get(roast_id, {}) if metadata_lookup else {}
        for row in rows:
            ts = row.get("timestamp")
            if not isinstance(ts, dt.datetime):
                continue
            out_row = {
                "roast_id": roast_id,
                "timestamp": format_timestamp(ts),
            }
            out_row.update(metadata)
            for key in CANONICAL_COLUMNS:
                if key == "timestamp":
                    continue
                out_row[key] = row.get(key, "")
            consolidated.append(out_row)

    if not consolidated:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["roast_id", "timestamp"] + extra_fields + [c for c in CANONICAL_COLUMNS if c != "timestamp"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(consolidated)

    return len(consolidated)
