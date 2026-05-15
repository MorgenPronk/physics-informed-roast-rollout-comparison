"""Column naming and schema helpers for the roaster dataset."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Set

CANONICAL_COLUMNS = [
    "timestamp",
    "tc",
    "t1",
    "t2",
    "t3",
    "flow_gas",
    "vac",
    "drum_speed",
    "vaf_close",
    "vat_open",
    "gas_pressure",
    "air_speed",
    "set_bf",
    "set_vac",
    "set_tt",
    "set_vaf",
    "set_vat",
]

MINIMUM_MODEL_COLUMNS = ["timestamp", "tc", "t1", "t2", "flow_gas", "drum_speed", "air_speed"]

ALIASES: Dict[str, str] = {
    "time stamp": "timestamp",
    "timestamp": "timestamp",
    "tc [0-300c] time": "timestamp",
    "actual value tc [c]": "tc",
    "tc [0-300c] valuey": "tc",
    "actual value t1 [c]": "t1",
    "t1 [0-900c] valuey": "t1",
    "actual value t2 [c]": "t2",
    "t2 [0-900c] valuey": "t2",
    "actual value t3 [c]": "t3",
    "flow gas bf [%]": "flow_gas",
    "present value vac [% * 0.1]": "vac",
    "actual value tt [rpm]": "drum_speed",
    "actual value closing vaf [%]": "vaf_close",
    "actual value opening vat [%]": "vat_open",
    "gas pressure bf [mbar]": "gas_pressure",
    "air speed meter fscp [m/sec]": "air_speed",
    "set % command bf [%]": "set_bf",
    "setpoint vac [% * 0.1]": "set_vac",
    "setpoint drum speed tt [% * 0.1]": "set_tt",
    "setpoint closing vaf [%]": "set_vaf",
    "setpoint opening vat [%]": "set_vat",
}


def normalize_column_name(name: str) -> str:
    name = (name or "").strip().strip('"').replace("\u00c2", "")
    name = re.sub(r"\s+", " ", name)
    name = name.lower()
    name = name.replace("\u00b0", "")
    return name


def canonicalize_header(header: Iterable[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    used_targets: Set[str] = set()
    for column in header:
        key = normalize_column_name(column)
        target = ALIASES.get(key)
        if target and target not in used_targets:
            mapping[column] = target
            used_targets.add(target)
    return mapping


def has_minimum_columns(canonical_columns: Iterable[str]) -> bool:
    available = set(canonical_columns)
    return all(column in available for column in MINIMUM_MODEL_COLUMNS)
