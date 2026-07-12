"""Helpers for identifying the modeled portion of a roast trajectory."""

from __future__ import annotations

from typing import Sequence


def _detect_drop_start_index(tc_values: Sequence[float]) -> int:
    """Find the first kept index on the falling side of a peak-to-minimum event."""
    if len(tc_values) < 20:
        return 0

    search_limit = min(len(tc_values), max(40, int(len(tc_values) * 0.7)))
    turning_start = min(5, search_limit - 1)
    if search_limit - turning_start < 5:
        return 0

    turning_index = min(range(turning_start, search_limit), key=lambda idx: tc_values[idx])
    if turning_index <= 2:
        return 0
    peak_index = max(range(turning_index), key=lambda idx: tc_values[idx])
    if turning_index - peak_index < 3:
        return 0

    drop_magnitude = tc_values[peak_index] - tc_values[turning_index]
    if drop_magnitude < 12.0:
        return 0

    drop_start = min(peak_index + 1, turning_index)
    for idx in range(peak_index + 1, min(turning_index + 1, peak_index + 6)):
        immediate_drop = tc_values[idx - 1] - tc_values[idx]
        cumulative_drop = tc_values[peak_index] - tc_values[idx]
        if immediate_drop >= 1.0 or cumulative_drop >= 2.0:
            drop_start = idx
            break
    return drop_start


def detect_charge_start_index(tc_values: Sequence[float]) -> int:
    """Return the index where the modeled roast should begin after bean drop."""
    return _detect_drop_start_index(tc_values)


def detect_dump_end_index(tc_values: Sequence[float], start_index: int = 0) -> int:
    """Return the exclusive end index just before final bean-dump cooling."""
    n = len(tc_values)
    if n < 20:
        return n

    peak_search_start = max(start_index + 15, int(n * 0.45))
    if peak_search_start >= n - 4:
        return n

    tail_peak = max(range(peak_search_start, n), key=lambda idx: tc_values[idx])
    if tail_peak >= n - 4:
        return n

    for idx in range(tail_peak + 1, n - 2):
        immediate_drop = tc_values[idx - 1] - tc_values[idx]
        cumulative_drop = tc_values[tail_peak] - tc_values[idx]
        window = tc_values[idx : min(n, idx + 4)]
        if len(window) < 3:
            continue
        sustained = sum(window[j + 1] <= window[j] + 0.5 for j in range(len(window) - 1))
        if cumulative_drop >= 8.0 and sustained >= len(window) - 2:
            return max(start_index + 5, tail_peak)
        if immediate_drop >= 3.0 and cumulative_drop >= 5.0 and sustained >= len(window) - 2:
            return max(start_index + 5, tail_peak)
    return n


def detect_modeled_window_bounds(tc_values: Sequence[float]) -> tuple[int, int]:
    """Return [start, end) bounds for the bean-charge-to-dump window."""
    n = len(tc_values)
    if n < 20:
        return 0, n

    start = detect_charge_start_index(tc_values)
    end = detect_dump_end_index(tc_values, start_index=start)
    return start, max(start + 5, end)
