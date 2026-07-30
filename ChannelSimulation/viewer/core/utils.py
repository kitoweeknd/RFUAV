"""
Common Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Provides common utility functions for file size formatting, progress percentage,
numeric display, and application data/log directory paths. The directory
helpers only resolve and create paths; they do not configure Python logging.
"""

from __future__ import annotations

import functools
import math
import os
import sys
from pathlib import Path
from typing import Any


def format_file_size(size_bytes: int) -> str:
    """Format file size as a readable string

    Args:
        size_bytes: File size in bytes

    Returns:
        Formatted string, e.g. "1.5 MB", "3.2 GB"
    """
    if size_bytes < 0:
        return "0 B"

    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit_idx = 0
    size = float(size_bytes)

    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024
        unit_idx += 1

    if unit_idx == 0:
        return f"{int(size)} {units[unit_idx]}"
    else:
        return f"{size:.2f} {units[unit_idx]}"


def format_duration(seconds: float) -> str:
    """Format duration as a readable string

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string, e.g. "1.23s", "234ms", "45.6µs"
    """
    if seconds < 0:
        return "0ms"

    if seconds >= 1:
        return f"{seconds:.2f}s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f}ms"
    else:
        return f"{seconds * 1e6:.2f}µs"


def format_samples(n_samples: int) -> str:
    """Format sample count as a readable string

    Args:
        n_samples: Number of samples

    Returns:
        Formatted string, e.g. "1.5M", "234K", "1234"
    """
    if n_samples < 0:
        return "0"

    if n_samples >= 1_000_000:
        return f"{n_samples / 1_000_000:.2f}M"
    elif n_samples >= 1_000:
        return f"{n_samples / 1_000:.2f}K"
    else:
        return str(n_samples)


def format_frequency(hz: float) -> str:
    """Format frequency as a readable string

    Args:
        hz: Frequency in Hz

    Returns:
        Formatted string, e.g. "1.5 GHz", "234 MHz", "1.23 kHz"
    """
    if hz < 0:
        return "0 Hz"

    if hz >= 1e9:
        return f"{hz / 1e9:.4f} GHz"
    elif hz >= 1e6:
        return f"{hz / 1e6:.4f} MHz"
    elif hz >= 1e3:
        return f"{hz / 1e3:.4f} kHz"
    else:
        return f"{hz:.4f} Hz"


def format_power(db_value: float, unit: str = "dB") -> str:
    """Format power value as a readable string

    Args:
        db_value: dB value
        unit: Unit suffix

    Returns:
        Formatted string, e.g. "-15.3 dB"
    """
    return f"{db_value:.2f} {unit}"


def parse_frequency_string(s: str) -> float | None:
    """Parse frequency string to Hz value

    Supported formats:
    - "1.5 GHz", "1.5G"
    - "100 MHz", "100M"
    - "1.23 kHz", "1.23k"
    - "500 Hz", "500"

    Args:
        s: Frequency string

    Returns:
        Hz value, returns None if parsing fails
    """
    s = s.strip().lower()

    # Drop a trailing "hz" unit so the documented full forms ("1.5 ghz",
    # "100 mhz", "1.23 khz", "500 hz") reach the magnitude logic below the same
    # way their bare-suffix forms ("1.5g", "100m", "1.23k", "500") do.
    if s.endswith("hz"):
        s = s[:-2].strip()

    # Magnitude suffix (longest first so single letters can't shadow a longer
    # match). After the "hz" strip above, "ghz"→"g", "mhz"→"m", "khz"→"k".
    multipliers = {
        "g": 1e9,
        "m": 1e6,
        "k": 1e3,
    }

    for suffix in sorted(multipliers, key=len, reverse=True):
        if s.endswith(suffix):
            try:
                num_str = s[:-len(suffix)].strip()
                val = float(num_str) * multipliers[suffix]
                # "inf"/"nan" parse as floats but are not a valid frequency;
                # treat them as a parse failure like the None returns below.
                return val if math.isfinite(val) else None
            except ValueError:
                continue

    # No magnitude suffix — parse the bare (already unit-stripped) number.
    try:
        val = float(s)
        return val if math.isfinite(val) else None
    except ValueError:
        return None


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value within specified range

    Args:
        value: Input value
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


@functools.lru_cache(maxsize=1)
def get_app_data_dir() -> Path:
    """Get application data directory

    Prefers Qt's platform-appropriate writable location (derived from the
    application's organization/app names), falling back to a hand-rolled
    per-OS path when Qt is unavailable or returns nothing:

    On Windows: %APPDATA%/channelsim_viewer
    On macOS: ~/Library/Application Support/channelsim_viewer
    On Linux: ~/.config/channelsim_viewer

    Cached with ``lru_cache`` so the mkdir/stat only runs once per process.

    Returns:
        Application data directory path
    """
    base_dir = ""
    try:
        from PySide6.QtCore import QStandardPaths
        base_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
    except Exception:
        base_dir = ""

    if base_dir:
        app_dir = Path(base_dir)
    else:
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        elif sys.platform == "darwin":
            base = Path.home() / "Library" / "Application Support"
        else:
            base = Path.home() / ".config"
        app_dir = base / "channelsim_viewer"

    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir


def get_log_dir() -> Path:
    """Get log directory

    Returns:
        Log directory path (app_data_dir/logs)
    """
    log_dir = get_app_data_dir() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def dict_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries

    Args:
        base: Base dictionary
        updates: Updates dictionary

    Returns:
        Merged new dictionary
    """
    result = dict(base)

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = dict_merge(result[key], value)
        else:
            result[key] = value

    return result
