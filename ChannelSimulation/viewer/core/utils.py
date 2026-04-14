"""
Common Utility Functions
~~~~~~~~~~~~~~~~~~~~~~~~

Provides common utility functions for file size formatting, progress percentage,
numeric display, etc.
"""

from __future__ import annotations

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


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format numeric value as percentage string

    Args:
        value: Value between 0 and 1
        decimals: Number of decimal places

    Returns:
        Formatted percentage string, e.g. "45.5%"
    """
    return f"{value * 100:.{decimals}f}%"


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

    # Try to parse unit
    multipliers = {
        "g": 1e9,
        "m": 1e6,
        "k": 1e3,
        "": 1,
    }

    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                num_str = s[:-len(suffix)].strip()
                return float(num_str) * mult
            except ValueError:
                continue

    # Try to parse number directly
    try:
        return float(s)
    except ValueError:
        return None


def parse_size_string(s: str) -> int | None:
    """Parse size string to bytes

    Supported formats:
    - "1.5GB", "1.5G"
    - "100MB", "100M"
    - "1KB", "1K"
    - "500"

    Args:
        s: Size string

    Returns:
        Bytes, returns None if parsing fails
    """
    s = s.strip().lower()

    multipliers = {
        "gb": 1024 ** 3,
        "g": 1024 ** 3,
        "mb": 1024 ** 2,
        "m": 1024 ** 2,
        "kb": 1024,
        "k": 1024,
        "b": 1,
        "": 1,
    }

    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            try:
                num_str = s[:-len(suffix)].strip()
                return int(float(num_str) * mult)
            except ValueError:
                continue

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


def get_app_data_dir() -> Path:
    """Get application data directory

    On Windows: %APPDATA%/channelsim_viewer
    On macOS: ~/Library/Application Support/channelsim_viewer
    On Linux: ~/.config/channelsim_viewer

    Returns:
        Application data directory path
    """
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


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists, create if it doesn't

    Args:
        path: Directory path

    Returns:
        Directory path
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_extension(path: str | Path) -> str:
    """Get file extension (without dot)

    Args:
        path: File path

    Returns:
        Extension, e.g. "iq", "yaml", "png"
    """
    return Path(path).suffix.lstrip(".").lower()


def guess_iq_format(file_path: str | Path) -> str:
    """Guess IQ data format based on file extension

    Args:
        file_path: File path

    Returns:
        Format name, e.g. "iq_float32", "int16", "numpy", etc.
    """
    ext = get_file_extension(file_path)

    format_map = {
        "iq": "iq_float32",
        "cf32": "iq_float32",
        "cfile": "raw_binary",
        "raw": "raw_binary",
        "bin": "raw_binary",
        "iq16": "int16",
        "s16": "int16",
        "mat": "matlab",
        "npy": "numpy",
        "npz": "numpy_zip",
    }

    return format_map.get(ext, "unknown")


class Timer:
    """Simple timer context manager"""

    def __init__(self):
        self.start_time: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        if self.start_time is not None:
            self.elapsed = time.perf_counter() - self.start_time


def safe_dict_get(d: dict[str, Any], key: str, default: Any = None) -> Any:
    """Safely get dictionary value with nested access support

    Args:
        d: Dictionary
        key: Key (supports dot-separated nested keys, e.g. "a.b.c")
        default: Default value

    Returns:
        Value or default
    """
    keys = key.split(".")
    current = d

    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
            if current is None:
                return default
        else:
            return default

    return current if current is not None else default


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
