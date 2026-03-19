from __future__ import annotations

import numpy as np


def apply_cfo(x: np.ndarray, fs: float, cfo_hz: float) -> np.ndarray:
    """载波频偏（CFO）。"""
    n = np.arange(len(x), dtype=np.float64)
    rot = np.exp(1j * 2.0 * np.pi * cfo_hz * n / fs)
    return x * rot
