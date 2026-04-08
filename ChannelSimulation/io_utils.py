from __future__ import annotations

from pathlib import Path

import numpy as np


def read_interleaved_iq(file_path: str | Path, dtype=np.float32) -> np.ndarray:
    """Read binary interleaved IQ data: I0,Q0,I1,Q1,..."""
    raw = np.fromfile(file_path, dtype=dtype)
    if raw.size < 2:
        raise ValueError(f"File data too short: {file_path}")
    if raw.size % 2 == 1:
        raw = raw[:-1]
    return raw[0::2] + 1j * raw[1::2]


def save_interleaved_iq(file_path: str | Path, iq: np.ndarray) -> None:
    out = np.empty(iq.size * 2, dtype=np.float32)
    out[0::2] = np.real(iq).astype(np.float32)
    out[1::2] = np.imag(iq).astype(np.float32)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    out.tofile(file_path)
