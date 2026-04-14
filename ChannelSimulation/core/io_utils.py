from __future__ import annotations

from pathlib import Path
from typing import Generator

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


# ----------------------------------------------------------------------
# Chunked / streaming interface
# ----------------------------------------------------------------------
def count_complex_samples(file_path: str | Path, dtype=np.float32) -> int:
    """Return the number of complex samples in an interleaved IQ file."""
    bytes_per_sample = np.dtype(dtype).itemsize
    total_bytes = Path(file_path).stat().st_size
    return total_bytes // (2 * bytes_per_sample)


def iter_iq_chunks(
    file_path: str | Path,
    chunk_size: int,
    dtype=np.float32,
) -> Generator[np.ndarray, None, None]:
    """Yield complex IQ chunks from an interleaved IQ file.

    Memory: only one chunk (2 × chunk_size × bytes_per_sample) is held at a time.
    For float32 IQ files at 100 MHz, 10 M samples ≈ 80 MB per chunk.
    """
    bytes_per_sample = np.dtype(dtype).itemsize
    bytes_per_complex = 2 * bytes_per_sample
    total_bytes = Path(file_path).stat().st_size
    total_complex = total_bytes // bytes_per_complex

    offset = 0
    while offset < total_complex:
        count = min(chunk_size, total_complex - offset)
        raw = np.fromfile(file_path, dtype=dtype,
                           offset=offset * bytes_per_complex,
                           count=count * 2)
        yield raw[0::2] + 1j * raw[1::2]
        offset += count


def append_interleaved_iq(file_path: str | Path, iq: np.ndarray) -> None:
    """Append a chunk to an interleaved IQ file (binary append)."""
    out = np.empty(iq.size * 2, dtype=np.float32)
    out[0::2] = np.real(iq).astype(np.float32)
    out[1::2] = np.imag(iq).astype(np.float32)
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "ab") as f:
        f.write(out.tobytes())
