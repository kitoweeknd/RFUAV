import os
import numpy as np


def _is_complex_dtype(dtype):
    return dtype in (np.complex64, np.complex128)


def _count_complex_samples(filepath: str, file_type=np.float32) -> int:
    """Return the total number of complex samples in the file."""
    total_bytes = os.path.getsize(filepath)
    if _is_complex_dtype(file_type):
        bytes_per_sample = np.dtype(file_type).itemsize
    else:
        bytes_per_sample = 2 * np.dtype(file_type).itemsize
    return total_bytes // bytes_per_sample


def _complex_from_file(filepath: str, start: int, count: int,
                       file_type=np.float32) -> np.ndarray:
    """Read `count` complex samples starting at `start` from a raw IQ file.

    Supports two layouts:
    - interleaved float (float32 / float64): alternating I, Q, I, Q, ...
    - direct complex   (complex64 / complex128): native complex binary

    Memory overhead: only the exact slice is kept in RAM.
    """
    if _is_complex_dtype(file_type):
        bytes_per_sample = np.dtype(file_type).itemsize
        skip_bytes = start * bytes_per_sample
        read_bytes = count * bytes_per_sample
        return np.fromfile(filepath, dtype=file_type, offset=skip_bytes, count=read_bytes)

    bytes_per_sample = np.dtype(file_type).itemsize
    skip_bytes = start * 2 * bytes_per_sample
    read_bytes = count * 2 * bytes_per_sample
    raw = np.fromfile(filepath, dtype=file_type, offset=skip_bytes, count=read_bytes)
    return raw[::2] + raw[1::2] * 1j


def _complex_memmap(filepath: str, file_type=np.float32):
    """Memory-mapped view of a raw IQ file as complex64 samples.

    Supports two layouts:
    - interleaved float: alternating I, Q, I, Q, ...
    - direct complex: native complex binary

    The returned array has dtype complex64 and length = total_complex_samples.
    NumPy memmap is read-only and backed by the OS page cache, so large files
    are never fully resident in RAM; only touched pages are loaded on demand.
    """
    if _is_complex_dtype(file_type):
        mm = np.memmap(filepath, dtype=file_type, mode='r')
        if file_type == np.complex128:
            return mm.astype(np.complex64)
        return mm

    bytes_per_sample = np.dtype(file_type).itemsize
    mm = np.memmap(filepath, dtype=file_type, mode='r')
    i = mm[::2].astype(np.complex64)
    q = mm[1::2].astype(np.complex64)
    return i + q * 1j
