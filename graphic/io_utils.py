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
        # numpy's `count` is an ELEMENT count (not bytes); `offset` is in bytes.
        return np.fromfile(filepath, dtype=file_type, offset=skip_bytes, count=count)

    bytes_per_sample = np.dtype(file_type).itemsize
    skip_bytes = start * 2 * bytes_per_sample
    # `count` elements = 2 * count interleaved floats (I, Q per complex sample);
    # `offset` stays in bytes.
    raw = np.fromfile(filepath, dtype=file_type, offset=skip_bytes, count=2 * count)
    return raw[::2] + raw[1::2] * 1j
