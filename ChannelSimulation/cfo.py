from __future__ import annotations

import numpy as np


def apply_cfo(
    x: np.ndarray,
    fs: float,
    cfo_hz: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Carrier Frequency Offset (CFO).

    Uses in-place multiplication by writing the rotation directly into
    the output array's view, avoiding creation of a separate rotation array.
    """
    if out is None:
        y = np.empty(len(x), dtype=np.complex128)
    else:
        y = np.ascontiguousarray(out, dtype=np.complex128)

    n = len(x)
    TWO_PI_OVER_FS = 2.0 * np.pi / fs
    TWO_PI_OVER_FS_CFO = TWO_PI_OVER_FS * cfo_hz

    # Compute phase directly into the imag view of the output
    t = np.arange(n, dtype=np.float64)
    phase = t * TWO_PI_OVER_FS_CFO
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)

    x_c = np.ascontiguousarray(x, dtype=np.complex128)
    y.real = x_c.real * cos_p - x_c.imag * sin_p
    y.imag = x_c.real * sin_p + x_c.imag * cos_p
    return y
