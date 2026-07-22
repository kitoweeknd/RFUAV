from __future__ import annotations

import numpy as np

try:
    from ChannelSimulation.core.common import normalize_power
except ImportError:  # Compatible with direct script execution
    from core.common import normalize_power

# Per-chunk processing bounds memory on large signals (see cfo.py).
_CHUNK = 1 << 21  # ~2.1M samples


def apply_phase_noise(
    x: np.ndarray,
    phase_std_per_sample: float = 2e-3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Phase noise: discrete Wiener process (random walk).

    Phase increments are generated in float32 (noise precision is fine in 32-bit),
    accumulated in float64, then applied via fused real/imag arithmetic.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(x)
    # Preserve the input's complex precision (complex64 in the pipeline).
    cdt = x.dtype if np.issubdtype(x.dtype, np.complexfloating) else np.complex64
    y = np.empty(n, dtype=cdt)

    # Wiener random walk applied per chunk (see cfo.py): keeps temporaries small
    # so large signals don't trigger memory compression. The accumulated phase
    # carries across chunk boundaries so the walk stays continuous.
    running = 0.0
    for s in range(0, n, _CHUNK):
        e = min(s + _CHUNK, n)
        cnt = e - s
        dphi = rng.normal(0.0, phase_std_per_sample, size=cnt).astype(np.float32)
        phi = np.cumsum(dphi, dtype=np.float64)  # accumulate in float64
        phi += running
        running = float(phi[-1])
        rot = np.empty(cnt, dtype=cdt)
        rot.real = np.cos(phi)
        rot.imag = np.sin(phi)
        y[s:e] = x[s:e] * rot
    return y
