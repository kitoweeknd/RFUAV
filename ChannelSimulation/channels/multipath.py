from __future__ import annotations

import numpy as np

try:
    from ChannelSimulation.core.common import EPS, normalize_power
except ImportError:  # Compatible with direct script execution
    from core.common import EPS, normalize_power


def apply_multipath(
    x: np.ndarray,
    max_delay_samp: int = 20,
    n_paths: int = 4,
    exp_decay: float = 0.35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Integer-sample multipath channel (random delay + exponential decay + random phase).

    Uses FFT convolution: O(n log n) time, O(n) extra memory instead of O(n) extra for
    np.convolve(mode='full').  For large arrays this reduces peak memory by ~2×.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_paths < 1 or max_delay_samp < 1:
        return x

    delays = np.sort(rng.integers(0, max_delay_samp + 1, size=n_paths))
    rel = np.exp(-exp_decay * delays)
    rel = rel / (np.linalg.norm(rel) + EPS)

    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_paths)
    taps = rel * np.exp(1j * phases)

    h = np.zeros(max_delay_samp + 1, dtype=np.complex128)
    for d, t in zip(delays, taps):
        h[d] += t

    # Overlap-add FFT convolution. h is a short kernel (a handful of taps) while x
    # is millions of samples, so oaconvolve — which convolves with small block-sized
    # FFTs — is ~2.5× faster than two full-length N-point FFTs and keeps memory low.
    # Stay in complex64 (the pipeline's working dtype); upcasting to complex128 just
    # doubled the FFT work for no display-relevant precision. numpy fallback for the
    # scipy-less case.
    n = len(x)
    h = h.astype(x.dtype, copy=False)
    try:
        from scipy.signal import oaconvolve

        y = oaconvolve(x, h, mode="full")[:n]
    except Exception:
        y = np.convolve(x, h, mode="full")[:n]
    return normalize_power(y)
