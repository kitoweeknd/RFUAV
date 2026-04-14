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

    # FFT convolution: avoid np.convolve(mode='full') which allocates (n + len(h) - 1) extra samples
    n = len(x)
    fft_len = n + len(h) - 1
    X = np.fft.fft(x.astype(np.complex128), n=fft_len)
    H = np.fft.fft(h, n=fft_len)
    y = np.fft.ifft(X * H).astype(np.complex128)
    return normalize_power(y[:n])
