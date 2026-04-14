from __future__ import annotations

import numpy as np

try:
    from ChannelSimulation.core.common import EPS
except ImportError:  # Compatible with direct script execution
    from core.common import EPS


def apply_awgn(
    x: np.ndarray,
    snr_db: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add complex Gaussian white noise at target SNR."""
    if rng is None:
        rng = np.random.default_rng()

    sig_power = float(np.mean(np.abs(x) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / max(snr_linear, EPS)
    sigma = np.sqrt(noise_power / 2.0)

    n = len(x)
    # Use complex128 output directly, avoid separate rotation array
    noise_real = rng.standard_normal(n, dtype=np.float64) * sigma
    noise_imag = rng.standard_normal(n, dtype=np.float64) * sigma

    x_c = np.ascontiguousarray(x, dtype=np.complex128)
    y = np.empty(n, dtype=np.complex128)
    y.real = x_c.real + noise_real
    y.imag = x_c.imag + noise_imag
    return y
