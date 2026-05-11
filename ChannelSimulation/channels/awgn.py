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
    ref_power: float | None = None,
) -> np.ndarray:
    """Add complex Gaussian white noise at target SNR.

    Args:
        x: Input signal array.
        snr_db: Target SNR in dB for the *output* signal.
        rng: Random number generator.
        ref_power: Reference signal power. If None, uses input signal power.
                   This should be the power of the original signal (after
                   preprocessing, before any AWGN stage) so that each AWGN
                   stage's SNR target is relative to the same reference.
    """
    if rng is None:
        rng = np.random.default_rng()

    sig_power = float(np.mean(np.abs(x) ** 2))
    # Use provided ref_power for SNR calculation; fall back to input signal power.
    # ref_power is the power of the original/preprocessed signal, ensuring
    # that multiple AWGN stages each degrade to the same target SNR relative
    # to the reference rather than accumulating noise.
    power_for_snr = ref_power if ref_power is not None else sig_power
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = power_for_snr / max(snr_linear, EPS)
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
