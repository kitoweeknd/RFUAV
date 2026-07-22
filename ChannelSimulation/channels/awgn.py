from __future__ import annotations

import numpy as np

try:
    from ChannelSimulation.core.common import EPS
except ImportError:  # Compatible with direct script execution
    from core.common import EPS

# Per-chunk processing bounds memory on large signals (see cfo.py).
_CHUNK = 1 << 21  # ~2.1M samples


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

    # Only compute the (O(N)) input power if we actually need it — in the
    # pipeline ref_power is always supplied, so skip the extra full-array pass.
    if ref_power is not None:
        power_for_snr = ref_power
    else:
        power_for_snr = float(np.mean(np.abs(x) ** 2))
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = power_for_snr / max(snr_linear, EPS)
    sigma = np.sqrt(noise_power / 2.0)

    n = len(x)
    # Preserve complex64 and add noise in cache-sized chunks so we never hold
    # two whole-signal float64 noise arrays at once (see cfo.py). Noise
    # precision is fine in 32-bit for the complex64 path.
    cdt = x.dtype if np.issubdtype(x.dtype, np.complexfloating) else np.complex64
    rdt = np.float32 if cdt == np.complex64 else np.float64
    sig = np.asarray(sigma, dtype=rdt)
    y = np.empty(n, dtype=cdt)
    for s in range(0, n, _CHUNK):
        e = min(s + _CHUNK, n)
        cnt = e - s
        yr = y[s:e]  # writable view into y
        yr.real = x[s:e].real + rng.standard_normal(cnt, dtype=rdt) * sig
        yr.imag = x[s:e].imag + rng.standard_normal(cnt, dtype=rdt) * sig
    return y
