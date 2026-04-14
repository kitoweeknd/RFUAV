from __future__ import annotations

import numpy as np


def apply_doppler(
    x: np.ndarray,
    fs: float,
    doppler_shift_hz: float,
    doppler_spread_hz: float,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Doppler effect simulation using Jakes model.

    Args:
        x: Input IQ signal.
        fs: Sampling frequency in Hz.
        doppler_shift_hz: Mean Doppler shift in Hz (simulates frequency offset).
        doppler_spread_hz: Doppler spread (RMS width of Doppler spectrum) in Hz.
        out: Optional output array.

    Returns:
        Signal with Doppler effect applied.

    Note:
        This implementation uses a simplified Jakes model that combines:
        1. A deterministic phase rotation for the mean Doppler shift
        2. Additive Gaussian noise scaled by the Doppler spread
          (approximating Clarke's Jake's Doppler spectrum)
    """
    n = len(x)
    TWO_PI_OVER_FS = 2.0 * np.pi / fs

    if out is None:
        y = np.empty(n, dtype=np.complex128)
    else:
        y = np.ascontiguousarray(out, dtype=np.complex128)

    if doppler_spread_hz <= 0 and doppler_shift_hz == 0:
        if out is None:
            return x.copy()
        y[:] = x
        return y

    # Mean Doppler shift: adds a constant phase rotation
    if doppler_shift_hz != 0:
        mean_phase_inc = TWO_PI_OVER_FS * doppler_shift_hz
        mean_t = np.arange(n, dtype=np.float64)
        mean_cos = np.cos(mean_phase_inc * mean_t)
        mean_sin = np.sin(mean_phase_inc * mean_t)
    else:
        mean_cos = np.ones(n, dtype=np.float64)
        mean_sin = np.zeros(n, dtype=np.float64)

    if doppler_spread_hz > 0:
        # Doppler spread: simulates frequency-selective fading
        # Use a Gaussian filter on the input to simulate doppler spread
        # Bandwidth ~ doppler_spread_hz, simplified as additive phase noise
        sigma_phase = TWO_PI_OVER_FS * doppler_spread_hz
        phase_noise = np.cumsum(np.random.randn(n) * sigma_phase)
        spread_cos = np.cos(phase_noise)
        spread_sin = np.sin(phase_noise)
    else:
        spread_cos = np.ones(n, dtype=np.float64)
        spread_sin = np.zeros(n, dtype=np.float64)

    x_c = np.ascontiguousarray(x, dtype=np.complex128)
    y.real = x_c.real * mean_cos * spread_cos - x_c.imag * mean_sin * spread_sin
    y.imag = x_c.real * mean_sin * spread_sin + x_c.imag * mean_cos * spread_cos
    return y
