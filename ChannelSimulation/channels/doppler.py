from __future__ import annotations

import numpy as np

# Per-sample rotation is applied in cache-sized chunks (see cfo.py) so a 100M
# sample run never materialises the ~8 whole-signal float64 arrays that used to
# tip the machine into memory compression (81s → linear in N).
_CHUNK = 1 << 21  # ~2.1M samples


def apply_doppler(
    x: np.ndarray,
    fs: float,
    doppler_shift_hz: float,
    doppler_spread_hz: float,
    out: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Doppler effect simulation using Jakes model.

    Args:
        x: Input IQ signal.
        fs: Sampling frequency in Hz.
        doppler_shift_hz: Mean Doppler shift in Hz (simulates frequency offset).
        doppler_spread_hz: Doppler spread (RMS width of Doppler spectrum) in Hz.
        out: Optional output array.
        rng: Random number generator (for reproducible Doppler spread).

    Returns:
        Signal with Doppler effect applied.

    Note:
        This implementation uses a simplified Jakes model that combines:
        1. A deterministic phase rotation for the mean Doppler shift
        2. Additive Gaussian noise scaled by the Doppler spread
          (approximating Clarke's Jake's Doppler spectrum)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(x)
    TWO_PI_OVER_FS = 2.0 * np.pi / fs

    # Preserve the input's complex precision (complex64 in the pipeline) — the
    # phase arrays stay float64 for accuracy, but the output and x_c copy no
    # longer double to complex128.
    cdt = x.dtype if np.issubdtype(x.dtype, np.complexfloating) else np.complex64

    if out is None:
        y = np.empty(n, dtype=cdt)
    else:
        y = np.ascontiguousarray(out, dtype=cdt)

    if doppler_spread_hz <= 0 and doppler_shift_hz == 0:
        if out is None:
            return x.copy()
        y[:] = x
        return y

    # Total phase = mean-shift ramp (deterministic) + spread random walk.
    # Built and applied per chunk; the random walk carries its running phase
    # across chunk boundaries so it stays continuous.
    mean_inc = TWO_PI_OVER_FS * doppler_shift_hz     # 0 when no mean shift
    sigma_phase = TWO_PI_OVER_FS * doppler_spread_hz  # 0 when no spread
    spread_running = 0.0
    for s in range(0, n, _CHUNK):
        e = min(s + _CHUNK, n)
        cnt = e - s
        ph = np.arange(s, e, dtype=np.float64)
        ph *= mean_inc
        if sigma_phase > 0:
            sp = np.cumsum(rng.standard_normal(cnt) * sigma_phase)
            sp += spread_running
            spread_running = float(sp[-1])
            ph += sp
        rot = np.empty(cnt, dtype=cdt)
        rot.real = np.cos(ph)
        rot.imag = np.sin(ph)
        y[s:e] = x[s:e] * rot
    return y
