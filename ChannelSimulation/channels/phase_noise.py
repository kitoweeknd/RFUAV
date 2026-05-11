from __future__ import annotations

import numpy as np

try:
    from ChannelSimulation.core.common import normalize_power
except ImportError:  # Compatible with direct script execution
    from core.common import normalize_power


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
    x_c = np.ascontiguousarray(x, dtype=np.complex128)

    # float32 is sufficient for per-sample phase noise; accumulate in float64
    dphi = rng.normal(0.0, phase_std_per_sample, size=n).astype(np.float32)
    phi = np.empty(n, dtype=np.float64)
    np.cumsum(dphi, out=phi)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Fused complex rotation: fill output directly
    y = np.empty(n, dtype=np.complex128)
    y.real = x_c.real * cos_phi - x_c.imag * sin_phi
    y.imag = x_c.real * sin_phi + x_c.imag * cos_phi
    return y
