from __future__ import annotations

import numpy as np

try:
    from ChannelSimulation.core.common import normalize_power
except ImportError:  # Compatible with direct script execution
    from core.common import normalize_power


def apply_iq_imbalance(
    x: np.ndarray,
    gain_imbalance_db: float = 1.0,
    phase_imbalance_deg: float = 3.0,
) -> np.ndarray:
    """Generalized IQ imbalance model: y = alpha*x + beta*conj(x)."""
    g = 10 ** (gain_imbalance_db / 20.0)
    phi = np.deg2rad(phase_imbalance_deg)

    # Cast the mixing coefficients to the input's complex precision so
    # alpha*x doesn't silently promote a complex64 signal to complex128.
    cdt = x.dtype if np.issubdtype(x.dtype, np.complexfloating) else np.complex64
    alpha = np.asarray(0.5 * (1.0 + g * np.exp(-1j * phi)), dtype=cdt)
    beta = np.asarray(0.5 * (1.0 - g * np.exp(1j * phi)), dtype=cdt)

    y = alpha * x + beta * np.conj(x)
    return normalize_power(y)
