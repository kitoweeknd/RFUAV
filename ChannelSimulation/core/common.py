from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize_power(
    x: np.ndarray,
    target_power: float = 1.0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Normalise complex IQ signal to target average power.

    Supports in-place operation via the `out` argument to avoid allocating
    a full copy when called repeatedly across all pipeline stages.
    """
    p = float(np.mean(np.abs(x) ** 2))
    if p < EPS:
        # Power too low to normalise: leave the values unchanged, but still honour
        # the documented `out` contract (write through and return `out`) and, when
        # allocating, return a fresh array like the normal path below rather than
        # aliasing the input. `.copy()` / `out[...] = x` both preserve dtype, so the
        # complex64 pipeline invariant is kept.
        if out is None:
            return x.copy()
        out[...] = x
        return out
    scale = np.sqrt(target_power / p)
    # Preserve the input's precision. A plain float64 `scale` promotes a
    # complex64 signal to complex128 — and because this runs at the very start
    # of the pipeline (and again inside fading/iq_imbalance), that single
    # promotion silently doubled the memory and halved the speed of every
    # downstream stage. Cast scale to the input's real dtype (float32 for
    # complex64) so the working precision stays complex64 end-to-end.
    if np.issubdtype(x.dtype, np.complexfloating):
        scale = x.real.dtype.type(scale)
    if out is None:
        return x * scale
    np.multiply(x, scale, out=out)
    return out
