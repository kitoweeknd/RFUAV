from __future__ import annotations

import numpy as np

EPS = 1e-12

# Canonical channel-stage order — the single source of truth shared by the
# pipeline (channel_pipeline, which re-exports it), the GUI viewer
# (pipeline_runner) and run metadata (metadata.build_run_metadata). It lives
# here, in a leaf module with no intra-package imports, so every consumer can
# import it directly without creating import cycles or hand-copying the list
# (hand-copies have already drifted once, silently dropping "doppler").
STAGE_ORDER = [
    "original",
    "fading",
    "doppler",
    "multipath",
    "cfo",
    "phase_noise",
    "iq_imbalance",
    "awgn",
    "crop_time_shift",
]


def normalize_power(
    x: np.ndarray,
    target_power: float = 1.0,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Normalise complex IQ signal to target average power.

    Supports in-place operation via the `out` argument to avoid allocating
    a full copy when called repeatedly across all pipeline stages.

    Note: when the input power is below ``EPS`` and ``out`` is None, the input
    array itself is returned (no copy) — callers must not assume the result is
    always a fresh array.
    """
    p = float(np.mean(np.abs(x) ** 2))
    if p < EPS:
        # Power too low to normalise: leave the values unchanged. Without
        # `out` the input is returned as-is (documented aliasing above — an
        # essentially all-zero capture should not pay a full-array copy that
        # no caller needs). With `out`, honour the write-through contract
        # using the same casting strictness as the np.multiply path below, so
        # an invalid `out` buffer (e.g. real dtype for complex input) fails
        # loudly here too instead of only for normal-power signals. Both
        # paths preserve dtype, keeping the complex64 pipeline invariant.
        if out is None:
            return x
        np.copyto(out, x, casting="same_kind")
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
