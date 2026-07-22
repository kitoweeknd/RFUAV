"""
Pluggable multithreaded FFT backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The viewer's spectrum computation is FFT-bound: the STFT precompute runs a
batched FFT (``~3000 × 4096``) for the original, the processed, and every
intermediate pipeline stage. The stock ``numpy.fft`` is single-threaded, so
this module routes those transforms through the fastest *validated* backend
available on the host, in preference order:

1. **pyFFTW** — FFTW planning/wisdom, multithreading, byte-aligned buffers and a
   plan cache (numpy-compatible drop-in). Best when installed.
2. **scipy.fft** — pocketfft with a ``workers=`` argument for parallel batched
   1-D transforms. Always present (scipy is a hard dependency) and needs no
   build toolchain.
3. **numpy.fft** — last-resort single-threaded fallback.

The public functions mirror the ``scipy.fft`` signature (``fft``/``ifft`` accept
``n``, ``axis`` and ``workers``), so a GPU backend can be dropped in later with
``scipy.fft.set_backend(cupyx.scipy.fft)`` without touching call sites. ``fftshift``,
``ifftshift`` and ``fftfreq`` are cheap index math and stay on numpy.

``WORKERS`` defaults to all cores; callers may override per-call.
"""

from __future__ import annotations

import os

import numpy as np

# fftshift / ifftshift / fftfreq are trivial index operations — keep them on
# numpy so call sites can use them directly regardless of the FFT backend.
from numpy.fft import fftshift, ifftshift, fftfreq  # noqa: F401

__all__ = [
    "fft", "ifft", "fftshift", "ifftshift", "fftfreq",
    "BACKEND_NAME", "WORKERS",
]

# Default worker count: all logical cores. Capped defensively so a machine with
# a huge core count does not oversubscribe on small transforms.
WORKERS: int = max(1, min(os.cpu_count() or 1, 32))


def _select_backend():
    """Return ``(name, fft_fn, ifft_fn)`` for the best available backend.

    ``fft_fn``/``ifft_fn`` are normalised to the signature
    ``(a, n=None, axis=-1, workers=None)``.
    """
    # 1) pyFFTW — enable the interface plan cache once so repeated same-shape
    #    transforms skip re-planning.
    try:
        import pyfftw  # type: ignore
        from pyfftw.interfaces import numpy_fft as _pfft  # type: ignore
        from pyfftw.interfaces import cache as _pcache  # type: ignore

        _pcache.enable()
        _pcache.set_keepalive_time(30.0)

        def _fft(a, n=None, axis=-1, workers=None):
            return _pfft.fft(a, n=n, axis=axis, threads=workers or WORKERS)

        def _ifft(a, n=None, axis=-1, workers=None):
            return _pfft.ifft(a, n=n, axis=axis, threads=workers or WORKERS)

        return "pyfftw", _fft, _ifft
    except Exception:
        pass

    # 2) scipy.fft — pocketfft with multithreaded batched transforms.
    try:
        from scipy import fft as _sfft  # type: ignore

        def _fft(a, n=None, axis=-1, workers=None):
            return _sfft.fft(a, n=n, axis=axis, workers=workers or WORKERS)

        def _ifft(a, n=None, axis=-1, workers=None):
            return _sfft.ifft(a, n=n, axis=axis, workers=workers or WORKERS)

        return "scipy", _fft, _ifft
    except Exception:
        pass

    # 3) numpy.fft — single-threaded fallback (ignores ``workers``).
    def _fft(a, n=None, axis=-1, workers=None):
        return np.fft.fft(a, n=n, axis=axis)

    def _ifft(a, n=None, axis=-1, workers=None):
        return np.fft.ifft(a, n=n, axis=axis)

    return "numpy", _fft, _ifft


BACKEND_NAME, _FFT, _IFFT = _select_backend()


def fft(a, n=None, axis=-1, workers=None):
    """Forward FFT via the active backend (multithreaded when supported)."""
    return _FFT(a, n=n, axis=axis, workers=workers)


def ifft(a, n=None, axis=-1, workers=None):
    """Inverse FFT via the active backend (multithreaded when supported)."""
    return _IFFT(a, n=n, axis=axis, workers=workers)
