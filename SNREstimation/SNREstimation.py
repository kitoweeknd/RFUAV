"""Command-line SNR estimation for OFDM / UAV IQ recordings.

Thin CLI over the maintained estimator in
``ChannelSimulation.viewer.signal_proc.snr_estimator.SNREstimator`` (the live SNR
path used by the viewer). This file previously carried a standalone Python port
of the MATLAB ``func/*.m`` algorithm, which duplicated that estimator; it now
delegates to the single maintained implementation.

Note: the maintained estimator uses a threshold-crossing band detector and is
*not* bit-identical to the older MATLAB sliding-energy-window method still kept
in ``func/*.m`` (the MATLAB set remains the reference research implementation).

Run from the repository root so that both ``ChannelSimulation`` and this package
are importable, e.g.::

    python -m SNREstimation.SNREstimation path/to/signal.iq --fs 100e6 --bw 20e6
"""
import argparse
import os
import sys

import numpy as np

# Allow running directly as `python SNREstimation/SNREstimation.py ...` (add the
# repo root to the path so `ChannelSimulation` is importable, not just `-m`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ChannelSimulation.viewer.signal_proc.snr_estimator import SNREstimator


def estimate_file_snr(path, fs=100e6, bw=20e6, nfft=None, dtype=np.float32,
                      max_samples=None):
    """Estimate the SNR (dB) of a raw interleaved-IQ file.

    The IQ lanes are read into a pre-allocated ``complex64`` buffer *in place*
    (writing the interleaved I/Q into its ``.real`` / ``.imag`` views) instead
    of via the shared reader's ``raw[0::2] + 1j*raw[1::2]``. That expression
    forces a full-length ``complex128`` temporary *and* result: for a 100 M-sample
    (800 MB) recording it briefly holds the 0.8 GB float buffer, a 1.6 GB
    complex128 intermediate, and a 1.6 GB complex128 result at once (~4 GB peak).
    The in-place ``complex64`` read holds only the float buffer (0.8 GB) plus the
    complex64 output (0.8 GB), frees the float buffer immediately, and never
    materialises the complex128 intermediate — halving the read's resident set.
    (The shared ``read_interleaved_iq`` is intentionally left untouched: it feeds
    the whole ChannelSimulation pipeline and its ``apply_*`` tests.)

    ``max_samples`` optionally caps how many leading complex samples are used.
    A stationary recording's SNR is well estimated from a few million samples, so
    bounding this turns a whole-file estimate (100 M samples, ~20 s, multi-GB)
    into a sub-second, sub-GB one. ``None`` (default) reads the whole file and
    keeps the previous behaviour.

    Returns the full ``SNRResult`` from the maintained estimator.
    """
    if max_samples is not None:
        # np.fromfile(count=...) reads only the leading 2*max_samples real values
        # (I0,Q0,I1,Q1,...) — a bounded prefix, without touching the rest.
        raw = np.fromfile(path, dtype=dtype, count=int(max_samples) * 2)
    else:
        raw = np.fromfile(path, dtype=dtype)
    if raw.size < 2:
        raise ValueError(f"File data too short: {path}")
    if raw.size % 2 == 1:
        raw = raw[:-1]

    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw  # release the float buffer before the estimator allocates its own

    return SNREstimator(fs=fs, bw=bw, nfft=nfft).estimate(iq)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the SNR of an OFDM/UAV interleaved-IQ recording.")
    parser.add_argument("input", help="Path to a raw interleaved-IQ file (I0,Q0,I1,Q1,...).")
    parser.add_argument("--fs", type=float, default=100e6, help="Sampling rate in Hz (default 100e6).")
    parser.add_argument("--bw", type=float, default=20e6, help="Signal bandwidth in Hz (default 20e6).")
    parser.add_argument("--nfft", type=int, default=None, help="Welch FFT size (default: auto).")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="Raw sample dtype (default float32).")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of leading complex samples used "
                             "(default: whole file). A stationary recording's SNR is "
                             "well estimated from a few million samples; bounding this "
                             "turns a multi-GB, ~20 s whole-file estimate into a "
                             "sub-GB, sub-second one.")
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    res = estimate_file_snr(args.input, fs=args.fs, bw=args.bw, nfft=args.nfft, dtype=dtype,
                            max_samples=args.max_samples)
    print(f"SNR: {res.snr_db:.2f} dB  "
          f"(signal band {res.f1 / 1e6:.3f}–{res.f2 / 1e6:.3f} MHz, "
          f"P_signal={res.p_signal:.3e}, P_noise={res.p_noise:.3e})")


if __name__ == "__main__":
    main()
