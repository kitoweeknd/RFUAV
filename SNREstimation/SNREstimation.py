"""Command-line SNR estimation for OFDM / UAV IQ recordings.

Primary estimator: adaptive FCME (Forward Consecutive Mean Excision),
replacing the previous PSD threshold-crossing method (kept as ``--legacy``).

Adaptive FCME iterates the Welch PSD, progressively tightening the noise-bin
set, and auto-tunes the threshold factor (T=2.5→2.0→1.5→1.0) until signal
bins are detected.  This makes it robust to both strong narrowband signals
and weak wideband signals — the previous estimator's band-detection failure
mode (locking to the full bandwidth and returning −200 dB) is eliminated.

The implementation lives in ``ChannelSimulation/viewer/signal_proc/snr_dual.py``;
this CLI is a thin wrapper.

Run from the repository root::

    python -m SNREstimation.SNREstimation path/to/signal.iq --fs 100e6
"""
import argparse
import os
import sys

import numpy as np

# Allow running directly as `python SNREstimation/SNREstimation.py ...`
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from ChannelSimulation.viewer.signal_proc.snr_dual import FCMEEstimator

# Datatype vocabulary
sys.path.insert(0, os.path.join(_ROOT, "tools", "rfuav_v2"))
import iq_dtype  # noqa: E402


def estimate_file_snr(path, fs=100e6, dtype=np.float32, max_samples=None):
    """Estimate the SNR (dB) of a raw interleaved-IQ file using adaptive FCME.

    Reads IQ into a pre-allocated ``complex64`` buffer, then delegates to
    ``FCMEEstimator`` (adaptive threshold, Welch PSD, iterative noise-bin
    convergence).

    ``max_samples`` caps how many leading complex samples are used.
    A stationary recording's SNR is well estimated from a few million samples,
    so bounding this turns a whole-file estimate into a sub-second one.

    ``dtype`` is the on-disk dtype of ONE interleaved component.  Integer
    components are cast into ``complex64`` — SNR is a ratio of powers so the
    absolute scale of raw ADC counts cancels out.

    Returns
    -------
    snr_db : float
        Estimated SNR in dB.
    diagnostics : dict
        FCME intermediate results (signal_bins, noise_bins, iterations, T_factor).
    """
    if max_samples is not None:
        raw = np.fromfile(path, dtype=dtype, count=int(max_samples) * 2)
    else:
        raw = np.fromfile(path, dtype=dtype)
    if raw.size < 2:
        raise ValueError("File data too short: %s" % path)
    if raw.size % 2 == 1:
        raw = raw[:-1]

    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw

    est = FCMEEstimator(fs=fs)
    snr_db, diag = est.estimate(iq)
    return snr_db, diag


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Estimate the SNR of an OFDM/UAV interleaved-IQ recording "
                    "using adaptive FCME.")
    parser.add_argument("input", help="Path to a raw interleaved-IQ file (I0,Q0,I1,Q1,...).")
    parser.add_argument("--fs", type=float, default=100e6,
                        help="Sampling rate in Hz (default 100e6).")
    parser.add_argument("--nfft", type=int, default=None,
                        help="Welch FFT size (default: auto).")
    parser.add_argument("--dtype",
                        choices=sorted(iq_dtype.VOCAB) + sorted(iq_dtype.ALIASES),
                        default="cf32_le", metavar="TOKEN",
                        help="On-disk sample datatype, as the manifest spells it "
                             "(default cf32_le). Canonical: %s. Also accepted: %s."
                             % (", ".join(sorted(iq_dtype.VOCAB)),
                                ", ".join(sorted(iq_dtype.ALIASES))))
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap the number of leading complex samples used "
                             "(default: whole file).")
    parser.add_argument("--legacy", action="store_true",
                        help="Use the old PSD threshold-crossing estimator "
                             "(deprecated; band detection fails on many signals).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print FCME diagnostics (signal bins, iterations, T factor).")
    return parser


def main():
    args = _build_parser().parse_args()
    dtype = iq_dtype.from_manifest_token(args.dtype).component

    if args.legacy:
        from ChannelSimulation.viewer.signal_proc.snr_estimator import SNREstimator
        if args.max_samples is not None:
            raw = np.fromfile(args.input, dtype=dtype, count=int(args.max_samples) * 2)
        else:
            raw = np.fromfile(args.input, dtype=dtype)
        if raw.size < 2:
            raise ValueError("File data too short: %s" % args.input)
        if raw.size % 2 == 1:
            raw = raw[:-1]
        n = raw.size // 2
        iq = np.empty(n, dtype=np.complex64)
        iq.real = raw[0::2]
        iq.imag = raw[1::2]
        del raw
        est = SNREstimator(fs=args.fs, bw=20e6, nfft=args.nfft)
        result = est.estimate(iq)
        print("SNR (legacy PSD threshold): %.2f dB" % result.snr_db)
        if args.verbose:
            print("  band: [%.1f, %.1f] MHz" % (result.f1 / 1e6, result.f2 / 1e6))
        return

    snr_db, diag = estimate_file_snr(
        args.input, fs=args.fs, dtype=dtype, max_samples=args.max_samples)

    print("SNR (adaptive FCME): %.2f dB" % snr_db)
    if args.verbose:
        print("  signal bins: %d  noise bins: %d  iterations: %d  T_factor: %.1f" % (
            diag["signal_bins"], diag["noise_bins"],
            diag["iterations"], diag.get("T_factor", 0)))


if __name__ == "__main__":
    main()