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
from ChannelSimulation.core.io_utils import read_interleaved_iq
from ChannelSimulation.viewer.signal_proc.snr_estimator import SNREstimator


def estimate_file_snr(path, fs=100e6, bw=20e6, nfft=None, dtype=np.float32):
    """Estimate the SNR (dB) of a raw interleaved-IQ file.

    Returns the full ``SNRResult`` from the maintained estimator.
    """
    signal = read_interleaved_iq(path, dtype=dtype)
    return SNREstimator(fs=fs, bw=bw, nfft=nfft).estimate(signal)


def main():
    parser = argparse.ArgumentParser(
        description="Estimate the SNR of an OFDM/UAV interleaved-IQ recording.")
    parser.add_argument("input", help="Path to a raw interleaved-IQ file (I0,Q0,I1,Q1,...).")
    parser.add_argument("--fs", type=float, default=100e6, help="Sampling rate in Hz (default 100e6).")
    parser.add_argument("--bw", type=float, default=20e6, help="Signal bandwidth in Hz (default 20e6).")
    parser.add_argument("--nfft", type=int, default=None, help="Welch FFT size (default: auto).")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="Raw sample dtype (default float32).")
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64
    res = estimate_file_snr(args.input, fs=args.fs, bw=args.bw, nfft=args.nfft, dtype=dtype)
    print(f"SNR: {res.snr_db:.2f} dB  "
          f"(signal band {res.f1 / 1e6:.3f}–{res.f2 / 1e6:.3f} MHz, "
          f"P_signal={res.p_signal:.3e}, P_noise={res.p_noise:.3e})")


if __name__ == "__main__":
    main()
