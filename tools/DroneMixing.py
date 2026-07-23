# Merge the data of two drones
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Allow running directly as `python tools/DroneMixing.py` (add repo root to path
# so the sibling `graphic` package is importable, not just via `python -m`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graphic.RawDataProcessor import STFT


def MergeData(UAV1: str,
              UAV2: str,
              targetFolderPath: str,
              name: str,
              duration_time: float = 0.1,
              fs: int = 100e6,
              stft_point: int = 1024
              ):

    """
    Merge and process data from two UAVs, and save the processed data as images.

    Args:
        UAV1 (str): Path to the first UAV data file.
        UAV2 (str): Path to the second UAV data file.
        targetFolderPath (str): Directory to save the processed images.
        name (str): Name prefix for the saved images.
        duration_time (float, optional): Duration of each segment in seconds. Defaults to 0.1.
        fs (int, optional): Sampling frequency in Hz. Defaults to 100e6.
        stft_point (int, optional): Number of points for the STFT. Defaults to 1024.
    """

    slice_point = int(fs * duration_time)

    data_UAV1 = np.fromfile(UAV1, dtype=np.float32)
    data_UAV2 =np.fromfile(UAV2, dtype=np.float32)

    data_UAV1_I = data_UAV1[::2]
    data_UAV1_Q = data_UAV1[1::2]

    data_UAV2_I = data_UAV2[::2]
    data_UAV2_Q = data_UAV2[1::2]

    # Zero-pad both recordings to the same length and combine, keeping the data
    # complex64 (float32 * 1j would silently promote the whole array to
    # complex128 — 2x memory, and against this repo's complex64 convention).
    n = max(len(data_UAV1_I), len(data_UAV2_I))
    _data_UAV1 = np.zeros(n, dtype=np.complex64)
    _data_UAV1.real[:len(data_UAV1_I)] = data_UAV1_I
    _data_UAV1.imag[:len(data_UAV1_Q)] = data_UAV1_Q
    _data_UAV2 = np.zeros(n, dtype=np.complex64)
    _data_UAV2.real[:len(data_UAV2_I)] = data_UAV2_I
    _data_UAV2.imag[:len(data_UAV2_Q)] = data_UAV2_Q

    i = 0
    data_merged = _data_UAV1 + _data_UAV2
    os.makedirs(os.path.join(targetFolderPath, name), exist_ok=True)
    while (i + 1) * slice_point <= len(data_merged):

        f, t, Zxx = STFT(data_merged[int(i * slice_point): int((i + 1) * slice_point)],
                         stft_point=stft_point,
                         fs=fs,
                         duration_time=duration_time,
                         onside=False)
        f = np.fft.fftshift(f)
        Zxx = np.fft.fftshift(Zxx, axes=0)
        # Floor the magnitude before log10 (log10(0) = -inf otherwise).
        aug = 10 * np.log10(np.clip(np.abs(Zxx), 1e-10, None))
        extent = [t.min(), t.max(), f.min(), f.max()]

        plt.figure()
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap='jet')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        plt.savefig(os.path.join(targetFolderPath, name, f'{name}_{i}.jpg'), dpi=300)
        plt.close()

        i += 1


# Usage-----------------------------------------------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Mix two UAV IQ recordings and save STFT spectrogram frames.")
    parser.add_argument("uav1", help="Path to the first UAV IQ file (float32 interleaved).")
    parser.add_argument("uav2", help="Path to the second UAV IQ file.")
    parser.add_argument("target", help="Output directory for the mixed spectrogram frames.")
    parser.add_argument("--name", default="mixed", help="Name prefix for the output subfolder/images.")
    parser.add_argument("--duration-time", type=float, default=0.1, help="Seconds per frame (default 0.1).")
    parser.add_argument("--fs", type=float, default=100e6, help="Sampling rate in Hz (default 100e6).")
    parser.add_argument("--stft-point", type=int, default=1024, help="STFT window size (default 1024).")
    args = parser.parse_args()

    MergeData(
        UAV1=args.uav1,
        UAV2=args.uav2,
        targetFolderPath=args.target,
        name=args.name,
        duration_time=args.duration_time,
        fs=args.fs,
        stft_point=args.stft_point)


if __name__ == '__main__':
    main()