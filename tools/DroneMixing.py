# Merge the data of two drones
import os
import matplotlib.pyplot as plt
import numpy as np

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

    data_UAV1_I2_padded = np.pad(data_UAV1_I, (0,  max(len(data_UAV1_I), len(data_UAV2_I)) - len(data_UAV1_I)), 'constant', constant_values=0)
    data_UAV1_Q2_padded = np.pad(data_UAV1_Q, (0,  max(len(data_UAV1_Q), len(data_UAV2_Q)) - len(data_UAV1_Q)), 'constant', constant_values=0)

    data_UAV2_I1_padded = np.pad(data_UAV2_I, (0,  max(len(data_UAV1_I), len(data_UAV2_I)) - len(data_UAV2_I)), 'constant', constant_values=0)
    data_UAV2_Q2_padded = np.pad(data_UAV2_Q, (0,  max(len(data_UAV1_Q), len(data_UAV2_Q)) - len(data_UAV2_Q)), 'constant', constant_values=0)

    _data_UAV1 = data_UAV1_I2_padded + data_UAV1_Q2_padded * 1j
    _data_UAV2 = data_UAV2_I1_padded + data_UAV2_Q2_padded * 1j

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
        aug = 10 * np.log10(np.abs(Zxx))
        extent = [t.min(), t.max(), f.min(), f.max()]

        plt.figure()
        plt.imshow(aug, extent=extent, aspect='auto', origin='lower', cmap='jet')
        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=None, hspace=None)
        plt.savefig(os.path.join(targetFolderPath, name, ' (' + str(i) + ').jpg'), dpi=300)
        plt.show()
        plt.close()

        i += 1


# Usage-----------------------------------------------------------------------------------------------------------------
def main():
    UAV1 = ''
    UAV2 = ''
    targetFolderPath = ''
    name = 'test'

    duration_time = 0.1
    fs = 100e6
    stft_point = 1024

    MergeData(
        UAV1=UAV1,
        UAV2=UAV2,
        targetFolderPath=targetFolderPath,
        name=name,
        duration_time=duration_time,
        fs=fs,
        stft_point=stft_point)


if __name__ == '__main__':
    main()