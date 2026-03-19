"""MATLAB 版本 SNR 估计算法的 Python 移植。

该实现对应以下 MATLAB 函数逻辑：
- dronesOFDMFreqShiftEsti
- dronesOFDMFreqCompensation
- positionFind (locate_signal.m)
- snrEsti (SNREsti.m)

输入数据格式：float32 交织 IQ（二进制文件，I0,Q0,I1,Q1,...）。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import get_window, resample_poly, welch


EPS = 1e-20


def db10(x: np.ndarray) -> np.ndarray:
    """功率转 dB。"""
    return 10.0 * np.log10(np.maximum(np.asarray(x), EPS))


def pwelch_centered(x: np.ndarray, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """近似 MATLAB pwelch(..., 'centered')。"""
    x = np.asarray(x)
    nperseg = max(8, min(len(x), int(round(len(x) / 10))))
    window = get_window("hamming", nperseg, fftbins=True)

    f, pxx = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=None,
        nfft=nfft,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )

    # MATLAB 的 centered 结果等价于频谱中心平移
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(pxx)
    return pxx, f


def drones_ofdm_freq_compensation(x: np.ndarray, fs: float, f: float) -> np.ndarray:
    """对应 MATLAB: dronesOFDMFreqCompensation。"""
    t = np.arange(1, len(x) + 1, dtype=np.float64) / fs
    return x * np.exp(-1j * 2.0 * np.pi * f * t)


def drones_ofdm_freq_shift_esti(
    x: np.ndarray,
    fs: float,
    bw: float,
    nfft: int,
) -> tuple[float, float]:
    """对应 MATLAB: dronesOFDMFreqShiftEsti。"""
    pxx, fvec = pwelch_centered(x, fs, nfft)

    pxx_db = db10(pxx)
    pxx_db = pxx_db / np.max(np.abs(pxx_db))
    pxx_db = pxx_db + (-np.min(pxx_db))

    bw_nfft = int(round(nfft * (bw / fs) * 0.9))
    bw_nfft = max(1, min(bw_nfft, nfft - 21))

    energy_len = nfft - bw_nfft - 20
    if energy_len <= 1:
        raise ValueError("参数组合导致能量搜索窗口无效，请检查 nfft / bw / fs。")

    energy = np.zeros(energy_len, dtype=np.float64)
    energy[0] = np.mean(np.square(pxx_db[0:bw_nfft]))

    for i in range(1, energy_len):
        energy[i] = energy[i - 1] - (pxx_db[i - 1] ** 2) / bw_nfft
        energy[i] += (pxx_db[i + bw_nfft - 1] ** 2) / bw_nfft

    max_idx = int(np.argmax(energy))
    f1 = float(fvec[max_idx])
    f2 = float(fvec[max_idx + bw_nfft])
    return f1, f2


def position_find(
    data_iq: np.ndarray,
    fs: float,
    bw: float,
    nfft: int,
) -> tuple[int, int, int, int, float, float]:
    """对应 MATLAB: positionFind (locate_signal.m)。"""
    f1, f2 = drones_ofdm_freq_shift_esti(data_iq, fs, bw, nfft)
    f_center = (f1 + f2) / 2.0
    sig = drones_ofdm_freq_compensation(data_iq, fs, f_center)

    f11 = f1 - f_center
    f22 = f2 - f_center

    refs = fs / 2.0
    re_nfft = nfft // 2

    # 对应 MATLAB resample(sig, refs, fs)，在当前参数下即降采样 2 倍
    sig_resample = resample_poly(sig, up=1, down=2)

    _, fvec3 = pwelch_centered(sig_resample, refs, re_nfft)

    idx1 = int(np.argmin(np.abs(fvec3 - f11)))
    idx2 = int(np.argmin(np.abs(fvec3 - f22)))

    bw_noise = 1e6
    f3 = 0.75e6

    bw_noise_nfft = int(round(re_nfft * (bw_noise / refs)))
    idx4 = idx1 - int(round(re_nfft * (f3 / refs)))
    idx3 = idx4 - bw_noise_nfft

    # 边界保护
    idx1 = int(np.clip(idx1, 0, re_nfft - 1))
    idx2 = int(np.clip(idx2, 0, re_nfft - 1))
    idx3 = int(np.clip(idx3, 0, re_nfft - 1))
    idx4 = int(np.clip(idx4, 0, re_nfft - 1))

    if idx1 > idx2:
        idx1, idx2 = idx2, idx1
    if idx3 > idx4:
        idx3, idx4 = idx4, idx3

    return idx1, idx2, idx3, idx4, f1, f2


def snr_esti(
    data_iq: np.ndarray,
    fs: float,
    nfft: int,
    f1: float,
    f2: float,
    idx1: int,
    idx2: int,
    idx3: int,
    idx4: int,
) -> float:
    """对应 MATLAB: snrEsti (SNREsti.m) 核心计算。"""
    f_center = (f1 + f2) / 2.0
    sig = drones_ofdm_freq_compensation(data_iq, fs, f_center)

    refs = fs / 2.0
    re_nfft = nfft // 2
    sig_resample = resample_poly(sig, up=1, down=2)

    fft_temp = np.abs(np.fft.fftshift(np.fft.fft(sig_resample, re_nfft) / re_nfft))

    sig_power = float(np.mean(np.square(fft_temp[idx1 : idx2 + 1])))
    nos_power = float(np.mean(np.square(fft_temp[idx3 : idx4 + 1])))

    snr_linear = (sig_power - nos_power) / max(nos_power, EPS)
    snr_linear = max(snr_linear, EPS)
    return float(np.real(10.0 * np.log10(snr_linear)))


def read_interleaved_iq(file_path: str | Path, dtype=np.float32) -> np.ndarray:
    """读取二进制交织 IQ 数据：I0,Q0,I1,Q1,..."""
    raw = np.fromfile(file_path, dtype=dtype)
    if raw.size < 2:
        raise ValueError(f"文件数据过短: {file_path}")
    if raw.size % 2 == 1:
        raw = raw[:-1]
    return raw[0::2] + 1j * raw[1::2]


def estimate_file_snr(file_path: str | Path, fs: float, bw: float, nfft: int) -> float:
    data_iq = read_interleaved_iq(file_path)
    idx1, idx2, idx3, idx4, f1, f2 = position_find(data_iq, fs, bw, nfft)
    return snr_esti(data_iq, fs, nfft, f1, f2, idx1, idx2, idx3, idx4)


def main() -> None:
    parser = argparse.ArgumentParser(description="OFDM 无人机信号 SNR 估计（MATLAB 版本移植）")
    parser.add_argument("input", type=str, help="输入 IQ 文件路径（float32 交织 IQ）")
    parser.add_argument("--fs", type=float, default=100e6, help="采样率，默认 100e6")
    parser.add_argument("--bw", type=float, default=20e6, help="符号带宽，默认 20e6")
    parser.add_argument("--nfft", type=int, default=409600, help="FFT 点数，默认 409600")

    args = parser.parse_args()

    snr_db = estimate_file_snr(args.input, fs=args.fs, bw=args.bw, nfft=args.nfft)
    print(f"Estimated SNR: {snr_db:.4f} dB")


if __name__ == "__main__":
    main()
