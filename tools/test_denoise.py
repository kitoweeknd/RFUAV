"""Test 3 denoising methods on a low-SNR recording."""
import sys, os, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator
from scipy.signal import get_window

fpath = r"E:\Drone_dataset\RFUAV\rawdata\FLYSKY_EL_18\FLYSKY_EL_18-daifei-60db_2440m_100m(1).iq"
max_samp = 10_000_000
raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
n = raw.size // 2
iq = np.empty(n, dtype=np.complex64)
iq.real = raw[0::2]
iq.imag = raw[1::2]
del raw

fcme = FCMEEstimator(fs=100e6, nfft=16384)
snr_native, diag = fcme.estimate(iq)
print("Native SNR: %.2f dB" % snr_native)

p_noise = float(np.mean(diag["psd_linear"][~diag["signal_mask"]]))
print("Noise floor (linear): %.8f" % p_noise)

# Method 1: Spectral subtraction (simple, per frame)
print("\n--- Method 1: Spectral Subtraction ---")
window = np.hanning(2048).astype(np.float32)
hop = 512
frame = iq[:2048] * window
fft_frame = np.fft.fft(frame)
psd_frame = np.abs(fft_frame) ** 2
# Subtract noise floor, clamp to zero
psd_clean = np.maximum(psd_frame - p_noise * 2048, 0)
fft_clean = fft_frame * np.sqrt(psd_clean / np.maximum(psd_frame, 1e-20))
iq_clean = np.fft.ifft(fft_clean)

snr_ss, _ = fcme.estimate(iq_clean.astype(np.complex64))
print("Spectral subtraction SNR: %.2f dB (gain: %.2f)" % (snr_ss, snr_ss - snr_native))

# Method 2: FFT-based Wiener (full signal, no STFT)
print("\n--- Method 2: FFT Wiener (full signal) ---")
fft_full = np.fft.fft(iq)
psd_full = np.abs(fft_full) ** 2
# Wiener per FFT bin: H = P_sig / (P_sig + P_noise)
# Estimate P_sig per bin = max(PSD_bin - P_noise * N, 0)
p_noise_per_bin = p_noise * len(iq)
psd_sig = np.maximum(psd_full - p_noise_per_bin, 0)
wiener_weights = psd_sig / np.maximum(psd_full, 1e-20)
fft_wiener = fft_full * wiener_weights
iq_wiener = np.fft.ifft(fft_wiener).astype(np.complex64)

snr_wiener, _ = fcme.estimate(iq_wiener)
print("FFT Wiener SNR: %.2f dB (gain: %.2f)" % (snr_wiener, snr_wiener - snr_native))

# Method 3: Soft thresholding in FFT domain
print("\n--- Method 3: FFT Soft Threshold ---")
mag = np.abs(fft_full)
phase = np.angle(fft_full)
threshold = np.sqrt(p_noise_per_bin) * 3  # 3-sigma threshold
mag_clean = np.maximum(mag - threshold, 0)
fft_soft = mag_clean * np.exp(1j * phase)
iq_soft = np.fft.ifft(fft_soft).astype(np.complex64)

snr_soft, _ = fcme.estimate(iq_soft)
print("FFT soft threshold SNR: %.2f dB (gain: %.2f)" % (snr_soft, snr_soft - snr_native))

print("\n=== SUMMARY ===")
print("Native:  %.2f dB" % snr_native)
print("SS:      %.2f dB  (gain %+.2f)" % (snr_ss, snr_ss - snr_native))
print("Wiener:  %.2f dB  (gain %+.2f)" % (snr_wiener, snr_wiener - snr_native))
print("Soft:    %.2f dB  (gain %+.2f)" % (snr_soft, snr_soft - snr_native))
