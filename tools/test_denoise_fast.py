"""Test denoising with 100K samples for speed."""
import sys, os, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

fpath = r"E:\Drone_dataset\RFUAV\rawdata\FLYSKY_EL_18\FLYSKY_EL_18-daifei-60db_2440m_100m(1).iq"
max_samp = 100_000
raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
n = raw.size // 2
iq = np.empty(n, dtype=np.complex64)
iq.real = raw[0::2]
iq.imag = raw[1::2]
del raw

fcme = FCMEEstimator(fs=100e6, nfft=4096)
snr_native, diag = fcme.estimate(iq)
p_noise = float(np.mean(diag["psd_linear"][~diag["signal_mask"]]))
print("Native SNR: %.2f dB  noise_floor: %.8f" % (snr_native, p_noise))

# Method 1: Spectral subtraction
fft = np.fft.fft(iq)
psd = np.abs(fft)**2
psd_clean = np.maximum(psd - p_noise * len(iq), 0)
fft_ss = fft * np.sqrt(psd_clean / np.maximum(psd, 1e-20))
iq_ss = np.fft.ifft(fft_ss).astype(np.complex64)
snr_ss, _ = fcme.estimate(iq_ss)
print("Spectral subtraction: %.2f dB (gain %+.2f)" % (snr_ss, snr_ss - snr_native))

# Method 2: Wiener
psd_sig = np.maximum(psd - p_noise * len(iq), 0)
wiener = psd_sig / np.maximum(psd, 1e-20)
iq_w = np.fft.ifft(fft * wiener).astype(np.complex64)
snr_w, _ = fcme.estimate(iq_w)
print("Wiener:              %.2f dB (gain %+.2f)" % (snr_w, snr_w - snr_native))

# Method 3: Soft threshold
mag = np.abs(fft)
phase = np.angle(fft)
thresh = np.sqrt(p_noise * len(iq)) * 3
mag_c = np.maximum(mag - thresh, 0)
iq_st = np.fft.ifft(mag_c * np.exp(1j*phase)).astype(np.complex64)
snr_st, _ = fcme.estimate(iq_st)
print("Soft threshold:      %.2f dB (gain %+.2f)" % (snr_st, snr_st - snr_native))

print("\n=== SUMMARY ===")
print("Native: %.2f dB  |  SS: %.2f  |  Wiener: %.2f  |  Soft: %.2f" % (
    snr_native, snr_ss, snr_w, snr_st))
