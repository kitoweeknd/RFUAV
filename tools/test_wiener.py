"""Test Wiener filter denoising on a low-SNR recording."""
import sys, os, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator
from scipy.signal import get_window, welch

# Pick a low-SNR recording
fpath = r"E:\Drone_dataset\RFUAV\rawdata\FLYSKY_EL_18\FLYSKY_EL_18-daifei-60db_2440m_100m(1).iq"
max_samp = 10_000_000
raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
n = raw.size // 2
iq = np.empty(n, dtype=np.complex64)
iq.real = raw[0::2]
iq.imag = raw[1::2]
del raw

# Step 1: Measure native SNR
fcme = FCMEEstimator(fs=100e6, nfft=16384)
snr_before, diag = fcme.estimate(iq)
print("Native SNR: %.2f dB (T=%.1f, sig_bins=%d, iters=%d)" % (
    snr_before, diag.get("T_factor", 0), diag["signal_bins"], diag["iterations"]))

# Step 2: Wiener filter
# Use the FCME PSD and signal/noise mask to build the filter
psd_linear = diag["psd_linear"]
signal_mask = diag["signal_mask"]

# Estimate P_signal and P_noise from the mask
p_sig = float(np.mean(psd_linear[signal_mask]))
p_noise = float(np.mean(psd_linear[~signal_mask]))

# Wiener: H(f) = P_signal / (P_signal + P_noise) per bin
# Apply per-bin: each bin gets its own Wiener weight
wiener = np.ones(len(psd_linear), dtype=np.float64)
# For signal bins: compute per-bin Wiener weights
for idx in np.where(signal_mask)[0]:
    wiener[idx] = psd_linear[idx] / (psd_linear[idx] + p_noise)
# For noise bins: suppress
wiener[~signal_mask] = psd_linear[~signal_mask] / (psd_linear[~signal_mask] + p_noise)

# Apply Wiener filter in frequency domain
# FFT the signal, multiply by sqrt(Wiener), IFFT back
fft_sig = np.fft.fft(iq)
# Interpolate wiener weights to FFT bins
# Wiener is on PSD bins (nfft=16384), FFT is on len(iq)=10M bins
# Strategy: use STFT-based approach instead
# Split into overlapping windows, apply Wiener per window, overlap-add

print("Applying STFT Wiener filter (window=2048, hop=512)...")
window_len = 2048
hop_len = 512
window = np.hanning(window_len).astype(np.float32)

n_frames = (len(iq) - window_len) // hop_len + 1
output = np.zeros(len(iq), dtype=np.complex128)
weight = np.zeros(len(iq), dtype=np.float64)

# Wiener weights in FCME PSD domain (16384 bins) -> interpolate to STFT bins (2048)
psd_bins = len(psd_linear)
stft_bins = window_len

for frame_idx in range(0, n_frames, 10):  # every 10th frame for speed
    start = frame_idx * hop_len
    end = start + window_len
    if end > len(iq):
        break
    frame = iq[start:end] * window
    frame_fft = np.fft.fft(frame)

    # Map FCME PSD bins to STFT bins
    ratio = stft_bins / psd_bins
    wiener_interp = np.interp(
        np.linspace(0, psd_bins - 1, stft_bins),
        np.arange(psd_bins), wiener)

    # Apply Wiener
    frame_clean = np.fft.ifft(frame_fft * np.sqrt(wiener_interp))
    output[start:end] += frame_clean.real
    weight[start:end] += window

# Normalize overlap-add
mask = weight > 0
output[mask] /= weight[mask]
output = output.astype(np.complex64)

# Step 3: Measure SNR after denoising
snr_after, diag2 = fcme.estimate(output[:max_samp])
print("Denoised SNR: %.2f dB (T=%.1f, sig_bins=%d, iters=%d)" % (
    snr_after, diag2.get("T_factor", 0), diag2["signal_bins"], diag2["iterations"]))
print("SNR gain: %.2f dB" % (snr_after - snr_before))
