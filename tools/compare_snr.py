"""Compare existing SNREstimator vs dual-path FCME+M2M4 on real box data."""
import sys, os, time, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")

from snr_dual import estimate_snr_dual
from snr_estimator import SNREstimator

# Pick diverse recordings: 3 classes, varied sizes
files = [
    r"E:\Drone_dataset\RFUAV\rawdata\DEVENTION DEVO\DEVENTION_DEVO_10-duima-60db_2440m_100m(1).iq",
    r"E:\Drone_dataset\RFUAV\rawdata\DJI AVATA2\DJI AVTA2-SNR0dB-85db_5765m_100m_10m(3).iq",
    r"E:\Drone_dataset\RFUAV\rawdata\SIYI_MK32\SIYI_MK32_2440_duima_gain80dB_34-36s.iq",
]

print("=" * 75)
print("SNR ESTIMATOR COMPARISON: Existing vs Dual-path (FCME + M2M4)")
print("=" * 75)

for fpath in files:
    name = os.path.basename(fpath)
    sz_mb = os.path.getsize(fpath) / 1e6
    print("\n--- %s (%.0f MB) ---" % (name, sz_mb))

    # Read first 5M samples for both estimators (fair comparison)
    max_samp = 5_000_000
    raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw

    # ── Existing estimator ──
    t0 = time.time()
    try:
        est = SNREstimator(fs=100e6, bw=20e6, nfft=16384)
        result = est.estimate(iq)
        snr_old = result.snr_db
        t_old = time.time() - t0
        print("  Existing (PSD threshold):  %.2f dB  (%.2fs, band=[%.1f, %.1f] MHz)" % (
            snr_old, t_old, result.f1/1e6, result.f2/1e6))
    except Exception as e:
        snr_old = None
        print("  Existing (PSD threshold):  FAILED (%s)" % e)

    # ── Dual-path (FCME + M2M4) ──
    t0 = time.time()
    try:
        dual = estimate_snr_dual(iq, fs=100e6, max_samples=max_samp)
        t_dual = time.time() - t0
        flag = "WARN: BAND LOCK FAILURE" if dual.band_warning else "OK"
        print("  FCME:                      %.2f dB  (%d iters, %d sig bins, %d noise bins)" % (
            dual.snr_fcme_db, dual.fcme_iterations, dual.fcme_signal_bins, dual.fcme_noise_bins))
        print("  M2M4:                      %.2f dB  (blind, no band detection)" % dual.snr_m2m4_db)
        print("  Cross-validation:          %s  |FCME-M2M4| = %.2f dB" % (flag, abs(dual.snr_fcme_db - dual.snr_m2m4_db)))
        print("  Dual-path:                  %.2fs" % t_dual)
    except Exception as e:
        print("  Dual-path:                 FAILED (%s)" % e)
        import traceback
        traceback.print_exc()

    # ── Comparison ──
    if snr_old is not None:
        diff = abs(snr_old - dual.snr_fcme_db)
        print("  Δ (existing vs FCME):     %.2f dB" % diff)

print("\n" + "=" * 75)
print("LEGEND: FCME (iterative, band-adaptive) | M2M4 (blind, moment-based)")
print("WARN = FCME and M2M4 disagree > 3 dB → band detection may be unreliable")
print("=" * 75)
