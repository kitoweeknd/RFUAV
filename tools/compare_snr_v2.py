"""Compare existing SNREstimator vs dual-path FCME + CP on real box data."""
import sys, os, time, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")

from snr_dual import estimate_snr_dual
from snr_estimator import SNREstimator

files = [
    r"E:\Drone_dataset\RFUAV\rawdata\DEVENTION DEVO\DEVENTION_DEVO_10-duima-60db_2440m_100m(1).iq",
    r"E:\Drone_dataset\RFUAV\rawdata\DJI AVATA2\DJI AVTA2-SNR0dB-85db_5765m_100m_10m(3).iq",
    r"E:\Drone_dataset\RFUAV\rawdata\SIYI_MK32\SIYI_MK32_2440_duima_gain80dB_34-36s.iq",
    r"E:\Drone_dataset\RFUAV\rawdata\FRSKY-X14\FRSKY-X14-daifei-60db_2440m_100m(2).iq",
    r"E:\Drone_dataset\RFUAV\rawdata\PARROT\PARROT_90db_5750m_100m(3).iq",
]

print("=" * 75)
print("SNR COMPARISON: Existing (PSD threshold) vs Dual-path (FCME + CP)")
print("=" * 75)

for fpath in files:
    if not os.path.exists(fpath):
        print("\n--- MISSING: %s ---" % os.path.basename(fpath))
        continue
    name = os.path.basename(fpath)
    sz_mb = os.path.getsize(fpath) / 1e6
    print("\n--- %s (%.0f MB) ---" % (name, sz_mb))

    max_samp = 5_000_000
    raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw

    t0 = time.time()
    try:
        est = SNREstimator(fs=100e6, bw=20e6, nfft=16384)
        result = est.estimate(iq)
        snr_old = result.snr_db
        t_old = time.time() - t0
        print("  Existing:    %7.2f dB  (%.2fs, band=[%.1f, %.1f] MHz)" % (
            snr_old, t_old, result.f1/1e6, result.f2/1e6))
    except Exception as e:
        snr_old = None
        print("  Existing:    FAILED (%s)" % e)

    t0 = time.time()
    try:
        dual = estimate_snr_dual(iq, fs=100e6, max_samples=max_samp)
        t_dual = time.time() - t0
        flag = "WARN" if dual.band_warning else "OK"
        print("  FCME:        %7.2f dB  (%d iters, %d sig bins)" % (
            dual.snr_fcme_db, dual.fcme_iterations, dual.fcme_signal_bins))
        print("  CP:          %7.2f dB  (blind, OFDM-specific)" % dual.snr_cp_db)
        print("  Cross-val:   %s  |FCME-CP| = %.2f dB" % (flag, abs(dual.snr_fcme_db - dual.snr_cp_db)))
        print("  Dual-path:   %.2fs" % t_dual)
    except Exception as e:
        print("  Dual-path:   FAILED (%s)" % e)

    if snr_old is not None:
        print("  Old vs FCME: %.2f dB" % abs(snr_old - dual.snr_fcme_db))

print("\n" + "=" * 75)
print("FCME = iterative band detection | CP = OFDM cyclic prefix correlation")
print("WARN = estimators disagree > 10 dB -> band detection may be unreliable")
print("=" * 75)
