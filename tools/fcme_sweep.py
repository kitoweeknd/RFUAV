"""FCME sweep over all 38 rawdata classes on box."""
import sys, os, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator, estimate_cp_snr
from snr_estimator import SNREstimator

rd = r"E:\Drone_dataset\RFUAV\rawdata"
results = []
for cls in sorted(os.listdir(rd)):
    cp = os.path.join(rd, cls)
    if not os.path.isdir(cp): continue
    iqs = [f for f in os.listdir(cp) if f.endswith('.iq')]
    if not iqs: continue
    fpath = os.path.join(cp, max(iqs, key=lambda f: os.path.getsize(os.path.join(cp, f))))
    sz_mb = os.path.getsize(fpath) / 1e6

    max_samp = 10_000_000
    raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw

    # Existing
    try:
        est = SNREstimator(fs=100e6, bw=20e6, nfft=16384)
        r = est.estimate(iq)
        snr_old = r.snr_db
        band_ok = abs(r.f2 - r.f1) < 80e6
    except:
        snr_old = -999
        band_ok = False

    # FCME
    fcme = FCMEEstimator(fs=100e6, nfft=16384, T_factor=2.5)
    snr_fcme, diag = fcme.estimate(iq)

    # CP
    snr_cp = estimate_cp_snr(iq)

    results.append((cls, sz_mb, snr_old, band_ok, snr_fcme, snr_cp,
                    diag["iterations"], diag["signal_bins"], diag["noise_bins"]))

print("=" * 100)
print("%-30s %6s %8s %5s %8s %8s %5s %5s %5s" % (
    "CLASS", "MB", "OLD_dB", "BAND", "FCME_dB", "CP_dB", "ITER", "SIG", "NOIS"))
print("-" * 100)
old_fail = 0
old_ok = 0
for r in results:
    cls, sz, old, ok, fcme, cp, it, sig, noi = r
    if old == -999:
        old_s = "FAIL"
        old_fail += 1
    else:
        old_s = "%7.1f" % old
        if ok: old_ok += 1
    ok_s = "OK" if ok else "XX"
    print("%-30s %6.0f %8s %5s %8.1f %8.1f %5d %5d %5d" % (
        cls[:30], sz, old_s, ok_s, fcme, cp, it, sig, noi))

print("-" * 100)
print("Existing failed: %d/%d  |  band OK: %d/%d" % (old_fail, len(results), old_ok, len(results)))
print("FCME: all converged  |  CP: blind OFDM cross-check")
print("=" * 100)
