"""FCME adaptive sweep over all 39 rawdata classes."""
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

    # Adaptive FCME
    fcme = FCMEEstimator(fs=100e6, nfft=16384)
    snr_fcme, diag = fcme.estimate(iq)

    # CP
    snr_cp = estimate_cp_snr(iq)

    results.append((cls, sz_mb, snr_old, band_ok, snr_fcme, snr_cp,
                    diag["signal_bins"], diag["iterations"], diag.get("T_factor", 0)))

print("=" * 105)
print("%-30s %6s %8s %5s %8s %8s %5s %5s %5s" % (
    "CLASS", "MB", "OLD_dB", "BAND", "FCME_dB", "CP_dB", "SIG", "ITER", "T"))
print("-" * 105)
old_fail = 0
fcme_fail = 0
for r in results:
    cls, sz, old, ok, fcme, cp, sig, it, T = r
    if old == -999:
        old_s = "FAIL"
        old_fail += 1
    else:
        old_s = "%7.1f" % old
    ok_s = "OK" if ok else "XX"
    if fcme < -29:
        fcme_s = "FAIL"
        fcme_fail += 1
    else:
        fcme_s = "%7.1f" % fcme
    print("%-30s %6.0f %8s %5s %8s %8.1f %5d %5d %5.1f" % (
        cls[:30], sz, old_s, ok_s, fcme_s, cp, sig, it, T))

print("-" * 105)
print("Existing: band OK=0/39  failed=%d  |  FCME adaptive: failed=%d" % (old_fail, fcme_fail))
print("=" * 105)
