"""Re-test FCME on the 5 failed classes with lower T_factor."""
import sys, os, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator
from snr_estimator import SNREstimator

rd = r"E:\Drone_dataset\RFUAV\rawdata"
failed = ["DJFPVCOMBO", "DJMINI3", "DJMINI4PRO", "FutabaT14SG", "JR PROPO_XG14"]
print("=" * 90)
print("%-25s %8s %8s %8s %8s %8s" % ("CLASS", "OLD", "T=2.5", "T=2.0", "T=1.5", "T=1.0"))
print("-" * 90)
for cls in failed:
    cp = os.path.join(rd, cls)
    if not os.path.isdir(cp): continue
    iqs = [f for f in os.listdir(cp) if f.endswith('.iq')]
    if not iqs: continue
    fpath = os.path.join(cp, iqs[0])
    max_samp = 10_000_000
    raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw

    try:
        est = SNREstimator(fs=100e6, bw=20e6, nfft=16384)
        r = est.estimate(iq)
        snr_old = r.snr_db
    except:
        snr_old = -999

    snrs = []
    for T in [2.5, 2.0, 1.5, 1.0]:
        fcme = FCMEEstimator(fs=100e6, nfft=16384, T_factor=T)
        s, _ = fcme.estimate(iq)
        snrs.append(s)
    print("%-25s %8.1f %8.1f %8.1f %8.1f %8.1f" % (
        cls[:25], snr_old, snrs[0], snrs[1], snrs[2], snrs[3]))
print("=" * 90)
