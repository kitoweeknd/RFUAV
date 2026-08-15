"""Verify the 8 DroneRFa classes whose FCME SNR is NaN: are slices real signal?"""
import os, json, sys
import numpy as np
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator, estimate_cp_snr

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
SUSPECT = ["rc.radiolink.at9s", "rc.yunzhuo.t10", "rc.yunzhuo.t12",
           "uav.dji.matrice200", "uav.dji.matrice300", "uav.dji.matrice30t",
           "uav.dji.matrice600", "uav.dji.phantom4_rtk"]
GOOD_REF = ["uav.dji.phantom3", "rc.vbar"]

def stats(tax):
    tdir = os.path.join(CANON, tax)
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        sp = os.path.join(rdir, "slice_0000.iq")
        if not os.path.exists(sp):
            continue
        raw = np.fromfile(sp, dtype=np.float32)
        n = raw.size // 2
        iq = np.empty(n, dtype=np.complex64)
        iq.real = raw[0::2]; iq.imag = raw[1::2]
        mag = np.abs(iq)
        rms = float(np.sqrt(np.mean(mag**2)))
        zfrac = float(np.mean(mag < 1e-12))
        # distinct value count on first 100k I samples (int16-converted data has few levels)
        distinct = len(np.unique(raw[:200000]))
        meta = json.load(open(os.path.join(rdir, "metadata.json")))
        fs = float(meta.get("sample_rate_hz") or 20e6)
        fcme = FCMEEstimator(fs=fs, nfft=min(16384, n))
        snr, diag = fcme.estimate(iq[:2_000_000])
        cp = estimate_cp_snr(iq[:2_000_000])
        print("  %-44s rms=%.3e zero%%=%.1f distinct=%6d fcme=%7.1f sigbins=%5d cp=%6.1f" % (
            rec[:44], rms, zfrac*100, distinct, snr, diag["signal_bins"], cp))

for tax in SUSPECT:
    print(tax + ":")
    stats(tax)
print("--- reference (known-good) ---")
for tax in GOOD_REF:
    print(tax + ":")
    stats(tax)
