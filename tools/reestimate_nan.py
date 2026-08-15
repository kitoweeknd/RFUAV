"""Re-estimate SNR for canonical recordings with null/NaN native_snr_db."""
import os, json, sys, math
import numpy as np
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
fixed = checked = 0
for tax in sorted(os.listdir(CANON)):
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        rdir = os.path.join(tdir, rec)
        mpath = os.path.join(rdir, "metadata.json")
        if not os.path.exists(mpath):
            continue
        with open(mpath) as f:
            meta = json.load(f)
        snr = meta.get("native_snr_db")
        checked += 1
        if snr is not None and not (isinstance(snr, float) and math.isnan(snr)):
            continue
        sp = os.path.join(rdir, "slice_0000.iq")
        if not os.path.exists(sp):
            continue
        raw = np.fromfile(sp, dtype=np.float32, count=10_000_000)
        n = raw.size // 2
        iq = np.empty(n, dtype=np.complex64)
        iq.real = raw[0::2]; iq.imag = raw[1::2]; del raw
        fs = float(meta.get("sample_rate_hz") or 100e6)
        fcme = FCMEEstimator(fs=fs, nfft=min(16384, n))
        snr_db, diag = fcme.estimate(iq)
        if not np.isfinite(snr_db):
            print("STILL NAN: %s/%s" % (tax, rec))
            continue
        meta["native_snr_db"] = round(float(snr_db), 1)
        meta["snr_method"] = "fcme_" + diag.get("method_fallback", "adaptive")
        with open(mpath, "w") as f:
            json.dump(meta, f, indent=2)
        fixed += 1
        print("FIXED %-24s %-40s SNR=%.1f dB  (%s, sig=%d noise=%d)" % (
            tax, rec[:40], snr_db, diag.get("method_fallback", "adaptive"),
            diag["signal_bins"], diag["noise_bins"]))
print("checked=%d fixed=%d" % (checked, fixed))
