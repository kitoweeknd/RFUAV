"""Decimate all DroneRF canonical slices 40 MHz -> 20 MHz complex (lossless:
the analytic signal only occupies the 20 MHz half-band; the other half is
numerically dead, so y[::2] cannot alias). Also re-estimates FCME SNR on the
decimated signal (the dead-band artifact disappears with the dead band) and
deletes the stale 40 MHz spectrograms so stage3 re-renders.
"""
import os, sys, json, glob
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
IMAGES = r"E:\Drone_dataset\RFUAV-v2\images_canonical"
FS_NEW = 20e6
SS_NEW = 2_000_000

fixed = skipped = 0
for tax in sorted(os.listdir(CANON)):
    tdir = os.path.join(CANON, tax)
    if not os.path.isdir(tdir):
        continue
    for rec in sorted(os.listdir(tdir)):
        mpath = os.path.join(tdir, rec, "metadata.json")
        if not os.path.exists(mpath):
            continue
        md = json.load(open(mpath, encoding="utf-8"))
        if md.get("source_dataset") != "DroneRF":
            continue
        if md.get("sample_rate_hz") == FS_NEW:
            skipped += 1
            continue
        rdir = os.path.join(tdir, rec)
        for spath in sorted(glob.glob(os.path.join(rdir, "slice_*.iq"))):
            raw = np.fromfile(spath, dtype=np.float32)
            n = raw.size // 2
            y = np.empty(n, dtype=np.complex64)
            y.real = raw[0::2]; y.imag = raw[1::2]
            del raw
            y2 = np.ascontiguousarray(y[::2])
            del y
            out = np.empty(2 * len(y2), dtype=np.float32)
            out[0::2] = y2.real; out[1::2] = y2.imag
            out.tofile(spath)
            del y2, out
        # correct SNR on the decimated (fully occupied band) signal
        raw = np.fromfile(os.path.join(rdir, "slice_0000.iq"),
                          dtype=np.float32, count=2 * SS_NEW)
        iq = np.empty(len(raw) // 2, dtype=np.complex64)
        iq.real = raw[0::2]; iq.imag = raw[1::2]
        snr, diag = FCMEEstimator(fs=FS_NEW, nfft=16384).estimate(iq)
        old_snr = md.get("native_snr_db")
        md["sample_rate_hz"] = FS_NEW
        md["slice_samples"] = SS_NEW
        md["native_snr_db"] = round(float(snr), 1) if np.isfinite(snr) else None
        md["real_to_complex"] = ("hilbert analytic + (-10 MHz) shift + 2:1 "
                                 "decimation (lossless, half-band was empty), "
                                 "complex64 @ 20 MHz")
        md["snr_method"] = "fcme_after_decimation"
        json.dump(md, open(mpath, "w", encoding="utf-8"), indent=2)
        # purge stale 40 MHz renders so stage3 re-renders
        idir = os.path.join(IMAGES, tax, rec)
        n_img = 0
        if os.path.isdir(idir):
            for j in glob.glob(os.path.join(idir, "*.jpg")):
                os.remove(j)
                n_img += 1
        fixed += 1
        print("DECIMATED %-22s %-20s SNR %s -> %s dB (purged %d imgs)"
              % (tax[:22], rec, old_snr, md["native_snr_db"], n_img))

print("decimation done: fixed=%d skipped=%d" % (fixed, skipped))
