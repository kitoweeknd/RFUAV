"""Measure native SNR of all 254 rawdata recordings."""
import sys, os, csv, io, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

rd = r"E:\Drone_dataset\RFUAV\rawdata"
m = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
with io.open(m, encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["source_root"] == "rawdata"]
print("recordings: %d" % len(rows))

results = []
for r in rows:
    rel = r["source_relpath"]
    fpath = os.path.normpath(os.path.join(rd, "..", rel))
    if not os.path.exists(fpath):
        continue
    max_samp = 5_000_000
    raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw
    fcme = FCMEEstimator(fs=100e6, nfft=16384)
    snr, diag = fcme.estimate(iq)
    results.append((r["taxonomy_id"], r["split"], snr, diag["signal_bins"], diag.get("T_factor", 0)))

# Summary by class
from collections import defaultdict
cls_snr = defaultdict(list)
for cls, split, snr, sig, T in results:
    if split in ("train", "test"):
        cls_snr[cls].append(snr)

print("\n%-35s %5s %7s %7s %7s" % ("CLASS", "N", "MIN", "MEDIAN", "MAX"))
print("-" * 60)
all_snr = []
for cls in sorted(cls_snr):
    snrs = cls_snr[cls]
    all_snr.extend(snrs)
    print("%-35s %5d %7.1f %7.1f %7.1f" % (cls, len(snrs), min(snrs), np.median(snrs), max(snrs)))

print("-" * 60)
all_arr = np.array(all_snr)
print("OVERALL: min=%.1f  median=%.1f  max=%.1f  mean=%.1f" % (
    all_arr.min(), np.median(all_arr), all_arr.max(), all_arr.mean()))
print("SNR < 20dB: %d/%d (%.0f%%)" % (
    (all_arr < 20).sum(), len(all_arr), 100*(all_arr < 20).sum()/len(all_arr)))
print("SNR < 15dB: %d/%d (%.0f%%)" % (
    (all_arr < 15).sum(), len(all_arr), 100*(all_arr < 15).sum()/len(all_arr)))
print("SNR < 10dB: %d/%d (%.0f%%)" % (
    (all_arr < 10).sum(), len(all_arr), 100*(all_arr < 10).sum()/len(all_arr)))
print("SNR < 5dB:  %d/%d (%.0f%%)" % (
    (all_arr < 5).sum(), len(all_arr), 100*(all_arr < 5).sum()/len(all_arr)))
