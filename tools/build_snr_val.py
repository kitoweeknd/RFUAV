"""Build SNR-stratified val set: add AWGN to IQ, render spectrograms."""
import sys, os, csv, io, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
from graphic.RawDataProcessor import generate_images
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

m = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
rd = r"E:\Drone_dataset\RFUAV"
out = r"E:\Drone_dataset\RFUAV-v2\split_snr_val"

with io.open(m, encoding="utf-8-sig") as f:
    test_recs = [r for r in csv.DictReader(f) if r["split"] == "test" and r["source_root"] == "rawdata"]

from collections import defaultdict
best = {}
for r in test_recs:
    cls = r["taxonomy_id"]
    dur = float(r.get("delivered_duration_s", 0) or 0)
    if cls not in best or dur > best[cls][1]:
        best[cls] = (r, dur)
print("classes: %d" % len(best))

for cls, (r, dur) in sorted(best.items()):
    rel = r["source_relpath"]
    fpath = os.path.normpath(os.path.join(rd, rel))
    iq_name = rel.replace("/", "\\").split("\\")[-1]
    if not os.path.exists(fpath):
        continue
    
    # Measure native SNR from first 5M samples
    raw = np.fromfile(fpath, dtype=np.float32, count=10_000_000)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]; iq.imag = raw[1::2]; del raw
    
    fcme = FCMEEstimator(fs=100e6, nfft=16384)
    native_snr, _ = fcme.estimate(iq)
    
    # Render native
    native_dir = os.path.join(out, "snr_native", cls)
    if not os.path.exists(native_dir) or not os.listdir(native_dir):
        generate_images(datapack=fpath, file=cls, pack=iq_name, location=os.path.join(out, "snr_native"),
                       fs=100e6, stft_point=256, duration_time=0.1, step=1)
    
    # Degrade and render: -5, -10, -15 dB from native
    for delta in [5, 10, 15]:
        tgt = max(round(native_snr - delta, 1), 0)
        tgt_dir = os.path.join(out, "snr_%d" % int(tgt), cls)
        if os.path.exists(tgt_dir) and os.listdir(tgt_dir):
            continue
        
        # Add AWGN to degrade
        p_sig = float(np.mean(np.abs(iq) ** 2))
        p_noise_native = p_sig / (10 ** (native_snr / 10))
        p_noise_target = p_sig / (10 ** (tgt / 10))
        p_noise_add = max(p_noise_target - p_noise_native, 0)
        
        if p_noise_add > 0:
            iq_degraded = iq.copy()
            iq_degraded.real += np.random.randn(n).astype(np.float32) * np.sqrt(p_noise_add / 2)
            iq_degraded.imag += np.random.randn(n).astype(np.float32) * np.sqrt(p_noise_add / 2)
        else:
            iq_degraded = iq
        
        # Write temp .iq and render
        tmp_path = os.path.join(out, "_tmp_%s_%d.iq" % (cls.replace(".", "_"), int(tgt)))
        data = np.empty(2 * n, dtype=np.float32)
        data[0::2] = iq_degraded.real; data[1::2] = iq_degraded.imag
        data.tofile(tmp_path)
        
        generate_images(datapack=tmp_path, file=cls, pack=iq_name, location=os.path.join(out, "snr_%d" % int(tgt)),
                       fs=100e6, stft_point=256, duration_time=0.1, step=1)
        os.remove(tmp_path)
    
    print("  %-30s  native=%.1f dB" % (cls[:30], native_snr))

print("Done")
