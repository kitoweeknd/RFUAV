"""Verify DroneRFa imported data: check slices, SNR, and renderability."""
import sys, os, json, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator
from graphic.RawDataProcessor import generate_images

canon = r"E:\Drone_dataset\RFUAV-v2\canonical"

# Find an imported DroneRFa class with data
imported = []
for cls in sorted(os.listdir(canon)):
    cp = os.path.join(canon, cls)
    if not os.path.isdir(cp):
        continue
    for rec in os.listdir(cp):
        rp = os.path.join(cp, rec)
        md = os.path.join(rp, "metadata.json")
        if os.path.exists(md):
            with open(md) as f:
                meta = json.load(f)
            if meta.get("source_dataset") == "DroneRFa":
                slices = [f for f in os.listdir(rp) if f.endswith(".iq")]
                imported.append((cls, rec, rp, meta, slices))
                if len(imported) >= 3:
                    break
    if len(imported) >= 3:
        break

if not imported:
    print("No DroneRFa data found!")
    raise SystemExit(1)

print("=== Verifying %d DroneRFa recordings ===" % len(imported))
for cls, rec, rp, meta, slices in imported:
    print("\n--- %s / %s ---" % (cls, rec))
    print("  metadata: %s" % json.dumps(meta))
    print("  slices: %d" % len(slices))
    
    # Verify slice 0
    s0 = os.path.join(rp, "slice_0000.iq")
    if not os.path.exists(s0):
        print("  MISSING slice_0000.iq!")
        continue
    
    sz = os.path.getsize(s0)
    raw = np.fromfile(s0, dtype=np.float32, count=min(5_000_000 * 2, sz // 4))
    if len(raw) < 2:
        print("  EMPTY slice!")
        continue
    
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw
    
    print("  slice size: %.1f MB  samples: %d" % (sz / 1e6, n))
    print("  IQ range: [%.4f, %.4f]  mean abs: %.4f" % (
        np.min(np.abs(iq)), np.max(np.abs(iq)), np.mean(np.abs(iq))))
    
    # Re-measure SNR
    fcme = FCMEEstimator(fs=meta["sample_rate_hz"], nfft=min(16384, n))
    snr, diag = fcme.estimate(iq[:min(n, 5_000_000)])
    print("  SNR: measured=%.1f dB  recorded=%.1f dB  match=%s" % (
        snr, meta["native_snr_db"], "OK" if abs(snr - meta["native_snr_db"]) < 3 else "DRIFT"))
    
    # Render a spectrogram
    try:
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            generate_images(datapack=s0, file=cls, pack=rec, location=td,
                          fs=meta["sample_rate_hz"], stft_point=256, duration_time=0.1, step=1)
            out_dir = os.path.join(td, cls, rec)
            if os.path.isdir(out_dir):
                jpgs = [f for f in os.listdir(out_dir) if f.endswith(".jpg")]
                print("  spectrogram: %d frames rendered  OK" % len(jpgs))
            else:
                print("  spectrogram: FAILED (no output dir)")
    except Exception as e:
        print("  spectrogram: FAILED (%s)" % e)

print("\n=== VERIFICATION COMPLETE ===")
