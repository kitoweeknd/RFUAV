"""SNR pipeline: slice, SNR, render, delete .iq, keep .jpg only."""
import sys, os, csv, io, json, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator
from graphic.RawDataProcessor import generate_images

MAX_SLICES = 2
SNR_TARGETS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
CANON = r"E:\Drone_dataset\RFUAV-v2\canonical"
OUT = r"E:\Drone_dataset\RFUAV-v2\split_snr"
RD = r"E:\Drone_dataset\RFUAV"
M = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"

with io.open(M, encoding="utf-8-sig") as f:
    recs = [r for r in csv.DictReader(f) if r["split"] == "test" and r["source_root"] == "rawdata"]

for idx, r in enumerate(recs):
    cls = r["taxonomy_id"]
    rel = r["source_relpath"]
    fpath = os.path.normpath(os.path.join(RD, rel))
    stem = os.path.splitext(rel.replace("/", "\\").split("\\")[-1])[0]
    if not os.path.exists(fpath): continue
    
    rec_dir = os.path.join(CANON, cls, stem)
    os.makedirs(rec_dir, exist_ok=True)
    sp = int(100e6 * 0.1)
    
    # Slice
    for si in range(MAX_SLICES):
        out_slice = os.path.join(rec_dir, "slice_%04d.iq" % si)
        if os.path.exists(out_slice): continue
        raw = np.fromfile(fpath, dtype=np.float32, offset=si * sp * 8, count=sp * 2)
        if raw.size < 2: break
        raw.tofile(out_slice)
    
    # SNR
    first_slice = os.path.join(rec_dir, "slice_0000.iq")
    snr_file = os.path.join(rec_dir, "slice_0000_snr.json")
    if os.path.exists(snr_file):
        with open(snr_file) as f: native_snr = json.load(f)["native_snr_db"]
    else:
        raw = np.fromfile(first_slice, dtype=np.float32, count=5_000_000 * 2)
        n = raw.size // 2
        iq = np.empty(n, dtype=np.complex64); iq.real = raw[0::2]; iq.imag = raw[1::2]; del raw
        fcme = FCMEEstimator(fs=100e6, nfft=16384)
        native_snr, diag = fcme.estimate(iq)
        with open(snr_file, "w") as f: json.dump({"native_snr_db": round(native_snr, 1)}, f)
    
    # Render native + SNR targets, then delete .iq
    for si in range(MAX_SLICES):
        slice_path = os.path.join(rec_dir, "slice_%04d.iq" % si)
        if not os.path.exists(slice_path): continue
        
        # Native
        native_dir = os.path.join(OUT, "native", cls)
        if not os.path.exists(native_dir) or not os.listdir(native_dir):
            generate_images(datapack=slice_path, file=cls, pack=stem, location=os.path.join(OUT, "native"),
                           fs=100e6, stft_point=256, duration_time=0.1, step=1)
        
        # SNR targets
        raw_data = np.fromfile(slice_path, dtype=np.float32)
        n = raw_data.size // 2
        iq = np.empty(n, dtype=np.complex64); iq.real = raw_data[0::2]; iq.imag = raw_data[1::2]; del raw_data
        p_sig = float(np.mean(np.abs(iq) ** 2))
        p_noise_native = p_sig / (10 ** (native_snr / 10))
        
        for tgt in SNR_TARGETS:
            if tgt > native_snr: continue
            tgt_name = "%+ddB" % tgt
            tgt_dir = os.path.join(OUT, tgt_name, cls)
            if os.path.exists(tgt_dir) and os.listdir(tgt_dir): continue
            
            p_noise_target = p_sig / (10 ** (tgt / 10))
            p_noise_add = max(p_noise_target - p_noise_native, 0)
            if p_noise_add > 0:
                iq_d = iq.copy()
                iq_d.real += np.random.randn(n).astype(np.float32) * np.sqrt(p_noise_add / 2)
                iq_d.imag += np.random.randn(n).astype(np.float32) * np.sqrt(p_noise_add / 2)
            else:
                iq_d = iq
            
            tmp = os.path.join(rec_dir, "_tmp.iq")
            data = np.empty(2 * n, dtype=np.float32); data[0::2] = iq_d.real; data[1::2] = iq_d.imag
            data.tofile(tmp)
            generate_images(datapack=tmp, file=cls, pack=stem, location=os.path.join(OUT, tgt_name),
                           fs=100e6, stft_point=256, duration_time=0.1, step=1)
            os.remove(tmp)
        
        # Delete .iq slice to save space
        os.remove(slice_path)
    
    print("[%d/%d] %-30s nat=%.1f dB" % (idx+1, len(recs), cls[:30], native_snr))
print("Done")
