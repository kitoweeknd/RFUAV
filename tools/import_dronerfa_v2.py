"""Import DroneRFa float32/complex data only (skip int32/int16 with headers)."""
import sys, os, json, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

RFA = r"E:\Drone_dataset\RFA\DroneRFa-IQ"
OUT = r"E:\Drone_dataset\RFUAV-v2\canonical"

# Only float32/complex64 files that are proven to work
FILES = [
    # Float32 complex files (native complex64 on disk)
    (r"float32\DJIAvata_float32_RF0_complex.dat", "uav.dji.avata2", np.complex64, 20e6, 0),
    (r"float32\DJIAvata_float32_RF1_complex.dat", "uav.dji.avata2", np.complex64, 20e6, 1),
    (r"float32\DJIAir2S_float32_RF0_complex.dat", "uav.dji.air2s", np.complex64, 20e6, 0),
    (r"float32\DJIAir2S_float32_RF1_complex.dat", "uav.dji.air2s", np.complex64, 20e6, 1),
    (r"float32\DJIMavic3_float32_RF0_complex.dat", "uav.dji.mavic3", np.complex64, 20e6, 0),
    (r"float32\DJIMavic3_float32_RF1_complex.dat", "uav.dji.mavic3", np.complex64, 20e6, 1),
    (r"float32\DJIMavicPro_float32_RF0_complex.dat", "uav.dji.mavic_pro", np.complex64, 20e6, 0),
    (r"float32\DJIMavicPro_float32_RF1_complex.dat", "uav.dji.mavic_pro", np.complex64, 20e6, 1),
    (r"float32\DJIMini3Pro_float32_RF0_complex.dat", "uav.dji.mini3_pro", np.complex64, 20e6, 0),
    (r"float32\DJIMini3Pro_float32_RF1_complex.dat", "uav.dji.mini3_pro", np.complex64, 20e6, 1),
    # Float32 interleaved files (proven to work)
    (r"float32\DJIMavic3_float32_RF0.dat", "uav.dji.mavic3", np.float32, 20e6, 0),
    (r"float32\DJIMavic3_float32_RF1.dat", "uav.dji.mavic3", np.float32, 20e6, 1),
    (r"float32\DJi_T1111_S1011_RF0_float32.dat", "uav.dji.fpv_combo", np.float32, 20e6, 0),
    (r"float32\DJi_T1111_S1011_RF1_float32.dat", "uav.dji.fpv_combo", np.float32, 20e6, 1),
    # Also the float32 AVATA/Air2S ones  
    (r"float32\AVATA_T1110_S1000_RF0_float32.dat", "uav.dji.avata2", np.float32, 20e6, 0),
    (r"float32\AVATA_T1110_S1000_RF1_float32.dat", "uav.dji.avata2", np.float32, 20e6, 1),
    (r"float32\Air2S_T0101_D01_S1010_RF0_float32.dat", "uav.dji.air2s", np.float32, 20e6, 0),
    (r"float32\Air2S_T0101_D01_S1010_RF1_float32.dat", "uav.dji.air2s", np.float32, 20e6, 1),
]

classes = sorted(set(f[1] for f in FILES))
print("DroneRFa float32 classes: %d" % len(classes))

for idx, (rel, tax_id, dtype, fs, rf) in enumerate(FILES):
    fpath = os.path.join(RFA, rel)
    if not os.path.exists(fpath):
        print("MISSING: %s" % fpath)
        continue

    stem = os.path.splitext(os.path.basename(rel))[0]
    rec_stem = "%s_rf%d" % (stem, rf)
    rec_dir = os.path.join(OUT, tax_id, rec_stem)

    # Skip if already done
    if os.path.exists(os.path.join(rec_dir, "metadata.json")):
        print("SKIP %s/%s" % (tax_id, rec_stem))
        continue

    os.makedirs(rec_dir, exist_ok=True)
    sz_mb = os.path.getsize(fpath) / 1e6

    # Read first 5M samples for SNR
    max_samp = 5_000_000
    if dtype == np.complex64:
        iq = np.fromfile(fpath, dtype=np.complex64, count=max_samp)
        n = len(iq)
    else:
        raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
        n = raw.size // 2
        iq = np.empty(n, dtype=np.complex64)
        iq.real = raw[0::2]; iq.imag = raw[1::2]; del raw

    fcme = FCMEEstimator(fs=fs, nfft=min(16384, n))
    snr_db, diag = fcme.estimate(iq)

    # Slice
    slice_s = int(fs * 0.1)
    n_slices = n // slice_s
    n_slices = min(n_slices, 2)

    metadata = {
        "source_dataset": "DroneRFa", "taxonomy_id": tax_id,
        "native_snr_db": round(snr_db, 1) if not np.isnan(snr_db) else None,
        "sample_rate_hz": fs, "rf_chain": rf,
        "n_slices": n_slices, "slice_samples": slice_s,
        "original_file": stem, "T_factor": diag.get("T_factor", 0),
    }
    with open(os.path.join(rec_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Write slices
    if dtype == np.complex64:
        iq_full = np.fromfile(fpath, dtype=np.complex64)
    else:
        read_full = np.fromfile(fpath, dtype=np.float32)
        nf = read_full.size // 2
        iq_full = np.empty(nf, dtype=np.complex64)
        iq_full.real = read_full[0::2]; iq_full.imag = read_full[1::2]; del read_full

    for si in range(n_slices):
        start = si * slice_s; end = start + slice_s
        if end > len(iq_full): break
        sl = iq_full[start:end]
        data = np.empty(2 * len(sl), dtype=np.float32)
        data[0::2] = sl.real; data[1::2] = sl.imag
        data.tofile(os.path.join(rec_dir, "slice_%04d.iq" % si))

    del iq_full
    print("[%d/%d] %-30s %6.0f MB  SNR=%.1f dB  slices=%d" % (
        idx+1, len(FILES), tax_id[:30], sz_mb, snr_db, n_slices))

print("Done. Float32 classes: %d" % len(classes))
