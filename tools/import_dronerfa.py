"""Import DroneRFa float32/complex data into canonical tree."""
import os, sys, numpy as np, json, math
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator

RFA = r"E:\Drone_dataset\RFA\DroneRFa-IQ"
OUT = r"E:\Drone_dataset\RFUAV-v2\canonical"

# Known classes from DroneRFa (extracted from filenames)
# Format: (relpath, taxonomy_id, dtype, sample_rate_hz, is_complex, rf_chain)
FILES = [
    # Float32 complex files (already complex64 on disk)
    (r"float32\DJIAvata_float32_RF0_complex.dat", "uav.dji.avata2", np.complex64, 20e6, True, 0),
    (r"float32\DJIAvata_float32_RF1_complex.dat", "uav.dji.avata2", np.complex64, 20e6, True, 1),
    (r"float32\DJIAir2S_float32_RF0_complex.dat", "uav.dji.air2s", np.complex64, 20e6, True, 0),
    (r"float32\DJIAir2S_float32_RF1_complex.dat", "uav.dji.air2s", np.complex64, 20e6, True, 1),
    (r"float32\DJIMavic3_float32_RF0_complex.dat", "uav.dji.mavic3", np.complex64, 20e6, True, 0),
    (r"float32\DJIMavic3_float32_RF1_complex.dat", "uav.dji.mavic3", np.complex64, 20e6, True, 1),
    (r"float32\DJIMavicPro_float32_RF0_complex.dat", "uav.dji.mavic_pro", np.complex64, 20e6, True, 0),
    (r"float32\DJIMavicPro_float32_RF1_complex.dat", "uav.dji.mavic_pro", np.complex64, 20e6, True, 1),
    (r"float32\DJIMini3Pro_float32_RF0_complex.dat", "uav.dji.mini3_pro", np.complex64, 20e6, True, 0),
    (r"float32\DJIMini3Pro_float32_RF1_complex.dat", "uav.dji.mini3_pro", np.complex64, 20e6, True, 1),
    # Float32 files (interleaved I/Q, float32)
    (r"float32\DJIMavic3_float32_RF0.dat", "uav.dji.mavic3", np.float32, 20e6, False, 0),
    (r"float32\DJIMavic3_float32_RF1.dat", "uav.dji.mavic3", np.float32, 20e6, False, 1),
    (r"float32\DJi_T1111_S1011_RF0_float32.dat", "uav.dji.fpv_combo", np.float32, 20e6, False, 0),
    (r"float32\DJi_T1111_S1011_RF1_float32.dat", "uav.dji.fpv_combo", np.float32, 20e6, False, 1),
    # int32 files (interleaved I/Q, int32)
    (r"int32\AVATA_T1110_S1000_RF0.dat", "uav.dji.avata2", np.int32, 20e6, False, 0),
    (r"int32\AVATA_T1110_S1000_RF1.dat", "uav.dji.avata2", np.int32, 20e6, False, 1),
    (r"int32\Air2S_T0101_D01_S1010_RF0.dat", "uav.dji.air2s", np.int32, 20e6, False, 0),
    (r"int32\Air2S_T0101_D01_S1010_RF1.dat", "uav.dji.air2s", np.int32, 20e6, False, 1),
    (r"int32\FrskyX20_FLY_T10010_S1000_RF0.dat", "rc.frsky.x20", np.int32, 20e6, False, 0),
    (r"int32\FrskyX20_FLY_T10010_S1000_RF1.dat", "rc.frsky.x20", np.int32, 20e6, False, 1),
    (r"int32\FutabaT14SG_FLY_T10110_S1010_RF0.dat", "rc.futaba.t14sg", np.int32, 20e6, False, 0),
    (r"int32\FutabaT14SG_FLY_T10110_S1010_RF1.dat", "rc.futaba.t14sg", np.int32, 20e6, False, 1),
    (r"int32\FutabaT61Z_FLY_T10011_S1011_RF0.dat", "rc.futaba.t16iz", np.int32, 20e6, False, 0),
    (r"int32\FutabaT61Z_FLY_T10011_S1011_RF1.dat", "rc.futaba.t16iz", np.int32, 20e6, False, 1),
    (r"int32\Inspire2_T0111_D01_S1011_RF0.dat", "uav.dji.inspire2", np.int32, 20e6, False, 0),
    (r"int32\Inspire2_T0111_D01_S1011_RF1.dat", "uav.dji.inspire2", np.int32, 20e6, False, 1),
    (r"int32\Matrice100_RF0.dat", "uav.dji.matrice100", np.int32, 20e6, False, 0),
    (r"int32\Matrice100_RF1.dat", "uav.dji.matrice100", np.int32, 20e6, False, 1),
    (r"int32\Matrice200_RF0.dat", "uav.dji.matrice200", np.int32, 20e6, False, 0),
    (r"int32\Matrice200_RF1.dat", "uav.dji.matrice200", np.int32, 20e6, False, 1),
    (r"int32\Matrice300_T1011_S1011_RF0.dat", "uav.dji.matrice300", np.int32, 20e6, False, 0),
    (r"int32\Matrice300_T1011_S1011_RF1.dat", "uav.dji.matrice300", np.int32, 20e6, False, 1),
    (r"int32\Matrice30T_RF0.dat", "uav.dji.matrice30t", np.int32, 20e6, False, 0),
    (r"int32\Matrice30T_RF1.dat", "uav.dji.matrice30t", np.int32, 20e6, False, 1),
    (r"int32\Matrice600Pro_T10000_S1010_RF0.dat", "uav.dji.matrice600", np.int32, 20e6, False, 0),
    (r"int32\Matrice600Pro_T10000_S1010_RF1.dat", "uav.dji.matrice600", np.int32, 20e6, False, 1),
    (r"int32\Mavic3_RF0.dat", "uav.dji.mavic3", np.int32, 20e6, False, 0),
    (r"int32\Mavic3_RF1.dat", "uav.dji.mavic3", np.int32, 20e6, False, 1),
    (r"int32\Mini3PRO_RF0.dat", "uav.dji.mini3_pro", np.int32, 20e6, False, 0),
    (r"int32\Mini3PRO_RF1.dat", "uav.dji.mini3_pro", np.int32, 20e6, False, 1),
    (r"int32\Phantom3_T0001_D00_S1011_RF0.dat", "uav.dji.phantom3", np.int32, 20e6, False, 0),
    (r"int32\Phantom3_T0001_D00_S1011_RF1.dat", "uav.dji.phantom3", np.int32, 20e6, False, 1),
    (r"int32\Phantom4ProRTK_T1100_S1000_RF0.dat", "uav.dji.phantom4_rtk", np.int32, 20e6, False, 0),
    (r"int32\Phantom4ProRTK_T1100_S1000_RF1.dat", "uav.dji.phantom4_rtk", np.int32, 20e6, False, 1),
    (r"int32\Phantom4Pro_T0010_D00_S1010_RF0.dat", "uav.dji.phantom4", np.int32, 20e6, False, 0),
    (r"int32\Phantom4Pro_T0010_D00_S1010_RF1.dat", "uav.dji.phantom4", np.int32, 20e6, False, 1),
    (r"int32\RadioLinkAT9S_FLY_T10101_S1010_RF0.dat", "rc.radiolink.at9s", np.int32, 20e6, False, 0),
    (r"int32\RadioLinkAT9S_FLY_T10101_S1010_RF1.dat", "rc.radiolink.at9s", np.int32, 20e6, False, 1),
    (r"int32\Tarains_Plus_FLY_T10100_S1100_RF0.dat", "uav.tarains.plus", np.int32, 20e6, False, 0),
    (r"int32\Tarains_Plus_FLY_T10100_S1100_RF1.dat", "uav.tarains.plus", np.int32, 20e6, False, 1),
    (r"int32\VBar_Fly_T10001_S1010_RF0.dat", "rc.vbar", np.int32, 20e6, False, 0),
    (r"int32\VBar_Fly_T10001_S1010_RF1.dat", "rc.vbar", np.int32, 20e6, False, 1),
    (r"int32\yunzhuoT10_FLY_T11000_S1010_RF0.dat", "rc.yunzhuo.t10", np.int32, 20e6, False, 0),
    (r"int32\yunzhuoT10_FLY_T11000_S1010_RF1.dat", "rc.yunzhuo.t10", np.int32, 20e6, False, 1),
    (r"int32\yunzhuoT12_FLY_T10111_S1010_RF0.dat", "rc.yunzhuo.t12", np.int32, 20e6, False, 0),
    (r"int32\yunzhuoT12_FLY_T10111_S1010_RF1.dat", "rc.yunzhuo.t12", np.int32, 20e6, False, 1),
]

# Unique classes
classes = sorted(set(f[1] for f in FILES))
print("DroneRFa classes: %d" % len(classes))
for c in classes:
    files = [os.path.basename(f[0]) for f in FILES if f[1] == c]
    print("  %-30s %d files" % (c, len(files)))

# Process each file
for idx, (rel, tax_id, dtype, fs, is_complex, rf) in enumerate(FILES):
    fpath = os.path.join(RFA, rel)
    if not os.path.exists(fpath):
        print("MISSING: %s" % fpath)
        continue

    sz_mb = os.path.getsize(fpath) / 1e6
    stem = os.path.splitext(os.path.basename(rel))[0]
    rec_stem = "%s_rf%d" % (stem, rf)
    rec_dir = os.path.join(OUT, tax_id, rec_stem)
    os.makedirs(rec_dir, exist_ok=True)

    # Read first 5M complex samples for SNR estimation
    max_samp = 5_000_000
    if is_complex:
        # File is already complex64
        raw = np.fromfile(fpath, dtype=np.complex64, count=max_samp)
        n = len(raw)
        iq = raw
    else:
        # Interleaved IQ
        raw = np.fromfile(fpath, dtype=dtype, count=max_samp * 2)
        n = raw.size // 2
        iq = np.empty(n, dtype=np.complex64)
        if dtype == np.int32 or dtype == np.int16:
            # Normalize integer data to float32 range
            max_val = np.iinfo(dtype).max
            norm = float(max_val)
            iq.real = raw[0::2].astype(np.float32) / norm
            iq.imag = raw[1::2].astype(np.float32) / norm
        else:
            iq.real = raw[0::2]
            iq.imag = raw[1::2]
        del raw

    # Estimate SNR
    fcme = FCMEEstimator(fs=fs, nfft=min(16384, len(iq)))
    snr_db, diag = fcme.estimate(iq)

    # Slice into 0.1s windows
    slice_samples = int(fs * 0.1)
    n_slices = n // slice_samples
    n_slices = min(n_slices, 2)  # Only first 2 slices for space

    metadata = {
        "source_dataset": "DroneRFa",
        "taxonomy_id": tax_id,
        "native_snr_db": round(snr_db, 1),
        "sample_rate_hz": fs,
        "rf_chain": rf,
        "n_slices": n_slices,
        "slice_samples": slice_samples,
        "original_file": stem,
        "T_factor": diag.get("T_factor", 0),
    }
    with open(os.path.join(rec_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Write slices
    read_full = np.fromfile(fpath, dtype=np.complex64 if is_complex else dtype)
    if is_complex:
        iq_full = read_full
    else:
        nf = read_full.size // 2
        iq_full = np.empty(nf, dtype=np.complex64)
        if dtype == np.int32 or dtype == np.int16:
            max_val = np.iinfo(dtype).max
            norm = float(max_val)
            iq_full.real = read_full[0::2].astype(np.float32) / norm
            iq_full.imag = read_full[1::2].astype(np.float32) / norm
        else:
            iq_full.real = read_full[0::2]
            iq_full.imag = read_full[1::2]
        del read_full

    for si in range(n_slices):
        start = si * slice_samples
        end = start + slice_samples
        if end > len(iq_full):
            break
        slice_iq = iq_full[start:end]
        # Write as float32 interleaved (our canonical format)
        data = np.empty(2 * len(slice_iq), dtype=np.float32)
        data[0::2] = slice_iq.real
        data[1::2] = slice_iq.imag
        data.tofile(os.path.join(rec_dir, "slice_%04d.iq" % si))

    del iq_full
    print("[%d/%d] %-30s %6.0f MB  SNR=%.1f dB  slices=%d  fs=%.0f MHz" % (
        idx+1, len(FILES), tax_id[:30], sz_mb, snr_db, n_slices, fs/1e6))

print("\nDone. Total new classes: %d" % len(classes))
