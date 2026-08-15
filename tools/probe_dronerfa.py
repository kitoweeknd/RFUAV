"""Probe DroneRFa .dat format: read first 20 bytes, infer dtype."""
import os, numpy as np

rfa = r"E:\Drone_dataset\RFA\DroneRFa-IQ"
probes = [
    (r"float32\DJIAvata_float32_RF0_complex.dat", "complex64"),
    (r"float32\DJIAvata_float32_RF1_complex.dat", "complex64"),
    (r"float32\DJIMavic3_float32_RF0.dat", "float32 interleaved"),
    (r"float32\DJIMavic3_float32_RF0_complex.dat", "complex64"),
    (r"int32\AVATA_T1110_S1000_RF0.dat", "int32"),
    (r"int32\DJi_RF0.dat", "int32"),
    (r"int16\DJi_RF0.dat", "int16"),
    (r"\avata2_fs15.36M.dat", "float32"),
]

for rel, expected in probes:
    fpath = os.path.join(rfa, rel.lstrip("\\"))
    if not os.path.exists(fpath):
        print("MISSING: %s" % fpath)
        continue
    sz = os.path.getsize(fpath)
    raw = np.fromfile(fpath, dtype=np.uint8, count=40)
    # Try reading as float32, int32, int16, complex64
    as_f32 = raw[:16].view(np.float32)
    as_i32 = raw[:16].view(np.int32)
    as_i16 = raw[:16].view(np.int16)
    as_c64 = raw[:16].view(np.complex64)
    base = os.path.basename(rel)
    print("%-50s %6.0f MB  expected=%s" % (base, sz/1e6, expected))
    print("  float32 view: %s" % as_f32)
    print("  int32 view:   %s" % as_i32)
    print("  int16 view:   %s" % as_i16)
    print("  complex64:    %s" % as_c64)
    # Check if file size is consistent with expected dtype
    if "complex" in expected.lower():
        bps = 8
    elif "float32" in expected.lower() or "int32" in expected.lower():
        bps = 4
    else:
        bps = 2
    print("  size=%d  expected bps=%d  -> %d complex samples" % (sz, bps, sz // (bps * 2)))
    print()
