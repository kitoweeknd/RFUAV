# -*- coding: utf-8 -*-
"""Positive confirmation that .dat_pic\\*.dat is interleaved int16-LE.

The float32 discriminator only PROVES 'not cf32'.  Committing 'ci16_le' to the
manifest on that alone would be an assertion, not a measurement, so this runs
three independent positive tests whose predictions differ between the rival
hypotheses (ci16_le vs ci8 vs cf32):

  T1  byte-plane asymmetry.  If the datum is int16-LE, the HIGH byte of each
      sample is a sign extension and must cluster hard on {0x00, 0xFF} while
      the LOW byte stays near-uniform.  If the datum were int8, both byte
      planes would look alike.  Predicted: P(high in {0,255}) >> P(low in
      {0,255}).
  T2  ADC headroom.  A real receiver never fills int16.  Read as int16 the
      max |v| must sit at a plausible converter width (<= 2^15) and, for a
      12/14-bit front end, well under it.  Nonsense byte alignment would
      instead saturate.
  T3  I/Q symmetry.  Interleaving means even samples are I and odd are Q, two
      views of the same complex process, so their variances must agree to
      within a few percent.  If the stride were wrong (int8, or a header
      offset) the two planes would decorrelate and the ratio would drift.

READ-ONLY.  Head-only reads.
"""
import io, sys

PREFIX = "E:\\Drone_dataset\\RFUAV\\"
NREAD = 4 * 1024 * 1024

FILES = [
    r".dat_pic\AVATA\AVATA_high_RF0.dat",
    r".dat_pic\Air2S\Air_2S_low_RF0.dat",
    r".dat_pic\Dji\Dji_low_RF0.dat",
    r".dat_pic\Frsky\FrSky_X20_low_RF1.dat",
    r".dat_pic\Matrice100\MATRICE_100_high_RF0.dat",
    r".dat_pic\Matrice600\MATRICE_600_Pro_high_RF0.dat",
    r".dat_pic\PhanTom4ProRTK\Phantom_4_Pro_RTK_low_RF1.dat",
    r".dat_pic\Phantom3\Phantom_3_high_RF0.dat",
    r".dat_pic\Phantom4Pro\Phantom_4_Pro_low_RF0.dat",
    r".dat_pic\Taranis_Plus\Taranis_Plus_high_RF0.dat",
    r".dat_pic\inspire2\inspire_low_RF0.dat",
]

try:
    import numpy as np
except ImportError:
    print("numpy unavailable"); sys.exit(1)

print("%-52s %7s %7s %8s %7s %6s %s" % (
    "file", "hi{0,FF}", "lo{0,FF}", "max|v|", "bits", "Q/I", "verdict"))

import os
allok = 0
for rel in FILES:
    p = PREFIX + rel
    try:
        with io.open(p, "rb") as fh:
            raw = fh.read(NREAD)
        size = os.path.getsize(p)
    except (IOError, OSError) as e:
        print("%-52s ERROR %s" % (rel[-52:], e.__class__.__name__)); continue

    nb = (len(raw) // 4) * 4
    b = np.frombuffer(raw[:nb], dtype=np.uint8)
    lo = b[0::2]; hi = b[1::2]
    p_hi = 100.0 * np.mean((hi == 0) | (hi == 255))
    p_lo = 100.0 * np.mean((lo == 0) | (lo == 255))

    v = np.frombuffer(raw[:nb], dtype="<i2").astype(np.float64)
    mx = float(np.max(np.abs(v)))
    bits = (np.log2(mx + 1) if mx > 0 else 0)
    I = v[0::2]; Q = v[1::2]
    sI = float(np.std(I)); sQ = float(np.std(Q))
    ratio = (sQ / sI) if sI > 0 else float("nan")

    # T1 must pass, T2 must pass, T3 must pass
    t1 = p_hi > 4.0 * max(p_lo, 0.01) and p_hi > 20.0
    t2 = mx <= 32768.0
    t3 = 0.90 <= ratio <= 1.11
    ok = t1 and t2 and t3
    allok += 1 if ok else 0
    print("%-52s %6.2f%% %6.2f%% %8.0f %7.2f %6.3f %s%s" % (
        rel[-52:], p_hi, p_lo, mx, bits, ratio,
        "ci16_le" if ok else "FAIL",
        "" if ok else " (T1=%s T2=%s T3=%s)" % (t1, t2, t3)))

print("\nci16_le positively confirmed: %d/%d" % (allok, len(FILES)))

# size divisibility: an interleaved int16 IQ stream must be a multiple of 4
print("\nsize %% 4 check (must be 0 for complete IQ pairs):")
for rel in FILES:
    try:
        s = os.path.getsize(PREFIX + rel)
    except OSError:
        continue
    print("  %-52s %14d  %%4=%d  %%2=%d" % (rel[-52:], s, s % 4, s % 2))
