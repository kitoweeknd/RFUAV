# -*- coding: utf-8 -*-
"""Decide the .dat dtype by ALIGNMENT, not by amplitude heuristics.

Why this test and not the previous one: my byte-plane test assumed the int16
high byte is a sign extension, which only holds for small-amplitude data.  These
files run at 12-15 bits of a 16-bit word, so the high byte legitimately spans a
wide range and the test fails on correct data.  Likewise an I/Q variance ratio
measures front-end imbalance, not dtype.  Both were bad tests.

Alignment cannot be faked.  An oversampled IQ stream read with the RIGHT dtype
and byte offset is band-limited and therefore strongly correlated sample to
sample; read one byte off, or with the wrong element width, the same bytes
become noise-like.  So the prediction is sharp and falsifiable:

  the TRUE (dtype, offset) maximises lag-1 autocorrelation and spectral
  concentration, and the byte-shifted rival must COLLAPSE.

If int16@0 does not clearly win, int16 is wrong and I must say so.
READ-ONLY, head-only.
"""
import io, os, sys

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


def score(z):
    """lag-1 |autocorr| and spectral concentration of a complex sequence."""
    z = z - z.mean()
    p = float(np.mean(np.abs(z) ** 2))
    if p <= 0:
        return 0.0, 0.0
    ac = abs(float(np.vdot(z[:-1], z[1:]).real) + 1j * 0) / (p * (z.size - 1)) \
        if False else abs(complex(np.vdot(z[:-1], z[1:]))) / (p * (z.size - 1))
    n = 1 << 16
    w = z[:n]
    if w.size < n:
        return ac, 0.0
    S = np.abs(np.fft.fft(w * np.hanning(n))) ** 2
    S = np.sort(S)[::-1]
    conc = float(S[:n // 10].sum() / S.sum())
    return ac, conc


def hyp(raw, dtype, off, itemsize):
    nb = ((len(raw) - off) // (2 * itemsize)) * (2 * itemsize)
    if nb <= 0:
        return None
    v = np.frombuffer(raw[off:off + nb], dtype=dtype)
    if dtype == "<f4":
        v = v.astype(np.float64)
        v = np.where(np.isfinite(v), v, 0.0)
    else:
        v = v.astype(np.float64)
    return v[0::2] + 1j * v[1::2]


HYPS = [("i16@0", "<i2", 0, 2), ("i16@1", "<i2", 1, 2), ("i16@2", "<i2", 2, 2),
        ("i8@0", "i1", 0, 1), ("f32@0", "<f4", 0, 4)]

print("%-46s %s" % ("file", "  ".join("%-13s" % h[0] for h in HYPS)))
print("%-46s %s" % ("", "  ".join("%-13s" % "ac / conc" for h in HYPS)))
wins = {}
for rel in FILES:
    try:
        with io.open(PREFIX + rel, "rb") as fh:
            raw = fh.read(NREAD)
    except (IOError, OSError) as e:
        print("%-46s ERROR %s" % (rel[-46:], e.__class__.__name__)); continue
    cells, best, bestname = [], -1.0, "?"
    for name, dt, off, isz in HYPS:
        z = hyp(raw, dt, off, isz)
        if z is None or z.size < 1 << 16:
            cells.append("%-13s" % "n/a"); continue
        ac, conc = score(z[:1 << 18])
        cells.append("%-13s" % ("%.3f/%.3f" % (ac, conc)))
        if ac > best:
            best, bestname = ac, name
    wins[bestname] = wins.get(bestname, 0) + 1
    print("%-46s %s  -> %s" % (rel[-46:], "  ".join(cells), bestname))

print("\nargmax(lag-1 autocorr) tally: %s" % wins)
