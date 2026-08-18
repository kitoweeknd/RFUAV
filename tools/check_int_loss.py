"""Measure int16/int32 -> float32 conversion loss on the actual sources."""
import numpy as np

I16 = r"E:\Drone_dataset\RFA\DroneRFa-IQ\int16\Matrice300_T1011_S1011_RF0.dat"
I32 = r"E:\Drone_dataset\RFA\DroneRFa-IQ\int32\Mavic3_RF0.dat"
N = 4_000_000

a = np.fromfile(I16, dtype=np.int16, count=N)
rt = np.float32(a).astype(np.int16)
print("int16: max|v|=%d  roundtrip mismatches=%d / %d"
      % (abs(a).max(), int((rt != a).sum()), len(a)))

b = np.fromfile(I32, dtype=np.int32, count=N)
big = int((np.abs(b) > 2**24).sum())
f = np.float32(b)
err = np.abs(f.astype(np.float64) - b.astype(np.float64))
nz = b[b != 0]
tz = 32
for v in nz[:100000]:
    t = (int(v) & -int(v)).bit_length() - 1
    if t < tz:
        tz = t
    if tz == 0:
        break
eff_bits = int(np.ceil(np.log2(abs(b).max() + 1))) + 1 - tz
print("int32: max|v|=%d  |v|>2^24: %d/%d (%.4f%%)" % (abs(b).max(), big, len(b), 100*big/len(b)))
print("int32: max abs f32-rounding error=%.1f LSB, max relative=%.2e (%.1f dB)"
      % (err.max(), (err[np.abs(b) > 0] / np.abs(b[np.abs(b) > 0])).max() if big else 0,
         20*np.log10((err[np.abs(b)>0]/np.abs(b[np.abs(b)>0])).max()) if err.max() > 0 else float('-inf')))
print("int32: trailing zero bits(min)=%d -> effective ADC bits <= %d" % (tz, eff_bits))
