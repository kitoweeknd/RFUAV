"""Smoke test: augment+render exactly one train slice, print timing."""
import os, sys, time
import numpy as np

sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\tools")
os.environ["SMOKE_AUG"] = "1"

t0 = time.time()
from graphic.RawDataProcessor import generate_images
from ChannelSimulation.channels.fading import apply_rayleigh_or_rician_fading
from ChannelSimulation.channels.doppler import apply_doppler
from ChannelSimulation.channels.multipath import apply_multipath
from ChannelSimulation.channels.cfo import apply_cfo
from ChannelSimulation.channels.phase_noise import apply_phase_noise
from ChannelSimulation.channels.iq_imbalance import apply_iq_imbalance
from ChannelSimulation.channels.awgn import apply_awgn
print("imports ok (%.1fs)" % (time.time() - t0))

SLICE = r"E:\Drone_dataset\RFUAV-v2\canonical\uav.dji.mini4_pro"
# pick first train-ish recording dir + slice
rec = sorted(os.listdir(SLICE))[0]
sp = os.path.join(SLICE, rec, "slice_0005.iq")
print("slice:", sp)

t0 = time.time()
raw = np.fromfile(sp, dtype=np.float32)
n = raw.size // 2
iq = np.empty(n, dtype=np.complex64)
iq.real = raw[0::2]; iq.imag = raw[1::2]
del raw
print("load %.1fs n=%d dtype=%s" % (time.time() - t0, n, iq.dtype))

rng = np.random.default_rng(42)
fs = 100e6
t0 = time.time()
y = apply_rayleigh_or_rician_fading(iq, fading_type="rician", k_factor=5.0,
                                    block_size=256, rng=rng)
print("fading %.1fs dtype=%s" % (time.time() - t0, y.dtype))
t0 = time.time()
y = apply_doppler(y, fs, doppler_shift_hz=120.0, doppler_spread_hz=60.0, rng=rng)
print("doppler %.1fs dtype=%s" % (time.time() - t0, y.dtype))
t0 = time.time()
y = apply_multipath(y, max_delay_samp=20, n_paths=4, exp_decay=0.35, rng=rng)
print("multipath %.1fs dtype=%s" % (time.time() - t0, y.dtype))
t0 = time.time()
y = apply_cfo(y, fs, cfo_hz=8000.0)
print("cfo %.1fs dtype=%s" % (time.time() - t0, y.dtype))
t0 = time.time()
y = apply_phase_noise(y, phase_std_per_sample=2e-3, rng=rng)
print("phase_noise %.1fs dtype=%s" % (time.time() - t0, y.dtype))
t0 = time.time()
y = apply_iq_imbalance(y, gain_imbalance_db=1.0, phase_imbalance_deg=2.0)
print("iq_imb %.1fs dtype=%s" % (time.time() - t0, y.dtype))
t0 = time.time()
y = apply_awgn(y, snr_db=15.0, rng=rng)
print("awgn %.1fs dtype=%s" % (time.time() - t0, y.dtype))
assert y.dtype == np.complex64, "COMPLEX64 INVARIANT BROKEN: %s" % y.dtype
assert np.isfinite(y[:1000]).all()

t0 = time.time()
tmp = r"E:\Drone_dataset\RFUAV-v2\augmented\_smoke_tmp.iq"
os.makedirs(os.path.dirname(tmp), exist_ok=True)
buf = np.empty(2 * len(y), dtype=np.float32)
buf[0::2] = y.real; buf[1::2] = y.imag
buf.tofile(tmp)
imgs = generate_images(datapack=tmp, fs=int(fs), stft_point=1024,
                       duration_time=min(0.1, n / fs), location='buffer',
                       file_type=np.float32)
os.remove(tmp)
out = r"E:\Drone_dataset\RFUAV-v2\augmented\_smoke_aug.jpg"
imgs[0].convert("RGB").save(out, "JPEG", quality=95)
print("render %.1fs -> %s" % (time.time() - t0, out))
print("SMOKE OK")
