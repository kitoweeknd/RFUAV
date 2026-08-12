"""Cross-validate low-SNR classes with 3 methods: FCME, PSD percentile, time-domain RMS."""
import sys, os, csv, io, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")
sys.path.insert(0, r"D:\ML\RFUAV\ChannelSimulation\viewer\signal_proc")
from snr_dual import FCMEEstimator
from snr_estimator import SNREstimator
from scipy.signal import get_window, welch

m = r"E:\Drone_dataset\RFUAV-v2\manifests\assignment.csv"
rd = r"E:\Drone_dataset\RFUAV"
with io.open(m, encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["source_root"] == "rawdata" and r["split"] in ("train", "test")]

low_snr_classes = [
    "rc.flysky.el18", "rc.flysky.nv14", "rc.futaba.t14sg", "rc.herelink.hx4",
    "rc.radiolink.at10_ii", "rc.siyi.mk15", "uav.dji.mavic3_pro", "uav.parrot.parrot_unspecified",
]

print("%-35s %8s %8s %8s %8s %8s %8s" % (
    "CLASS", "FCME", "OLD_EST", "P15_FLR", "P99_FLR", "RMS_dB", "VERDICT"))
print("-" * 100)

for cls in low_snr_classes:
    # Pick one recording for this class
    cls_recs = [r for r in rows if r["taxonomy_id"] == cls]
    if not cls_recs:
        continue
    r = cls_recs[0]
    rel = r["source_relpath"]
    fpath = os.path.normpath(os.path.join(rd, rel))
    if not os.path.exists(fpath):
        continue

    max_samp = 5_000_000
    raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]; iq.imag = raw[1::2]; del raw

    # Method 1: FCME
    fcme = FCMEEstimator(fs=100e6, nfft=16384)
    snr_fcme, diag = fcme.estimate(iq)

    # Method 2: Old estimator
    try:
        est = SNREstimator(fs=100e6, bw=20e6, nfft=16384)
        result = est.estimate(iq)
        snr_old = result.snr_db
    except:
        snr_old = -999

    # Method 3: PSD percentile-based noise floor
    nfft = 16384
    nperseg = max(8, min(nfft, int(round(n / 10))))
    window = get_window("hamming", nperseg, fftbins=True)
    f, psd = welch(iq, fs=100e6, window=window, nperseg=nperseg, nfft=nfft,
                   return_onesided=False, scaling="density", detrend=False)
    psd = np.fft.fftshift(psd)
    psd_linear = np.maximum(psd, 1e-20)
    psd_db = 10 * np.log10(psd_linear)
    
    # P15 = 15th percentile as noise floor estimate
    p15 = float(np.percentile(psd_db, 15))
    p99 = float(np.percentile(psd_db, 99))
    # Signal = mean of top 1% bins, noise = mean of bottom 15% bins
    threshold = np.percentile(psd_db, 85)
    sig_mask = psd_db > threshold
    noise_mask = psd_db < np.percentile(psd_db, 50)
    if sig_mask.sum() > 0 and noise_mask.sum() > 0:
        p_sig = float(np.mean(psd_linear[sig_mask]))
        p_noise = float(np.mean(psd_linear[noise_mask]))
        snr_percentile = float(10 * np.log10(max(p_sig - p_noise, 1e-20) / max(p_noise, 1e-20)))
    else:
        snr_percentile = -999

    # Method 4: Time-domain RMS ratio (signal vs noise periods)
    # Simple: split signal into chunks, find chunks with high/low power
    chunk_size = 10000
    n_chunks = n // chunk_size
    chunk_power = np.array([float(np.mean(np.abs(iq[i*chunk_size:(i+1)*chunk_size])**2))
                            for i in range(min(n_chunks, 1000))])
    if len(chunk_power) > 10:
        p_low = float(np.percentile(chunk_power, 10))
        p_high = float(np.percentile(chunk_power, 90))
        snr_rms = float(10 * np.log10(max(p_high - p_low, 1e-20) / max(p_low, 1e-20)))
    else:
        snr_rms = -999

    # Verdict
    snrs = [s for s in [snr_fcme, snr_percentile, snr_rms] if s > -999 and s < 100]
    mean_snr = np.mean(snrs) if snrs else -999
    verdict = "CONFIRMED < 20dB" if mean_snr < 20 else "BORDERLINE"

    print("%-35s %8.1f %8.1f %8.1f %8.1f %8.1f %s" % (
        cls[:35], snr_fcme, snr_old, snr_percentile, p15, snr_rms, verdict))

print("-" * 100)
print("FCME = adaptive FCME | OLD_EST = old PSD threshold | P15_FLR = 15%% percentile noise floor")
print("RMS_dB = time-domain RMS ratio | VERDICT = confirmed < 20dB if all methods agree")
