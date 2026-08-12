"""Check the 5 FCME-failed classes: are they genuinely weak or is FCME wrong?"""
import sys, os, numpy as np
sys.path.insert(0, r"D:\ML\RFUAV")

rd = r"E:\Drone_dataset\RFUAV\rawdata"
failed = ["DJFPVCOMBO", "DJMINI3", "DJMINI4PRO", "FutabaT14SG", "JR PROPO_XG14"]
for cls in failed:
    cp = os.path.join(rd, cls)
    if not os.path.isdir(cp): continue
    iqs = [f for f in os.listdir(cp) if f.endswith('.iq')]
    if not iqs: continue
    fpath = os.path.join(cp, iqs[0])
    sz_mb = os.path.getsize(fpath) / 1e6
    
    max_samp = 10_000_000
    raw = np.fromfile(fpath, dtype=np.float32, count=max_samp * 2)
    n = raw.size // 2
    iq = np.empty(n, dtype=np.complex64)
    iq.real = raw[0::2]
    iq.imag = raw[1::2]
    del raw
    
    pwr = float(np.mean(np.abs(iq) ** 2))
    peak = float(np.max(np.abs(iq)))
    zeros = int(np.sum(np.abs(iq) < 1e-10))
    pct_zero = 100 * zeros / len(iq)
    
    # Check if there's any frequency structure
    from scipy.signal import welch, get_window
    f, psd = welch(iq, fs=100e6, window=get_window("hamming", 4096, fftbins=True),
                   nperseg=4096, nfft=16384, return_onesided=False, scaling="density")
    f = np.fft.fftshift(f)
    psd = np.fft.fftshift(psd)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
    psd_range = float(psd_db.max() - psd_db.min())
    
    print("%-30s %6.0f MB  pwr=%.6f  peak=%.4f  zeros=%.1f%%  PSD_range=%.1f dB" % (
        cls[:30], sz_mb, pwr, peak, pct_zero, psd_range))
