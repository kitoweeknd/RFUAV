import os
d = r"E:\Drone_dataset\RFUAV-v2\canonical"
total_iq = 0; total_adj = 0; total_recs = 0
for cls in os.listdir(d):
    cp = os.path.join(d, cls)
    if not os.path.isdir(cp): continue
    for rec in os.listdir(cp):
        rp = os.path.join(cp, rec)
        if not os.path.isdir(rp): continue
        total_recs += 1
        for f in os.listdir(rp):
            if f.endswith(".iq") and "dB" not in f:
                total_iq += 1
            elif f.endswith(".iq"):
                total_adj += 1
print("recordings: %d  original slices: %d  SNR adjusted: %d" % (total_recs, total_iq, total_adj))
