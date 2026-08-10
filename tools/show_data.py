import os
d = r"E:\Drone_dataset\RFUAV-v2\split_flat"
for s in ["train", "val"]:
    sp = os.path.join(d, s)
    if not os.path.isdir(sp):
        continue
    total = 0
    for c in sorted(os.listdir(sp)):
        n = len(os.listdir(os.path.join(sp, c)))
        total += n
        print("%-6s  %-35s  %5d images" % (s, c, n))
    print("%-6s  %-35s  %5d images" % (s, "TOTAL", total))
    print()
