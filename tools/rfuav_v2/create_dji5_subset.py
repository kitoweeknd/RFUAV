import os, shutil
src = r"E:\Drone_dataset\RFUAV-v2\split_flat"
dst = r"E:\Drone_dataset\RFUAV-v2\split_dji5"
classes = ["uav.dji.avata2", "uav.dji.fpv_combo", "uav.dji.mavic3_pro", "uav.dji.mini3", "uav.dji.mini4_pro"]
for s in ["train", "val"]:
    for c in classes:
        sp = os.path.join(src, s, c)
        if os.path.isdir(sp):
            tp = os.path.join(dst, s, c)
            os.makedirs(tp, exist_ok=True)
            for f in os.listdir(sp):
                if f.lower().endswith(".jpg"):
                    shutil.copy2(os.path.join(sp, f), os.path.join(tp, f))
total = sum(1 for _ in os.walk(dst) for f in _[2] if f.lower().endswith(".jpg"))
print("DJI5 subset: %d files" % total)
