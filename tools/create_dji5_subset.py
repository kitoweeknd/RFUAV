import os, shutil
src = r"E:\Drone_dataset\RFUAV-v2\split_flat"
dst = r"E:\Drone_dataset\RFUAV-v2\split_dji5_flat"
classes = ["uav.dji.avata2", "uav.dji.fpv_combo", "uav.dji.mavic3_pro", "uav.dji.mini3", "uav.dji.mini4_pro"]
if os.path.exists(dst):
    shutil.rmtree(dst)
for s in ["train", "val"]:
    for c in classes:
        sp = os.path.join(src, s, c)
        if not os.path.isdir(sp):
            continue
        tp = os.path.join(dst, s, c)
        os.makedirs(tp, exist_ok=True)
        for f in os.listdir(sp):
            if f.lower().endswith(".jpg"):
                shutil.copy2(os.path.join(sp, f), os.path.join(tp, f))
for s in ["train", "val"]:
    sp = os.path.join(dst, s)
    n = sum(len(os.listdir(os.path.join(sp, c))) for c in os.listdir(sp) if os.path.isdir(os.path.join(sp, c)))
    print(f"{s}: {len(os.listdir(sp))} classes, {n} images")
