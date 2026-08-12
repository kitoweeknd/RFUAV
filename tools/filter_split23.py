import os, shutil
src = r"E:\Drone_dataset\RFUAV-v2\split_flat"
dst = r"E:\Drone_dataset\RFUAV-v2\split_23"

valid = [
    "rc.devention.devo_10", "rc.flysky.el18", "rc.flysky.fs_i6x", "rc.flysky.nv14",
    "rc.frsky.x9dp2019", "rc.futaba.t10j", "rc.futaba.t16iz", "rc.futaba.t18sz",
    "rc.jrpropo.xg14", "rc.radiolink.at10_ii", "rc.siyi.ft24", "rc.siyi.mk15",
    "rc.siyi.mk32", "rc.skydroid.t10", "rc.wfly.et10", "rc.yunzhuo.h16",
    "uav.autel.evo_nano", "uav.dji.avata2", "uav.dji.fpv_combo",
    "uav.dji.mavic3_pro", "uav.dji.mini3", "uav.dji.mini4_pro",
    "uav.parrot.parrot_unspecified",
]

if os.path.exists(dst):
    shutil.rmtree(dst)

for s in ["train", "val"]:
    sp = os.path.join(src, s)
    for c in os.listdir(sp):
        if c not in valid:
            continue
        cp = os.path.join(sp, c)
        if not os.path.isdir(cp):
            continue
        tp = os.path.join(dst, s, c)
        os.makedirs(tp, exist_ok=True)
        for f in os.listdir(cp):
            shutil.copy2(os.path.join(cp, f), os.path.join(tp, f))

for s in ["train", "val"]:
    sp = os.path.join(dst, s)
    n = sum(len(os.listdir(os.path.join(sp, c))) for c in os.listdir(sp))
    print("%s: %d classes, %d images" % (s, len(os.listdir(sp)), n))
