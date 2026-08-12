"""Detached training: writes to log, survives SSH disconnect."""
import os, subprocess, sys, time

py = r"C:\ML\anaconda\envs\p0\python.exe"
log = r"D:\ML\RFUAV\tools\expB_scaling_log.txt"

models = [
    ("resnet101", 640), ("resnet152", 640),
    ("vit_l_16", 224), ("vit_l_32", 224),
    ("swin_t", 256), ("swin_s", 256), ("swin_b", 256),
    ("eff_b0", 224), ("eff_b1", 224), ("eff_b2", 224), ("eff_b3", 224),
]

with open(log, "a") as lf:
    lf.write("=== Exp B scaling started at %s ===\n" % time.ctime())
    for name, sz in models:
        ckpt = r"E:\Drone_dataset\RFUAV-v2\checkpoints\expB_%s\best_model.pth" % name
        if os.path.exists(ckpt):
            lf.write("SKIP %s (exists)\n" % name)
            continue
        cfg = r"D:\ML\RFUAV\configs\expB_%s.yaml" % name
        lf.write(">>> %s <<<\n" % name)
        lf.flush()
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        ret = subprocess.run([py, r"D:\ML\RFUAV\train.py", "--cfg", cfg],
                           check=False, env=env,
                           stdout=lf, stderr=subprocess.STDOUT)
        lf.write("EXIT %d: %s\n" % (ret.returncode, name))
        lf.flush()
    lf.write("=== DONE at %s ===\n" % time.ctime())
