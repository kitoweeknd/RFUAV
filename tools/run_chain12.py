"""Chain 12: fast-loader path. Thumbnails -> t256 grouped tree -> validation
train (ResNet50, c8a recipe) to certify equivalence."""
import glob, os, subprocess, sys, time
PY = r"C:\ML\anaconda\envs\p0\python.exe"
TOOLS = r"D:\ML\RFUAV\tools"
LOG = os.path.join(TOOLS, "thumbs_chain_log.txt")
def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), m))
def run(args, tag):
    log("--- %s start ---" % tag); t0 = time.time()
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call([PY, "-X", "utf8", "-u"] + args, stdout=f, stderr=f)
    log("--- %s exit=%d (%.2f h) ---" % (tag, rc, (time.time()-t0)/3600))
    if rc != 0:
        log("=== CHAIN 12 ABORTED at %s ===" % tag); sys.exit(rc)
log("=== THUMBS CHAIN START ===")
run([os.path.join(TOOLS, "build_thumbs256.py")], "build_thumbs256")
run([os.path.join(TOOLS, "build_thumb_tree.py")], "build_thumb_tree")
run([os.path.join(TOOLS, "make_vt_cfg.py")], "make_vt_cfg")
run([os.path.join(TOOLS, "train_runner.py"), r"D:\ML\RFUAV\configs\vt_resnet50_t256.yaml",
     os.path.join(TOOLS, "train_vt_resnet50.txt")], "train_vt_resnet50")
for f in glob.glob(r"E:\Drone_dataset\RFUAV-v2\benchmark\runs\vt_resnet50_t256\*_epoch_*.pth"):
    os.remove(f)
log("=== THUMBS CHAIN COMPLETE ===")
