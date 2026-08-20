"""Box-side waiter: when chain 7 logs COMPLETE, run chain 7b. Times out in 3h."""
import subprocess, sys, time

LOG = r"D:\ML\RFUAV\tools\dronerf_chain_log.txt"
PY = r"C:\ML\anaconda\envs\p0\python.exe"
for _ in range(180):
    try:
        txt = open(LOG, encoding="utf-8", errors="replace").read()
    except OSError:
        txt = ""
    if "=== CHAIN ABORTED" in txt:
        sys.exit(1)
    if "=== DRONERF CHAIN COMPLETE ===" in txt:
        sys.exit(subprocess.call([PY, "-X", "utf8", r"D:\ML\RFUAV\tools\run_chain7b.py"]))
    time.sleep(60)
sys.exit(2)
