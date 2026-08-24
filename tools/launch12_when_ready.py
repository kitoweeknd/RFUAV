"""Box-side waiter: when the naive chain logs COMPLETE, run chain 12."""
import subprocess, sys, time
LOG = r"D:\ML\RFUAV\tools\naive_chain_log.txt"
PY = r"C:\ML\anaconda\envs\p0\python.exe"
for _ in range(360):
    try:
        txt = open(LOG, encoding="utf-8", errors="replace").read()
    except OSError:
        txt = ""
    if "CHAIN COMPLETE" in txt:
        sys.exit(subprocess.call([PY, "-X", "utf8", r"D:\ML\RFUAV\tools\run_chain12.py"]))
    time.sleep(60)
sys.exit(2)
