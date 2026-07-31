# -*- coding: utf-8 -*-
"""Status probe: live python processes (with full CommandLine so nothing is
ever killed by pattern), E1 hash progress, and scan log tail."""
import io
import os
import subprocess

PS = (
    "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
    "ForEach-Object { '=== PID {0}  cpu={1}s  mem={2}MB  start={3}' -f "
    "$_.ProcessId, [math]::Round($_.UserModeTime/1e7), "
    "[math]::Round($_.WorkingSetSize/1MB), $_.CreationDate; "
    "'    ' + $_.CommandLine }"
)
r = subprocess.run(["powershell", "-NoProfile", "-Command", PS],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print("--- PYTHON PROCESSES ---")
print(r.stdout.decode("utf-8", "replace").strip())

M = r"E:\Drone_dataset\RFUAV-v2\manifests"
for name in ("e1_hashes.csv", "e1_inventory.csv"):
    p = os.path.join(M, name)
    if os.path.exists(p):
        n = 0
        with io.open(p, "r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                n += 1
        print("\n%s : %d rows  (%.1f MB, mtime %s)" % (
            name, n, os.path.getsize(p) / 1e6,
            __import__("datetime").datetime.fromtimestamp(
                os.path.getmtime(p)).isoformat(timespec="seconds")))

log = os.path.join(M, "e1_scan.log")
if os.path.exists(log):
    with io.open(log, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    print("\n--- e1_scan.log tail ---")
    for ln in lines[-8:]:
        print("  " + ln)
