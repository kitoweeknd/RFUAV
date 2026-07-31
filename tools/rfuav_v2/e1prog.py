# -*- coding: utf-8 -*-
"""Honest E1 progress: numerator and denominator drawn from the SAME set.

The previous version divided ALL hashed bytes (which include ~50 large non-signal
archives) by the signal-only total, and so reported >100 %.  Progress is now
measured strictly over the signal set that build_manifests.py actually consumes.
"""
import csv, io, os

M = r"E:\Drone_dataset\RFUAV-v2\manifests"
SIG = (".iq", ".dat", ".bin", ".mat")

hashed = set()
with io.open(M + r"\e1_hashes.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.reader(f):                     # HEADERLESS
        if row:
            hashed.add(row[0])

sig, done_b, todo_b, done_n = {}, 0, 0, 0
with io.open(M + r"\e1_inventory.csv", "r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):                 # HAS header
        p = row["path"]
        if not p.lower().endswith(SIG):
            continue
        s = int(row["size"])
        sig[p] = s
        if p in hashed:
            done_b += s; done_n += 1
        else:
            todo_b += s

tot_b = done_b + todo_b
tot_n = len(sig)
G = 2.0 ** 30
print("signal set     : %d files   %.1f GB" % (tot_n, tot_b / G))
print("hashed         : %d files   %.1f GB   %.1f%% by bytes  %.1f%% by count"
      % (done_n, done_b / G, 100.0 * done_b / tot_b, 100.0 * done_n / tot_n))
print("remaining      : %d files   %.1f GB" % (tot_n - done_n, todo_b / G))
print("non-signal hashed (archives etc): %d files"
      % len([p for p in hashed if not p.lower().endswith(SIG)]))
print("mtime          : %s"
      % __import__("time").ctime(os.path.getmtime(M + r"\e1_hashes.csv")))
