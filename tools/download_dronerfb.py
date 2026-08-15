"""Download all DroneRFb files from SciDB FTP."""
import ftplib, os, sys, time

HOST = "ftp.scidb.cn"
PORT = 2121
USER = "qiq2Y3"
PWD = "ayMRFb"
DST = r"E:\Drone_dataset\DroneRFb"

ftp = ftplib.FTP()
ftp.connect(HOST, PORT, timeout=30)
ftp.login(USER, PWD)
print("Connected. Listing tree...")

# Recursive listing
def list_tree(path=""):
    items = []
    ftp.cwd(path)
    ftp.dir(items.append)
    files = []
    dirs = []
    for item in items:
        parts = item.split()
        if len(parts) < 9:
            continue
        name = " ".join(parts[8:])
        if item.startswith("d"):
            dirs.append((path + "/" + name).lstrip("/"))
        elif item.startswith("-"):
            sz = int(parts[4])
            files.append((path + "/" + name, sz))
    for d in sorted(dirs):
        files.extend(list_tree(d))
    return files

all_files = list_tree("twin_droneRFa")
total_sz = sum(f[1] for f in all_files)
print("Files: %d  Total size: %.1f GB" % (len(all_files), total_sz / 1e9))
for f, sz in sorted(all_files)[:20]:
    print("  %-80s %8.1f MB" % (f, sz / 1e6))
if len(all_files) > 20:
    print("  ... and %d more" % (len(all_files) - 20))

# Download
print("\n=== Downloading ===")
ok = err = 0
for idx, (fpath, sz) in enumerate(all_files):
    local = os.path.join(DST, fpath.replace("/", "\\"))
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if os.path.exists(local) and os.path.getsize(local) == sz:
        print("[%d/%d] SKIP %s" % (idx+1, len(all_files), fpath))
        ok += 1
        continue
    t0 = time.time()
    try:
        with open(local, "wb") as f:
            ftp.retrbinary("RETR " + fpath, f.write, blocksize=1024*1024)
        elapsed = time.time() - t0
        print("[%d/%d] OK   %s (%.1f MB, %.0fs)" % (idx+1, len(all_files), fpath, sz/1e6, elapsed))
        ok += 1
    except Exception as e:
        print("[%d/%d] ERR  %s (%s)" % (idx+1, len(all_files), fpath, e))
        err += 1

ftp.quit()
print("Done: %d ok, %d err" % (ok, err))
