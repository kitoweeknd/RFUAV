"""Download DroneRFb zip parts from SciDB direct URLs."""
import os, urllib.request, time, sys

URLS = [
    "https://download.scidb.cn/download?fileId=e48745d29ea7084683a59bd4a1b7687c&path=/V1/twin_droneRFa/twin_droneRF.z01&username=2403329962@qq.com&fileName=twin_droneRF.z01",
    "https://download.scidb.cn/download?fileId=e5f96943f9d822b7c305ee9ee4492305&path=/V1/twin_droneRFa/twin_droneRF.z02&username=2403329962@qq.com&fileName=twin_droneRF.z02",
    "https://download.scidb.cn/download?fileId=9cf05e5079f98b51a11a30945306a73c&path=/V1/twin_droneRFa/twin_droneRF.z03&username=2403329962@qq.com&fileName=twin_droneRF.z03",
    "https://download.scidb.cn/download?fileId=e5dae73bbac4a81ae3ffaf8f19d0deec&path=/V1/twin_droneRFa/twin_droneRF.z04&username=2403329962@qq.com&fileName=twin_droneRF.z04",
    "https://download.scidb.cn/download?fileId=cb32d0abc66ae535d3745a856e3e0eba&path=/V1/twin_droneRFa/twin_droneRF.z05&username=2403329962@qq.com&fileName=twin_droneRF.z05",
    "https://download.scidb.cn/download?fileId=23d91d15bbe85ae033fa5a701892c18a&path=/V1/twin_droneRFa/twin_droneRF.z06&username=2403329962@qq.com&fileName=twin_droneRF.z06",
    "https://download.scidb.cn/download?fileId=e3996c7bf0e69fb06b19d58977f552e7&path=/V1/twin_droneRFa/twin_droneRF.z07&username=2403329962@qq.com&fileName=twin_droneRF.z07",
    "https://download.scidb.cn/download?fileId=5dfb77b20abafc38d83aa8ffd2e98331&path=/V1/twin_droneRFa/twin_droneRF.z08&username=2403329962@qq.com&fileName=twin_droneRF.z08",
    "https://download.scidb.cn/download?fileId=871fb8256b52676a9f636c840a1abd96&path=/V1/twin_droneRFa/twin_droneRF.z09&username=2403329962@qq.com&fileName=twin_droneRF.z09",
    "https://download.scidb.cn/download?fileId=b3cf7a3a2d9cdd60e6dc4e5eb05166d0&path=/V1/twin_droneRFa/twin_droneRF.z10&username=2403329962@qq.com&fileName=twin_droneRF.z10",
    "https://download.scidb.cn/download?fileId=42169dceb9d093138a1cb13beebb0e3b&path=/V1/twin_droneRFa/twin_droneRF.z11&username=2403329962@qq.com&fileName=twin_droneRF.z11",
    "https://download.scidb.cn/download?fileId=513de50a819f09b0ada14e0f650e3be0&path=/V1/twin_droneRFa/twin_droneRF.z12&username=2403329962@qq.com&fileName=twin_droneRF.z12",
    "https://download.scidb.cn/download?fileId=1b12ba0f120f0d29ba89f8b6ff146707&path=/V1/twin_droneRFa/twin_droneRF.z13&username=2403329962@qq.com&fileName=twin_droneRF.z13",
    "https://download.scidb.cn/download?fileId=c63079fb0edef3c9408da4652749770c&path=/V1/twin_droneRFa/twin_droneRF.z14&username=2403329962@qq.com&fileName=twin_droneRF.z14",
    "https://download.scidb.cn/download?fileId=08956a6555c53a08abaca460760dcb0c&path=/V1/twin_droneRFa/twin_droneRF.z15&username=2403329962@qq.com&fileName=twin_droneRF.z15",
    "https://download.scidb.cn/download?fileId=44b309bb80920d872fc633e92e6b541e&path=/V1/twin_droneRFa/twin_droneRF.z16&username=2403329962@qq.com&fileName=twin_droneRF.z16",
    "https://download.scidb.cn/download?fileId=3e8421fbb6d916361a4176c55518efb4&path=/V1/twin_droneRFa/twin_droneRF.z17&username=2403329962@qq.com&fileName=twin_droneRF.z17",
    "https://download.scidb.cn/download?fileId=45bba9665da32edbc943ab71b311b673&path=/V1/twin_droneRFa/twin_droneRF.z18&username=2403329962@qq.com&fileName=twin_droneRF.z18",
    "https://download.scidb.cn/download?fileId=2014b496e41e798d57828d9543019c94&path=/V1/twin_droneRFa/twin_droneRF.z19&username=2403329962@qq.com&fileName=twin_droneRF.z19",
    "https://download.scidb.cn/download?fileId=5ed534d26696619ca7d39811120465bb&path=/V1/twin_droneRFa/twin_droneRF.z20&username=2403329962@qq.com&fileName=twin_droneRF.z20",
    "https://download.scidb.cn/download?fileId=58910bf97d9349126861d7fa9902516b&path=/V1/twin_droneRFa/twin_droneRF.z21&username=2403329962@qq.com&fileName=twin_droneRF.z21",
    "https://download.scidb.cn/download?fileId=4165f0e2816d1229de5bf18ecc22a9ed&path=/V1/twin_droneRFa/twin_droneRF.z22&username=2403329962@qq.com&fileName=twin_droneRF.z22",
    "https://download.scidb.cn/download?fileId=9b89ce2ad77eeb7e3707ec3b5ab2dd46&path=/V1/twin_droneRFa/twin_droneRF.z23&username=2403329962@qq.com&fileName=twin_droneRF.z23",
    "https://download.scidb.cn/download?fileId=0007269439e0ce66f90b6d4f9ccda64f&path=/V1/twin_droneRFa/twin_droneRF.z24&username=2403329962@qq.com&fileName=twin_droneRF.z24",
    "https://download.scidb.cn/download?fileId=cf7ed4a87c7cbf5193f0d86c5316687b&path=/V1/twin_droneRFa/twin_droneRF.z25&username=2403329962@qq.com&fileName=twin_droneRF.z25",
    "https://download.scidb.cn/download?fileId=b1373d3ccd83769e18ec45c7db1841c3&path=/V1/twin_droneRFa/twin_droneRF.z26&username=2403329962@qq.com&fileName=twin_droneRF.z26",
    "https://download.scidb.cn/download?fileId=95e4ac8d4b292af162f17eaa07f83848&path=/V1/twin_droneRFa/twin_droneRF.z27&username=2403329962@qq.com&fileName=twin_droneRF.z27",
    "https://download.scidb.cn/download?fileId=0c5e63254981eac22fe189a0eb7fba7f&path=/V1/twin_droneRFa/twin_droneRF.z28&username=2403329962@qq.com&fileName=twin_droneRF.z28",
    "https://download.scidb.cn/download?fileId=5bfd4ed49d11b73e89dca77b04fdb226&path=/V1/twin_droneRFa/twin_droneRF.z29&username=2403329962@qq.com&fileName=twin_droneRF.z29",
    "https://download.scidb.cn/download?fileId=2f31558b430f7cd2771a797d597d14a8&path=/V1/twin_droneRFa/twin_droneRF.z30&username=2403329962@qq.com&fileName=twin_droneRF.z30",
    "https://download.scidb.cn/download?fileId=851d0d337b5630869c2ee610438f43a0&path=/V1/twin_droneRFa/twin_droneRF.z31&username=2403329962@qq.com&fileName=twin_droneRF.z31",
    "https://download.scidb.cn/download?fileId=1f5eb3f9ed545dd6c57fe6599ed606a8&path=/V1/twin_droneRFa/twin_droneRF.zip&username=2403329962@qq.com&fileName=twin_droneRF.zip",
]

DST = r"E:\Drone_dataset\DroneRFb"
os.makedirs(DST, exist_ok=True)

# Download first file to test speed
test_url = URLS[0]
fname = "twin_droneRF.z01"
local = os.path.join(DST, fname)
print("Testing download: %s" % fname)
t0 = time.time()
try:
    urllib.request.urlretrieve(test_url, local)
    elapsed = time.time() - t0
    sz = os.path.getsize(local) / 1e6
    print("OK: %.0f MB in %.0fs (%.1f MB/s)" % (sz, elapsed, sz / elapsed))
    print("Starting full download of %d files..." % len(URLS))
except Exception as e:
    print("Test failed: %s" % e)
    sys.exit(1)

# Download rest
for idx, url in enumerate(URLS[1:], 1):
    fname = "twin_droneRF.z%02d" % (idx + 1) if idx < 31 else "twin_droneRF.zip"
    local = os.path.join(DST, fname)
    if os.path.exists(local):
        print("[%d/%d] SKIP %s" % (idx + 1, len(URLS), fname))
        continue
    t0 = time.time()
    try:
        urllib.request.urlretrieve(url, local)
        elapsed = time.time() - t0
        sz = os.path.getsize(local) / 1e6
        print("[%d/%d] OK   %s (%.0f MB, %.0fs, %.1f MB/s)" % (idx + 1, len(URLS), fname, sz, elapsed, sz / elapsed))
    except Exception as e:
        print("[%d/%d] ERR  %s (%s)" % (idx + 1, len(URLS), fname, e))

print("Done")
