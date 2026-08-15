import h5py, numpy as np, os
SRC = r"E:\Drone_dataset\DroneRFb\twin_droneRF\test"
for name in ["0.mat", "1.mat", "1000.mat"]:
    p = os.path.join(SRC, name)
    with h5py.File(p, "r") as hf:
        keys = list(hf.keys())
        info = {k: (hf[k].shape, str(hf[k].dtype)) for k in keys}
    print(name, info)
