"""Run ResNet18 training on the P0 clean split using the venv_p0 environment."""
import subprocess, sys, os
os.chdir(r"D:\ML\RFUAV")
venv_python = r".venv_p0\Scripts\python.exe"

# First verify environment
print("=== Checking environment ===")
subprocess.run([venv_python, "-c", 
    "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); "
    "import timm; print('timm', timm.__version__); "
    "import cv2; print('cv2 ok')"], check=True)

# Check data
print("=== Checking data ===")
for split in ["train", "val"]:
    path = os.path.join(r"E:\Drone_dataset\RFUAV-v2\split_flat", split)
    classes = sorted(os.listdir(path))
    n = sum(len(os.listdir(os.path.join(path, c))) for c in classes)
    print(f"  {split}: {len(classes)} classes, {n} images")
    for c in classes[:5]:
        print(f"    {c}: {len(os.listdir(os.path.join(path, c)))} images")

# Run training
print("=== Starting training ===")
subprocess.run([venv_python, "train.py", "--cfg", r"configs\p0_dji5_resnet18.yaml"], check=True)
print("=== DONE ===")
