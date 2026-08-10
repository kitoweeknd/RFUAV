import subprocess, sys
pip = r"D:\ML\RFUAV\.venv_p0\Scripts\pip.exe"
subprocess.run([pip, "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu124"], check=True)
subprocess.run([pip, "install", "timm", "tqdm", "pyyaml", "opencv-python", "matplotlib", "pandas", "numpy==1.26.4"], check=True)
print("DONE")
