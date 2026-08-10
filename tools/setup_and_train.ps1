Write-Output "=== Uninstalling old torch ==="
pip uninstall torch torchvision -y 2>&1 | Select-Object -Last 3

Write-Output "=== Installing torch 2.5.1+cu124 ==="
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | Select-Object -Last 5

Write-Output "=== Installing other deps ==="
pip install timm tqdm pyyaml opencv-python matplotlib pandas numpy==1.26.4 2>&1 | Select-Object -Last 3

Write-Output "=== Verifying env ==="
python -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); import timm; print('timm', timm.__version__); import cv2; print('cv2 ok')"

Write-Output "=== Running training ==="
Set-Location D:\ML\RFUAV
python train.py --cfg configs\p0_dji5_resnet18.yaml 2>&1
