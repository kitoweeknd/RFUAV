$ErrorActionPreference = "Continue"
$py = "C:\ML\anaconda\envs\p0\python.exe"
$pip = "C:\ML\anaconda\envs\p0\Scripts\pip.exe"
$log = "D:\ML\RFUAV\tools\train_log.txt"

Function Log($msg) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "$ts  $msg"
    Write-Output $line
    Add-Content -Path $log -Value $line
}

Log "=== STEP 1: Install torch ==="
& $pip install torch torchvision --index-url https://pypi.org/simple/ 2>&1 | Select-Object -Last 5 | ForEach-Object { Log $_ }

Log "=== STEP 2: Install other deps ==="
& $pip install timm tqdm pyyaml opencv-python matplotlib pandas 2>&1 | Select-Object -Last 5 | ForEach-Object { Log $_ }

Log "=== STEP 3: Verify env ==="
& $py -c "import torch; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available()); import timm; print('timm', timm.__version__); import cv2; print('cv2 ok')" 2>&1 | ForEach-Object { Log $_ }

Log "=== STEP 4: Train ==="
Set-Location D:\ML\RFUAV
& $py train.py --cfg configs\p0_dji5_resnet18.yaml 2>&1 | ForEach-Object { Log $_ }

Log "=== DONE ==="
