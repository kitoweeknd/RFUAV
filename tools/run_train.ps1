Set-Location D:\ML\RFUAV
$env:PATH = "D:\ML\RFUAV\.venv_p0\Scripts;" + $env:PATH
python tools\check_venv.py
python train.py --cfg configs\p0_dji5_resnet18.yaml 2>&1
