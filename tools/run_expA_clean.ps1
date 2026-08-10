$py = "C:\ML\anaconda\envs\p0\python.exe"
$train = "D:\ML\RFUAV\train.py"
$models = @("resnet50", "vit_base_patch16_224", "swinv2_tiny_window8_256")
$log = "D:\ML\RFUAV\tools\expA_clean_log.txt"

foreach ($m in $models) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $msg = "$ts  === Starting Exp A clean: $m ==="
    Write-Output $msg
    Add-Content -Path $log -Value $msg
    & $py $train --cfg "D:\ML\RFUAV\configs\p0_dji5_${m}.yaml" 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Output $line
        Add-Content -Path $log -Value $line
    }
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $msg = "$ts  === Done Exp A clean: $m ==="
    Write-Output $msg
    Add-Content -Path $log -Value $msg
}
Write-Output "=== ALL DONE ==="
