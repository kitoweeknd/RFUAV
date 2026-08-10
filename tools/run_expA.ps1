$py = "C:\ML\anaconda\envs\p0\python.exe"
$train = "D:\ML\RFUAV\train.py"
$models = @(
    @{name="resnet50"; cfg="p0_dji5_resnet50.yaml"},
    @{name="vit_base_patch16_224"; cfg="p0_dji5_vit.yaml"},
    @{name="swinv2_tiny_window8_256"; cfg="p0_dji5_swin.yaml"}
)
$log = "D:\ML\RFUAV\tools\expA_clean_log.txt"

foreach ($m in $models) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $msg = "$ts  === Exp A clean: $($m.name) ==="
    Write-Output $msg
    Add-Content -Path $log -Value $msg
    & $py $train --cfg "D:\ML\RFUAV\configs\$($m.cfg)" 2>&1 | ForEach-Object {
        $line = "$_"
        Write-Output $line
        Add-Content -Path $log -Value $line
    }
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $msg = "$ts  === Done: $($m.name) ==="
    Write-Output $msg
    Add-Content -Path $log -Value $msg
}
Write-Output "=== ALL DONE ==="
