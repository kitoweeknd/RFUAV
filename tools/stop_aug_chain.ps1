$procs = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -like 'python*' -and (
        $_.CommandLine -match 'run_chain6\.py' -or
        $_.CommandLine -match 'augment_render_train\.py' -or
        $_.CommandLine -match 'build_train_aug_tree\.py' -or
        ($_.CommandLine -match 'train_runner\.py' -and $_.CommandLine -match 'exp_c5')
    )
}
foreach ($p in $procs) {
    Write-Output ("KILL PID=" + $p.ProcessId + " CMD=" + $p.CommandLine.Substring(0, [Math]::Min(100, $p.CommandLine.Length)))
    Stop-Process -Id $p.ProcessId -Force
}
if (-not $procs) { Write-Output "no matching processes" }
schtasks /End /TN RFUAV_AUG_C5 2>$null
Write-Output "stopped."
