param([switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$demoRoot = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$runRoot = Join-Path $demoRoot 'results\demo_accuracy_20261021_040741'
$manifest = Get-Content -LiteralPath (Join-Path $runRoot 'rollback_manifest.json') -Raw | ConvertFrom-Json
if ($manifest.project_root -ne $demoRoot) { throw 'Rollback project root mismatch.' }
function Scoped-Path([string]$base, [string]$relative) {
    $resolved = [IO.Path]::GetFullPath((Join-Path $base $relative))
    if (-not $resolved.StartsWith($base + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { throw "Unsafe rollback path: $resolved" }
    return $resolved
}
$backupRoot = Join-Path $runRoot 'baseline\files'
foreach ($relative in $manifest.restore_files) {
    $saved = Scoped-Path $backupRoot $relative
    $expected = $manifest.files | Where-Object { $_.path -eq $relative }
    if (-not $expected -or (Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected.sha256) { throw "Invalid backup: $relative" }
}
if ($VerifyOnly) { Write-Output 'Rollback backup verified; no files or processes changed.'; exit 0 }
# PIDs are written by the experiment itself. Check its exact run-directory
# argument and module name before stopping anything, including after PID reuse.
foreach ($worker in (Get-Content -LiteralPath (Join-Path $runRoot 'processes.json') -Raw | ConvertFrom-Json)) {
    $registeredPid = [int]$worker.pid
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $registeredPid" -ErrorAction SilentlyContinue
    $experimentCommand = $process -and ($process.CommandLine -like '*training.demo_accuracy*' -or $process.CommandLine -like '*tools.verify_demo_accuracy*')
    if ($experimentCommand -and $process.CommandLine.Contains($runRoot)) {
        foreach ($child in @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $registeredPid" -ErrorAction SilentlyContinue)) {
            if ($child.CommandLine -like '*multiprocessing.spawn*') { Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue }
        }
        Stop-Process -Id $registeredPid -Force -ErrorAction SilentlyContinue
        Write-Output "Stopped demo experiment process $registeredPid"
    }
}
foreach ($relative in $manifest.restore_files) {
    $saved = Scoped-Path $backupRoot $relative
    $target = Scoped-Path $demoRoot $relative
    Copy-Item -LiteralPath $saved -Destination $target -Force
    if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath $saved -Algorithm SHA256).Hash) { throw "Restore failed: $relative" }
}
foreach ($relative in $manifest.absent_paths) {
    $target = Scoped-Path $demoRoot $relative
    if (Test-Path -LiteralPath $target -PathType Leaf) { Remove-Item -LiteralPath $target -Force }
}
$python = Join-Path $demoRoot 'dfenv\Scripts\python.exe'
$connections = @(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)
foreach ($connection in $connections) {
    $serverProcess = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $connection.OwningProcess)
    $launcher = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $serverProcess.ParentProcessId) -ErrorAction SilentlyContinue
    $owned = $serverProcess.CommandLine.Contains($demoRoot) -or ($launcher -and $launcher.CommandLine.Contains($demoRoot) -and $launcher.CommandLine -like '*streamlit*app.py*')
    if ($serverProcess.CommandLine -like '*streamlit*app.py*' -and $owned) {
        Stop-Process -Id $connection.OwningProcess -Force
    } else { throw 'Port 8501 belongs to another application; baseline files restored, server not changed.' }
}
$restartStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
Start-Process -FilePath $python -ArgumentList '-B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false' -WorkingDirectory $demoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot "rollback_server_$restartStamp.stdout.log") -RedirectStandardError (Join-Path $runRoot "rollback_server_$restartStamp.stderr.log") | Out-Null
$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) { $healthy = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $healthy) { throw 'Baseline restored; server health check did not complete. Run tools\start_demo.ps1.' }
Write-Output 'Baseline restored and local demo healthy: http://127.0.0.1:8501'
