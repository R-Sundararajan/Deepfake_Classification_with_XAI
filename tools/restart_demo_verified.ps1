$ErrorActionPreference = 'Stop'
$demoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$runRoot = Join-Path $demoRoot 'results\demo_accuracy_20261021_040741'
foreach ($connection in @(Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue)) {
    $server = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $connection.OwningProcess)
    $launcher = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $server.ParentProcessId) -ErrorAction SilentlyContinue
    $owned = $server.CommandLine.Contains($demoRoot) -or ($launcher -and $launcher.CommandLine.Contains($demoRoot) -and $launcher.CommandLine -like '*streamlit*app.py*')
    if ($server.CommandLine -notlike '*streamlit*app.py*' -or -not $owned) { throw 'Port 8501 belongs to another application.' }
    Stop-Process -Id $connection.OwningProcess -Force
}
$restartStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
Start-Process -FilePath (Join-Path $demoRoot 'dfenv\Scripts\python.exe') -ArgumentList '-B -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false' -WorkingDirectory $demoRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $runRoot "candidate_server_$restartStamp.stdout.log") -RedirectStandardError (Join-Path $runRoot "candidate_server_$restartStamp.stderr.log") | Out-Null
$healthy = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8501/_stcore/health' -UseBasicParsing -TimeoutSec 2
        if ($response.StatusCode -eq 200) { $healthy = $true; break }
    } catch { Start-Sleep -Milliseconds 500 }
}
if (-not $healthy) { throw 'Demo did not become healthy; run rollback_demo.ps1.' }
Write-Output 'Verified local demo server is healthy: http://127.0.0.1:8501'
