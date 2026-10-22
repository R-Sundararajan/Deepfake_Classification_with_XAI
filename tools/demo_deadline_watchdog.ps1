param([Parameter(Mandatory=$true)][string]$RunRoot)
$ErrorActionPreference = 'Continue'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$expectedRoot = Join-Path $projectRoot 'results\demo_accuracy_20261021_040741'
if ([IO.Path]::GetFullPath($RunRoot) -ne $expectedRoot) { throw 'Unexpected deadline run directory.' }
$manifest = Get-Content -LiteralPath (Join-Path $RunRoot 'rollback_manifest.json') -Raw | ConvertFrom-Json
$stopTraining = [DateTimeOffset]::Parse($manifest.deadlines_utc.training)
$restoreAt = [DateTimeOffset]::Parse($manifest.deadlines_utc.finish).AddSeconds(-30)
while ([DateTimeOffset]::UtcNow -lt $restoreAt) {
    if (Test-Path -LiteralPath (Join-Path $RunRoot 'handoff_complete.json')) { exit 0 }
    if ([DateTimeOffset]::UtcNow -ge $stopTraining) {
        foreach ($worker in (Get-Content -LiteralPath (Join-Path $RunRoot 'processes.json') -Raw | ConvertFrom-Json)) {
            if ($worker.role -ne 'train') { continue }
            $registeredPid = [int]$worker.pid
            $process = Get-CimInstance Win32_Process -Filter "ProcessId = $registeredPid" -ErrorAction SilentlyContinue
            if ($process -and $process.CommandLine -like '*training.demo_accuracy*' -and $process.CommandLine.Contains($RunRoot)) {
                foreach ($child in @(Get-CimInstance Win32_Process -Filter "ParentProcessId = $registeredPid" -ErrorAction SilentlyContinue)) {
                    if ($child.CommandLine -like '*multiprocessing.spawn*') { Stop-Process -Id $child.ProcessId -Force -ErrorAction SilentlyContinue }
                }
                Stop-Process -Id $registeredPid -Force -ErrorAction SilentlyContinue
                @{ event='training_deadline'; utc=[DateTimeOffset]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunRoot 'watchdog_stop.json')
            }
        }
    }
    Start-Sleep -Seconds 1
}
if (-not (Test-Path -LiteralPath (Join-Path $RunRoot 'handoff_complete.json'))) {
    & powershell -ExecutionPolicy Bypass -File (Join-Path $projectRoot 'rollback_demo.ps1')
    @{ event='automatic_baseline_restore'; utc=[DateTimeOffset]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $RunRoot 'watchdog_restore.json')
}
