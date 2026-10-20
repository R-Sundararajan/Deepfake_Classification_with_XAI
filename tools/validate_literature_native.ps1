# Recalculate only a read-only backup in a separate hidden Excel instance.
$ErrorActionPreference='Stop'
$phaseProject=Split-Path -Parent $PSScriptRoot
$sourceRoot=Split-Path -Parent $phaseProject
$record=Get-ChildItem -LiteralPath (Join-Path $sourceRoot 'archive/literature_backups') -Filter 'backup_*.json' | Sort-Object Name | Select-Object -Last 1
$source=(Get-Content -LiteralPath $record.FullName -Raw | ConvertFrom-Json).Backup
$excelApp=$null; $workbook=$null; $ownInstance=$false
function ColumnName([int]$number) {
    $letters=''
    while($number -gt 0){$number--; $letters=[char](65+($number%26))+$letters; $number=[math]::Floor($number/26)}
    return $letters
}
try {
    $excelApp=New-Object -ComObject Excel.Application
    if($excelApp.Workbooks.Count -ne 0){throw 'Refusing to use an Excel instance with existing open workbooks'}
    $ownInstance=$true
    $excelApp.Visible=$false; $excelApp.DisplayAlerts=$false; $excelApp.AutomationSecurity=3
    $workbook=$excelApp.Workbooks.Open($source,0,$true)
    if(-not $workbook.ReadOnly){throw 'Read-only validation contract failed'}
    $excelApp.CalculateFullRebuild()
    $sheets=[ordered]@{}; $errorCells=@()
    foreach($sheet in $workbook.Worksheets) {
        $used=$sheet.UsedRange; $values=$used.Value2; $cells=[ordered]@{}
        for($rowIndex=1;$rowIndex -le $used.Rows.Count;$rowIndex++){
            for($columnIndex=1;$columnIndex -le $used.Columns.Count;$columnIndex++){
                $value=$values[$rowIndex,$columnIndex]
                if($null -ne $value){
                    $address=(ColumnName ($used.Column+$columnIndex-1))+($used.Row+$rowIndex-1)
                    $cells[$address]=$value
                    if($value -is [int] -and $value -lt -2000000000){$errorCells+=($sheet.Name+'!'+$address)}
                }
            }
        }
        $sheets[$sheet.Name]=[ordered]@{Cells=$cells;Charts=$sheet.ChartObjects().Count;Rows=$used.Rows.Count;Columns=$used.Columns.Count}
    }
    $result=[ordered]@{ExcelVersion=$excelApp.Version;ReadOnly=$workbook.ReadOnly;Workbook=$source;Worksheets=$workbook.Worksheets.Count;FormulaErrors=$errorCells;Sheets=$sheets}
    $output=Join-Path $phaseProject '../research/working/expansion/native_excel_validation.json'
    $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $output -Encoding utf8
    [ordered]@{ExcelVersion=$excelApp.Version;Worksheets=$workbook.Worksheets.Count;ReadOnly=$workbook.ReadOnly;FormulaErrors=$errorCells;Output=$output} | ConvertTo-Json
} finally {
    if($workbook){$workbook.Close($false)}
    if($excelApp){if($ownInstance){$excelApp.Quit()}; [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($excelApp)}
}
