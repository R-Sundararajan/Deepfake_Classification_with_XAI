# Validate only the new output, in a separate hidden Excel instance, without saving.
$ErrorActionPreference='Stop'
$phaseRoot=Split-Path -Parent $PSScriptRoot
$phaseFile=Join-Path $phaseRoot '../research/workbooks/Existing_Six_Papers_Detailed_Analysis.xlsx'
$excelApp=$null; $book=$null; $ownInstance=$false
try {
    $excelApp=New-Object -ComObject Excel.Application
    if($excelApp.Workbooks.Count -ne 0){throw 'Refusing an Excel instance with existing workbooks'}
    $ownInstance=$true
    $excelApp.Visible=$false; $excelApp.DisplayAlerts=$false; $excelApp.AutomationSecurity=3
    $book=$excelApp.Workbooks.Open($phaseFile,0,$true)
    if(-not $book.ReadOnly){throw 'Expected a read-only validation workbook'}
    $excelApp.CalculateFullRebuild()
    $summary=@(); $formulaErrors=@()
    foreach($sheet in $book.Worksheets){
        $cells=$sheet.UsedRange.Value2
        for($ri=1;$ri -le $cells.GetLength(0);$ri++){
            for($ci=1;$ci -le $cells.GetLength(1);$ci++){
                $v=$cells[$ri,$ci]
                if(($v -is [int] -and $v -lt -2000000000) -or ($v -is [string] -and $v -match '^#(REF!|DIV/0!|VALUE!|NAME\?|N/A|NUM!|NULL!|SPILL!|CALC!)$')){
                    $formulaErrors+=($sheet.Name+':'+$ri+','+$ci)
                }
            }
        }
        $summary+=[ordered]@{Name=$sheet.Name;Rows=$sheet.UsedRange.Rows.Count;Columns=$sheet.UsedRange.Columns.Count}
    }
    $hardware=$book.Worksheets.Item('Hardware Feasibility')
    $baseHours=$hardware.Range('C38').Value2
    $hardware.Range('C37').Value2=200
    $excelApp.CalculateFullRebuild()
    $changedHours=$hardware.Range('C38').Value2
    if([math]::Abs($changedHours-$baseHours/2) -gt 0.000001){throw 'Native scenario dependency failed'}
    $hardware.Range('C37').Value2=100
    $excelApp.CalculateFullRebuild()
    $performance=$book.Worksheets.Item('DFDC Verified')
    $dfdc=@()
    for($rowIndex=6;$rowIndex -le 13;$rowIndex++){
        $dfdc+=[ordered]@{PaperID=$performance.Cells.Item($rowIndex,1).Value2;Accuracy=$performance.Cells.Item($rowIndex,4).Value2;AUC=$performance.Cells.Item($rowIndex,8).Value2}
    }
    $checks=[ordered]@{
        ExcelVersion=$excelApp.Version;ReadOnly=$book.ReadOnly;Sheets=$summary;FormulaErrors=$formulaErrors
        NativeBaseHours=$baseHours;NativeChangedHours=$changedHours;ScenarioRestored=$hardware.Range('C37').Value2
        CacheGiB=$hardware.Range('C23').Value2;DFDC=$dfdc;Saved=$false
    }
    $out=Join-Path $phaseRoot '../research/working/existing_six/native_validation.json'
    $checks | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath $out -Encoding utf8
    $checks | ConvertTo-Json -Depth 7
} finally {
    if($book){$book.Close($false)}
    if($excelApp){if($ownInstance){$excelApp.Quit()};[void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($excelApp)}
}
