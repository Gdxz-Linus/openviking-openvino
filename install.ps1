$ErrorActionPreference = 'Stop'
$scriptDir = $PSScriptRoot
$pyexe = $null
$candidates = @()
if (-not [string]::IsNullOrWhiteSpace($env:OPENVIKING_HOME)) {
    $candidates += Join-Path $env:OPENVIKING_HOME 'venv\Scripts\python.exe'
}
if (-not [string]::IsNullOrWhiteSpace($env:OV_HOME)) {
    $candidates += Join-Path $env:OV_HOME 'venv\Scripts\python.exe'
}
$candidates += Join-Path $scriptDir '..\..\venv\Scripts\python.exe'
$candidates += Join-Path $scriptDir '..\venv\Scripts\python.exe'
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $pyexe = $candidate
        break
    }
}
if ($null -eq $pyexe) {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $pythonCommand) {
        $pythonSource = $pythonCommand.Source
        if ((-not [string]::IsNullOrWhiteSpace($pythonSource)) -and (Test-Path -LiteralPath $pythonSource -PathType Leaf)) {
            $pyexe = $pythonSource
        }
    }
}
if ($null -eq $pyexe) {
    Write-Output 'Python not found'
    exit 1
}
& $pyexe -I (Join-Path $scriptDir 'install.py') @args
exit $LASTEXITCODE
