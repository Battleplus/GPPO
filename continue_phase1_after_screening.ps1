param(
    [Parameter(Mandatory = $true)]
    [int]$ScreeningProcessId,
    [string]$Repository = $PSScriptRoot,
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $Repository
$log = Join-Path $Repository "outputs\gate_screening\postprocess.log"

try {
    Wait-Process -Id $ScreeningProcessId
    $manifestPath = Join-Path $Repository "outputs\gate_screening\screening\run_manifest.json"
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    if ($manifest.valid -ne $true) {
        throw "Screening process ended without a valid training manifest."
    }
    python finalize_phase1_gate_screening.py --jobs $Jobs --device cpu *>> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Gate screening finalizer exited with code $LASTEXITCODE."
    }
    "[$(Get-Date -Format o)] Gate screening post-processing completed." | Add-Content -LiteralPath $log
}
catch {
    "[$(Get-Date -Format o)] ERROR: $($_.Exception.Message)" | Add-Content -LiteralPath $log
    exit 1
}
