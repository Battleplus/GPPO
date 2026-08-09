param(
    [Parameter(Mandatory = $true)]
    [int]$GatePostprocessProcessId,
    [string]$Repository = $PSScriptRoot,
    [string]$FormalWorkspace = "E:\Z博士"
)

$ErrorActionPreference = "Stop"
$log = Join-Path $Repository "outputs\gate_screening\formal_handoff.log"

try {
    Wait-Process -Id $GatePostprocessProcessId
    $summaryPath = Join-Path $Repository "outputs\gate_screening\summary.json"
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ($summary.valid -ne $true) {
        throw "Gate screening post-processing ended without a valid summary."
    }

    python (Join-Path $Repository "freeze_phase1_protocol.py") *>> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Phase-1 protocol freeze failed with code $LASTEXITCODE."
    }

    $python = "D:\anaconda\python.exe"
    $literalRoot = Join-Path $FormalWorkspace "outputs\paper_faithful\formal\T5-10-48_literal_event"
    "[$(Get-Date -Format o)] Resuming preserved 2000-iteration Literal matrix with published code." | Add-Content -LiteralPath $log
    & $python (Join-Path $Repository "run_paper_faithful_formal.py") `
        --scale T5-10-48 --methods literal:event `
        --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 `
        --iterations 2000 --rollout-steps 512 --batch-size 512 --update-epochs 4 `
        --validation-interval 50 --validation-instances 100 --validation-split validation_a `
        --rrelu-mode expected --gate-bias-init 0.0 --gate-scope task_message `
        --device cpu --jobs 4 --output-root $literalRoot *>> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Preserved Literal matrix resume failed with code $LASTEXITCODE."
    }

    $checkpoints = @(Get-ChildItem -LiteralPath $literalRoot -Recurse -Filter checkpoint.pt -File)
    if ($checkpoints.Count -ne 5) {
        throw "Expected five final Literal checkpoints after exact resume; found $($checkpoints.Count)."
    }

    "[$(Get-Date -Format o)] Reselecting all preserved candidates on validation-A." | Add-Content -LiteralPath $log
    & $python (Join-Path $Repository "reselect_phase1_formal_candidates.py") `
        --root $literalRoot --instances 100 --jobs 4 *>> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Formal candidate validation-A reselection failed with code $LASTEXITCODE."
    }

    "[$(Get-Date -Format o)] Starting remaining four-scale formal matrix." | Add-Content -LiteralPath $log
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
        (Join-Path $FormalWorkspace "continue_paper_faithful_matrix.ps1") *>> $log
    if ($LASTEXITCODE -ne 0) {
        throw "Formal matrix continuation failed with code $LASTEXITCODE."
    }
    "[$(Get-Date -Format o)] Formal matrix continuation completed." | Add-Content -LiteralPath $log
}
catch {
    "[$(Get-Date -Format o)] ERROR: $($_.Exception.Message)" | Add-Content -LiteralPath $log
    exit 1
}
