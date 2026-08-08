param(
    [string]$Root = "outputs/paper_faithful/quick_seed1_100",
    [string]$FormalRoot = "outputs/paper_faithful/formal/T5-10-48_literal_event/T5-10-48",
    [switch]$WaitForFormal
)

$ErrorActionPreference = "Stop"
$work = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "D:\anaconda\python.exe"
$testPython = Join-Path $work ".venv\Scripts\python.exe"
$rootPath = [System.IO.Path]::GetFullPath((Join-Path $work $Root))
$formalPath = [System.IO.Path]::GetFullPath((Join-Path $work $FormalRoot))
New-Item -ItemType Directory -Force -Path $rootPath | Out-Null
$log = Join-Path $rootPath "quick_pipeline.log"
"quick pipeline queued $(Get-Date -Format o)" | Out-File -LiteralPath $log -Encoding utf8

# The gate is opt-in. It is useful when formal trainers are actually alive,
# but must not make the quick suite wait on stale interrupted outputs.
if ($WaitForFormal) {
    $deadline = (Get-Date).AddMinutes(60)
    while (@(Get-ChildItem -LiteralPath $formalPath -Recurse -Filter checkpoint.pt -File -ErrorAction SilentlyContinue).Count -lt 5) {
        if ((Get-Date) -gt $deadline) { throw "formal T5 batch did not finish within 60 minutes" }
        Start-Sleep -Seconds 30
    }
    "formal resource gate passed $(Get-Date -Format o)" | Add-Content -LiteralPath $log -Encoding utf8
} else {
    "formal resource gate bypassed $(Get-Date -Format o)" | Add-Content -LiteralPath $log -Encoding utf8
}

& $testPython -m pytest tests/test_paper_faithful.py tests/test_paper_faithful_resume.py -q *>> $log
if ($LASTEXITCODE -ne 0) { throw "paper-faithful smoke failed with code $LASTEXITCODE" }

& $python run_paper_faithful_formal.py `
    --scale T5-10-48 `
    --methods literal:event ppo_mlp:none ppo_mlp:event literal:none literal_no_gate:event literal_single_head:event `
    --seed 1 `
    --iterations 100 `
    --rollout-steps 512 `
    --batch-size 512 `
    --update-epochs 4 `
    --validation-interval 50 `
    --validation-instances 20 `
    --rrelu-mode expected `
    --gate-scope task_message `
    --jobs 4 `
    --output-root $rootPath *>> $log
if ($LASTEXITCODE -ne 0) { throw "quick training matrix failed with code $LASTEXITCODE" }

& $python evaluate_paper_faithful_formal.py --root $rootPath --instances 100 --split test --jobs 1 --trace *>> $log
if ($LASTEXITCODE -ne 0) { throw "quick test100 evaluation failed with code $LASTEXITCODE" }

$gppoCheckpoint = Join-Path $rootPath "T5-10-48\literal_event_seed1\checkpoint.pt"
$fullOutput = Join-Path $rootPath "T5-10-48\literal_event_seed1\evaluations\test_native_always_100.json"
& $python evaluate_paper_faithful.py --checkpoint $gppoCheckpoint --instances 100 --split test --trace --sync-mode always --output $fullOutput *>> $log
if ($LASTEXITCODE -ne 0) { throw "Event/Full replay failed with code $LASTEXITCODE" }

$baselineOutput = Join-Path $rootPath "baselines_test100.json"
& $python evaluate_paper_faithful_baselines.py --scale T5-10-48 --instances 100 --policy-seed 1 --output $baselineOutput *>> $log
if ($LASTEXITCODE -ne 0) { throw "quick baseline evaluation failed with code $LASTEXITCODE" }

& $python summarize_paper_faithful_formal.py --root $rootPath --output (Join-Path $rootPath "quick_summary.json") *>> $log
if ($LASTEXITCODE -ne 0) { throw "quick summary failed with code $LASTEXITCODE" }

& $python report_paper_faithful_quick.py `
    --root $rootPath `
    --baselines $baselineOutput `
    --output (Join-Path $rootPath "QUICK_MECHANISM_REPORT_ZH.md") `
    --json-output (Join-Path $rootPath "quick_mechanism_result.json") *>> $log
if ($LASTEXITCODE -ne 0) { throw "quick report failed with code $LASTEXITCODE" }

"quick pipeline finished $(Get-Date -Format o)" | Add-Content -LiteralPath $log -Encoding utf8
