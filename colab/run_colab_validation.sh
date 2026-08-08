#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash colab/run_colab_validation.sh smoke
#   bash colab/run_colab_validation.sh benchmark
#   bash colab/run_colab_validation.sh one-click
#   DEVICE=cpu bash colab/run_colab_validation.sh quick
#   SCALE=T5-10-48 DEVICE=cpu bash colab/run_colab_validation.sh formal-scale
#   DEVICE=cpu bash colab/run_colab_validation.sh formal-all
#
# OUTPUT_ROOT should normally be on mounted Google Drive so resume_latest.pt
# survives a Colab runtime reset.

PHASE="${1:-smoke}"
DEVICE="${DEVICE:-auto}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/content/drive/MyDrive/GPPO_colab}"
SCALE="${SCALE:-T5-10-48}"
JOBS="${JOBS:-1}"
export PAPER_TORCH_THREADS="${PAPER_TORCH_THREADS:-2}"

QUICK_ROOT="${OUTPUT_ROOT}/quick_seed1_100"
FORMAL_ROOT="${OUTPUT_ROOT}/formal"
METHODS=(
  literal:event
  ppo_mlp:none
  ppo_mlp:event
  literal:none
  literal_no_gate:event
  literal_single_head:event
)
FULL_METHODS=(
  "${METHODS[@]}"
  literal:periodic
  literal:always
)

python -m pip install -q -e '.[dev]'
python - <<'PY'
import json
import os
import platform
import torch

print(json.dumps({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cpu_count": os.cpu_count(),
}, indent=2))
PY

mkdir -p "${OUTPUT_ROOT}"

run_smoke() {
  python -m pytest \
    tests/test_paper_faithful.py \
    tests/test_paper_faithful_resume.py \
    tests/test_paper_faithful_cuda.py -q
  python train_paper_faithful.py \
    --mode literal --sync-mode event --scale T5-10-48 --seed 1 \
    --iterations 2 --rollout-steps 64 --batch-size 64 --update-epochs 1 \
    --validation-interval 2 --validation-instances 2 --device "${DEVICE}" \
    --output "${OUTPUT_ROOT}/smoke"
}

run_benchmark() {
  python colab/benchmark_device.py \
    --iterations 5 --rollout-steps 256 \
    --output "${OUTPUT_ROOT}/DEVICE_BENCHMARK.json"
}

run_quick() {
  python run_paper_faithful_formal.py \
    --scale T5-10-48 --seed 1 --methods "${METHODS[@]}" \
    --iterations 100 --rollout-steps 512 --batch-size 512 --update-epochs 4 \
    --validation-interval 50 --validation-instances 20 \
    --rrelu-mode expected --gate-scope task_message --gate-activation sigmoid \
    --device "${DEVICE}" --jobs "${JOBS}" --output-root "${QUICK_ROOT}"

  python evaluate_paper_faithful_formal.py \
    --root "${QUICK_ROOT}" --instances 100 --split test --jobs 1 --trace

  local checkpoint="${QUICK_ROOT}/T5-10-48/literal_event_seed1/checkpoint.pt"
  python evaluate_paper_faithful.py \
    --checkpoint "${checkpoint}" --instances 100 --split test --trace \
    --sync-mode always \
    --output "${QUICK_ROOT}/T5-10-48/literal_event_seed1/evaluations/test_native_always_100.json"

  python evaluate_paper_faithful_baselines.py \
    --scale T5-10-48 --instances 100 --policy-seed 1 \
    --output "${QUICK_ROOT}/baselines_test100.json"
  python summarize_paper_faithful_formal.py \
    --root "${QUICK_ROOT}" --output "${QUICK_ROOT}/quick_summary.json"
  python report_paper_faithful_quick.py \
    --root "${QUICK_ROOT}" \
    --baselines "${QUICK_ROOT}/baselines_test100.json" \
    --output "${QUICK_ROOT}/QUICK_MECHANISM_REPORT_ZH.md" \
    --json-output "${QUICK_ROOT}/quick_mechanism_result.json"
}

run_formal_scale() {
  python run_paper_faithful_formal.py \
    --scale "${SCALE}" --methods "${FULL_METHODS[@]}" \
    --iterations 2000 --rollout-steps 512 --batch-size 512 --update-epochs 4 \
    --validation-interval 50 --validation-instances 100 \
    --rrelu-mode expected --gate-scope task_message --gate-activation sigmoid \
    --device "${DEVICE}" --jobs "${JOBS}" --output-root "${FORMAL_ROOT}"
  python evaluate_paper_faithful_formal.py \
    --root "${FORMAL_ROOT}/${SCALE}" --instances 100 --split test --jobs 1 --trace
}

run_formal_all() {
  python run_paper_faithful_formal.py \
    --methods "${FULL_METHODS[@]}" \
    --iterations 2000 --rollout-steps 512 --batch-size 512 --update-epochs 4 \
    --validation-interval 50 --validation-instances 100 \
    --rrelu-mode expected --gate-scope task_message --gate-activation sigmoid \
    --device "${DEVICE}" --jobs "${JOBS}" --output-root "${FORMAL_ROOT}"
  python evaluate_paper_faithful_formal.py \
    --root "${FORMAL_ROOT}" --instances 100 --split test --jobs 1 --trace
  python summarize_paper_faithful_formal.py \
    --root "${FORMAL_ROOT}" --output "${FORMAL_ROOT}/formal_summary.json"
}

run_one_click() {
  echo "[one-click] 1/4 formula, resume, and save smoke tests"
  run_smoke

  echo "[one-click] 2/4 end-to-end CPU/CUDA benchmark"
  run_benchmark
  DEVICE="$(python -c "import json; print(json.load(open(r'${OUTPUT_ROOT}/DEVICE_BENCHMARK.json', encoding='utf-8'))['recommended_device'])")"
  if [[ "${DEVICE}" == "cuda" ]]; then
    # The model is small. Two jobs overlap CPU rollout generation while sharing
    # the L4; override with ONE_CLICK_JOBS=1 if the runtime has only one vCPU.
    JOBS="${ONE_CLICK_JOBS:-2}"
    export PAPER_TORCH_THREADS="${ONE_CLICK_TORCH_THREADS:-1}"
  else
    JOBS="${ONE_CLICK_JOBS:-2}"
    export PAPER_TORCH_THREADS="${ONE_CLICK_TORCH_THREADS:-1}"
  fi
  echo "[one-click] selected DEVICE=${DEVICE}, JOBS=${JOBS}, PAPER_TORCH_THREADS=${PAPER_TORCH_THREADS}"

  echo "[one-click] 3/4 six-model training, test100, baselines, and report"
  run_quick

  echo "[one-click] 4/4 result manifest and ZIP archive"
  python - "${OUTPUT_ROOT}" "${QUICK_ROOT}" <<'PY'
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

output_root = Path(sys.argv[1]).resolve()
quick_root = Path(sys.argv[2]).resolve()
report = quick_root / "QUICK_MECHANISM_REPORT_ZH.md"
result = quick_root / "quick_mechanism_result.json"
if not report.exists() or not result.exists():
    raise FileNotFoundError("one-click report artifacts are incomplete")
checkpoints = sorted(quick_root.glob("**/checkpoint.pt"))
manifest = {
    "version": "paper-faithful-colab-one-click-v1",
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "quick_root": str(quick_root),
    "checkpoint_count": len(checkpoints),
    "checkpoints": [
        {
            "path": str(path.relative_to(quick_root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in checkpoints
    ],
    "report": str(report),
    "machine_result": str(result),
}
(output_root / "ONE_CLICK_MANIFEST.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)
archive = shutil.make_archive(str(output_root), "zip", root_dir=output_root)
print(json.dumps({"completed": True, "archive": archive, **manifest}, indent=2, ensure_ascii=False))
PY
  echo "[one-click] complete: ${OUTPUT_ROOT}.zip"
}

case "${PHASE}" in
  smoke) run_smoke ;;
  benchmark) run_benchmark ;;
  quick) run_quick ;;
  one-click) run_one_click ;;
  formal-scale) run_formal_scale ;;
  formal-all) run_formal_all ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    echo "Expected: smoke, benchmark, quick, one-click, formal-scale, formal-all" >&2
    exit 2
    ;;
esac
