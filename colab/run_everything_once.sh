#!/usr/bin/env bash
set -euo pipefail

# One command runs the complete single-seed mechanism validation suite.
# Default output is local to the cloned repository. Set OUTPUT_ROOT to a
# mounted Google Drive directory when the results must survive a runtime reset.

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPOSITORY_ROOT}"

export OUTPUT_ROOT="${OUTPUT_ROOT:-${REPOSITORY_ROOT}/outputs/colab_one_click}"
mkdir -p "${OUTPUT_ROOT}"

LOG="${OUTPUT_ROOT}/one_click.log"
exec > >(tee -a "${LOG}") 2>&1

echo "One-click GPPO validation started at $(date -Iseconds)"
echo "Repository: ${REPOSITORY_ROOT}"
echo "Output: ${OUTPUT_ROOT}"

bash colab/run_colab_validation.sh one-click

echo "One-click GPPO validation finished at $(date -Iseconds)"
echo "Report: ${OUTPUT_ROOT}/quick_seed1_100/QUICK_MECHANISM_REPORT_ZH.md"
echo "Archive: ${OUTPUT_ROOT}.zip"

if [[ "${AUTO_DOWNLOAD:-0}" == "1" ]]; then
  python - "${OUTPUT_ROOT}.zip" <<'PY'
import sys

try:
    from google.colab import files
except ImportError:
    print("AUTO_DOWNLOAD was requested outside Colab; archive remains at", sys.argv[1])
else:
    files.download(sys.argv[1])
PY
fi
