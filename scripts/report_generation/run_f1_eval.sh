#!/usr/bin/env bash
# F1 / CRG evaluation for existing CT-RATE report predictions (AutoExps/RadBERT).
set -eo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/report_generation/ctrate_valid_8b/full}"

PY_F1="${PY_F1:-python}"

AUTOEXPS_ROOT="${AUTOEXPS_ROOT:-${ROOT}/shared/auto_exps}"
export PYTHONPATH="${AUTOEXPS_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"${PY_F1}" "${ROOT}/scripts/report_generation/eval_f1.py" \
  --gt-csv "${RUN_DIR}/gt.csv" \
  --pred-csv "${RUN_DIR}/predictions.csv" \
  --output-dir "${RUN_DIR}"
