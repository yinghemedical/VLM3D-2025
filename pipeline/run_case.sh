#!/usr/bin/env bash
# End-to-end inference for one CT case:
#   CT_CLS8 classification -> 18-class prompt -> MedM-VL 8B report -> case.json
#
# Usage:
#   bash pipeline/run_case.sh /path/to/case.nii.gz CASE_ID [OUTPUT_DIR]
set -eo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 IMAGE CASE_ID [OUTPUT_DIR]" >&2
  exit 2
fi

PIPE_ROOT="$(cd "$(dirname "$0")" && pwd)"
VLM3D_ROOT="$(cd "${PIPE_ROOT}/.." && pwd)"
IMAGE="$(readlink -f "$1")"
CASE_ID="$2"
OUT_ROOT="${3:-${PIPE_ROOT}/outputs}"
CASE_DIR="${OUT_ROOT}/${CASE_ID}"

PIPELINE_ENV="${PIPELINE_ENV:-${PIPE_ROOT}/configs/pipeline.env}"
if [ -f "${PIPELINE_ENV}" ]; then
  # shellcheck disable=SC1090
  source "${PIPELINE_ENV}"
fi

: "${VLM3D_ROOT:=${VLM3D_ROOT}}"
: "${CT_CLS_ROOT:=${VLM3D_ROOT}/code/classification/inference}"
: "${CT_CLS_CKPT_DIR:=${WEIGHTS_ROOT:-${VLM3D_ROOT}/../VLM3D-2025-weights}/classification/ct_cls8}"
: "${REPORT_SRC_MODEL:=${WEIGHTS_ROOT:-${VLM3D_ROOT}/../VLM3D-2025-weights}/report_generation/final-8b}"
: "${REPORT_RUNTIME_MODEL:=${WEIGHTS_ROOT:-${VLM3D_ROOT}/../VLM3D-2025-weights}/report_generation/runtime-8b-ctrate}"
: "${CACHE_DIR:=/path/to/hf_cache}"
: "${SIGLIP_DIR:=${CACHE_DIR}/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f}"
: "${QWEN_DIR:=${CACHE_DIR}/Qwen3-8B}"
: "${MEDMVL_ROOT:=${VLM3D_ROOT}/code/report_generation/MedM-VL}"

for p in "${IMAGE}" "${CT_CLS_ROOT}" "${REPORT_SRC_MODEL}" "${MEDMVL_ROOT}"; do
  [ -e "${p}" ] || { echo "missing required path: ${p}" >&2; exit 1; }
done

mkdir -p "${CASE_DIR}/logs"
export CT_CLS_ROOT CT_CLS_CKPT_DIR

PY_MEDMVL="${PY_MEDMVL:-python}"
PY_INFER="${PY_INFER:-python}"

echo "[1/4] classification manifest"
"${PY_MEDMVL}" "${PIPE_ROOT}/scripts/make_case_manifest.py" \
  --image "${IMAGE}" --case-id "${CASE_ID}" --out "${CASE_DIR}/cases.csv"

echo "[2/4] CT_CLS8"
RUN_DIR="${CASE_DIR}/classification" \
GT_CSV="${CASE_DIR}/cases.csv" \
LIMIT=1 NUM_SHARDS=1 \
bash "${VLM3D_ROOT}/scripts/classification/run_ct_cls8_all.sh" smoke \
  > "${CASE_DIR}/logs/classification.log" 2>&1

echo "[3/4] build report prompt"
"${PY_MEDMVL}" "${PIPE_ROOT}/scripts/make_case_report_input.py" \
  --image "${IMAGE}" \
  --case-id "${CASE_ID}" \
  --classification-csv "${CASE_DIR}/classification/ct_cls8_predictions.csv" \
  --out "${CASE_DIR}/report_input.json"

echo "[4/4] MedM-VL 8B report"
"${PY_MEDMVL}" "${VLM3D_ROOT}/scripts/report_generation/prepare_runtime_model.py" \
  --src "${REPORT_SRC_MODEL}" \
  --dst "${REPORT_RUNTIME_MODEL}" \
  --cache-dir "${CACHE_DIR}" \
  --siglip-dir "${SIGLIP_DIR}" \
  --qwen-dir "${QWEN_DIR}" \
  --attn-implementation sdpa \
  > "${CASE_DIR}/logs/runtime_model.log" 2>&1

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PY_INFER}" "${VLM3D_ROOT}/scripts/report_generation/run_medmvl_report.py" \
  --model-dir "${REPORT_RUNTIME_MODEL}" \
  --input-json "${CASE_DIR}/report_input.json" \
  --output-json "${CASE_DIR}/report.json" \
  --output-csv "${CASE_DIR}/report.csv" \
  --medmvl-root "${MEDMVL_ROOT}" \
  --limit 1 \
  > "${CASE_DIR}/logs/report.log" 2>&1

"${PY_MEDMVL}" "${PIPE_ROOT}/scripts/collect_case.py" \
  --case-id "${CASE_ID}" \
  --image "${IMAGE}" \
  --classification-csv "${CASE_DIR}/classification/ct_cls8_predictions.csv" \
  --report-json "${CASE_DIR}/report.json" \
  --report-input "${CASE_DIR}/report_input.json" \
  --out "${CASE_DIR}/case.json"

echo "=== done: ${CASE_DIR}/case.json ==="
