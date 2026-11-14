#!/usr/bin/env bash
# Run the CT_CLS8 ensemble on CT-RATE valid cases (smoke or full).
set -eo pipefail

MODE="${1:-full}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUN_ROOT="${ROOT}/runs/classification/ctrate_valid_ct_cls8"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${MODE}}"
GT_CSV="${GT_CSV:-${ROOT}/runs/report_generation/ctrate_valid_8b/full/gt.csv}"

if [ "${MODE}" = "smoke" ]; then
  NUM_SHARDS="${NUM_SHARDS:-1}"
  LIMIT="${LIMIT:-1}"
else
  NUM_SHARDS="${NUM_SHARDS:-4}"
  LIMIT="${LIMIT:-}"
fi

mkdir -p "${RUN_DIR}/logs"
rm -f "${RUN_DIR}"/predictions.shard*.json "${RUN_DIR}"/ct_cls8_predictions.csv

PY_CLS="${PY_CLS:-python}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
CT_CLS_ROOT="${CT_CLS_ROOT:-${ROOT}/code/classification/inference}"
export PYTHONPATH="${CT_CLS_ROOT}:${PYTHONPATH:-}"

PREP_ARGS=(--gt-csv "${GT_CSV}" --out-dir "${RUN_DIR}" --num-shards "${NUM_SHARDS}")
if [ -n "${LIMIT}" ]; then
  PREP_ARGS+=(--limit "${LIMIT}")
fi
"${PY_CLS}" "${ROOT}/scripts/classification/prepare_ct_cls8_shards.py" "${PREP_ARGS[@]}"

pids=()
for rank in $(seq 0 $((NUM_SHARDS - 1))); do
  shard=$(printf '%02d' "${rank}")
  echo "launch CT_CLS8 shard ${shard} on GPU ${rank}"
  CUDA_VISIBLE_DEVICES="${rank}" "${PY_CLS}" "${ROOT}/scripts/classification/run_ct_cls8_shard.py" \
    --shard-dir "${RUN_DIR}/shards/shard${shard}" \
    --output-json "${RUN_DIR}/predictions.shard${shard}.json" \
    > "${RUN_DIR}/logs/shard${shard}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [ "${status}" -ne 0 ]; then
  echo "at least one CT_CLS8 shard failed; see ${RUN_DIR}/logs/" >&2
  exit 1
fi

"${PY_CLS}" "${ROOT}/scripts/classification/merge_ct_cls8_predictions.py" \
  --run-dir "${RUN_DIR}" \
  --gt-csv "${GT_CSV}" \
  --output-csv "${RUN_DIR}/ct_cls8_predictions.csv"

echo "=== done: ${RUN_DIR} ==="
