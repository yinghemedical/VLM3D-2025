#!/usr/bin/env bash
# CT-RATE validation/test report generation with the VLM3D 2025 8B model.
# Usage: bash run_ctrate_eval_8b.sh smoke|full
set -eo pipefail

MODE="${1:-full}"
ROOT="/path/to/VLM3D"
RUN_ROOT="${ROOT}/runs/report_generation/ctrate_valid_8b"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${MODE}}"

PROMPT_LABELS="${PROMPT_LABELS:-}"
if [ "${MODE}" = "smoke" ]; then
  NUM_SHARDS="${NUM_SHARDS:-1}"
  MAX_CASES="${MAX_CASES:-8}"
elif [ "${MODE}" = "full_predcls" ]; then
  NUM_SHARDS="${NUM_SHARDS:-8}"
  MAX_CASES="${MAX_CASES:-}"
  PROMPT_LABELS="${PROMPT_LABELS:-/path/to/shipengcheng/code/CT-RATE_eval_scripts/eval_cls_csv/nnXNetTrainer_ResEncoderUVL_v6_with_reportgen_only_cls_balanced_nnUNetAug_w_seg_pretrained_256_384_384_Plans_epoch410.csv}"
elif [ "${MODE}" = "full_predcls_ctcls8" ]; then
  NUM_SHARDS="${NUM_SHARDS:-4}"
  MAX_CASES="${MAX_CASES:-}"
  PROMPT_LABELS="${PROMPT_LABELS:-${ROOT}/runs/classification/ctrate_valid_ct_cls8/full/ct_cls8_predictions.csv}"
else
  NUM_SHARDS="${NUM_SHARDS:-8}"
  MAX_CASES="${MAX_CASES:-}"
fi

SRC_MODEL="${ROOT}/weights/report_generation/final-MedM-VL-siglip-384-qwen3-8b-finetune"
RUNTIME_MODEL="${ROOT}/weights/report_generation/runtime-8b-ctrate"
CACHE_DIR="/path/to/luyan_mllm/VLM3D_challenge/jiawei_workspace/cache_dir_hf"
SIGLIP_DIR="${CACHE_DIR}/models--google--siglip2-so400m-patch14-384/snapshots/e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
QWEN_DIR="${CACHE_DIR}/Qwen3-8B"

mkdir -p "${RUN_DIR}/logs"
rm -f "${RUN_DIR}"/predictions.shard*.json "${RUN_DIR}"/predictions.csv \
      "${RUN_DIR}"/predictions.json "${RUN_DIR}"/metrics_nlg.json

# Use absolute interpreters; conda activate by path is fragile when
# CONDA_ENVS_PATH is inherited from the caller.
PY_INFER="/path/to/shipengcheng/.conda/envs/medals_local_test/bin/python"
PY_EVAL="/path/to/miniconda3/envs/medmvl/bin/python"
MEDMVL_ROOT="${MEDMVL_ROOT:-${ROOT}/code/report_generation/MedM-VL}"
export PYTHONPATH="${MEDMVL_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Keep each shard from spawning a large BLAS thread pool.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "=== mode=${MODE} shards=${NUM_SHARDS} max_cases=${MAX_CASES:-all} ==="
echo "run_dir=${RUN_DIR}"

"${PY_INFER}" "${ROOT}/scripts/report_generation/prepare_runtime_model.py" \
  --src "${SRC_MODEL}" \
  --dst "${RUNTIME_MODEL}" \
  --cache-dir "${CACHE_DIR}" \
  --siglip-dir "${SIGLIP_DIR}" \
  --qwen-dir "${QWEN_DIR}" \
  --attn-implementation sdpa

PREP_ARGS=(--out-dir "${RUN_DIR}" --num-shards "${NUM_SHARDS}")
if [ -n "${MAX_CASES}" ]; then
  PREP_ARGS+=(--max-cases "${MAX_CASES}")
fi
if [ -n "${PROMPT_LABELS}" ]; then
  PREP_ARGS+=(--prompt-labels "${PROMPT_LABELS}")
fi
"${PY_INFER}" "${ROOT}/scripts/report_generation/prepare_ctrate_eval.py" "${PREP_ARGS[@]}"

pids=()
for rank in $(seq 0 $((NUM_SHARDS - 1))); do
  shard=$(printf '%02d' "${rank}")
  echo "launch shard ${shard} on GPU ${rank}"
  CUDA_VISIBLE_DEVICES="${rank}" "${PY_INFER}" "${ROOT}/scripts/report_generation/run_medmvl_report.py" \
    --model-dir "${RUNTIME_MODEL}" \
    --input-json "${RUN_DIR}/inputs.shard${shard}.json" \
    --output-json "${RUN_DIR}/predictions.shard${shard}.json" \
    --output-csv "${RUN_DIR}/predictions.shard${shard}.csv" \
    --save-every 10 \
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
  echo "at least one shard failed; see ${RUN_DIR}/logs/" >&2
  exit 1
fi

# Evaluation needs pycocoevalcap; use the medmvl interpreter directly
# (conda activate by absolute path can fail when CONDA_ENVS_PATH is set).
"${PY_EVAL}" "${ROOT}/scripts/report_generation/merge_results.py" \
  --run-dir "${RUN_DIR}" \
  --gt-csv "${RUN_DIR}/gt.csv" \
  --output-json "${RUN_DIR}/predictions.json" \
  --output-csv "${RUN_DIR}/predictions.csv"

"${PY_EVAL}" "${ROOT}/scripts/report_generation/eval_nlg.py" \
  --gt-csv "${RUN_DIR}/gt.csv" \
  --pred-csv "${RUN_DIR}/predictions.csv" \
  --output-dir "${RUN_DIR}"

case "${MODE}" in
  full_predcls|full_predcls_ctcls8)
    RUN_DIR="${RUN_DIR}" bash "${ROOT}/scripts/report_generation/run_f1_eval.sh"
    ;;
esac

echo "=== done: ${RUN_DIR} ==="
