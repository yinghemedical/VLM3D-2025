#!/usr/bin/env bash
# Run the full CT-RATE 3,039-case classification + report pipeline.
# This assumes 4 GPUs are visible; use the cluster YAMLs for submission.
set -eo pipefail

PIPE_ROOT="$(cd "$(dirname "$0")" && pwd)"
VLM3D_ROOT="$(cd "${PIPE_ROOT}/.." && pwd)"

echo "[1/2] CT_CLS8 classification on 3,039 cases"
bash "${VLM3D_ROOT}/scripts/classification/run_ct_cls8_all.sh" full

echo "[2/2] 8B report generation conditioned on CT_CLS8 output"
bash "${VLM3D_ROOT}/scripts/report_generation/run_ctrate_eval_8b.sh" full_predcls_ctcls8

echo "=== done:"
echo "  classification: ${VLM3D_ROOT}/runs/classification/ctrate_valid_ct_cls8/full/"
echo "  report:         ${VLM3D_ROOT}/runs/report_generation/ctrate_valid_8b/full_predcls_ctcls8/"
