#!/usr/bin/env bash
# Run the full CT_CLS8 + report pipeline on a few example cases.
set -eo pipefail

PIPE_ROOT="$(cd "$(dirname "$0")" && pwd)"
CASES_CSV="${1:-${PIPE_ROOT}/examples/cases.example.csv}"
OUT_ROOT="${2:-${PIPE_ROOT}/outputs/examples}"
mkdir -p "${OUT_ROOT}"

# Skip header; blank lines and comments ignored.
tail -n +2 "${CASES_CSV}" | while IFS=, read -r case_id image_path; do
  [ -n "${case_id}" ] || continue
  case "${case_id}" in \#*) continue ;; esac
  [ -n "${image_path}" ] || continue
  echo "=== ${case_id} ==="
  bash "${PIPE_ROOT}/run_case.sh" "${image_path}" "${case_id}" "${OUT_ROOT}"
done

echo "=== examples done: ${OUT_ROOT} ==="
