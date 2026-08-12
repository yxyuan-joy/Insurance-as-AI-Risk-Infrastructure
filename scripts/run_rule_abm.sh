#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/rule_abm}"
extra_args=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  extra_args+=(--overwrite)
fi

for seed in 42 77 202; do
  python3 run_simulation.py \
    --config configs/rule_abm.yaml \
    --runs-dir "${RUNS_DIR}" \
    --run-name "seed${seed}" \
    --days 100 \
    --firms 300 \
    --seed "${seed}" \
    --decision-mode rule_heuristic \
    "${extra_args[@]}"
done
