#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PROFILE="${1:?Usage: scripts/run_sensitivity_pair.sh PROFILE [SEED]}"
SEED="${2:-42}"
CONFIG="configs/sensitivity/${PROFILE}.yaml"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/sensitivity/${PROFILE}}"
VLLM_MODEL="${VLLM_MODEL:-qwen3-agent-local}"
: "${VLLM_BASE_URLS:?Set VLLM_BASE_URLS to one or more comma-separated OpenAI-compatible endpoints}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Unknown sensitivity profile: ${PROFILE}" >&2
  exit 2
fi

common_args=(
  --config "${CONFIG}"
  --runs-dir "${RUNS_DIR}"
  --days 100
  --firms 300
  --seed "${SEED}"
  --decision-mode vllm_openai
  --vllm-base-urls "${VLLM_BASE_URLS}"
  --vllm-model "${VLLM_MODEL}"
  --model-fallback-to-rule 0
)

python3 run_simulation.py \
  "${common_args[@]}" \
  --run-name "seed${SEED}_on"

python3 run_simulation.py \
  "${common_args[@]}" \
  --run-name "seed${SEED}_off" \
  --disable-insurance-market
