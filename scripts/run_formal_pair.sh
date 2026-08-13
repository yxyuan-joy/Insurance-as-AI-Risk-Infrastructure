#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

SEED="${1:?Usage: scripts/run_formal_pair.sh SEED}"
RUNS_DIR="${RUNS_DIR:-${REPO_ROOT}/runs/formal}"
VLLM_MODEL="${VLLM_MODEL:-qwen3-agent-local}"
: "${VLLM_BASE_URLS:?Set VLLM_BASE_URLS to one or more comma-separated OpenAI-compatible endpoints}"

common_args=(
  --config configs/formal.yaml
  --runs-dir "${RUNS_DIR}"
  --days 100
  --firms 300
  --seed "${SEED}"
  --decision-mode vllm_openai
  --vllm-base-urls "${VLLM_BASE_URLS}"
  --vllm-model "${VLLM_MODEL}"
  --model-fallback-to-rule 0
)

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  common_args+=(--overwrite)
fi

python3 run_simulation.py \
  "${common_args[@]}" \
  --run-name "seed${SEED}_on"

python3 run_simulation.py \
  "${common_args[@]}" \
  --run-name "seed${SEED}_off" \
  --disable-insurance-market
