#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

for seed in 42 77 202; do
  "${REPO_ROOT}/scripts/run_formal_pair.sh" "${seed}"
done
