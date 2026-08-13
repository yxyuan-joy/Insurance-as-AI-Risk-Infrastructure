# Cluster Execution

## Compute Architecture

The simulator is one Python process that sends independent structured decision requests to OpenAI-compatible model endpoints. The formal setup used eight H100-class GPUs and one Qwen3-8B worker per GPU. Calls are distributed across the endpoint list rather than using tensor parallelism across workers.

## Portable Environment Contract

A cluster job requires:

- a checkout of this repository on persistent storage;
- the companion dataset downloaded with `scripts/download_data.py`;
- Python 3.10 or later with the project installed;
- Qwen3-8B available through one or more OpenAI-compatible endpoints;
- a writable output directory.

No released configuration depends on a specific account, mount prefix, host, or port.

## Worker Setup

The exact scheduler wrapper and network addresses are platform-specific. Start one model worker per assigned GPU, wait for every `/v1/models` endpoint to respond, and expose the resulting comma-separated URLs through `VLLM_BASE_URLS`.

## Submit One Paired Seed

From the repository root inside the compute job:

```bash
export VLLM_MODEL=qwen3-agent-local
export VLLM_BASE_URLS=<comma-separated-endpoints>
export RUNS_DIR=<persistent-output-directory>
bash scripts/run_formal_pair.sh 42
```

Seeds 42, 77, and 202 may be submitted as separate jobs. Each job should use a distinct output directory or non-overlapping run names.

## Failure and Resume Policy

Formal model fallback is disabled. If a worker is preempted or an endpoint fails, retain the run directory and resume with `run_simulation.py --resume` after restoring the endpoint set. Do not set `OVERWRITE=1` unless an existing run is intentionally being replaced.

Checkpoints are intended for active jobs and remain outside Git. Keep completed outputs on persistent storage before deleting a compute container.

## Shared-Cluster Hygiene

- Work only inside allocated project and output directories.
- Never edit, move, delete, or reuse another user's files, model cache, environment, or process.
- Use scheduler-assigned GPU identifiers rather than assuming devices are free.
- Keep credentials in environment variables or the platform secret store.
- Confirm free storage before starting a multi-seed run.
- Record model revision, container image, Python environment, GPU type, and endpoint count with each archival run.
