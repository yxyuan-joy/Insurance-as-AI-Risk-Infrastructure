# Reproducibility Guide

## Release Model

Reproduction uses two coordinated resources. The GitHub repository provides executable code, frozen configurations, tests, and analysis utilities. The companion Hugging Face dataset provides the firm population and prepared AutoCLAW operational-risk inputs.

Precomputed simulation outcomes and model traces are not distributed. Reproduction therefore consists of validating the implementation and generating new trajectories from the released code and inputs.

## Environment

Use Python 3.10 or later from the repository root. The formal simulator process used Python 3.11. vLLM is an external serving dependency and is not installed by this repository.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,test]"
```

## Code Verification

The unit and mechanism suite uses an automatically generated synthetic fixture, so it runs without research data or a GPU:

```bash
bash scripts/run_smoke.sh
```

The tests cover input loading, prior-update information boundaries, vendor visibility, contract lifecycle, insurance pricing and selection, claims, bankruptcy ordering, memory, panic transmission, network context, checkpoint resume, and AutoCLAW panel preparation.

## Research Input Data

Download and validate the companion dataset:

```bash
python scripts/download_data.py
python scripts/validate_data.py
```

The downloader writes the eight required files into the ignored `data/` directory. `HF_DATASET_REPO`, `HF_DATASET_REVISION`, and `HF_TOKEN` can be supplied through the environment when using another namespace, revision, or private repository.

## Rule-Based ABM Run

The rule-policy comparison does not require an LLM service:

```bash
bash scripts/run_rule_abm.sh
```

It uses the same firms, market institutions, accounting, risk mapping, contract lifecycle, and horizon as the formal configuration. Only the decision policy is replaced by `rule_heuristic` together with its documented score-scale cutoffs.

## Formal Paired Experiment

The formal configuration is frozen in `configs/formal.yaml`. Start one or more OpenAI-compatible endpoints serving Qwen3-8B and expose their comma-separated URLs through `VLLM_BASE_URLS`. The launcher deliberately contains no machine-specific endpoint list.

```bash
VLLM_MODEL=qwen3-agent-local \
VLLM_BASE_URLS=<comma-separated-endpoints> \
bash scripts/run_formal_pair.sh 42
```

Use `scripts/run_formal_seeds.sh` to run seeds 42, 77, and 202. New outputs are written to `runs/formal/` unless `RUNS_DIR` is set. Existing runs are protected unless `OVERWRITE=1` is explicitly supplied.

The paired launcher fixes `--model-fallback-to-rule 0`. The insurance-disabled arm applies only `--disable-insurance-market`; it retains the firm panel, vendor market, traditional-return process, network, operational-risk inputs, model settings, and random seed.

## Sensitivity Runs

Fourteen one-factor profiles are provided under `configs/sensitivity/`:

```bash
VLLM_MODEL=qwen3-agent-local \
VLLM_BASE_URLS=<comma-separated-endpoints> \
bash scripts/run_sensitivity_pair.sh ai_loss_high 42
```

Each profile inherits the formal configuration and changes one parameter family. The launcher generates matched insurance-enabled and insurance-disabled trajectories under `runs/sensitivity/<profile>/`.

## Analysis

Summarize locally generated formal and rule runs:

```bash
python analysis/summarize_results.py \
  --formal-dir runs/formal \
  --rule-dir runs/rule_abm
```

Audit newly generated model traces:

```bash
python analysis/audit_model_traces.py --formal-dir runs/formal
```

Generate manuscript-style figures from local runs:

```bash
python analysis/generate_paper_figures.py
```

Figures are written to the ignored `figures/` directory.

## Optional AutoCLAW Regeneration

The companion dataset supplies the exact prepared panel, so rebuilding upstream task episodes is optional. After configuring an AutoCLAW-compatible command and model service, run:

```bash
python autoclaw/engine/benchmark_runner.py
```

Aggregate the resulting task table with:

```bash
python autoclaw/engine/prepare_autoclaw_panel.py \
  --task-input runs/autoclaw/raw/firm_daily_action_risk.csv \
  --error-input runs/autoclaw/raw/benchmark_errors.csv \
  --buyer-population data/buyers_population.json \
  --output-dir runs/autoclaw/prepared \
  --days 100 \
  --run-tag replication
```

The formal upstream settings used 300 firms, 100 operational updates, mixed task difficulty, sector-specific task counts, and deterministic verifier-based loss classification.

## Time Convention

The simulation executes 100 operational updates indexed 0 through 99. Each update represents three calendar days, giving a 300-calendar-day interpretation. The historical input column name `day` stores the operational-update index.
