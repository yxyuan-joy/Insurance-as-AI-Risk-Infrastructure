# Insurance as AI Risk Infrastructure

<p align="center">
  <img src="assets/motivation.png" width="100%" alt="Insurance as a market-based risk-transfer layer for operational AI losses.">
</p>

Code for a large language model based social simulation of enterprise AI adoption, operational risk, and insurance-mediated risk transfer.

## Release Scope

This repository contains source code, frozen experiment configurations, tests, and analysis utilities. Research input data are distributed separately through the companion Hugging Face dataset. Simulation outputs, paper results, model-decision traces, and calibration runs are not distributed in either repository.

| Resource | Contents |
|---|---|
| GitHub | Simulator, AutoCLAW pipeline, configurations, tests, launchers, and analysis code |
| Hugging Face | Firm population and simulation-ready AutoCLAW input panel |
| User-generated `runs/` | New simulation outputs created locally and excluded from Git |

## Repository Layout

```text
src/action_risk_v2/  Core market, decision, contract, insurance, and accounting logic
configs/             Formal, rule-based ABM, stress, and sensitivity configurations
autoclaw/            Enterprise-task generation, verification, and risk-panel preparation
analysis/            Result summarization, trace auditing, and manuscript-figure code
scripts/             Data download, validation, tests, and experiment launchers
tests/               Unit and mechanism tests using a generated synthetic fixture
docs/                Configuration, data, prompt, execution, and release documentation
assets/              README artwork
```

## Installation

Python 3.10 or later is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[analysis,test]"
```

Run the 67 code and mechanism tests without downloading research data:

```bash
bash scripts/run_smoke.sh
```

The test suite creates a small deterministic fixture under the ignored `.test-data/` directory. It does not use or reproduce the formal research panel.

## Input Data

Download the companion Hugging Face dataset into the paths expected by the frozen configurations:

```bash
python scripts/download_data.py
python scripts/validate_data.py
```

Set `HF_DATASET_REPO` or pass `--repo-id` if the dataset is hosted under another namespace. For a private dataset, authenticate with `HF_TOKEN` in the environment. Downloaded files are placed under the ignored `data/` directory.

The main input is a complete 300-firm by 100-update risk panel. Each operational update represents three calendar days, so the simulation horizon corresponds to 300 calendar days. See [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) for schemas and [docs/RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md) for the exact publication boundary.

## Running the Code

A CPU rule-policy run can be launched after downloading the data:

```bash
python run_simulation.py \
  --config configs/rule_abm.yaml \
  --runs-dir runs/example \
  --run-name seed42 \
  --seed 42
```

Formal LLM-agent runs require one or more OpenAI-compatible model endpoints. The launch scripts read endpoint URLs from `VLLM_BASE_URLS` rather than embedding machine-specific addresses. Available entry points are:

```text
scripts/run_formal_pair.sh SEED
scripts/run_formal_seeds.sh
scripts/run_rule_abm.sh
scripts/run_sensitivity_pair.sh PROFILE [SEED]
```

New outputs are written below `runs/` by default. The analysis utilities accept these locally generated run directories; no precomputed result is required by the code test suite.

## Documentation

- [Reproducibility guide](docs/REPRODUCIBILITY.md)
- [Configuration reference](docs/CONFIGURATION.md)
- [Data dictionary](docs/DATA_DICTIONARY.md)
- [Prompt and response-schema index](docs/PROMPTS.md)
- [Cluster execution](docs/CLUSTER_EXECUTION.md)
- [Release scope](docs/RELEASE_SCOPE.md)

## License

The code is distributed under the scholarly peer-review terms in [LICENSE](LICENSE). The companion dataset carries the same access and reuse terms unless its repository record states otherwise.
