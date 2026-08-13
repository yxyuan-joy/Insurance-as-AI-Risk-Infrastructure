# Release Scope

## GitHub Code Repository

The code repository contains:

- the simulator and all executable prompt templates;
- the frozen formal, rule-based ABM, stress, and sensitivity configurations;
- the AutoCLAW task, sandbox, verification, risk-mapping, and panel-preparation pipeline;
- unit and mechanism tests with a generated synthetic fixture;
- experiment launchers and result-analysis code;
- documentation for configuration, data schemas, prompts, and execution.

It does not contain research input data, simulation outputs, manuscript result tables, model-decision traces, checkpoints, calibration runs, or paper source files. The ignored `data/`, `runs/`, `results/`, and `figures/` directories prevent generated artifacts from being committed accidentally.

## Hugging Face Data Repository

The companion dataset contains only the research inputs consumed by the frozen configurations:

- 300 simulated firm profiles;
- 300 reduced firm records used to construct AutoCLAW sandboxes;
- the 30,000-row firm-update operational-risk panel;
- the 1,100-row industry-update aggregate panel;
- the selected-firm table, benchmark-error summary, and data-quality reports.

It does not contain simulation outcomes, AI adoption trajectories, insurance outcomes, bankruptcy records, sensitivity outputs, paper figures, model prompts, raw responses, or negotiation traces.

## Locally Generated Artifacts

Running the code creates outputs under `runs/` unless another directory is specified. Analysis scripts operate on these user-generated outputs. Such files remain local and outside both public release repositories.

## Excluded Upstream Intermediates

Raw task-level AutoCLAW episodes, task sandboxes, and per-update checkpoints are not part of the data release. The distributed firm-update panel, aggregate error counts, and quality report record the processing boundary used by the simulator.
