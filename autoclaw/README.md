# AutoCLAW Action-Risk Benchmark

This module constructs enterprise task sandboxes, asks an AI agent to complete industry-specific tasks, verifies the resulting workspace, maps verified failures into operational-loss observations, and aggregates task episodes into the firm-day panel consumed by the social simulation.

## Pipeline

```text
firm population
  -> industry-specific task sampling
  -> isolated sandbox construction
  -> AutoCLAW execution
  -> deterministic artifact verification
  -> incident severity and loss mapping
  -> task-level output
  -> firm-day panel preparation
```

The formal task mix is defined in `autoclaw/configs/task_profiles.yaml`; runtime settings are in `autoclaw/configs/benchmark_config.yaml`. `autoclaw/engine/benchmark_runner.py` executes resumable task episodes. The merge utilities consolidate distributed or resumed outputs, and `prepare_autoclaw_panel.py` validates and aggregates the resulting episode table.

The simulation-ready panel is distributed through the companion Hugging Face dataset, so AutoCLAW is not required for simulator execution. Unit and mechanism tests use a generated fixture and require neither AutoCLAW nor the research data. See `docs/REPRODUCIBILITY.md` for optional upstream regeneration.
