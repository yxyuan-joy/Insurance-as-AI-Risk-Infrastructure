# Input Data Dictionary

Research inputs are hosted in the companion Hugging Face dataset and downloaded into the ignored `data/` directory by `scripts/download_data.py`.

## Firm Metadata

### `data/buyers_population.json`

Three hundred simulated firm profiles. Each record contains the firm identifier, sector, size class, cash and asset scale, and behavioral attributes used to initialize the simulator, including risk tolerance, technology urgency, AI dependence, inertia, innovativeness, and contagion sensitivity.

### `data/real_firms.json`

A reduced view of the same 300 simulated firms containing identifiers, display names, sectors, and cash. The AutoCLAW pipeline uses these records to construct task sandboxes.

### `data/autoclaw/selected_firms.csv`

The frozen order and metadata for firms represented in the prepared risk panel. It also reports task-coverage and incident summaries used for input auditing.

## Operational-Risk Panel

### `data/autoclaw/firm_daily_action_risk.csv`

The simulation-ready 300-firm by 100-update panel contains 30,000 rows. The historical column name `day` is the operational-update index from 0 through 99; each update represents three calendar days.

| Field | Definition |
|---|---|
| `firm_id`, `day`, `industry` | Firm, update index, and sector keys |
| `num_tasks` | AutoCLAW task executions represented by the row |
| `incident_any_flag` | Indicator for at least one verified incident |
| `incident_task_count` | Number of incident-bearing tasks |
| `avg_severity` | Mean verified incident severity |
| `sum_direct_loss_base` | Direct-loss component before simulation scaling |
| `sum_total_loss` | Total operational loss before simulation scaling |
| `avg_risk_score`, `max_risk_score` | Mean and maximum action-risk scores |
| `task_type_mix` | Semicolon-delimited task counts |
| `autoclaw_missing_flag`, `autoclaw_missing_reason` | Missing-episode treatment audit |
| `source_task_rows` | Number of task-level records aggregated into the row |

Fifteen missing firm-update observations are retained as explicit zero-risk rows under the documented `missing_policy=zero` rule rather than silently dropped.

### `data/autoclaw/industry_action_risk_series.csv`

Sector-level aggregates for 11 industries and 100 operational updates. Fields include observations, missing firm updates, task volume, incident rate, loss moments, risk-score moments, and 95th-percentile stress measures.

## Quality Audit

### `data/autoclaw/data_quality_report.json`

Machine-readable provenance, dimensions, missing-row treatment, incident rates, loss quantiles, task and difficulty distributions, and sector-level diagnostics.

### `data/autoclaw/data_quality_report.md`

Compact human-readable summary of the same quality audit.

### `data/autoclaw/benchmark_error_summary.csv`

Aggregate counts of benchmark execution errors by task type, difficulty, and error class. No raw prompts, responses, or stack traces are included.

## Validation

Run the structural validation after downloading:

```bash
python scripts/validate_data.py
```

The validator checks required files and columns, row counts, complete update indices, unique industry-update keys, and identifier consistency across all firm metadata and panel files.
