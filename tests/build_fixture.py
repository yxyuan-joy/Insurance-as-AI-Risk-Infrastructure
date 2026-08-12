from __future__ import annotations

import csv
import json
from pathlib import Path


INDUSTRIES = (
    "communication_services",
    "cons_discretionary",
    "cons_staples",
    "energy",
    "financials",
    "health_care",
    "industrials",
    "information_technology",
    "materials",
    "real_estate",
    "utilities",
)


def build_fixture(output_dir: Path) -> None:
    """Create a deterministic, non-research fixture for simulator tests."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles = []
    selected_rows = []
    risk_rows = []
    for industry_index, industry in enumerate(INDUSTRIES):
        for within_index in range(4):
            firm_index = industry_index * 4 + within_index
            firm_id = f"Test_{industry_index:02d}_{within_index:02d}"
            size_label = ("small", "small", "medium", "large")[within_index]
            cash = 450_000.0 + 25_000.0 * firm_index
            profile = {
                "id": firm_id,
                "name": f"Synthetic test firm {firm_index:02d}",
                "industry_code": industry,
                "cash": cash,
                "burn_rate": 0.006 + 0.0001 * within_index,
                "asset_value": cash,
                "risk_tolerance": 0.25 + 0.15 * within_index,
                "tech_urgency": 0.30 + 0.12 * ((firm_index + 1) % 4),
                "ai_dependency": 0.35 + 0.10 * (firm_index % 4),
                "inertia": 0.65 - 0.10 * (firm_index % 4),
                "innovativeness": 0.35 + 0.10 * (firm_index % 4),
                "has_ai": False,
                "upstream_dependency": 0.35,
                "downstream_dependency": 0.45,
                "contagion_sensitivity": 0.45 + 0.05 * (firm_index % 3),
                "size_label": size_label,
            }
            profiles.append(profile)
            selected_rows.append(
                {
                    "firm_id": firm_id,
                    "industry": industry,
                    "name": profile["name"],
                    "cash": cash,
                    "asset_value": cash,
                    "size_label": size_label,
                    "active_days": 12,
                    "missing_days": 0,
                    "assigned_tasks": 24,
                    "incident_days": 1,
                    "total_loss": 2_500.0,
                }
            )
            for day in range(12):
                incident = int((firm_index * 5 + day * 7) % 19 == 0)
                severity = (0.35 + 0.05 * ((firm_index + day) % 5)) if incident else 0.0
                total_loss = (1_500.0 + 175.0 * ((firm_index + day) % 7)) if incident else 0.0
                risk_score = min(0.95, severity + 0.12) if incident else 0.0
                risk_rows.append(
                    {
                        "firm_id": firm_id,
                        "day": day,
                        "industry": industry,
                        "num_tasks": 2,
                        "incident_any_flag": incident,
                        "incident_task_count": incident,
                        "avg_severity": severity,
                        "sum_direct_loss_base": total_loss * 0.8,
                        "sum_total_loss": total_loss,
                        "avg_risk_score": risk_score,
                        "max_risk_score": risk_score,
                        "task_type_mix": "document_revision:1;record_update:1",
                        "autoclaw_missing_flag": 0,
                        "autoclaw_missing_reason": "",
                        "source_task_rows": 2,
                    }
                )

    _write_json(output_dir / "buyers_population.json", profiles)
    _write_json(
        output_dir / "real_firms.json",
        [
            {
                "id": row["id"],
                "name": row["name"],
                "industry_code": row["industry_code"],
                "cash": row["cash"],
            }
            for row in profiles
        ],
    )
    _write_csv(output_dir / "selected_firms.csv", selected_rows)
    _write_csv(output_dir / "firm_daily_action_risk.csv", risk_rows)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    build_fixture(Path(__file__).resolve().parents[1] / ".test-data")
