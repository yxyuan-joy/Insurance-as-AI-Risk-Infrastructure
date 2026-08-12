#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


REQUIRED_FILES = (
    "data/buyers_population.json",
    "data/real_firms.json",
    "data/autoclaw/firm_daily_action_risk.csv",
    "data/autoclaw/industry_action_risk_series.csv",
    "data/autoclaw/selected_firms.csv",
    "data/autoclaw/benchmark_error_summary.csv",
    "data/autoclaw/data_quality_report.json",
    "data/autoclaw/data_quality_report.md",
)

PANEL_COLUMNS = {
    "firm_id",
    "day",
    "industry",
    "num_tasks",
    "incident_any_flag",
    "incident_task_count",
    "avg_severity",
    "sum_total_loss",
    "avg_risk_score",
    "max_risk_score",
}


def validate_data(root: Path) -> dict[str, int]:
    root = Path(root).resolve()
    missing = [path for path in REQUIRED_FILES if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required data files: " + ", ".join(missing))

    buyers = _read_json_list(root / "data/buyers_population.json")
    real_firms = _read_json_list(root / "data/real_firms.json")
    selected = _read_csv(root / "data/autoclaw/selected_firms.csv")
    panel = _read_csv(root / "data/autoclaw/firm_daily_action_risk.csv")
    industries = _read_csv(root / "data/autoclaw/industry_action_risk_series.csv")
    report = json.loads((root / "data/autoclaw/data_quality_report.json").read_text(encoding="utf-8"))

    if len(buyers) != 300 or len(real_firms) != 300 or len(selected) != 300:
        raise ValueError("Expected 300 records in each firm metadata file.")
    if len(panel) != 30_000:
        raise ValueError(f"Expected 30,000 firm-update rows, found {len(panel):,}.")
    if len(industries) != 1_100:
        raise ValueError(f"Expected 1,100 industry-update rows, found {len(industries):,}.")
    if panel and not PANEL_COLUMNS.issubset(panel[0]):
        missing_columns = sorted(PANEL_COLUMNS - set(panel[0]))
        raise ValueError("Risk panel is missing columns: " + ", ".join(missing_columns))

    buyer_ids = {str(row["id"]) for row in buyers}
    real_ids = {str(row["id"]) for row in real_firms}
    selected_ids = {str(row["firm_id"]) for row in selected}
    panel_ids = {str(row["firm_id"]) for row in panel}
    if not (buyer_ids == real_ids == selected_ids == panel_ids):
        raise ValueError("Firm identifiers are inconsistent across input files.")

    counts = Counter(str(row["firm_id"]) for row in panel)
    if set(counts.values()) != {100}:
        raise ValueError("Every firm must have exactly 100 operational updates.")
    panel_days = {int(row["day"]) for row in panel}
    if panel_days != set(range(100)):
        raise ValueError("Risk-panel update indices must span 0 through 99.")
    industry_days = {(str(row["industry"]), int(row["day"])) for row in industries}
    if len(industry_days) != 1_100:
        raise ValueError("Industry-update keys are not unique and complete.")

    expected = {
        "firms": 300,
        "days": 100,
        "expected_firm_days": 30_000,
    }
    for key, value in expected.items():
        if int(report.get(key, -1)) != value:
            raise ValueError(f"Quality report field {key!r} does not match the released data.")

    return {
        "files": len(REQUIRED_FILES),
        "firms": len(panel_ids),
        "updates": len(panel_days),
        "firm_update_rows": len(panel),
        "industry_update_rows": len(industries),
    }


def _read_json_list(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list at {path}.")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the companion research input dataset.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    summary = validate_data(args.root)
    print(
        "Data validation passed: "
        f"{summary['files']} files, {summary['firms']} firms, "
        f"{summary['updates']} updates, {summary['firm_update_rows']:,} firm-update rows."
    )


if __name__ == "__main__":
    main()
