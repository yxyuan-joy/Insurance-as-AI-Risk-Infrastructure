from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_TASK_COLUMNS = {
    "firm_id",
    "day",
    "industry",
    "task_type",
    "incident_flag",
    "severity",
    "total_loss",
    "risk_score",
}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
            return default
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "" or (isinstance(value, float) and math.isnan(value)):
            return default
        return int(float(value))
    except Exception:
        return default


def _read_json_rows(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {path}")
    return [dict(row) for row in data]


def _read_selected_rows(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    df = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        rows.append({str(k): row[k] for k in df.columns})
    return rows


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("firm_id", row.get("id", "")))


def _task_mix(values: pd.Series) -> str:
    counts = values.value_counts().to_dict()
    return ";".join(f"{key}:{int(counts[key])}" for key in sorted(counts))


def _quantile(series: pd.Series, q: float) -> float:
    if series.empty:
        return 0.0
    return float(series.quantile(q))


def _rate_table(df: pd.DataFrame, by: str, flag: str = "incident_flag") -> list[dict[str, Any]]:
    if by not in df.columns or flag not in df.columns or df.empty:
        return []
    table = (
        df.groupby(by)[flag]
        .agg(["count", "sum", "mean"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    return table.to_dict(orient="records")


def prepare_panel(
    task_input: Path,
    output_dir: Path,
    run_tag: str,
    error_input: Path | None = None,
    buyer_population_path: Path | None = None,
    selected_firms_input: Path | None = None,
    total_days: int | None = None,
    missing_policy: str = "zero",
    copy_task_level: bool = True,
) -> dict[str, Any]:
    task_input = Path(task_input)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_df = pd.read_csv(task_input)
    missing_cols = sorted(REQUIRED_TASK_COLUMNS - set(task_df.columns))
    if missing_cols:
        raise ValueError(f"Task-level AutoCLAW CSV is missing required columns: {missing_cols}")

    for col in ["day", "incident_flag", "severity", "total_loss", "risk_score"]:
        task_df[col] = pd.to_numeric(task_df[col], errors="coerce").fillna(0)
    if "direct_loss_base" in task_df.columns:
        task_df["direct_loss_base"] = pd.to_numeric(task_df["direct_loss_base"], errors="coerce").fillna(0)
    else:
        task_df["direct_loss_base"] = 0.0

    task_df["firm_id"] = task_df["firm_id"].astype(str)
    task_df["industry"] = task_df["industry"].astype(str)
    task_df["task_type"] = task_df["task_type"].astype(str)
    task_df["day"] = task_df["day"].map(_as_int)
    task_df["incident_flag"] = task_df["incident_flag"].map(_as_int)

    buyer_rows = _read_json_rows(buyer_population_path)
    selected_rows = _read_selected_rows(selected_firms_input)

    metadata: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    for row in buyer_rows + selected_rows:
        firm_id = _row_id(row)
        if not firm_id:
            continue
        if firm_id not in metadata:
            ordered_ids.append(firm_id)
            metadata[firm_id] = {}
        metadata[firm_id].update(row)

    task_firms = sorted(task_df["firm_id"].unique())
    if ordered_ids:
        firm_ids = [firm_id for firm_id in ordered_ids if firm_id in set(task_firms)]
        for firm_id in task_firms:
            if firm_id not in firm_ids:
                firm_ids.append(firm_id)
    else:
        firm_ids = task_firms

    if not firm_ids:
        raise ValueError("No firms found in AutoCLAW task-level data.")

    if total_days is None:
        day_min = int(task_df["day"].min())
        day_max = int(task_df["day"].max())
        if day_min != 0:
            raise ValueError(f"Expected day index to start at 0; found {day_min}")
        total_days = day_max + 1
    days = list(range(int(total_days)))

    industry_by_task = (
        task_df[["firm_id", "industry"]]
        .drop_duplicates(subset=["firm_id"], keep="last")
        .set_index("firm_id")["industry"]
        .to_dict()
    )
    industry_by_meta = {
        firm_id: str(row.get("industry_code", row.get("industry", "")))
        for firm_id, row in metadata.items()
        if row.get("industry_code", row.get("industry", ""))
    }

    grouped = (
        task_df.groupby(["firm_id", "day", "industry"], sort=True)
        .agg(
            num_tasks=("task_type", "count"),
            incident_any_flag=("incident_flag", "max"),
            incident_task_count=("incident_flag", "sum"),
            avg_severity=("severity", "mean"),
            sum_direct_loss_base=("direct_loss_base", "sum"),
            sum_total_loss=("total_loss", "sum"),
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            task_type_mix=("task_type", _task_mix),
        )
        .reset_index()
    )
    grouped["autoclaw_missing_flag"] = 0
    grouped["autoclaw_missing_reason"] = ""
    grouped["source_task_rows"] = grouped["num_tasks"]

    expected = pd.MultiIndex.from_product([firm_ids, days], names=["firm_id", "day"]).to_frame(index=False)
    expected["industry"] = expected["firm_id"].map(lambda x: industry_by_task.get(x, industry_by_meta.get(x, "unknown")))

    panel = expected.merge(grouped, how="left", on=["firm_id", "day", "industry"])
    missing_mask = panel["num_tasks"].isna()
    if missing_policy != "zero":
        raise ValueError(f"Unsupported missing policy: {missing_policy}")

    int_cols = ["num_tasks", "incident_any_flag", "incident_task_count", "source_task_rows"]
    float_cols = ["avg_severity", "sum_direct_loss_base", "sum_total_loss", "avg_risk_score", "max_risk_score"]
    for col in int_cols:
        panel[col] = panel[col].fillna(0).map(_as_int)
    for col in float_cols:
        panel[col] = panel[col].fillna(0.0).map(_as_float)
    panel["task_type_mix"] = panel["task_type_mix"].fillna("")
    panel["autoclaw_missing_flag"] = panel["autoclaw_missing_flag"].fillna(0).map(_as_int)
    panel.loc[missing_mask, "autoclaw_missing_flag"] = 1
    panel["autoclaw_missing_reason"] = panel["autoclaw_missing_reason"].fillna("")
    panel.loc[missing_mask, "autoclaw_missing_reason"] = "no_completed_task_after_autoclaw_error"

    panel = panel.sort_values(["day", "industry", "firm_id"]).reset_index(drop=True)

    industry = (
        panel.groupby(["industry", "day"], sort=True)
        .agg(
            observations=("firm_id", "count"),
            missing_firm_days=("autoclaw_missing_flag", "sum"),
            total_tasks=("num_tasks", "sum"),
            incident_days=("incident_any_flag", "sum"),
            incident_rate=("incident_any_flag", "mean"),
            avg_severity=("avg_severity", "mean"),
            avg_loss=("sum_total_loss", "mean"),
            total_loss=("sum_total_loss", "sum"),
            avg_risk_score=("avg_risk_score", "mean"),
            max_risk_score=("max_risk_score", "max"),
            stress_loss_p95=("sum_total_loss", lambda s: _quantile(s, 0.95)),
            stress_risk_score_p95=("max_risk_score", lambda s: _quantile(s, 0.95)),
        )
        .reset_index()
    )

    selected_rows_out = []
    firm_summary = panel.groupby("firm_id").agg(
        industry=("industry", "first"),
        active_days=("source_task_rows", lambda s: int((s > 0).sum())),
        missing_days=("autoclaw_missing_flag", "sum"),
        assigned_tasks=("num_tasks", "sum"),
        incident_days=("incident_any_flag", "sum"),
        total_loss=("sum_total_loss", "sum"),
    )
    for firm_id, row in firm_summary.iterrows():
        meta = metadata.get(str(firm_id), {})
        selected_rows_out.append(
            {
                "firm_id": str(firm_id),
                "industry": str(row["industry"]),
                "name": str(meta.get("name", firm_id)),
                "cash": _as_float(meta.get("cash"), 100000.0),
                "asset_value": _as_float(meta.get("asset_value"), meta.get("cash", 100000.0)),
                "size_label": str(meta.get("size_label", "unknown")),
                "active_days": int(row["active_days"]),
                "missing_days": int(row["missing_days"]),
                "assigned_tasks": int(row["assigned_tasks"]),
                "incident_days": int(row["incident_days"]),
                "total_loss": float(row["total_loss"]),
            }
        )
    selected_out = pd.DataFrame(selected_rows_out).sort_values(["industry", "firm_id"])

    task_output = output_dir / "firm_task_action_risk.csv"
    if copy_task_level:
        task_df.to_csv(task_output, index=False)

    firm_day_output = output_dir / "firm_daily_action_risk.csv"
    industry_output = output_dir / "industry_action_risk_series.csv"
    selected_output = output_dir / "selected_firms.csv"
    panel.to_csv(firm_day_output, index=False)
    industry.to_csv(industry_output, index=False)
    selected_out.to_csv(selected_output, index=False)

    errors = pd.DataFrame()
    error_summary = pd.DataFrame()
    error_output = output_dir / "benchmark_errors.csv"
    error_summary_output = output_dir / "benchmark_error_summary.csv"
    if error_input and Path(error_input).exists():
        errors = pd.read_csv(error_input)
        errors.to_csv(error_output, index=False)
        if not errors.empty:
            group_cols = [c for c in ["task_type", "task_difficulty", "error_type"] if c in errors.columns]
            if group_cols:
                error_summary = errors.groupby(group_cols).size().reset_index(name="count").sort_values("count", ascending=False)
                error_summary.to_csv(error_summary_output, index=False)
            else:
                pd.DataFrame([{"count": len(errors)}]).to_csv(error_summary_output, index=False)
        else:
            errors.to_csv(error_summary_output, index=False)
    else:
        errors.to_csv(error_output, index=False)
        errors.to_csv(error_summary_output, index=False)

    if not copy_task_level:
        task_output = task_input

    quality = {
        "run_tag": run_tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task_input": str(task_input),
        "error_input": str(error_input) if error_input else "",
        "missing_policy": missing_policy,
        "output_dir": str(output_dir),
        "outputs": {
            "firm_task_action_risk": str(task_output),
            "firm_daily_action_risk": str(firm_day_output),
            "industry_action_risk_series": str(industry_output),
            "selected_firms": str(selected_output),
            "benchmark_errors": str(error_output),
            "benchmark_error_summary": str(error_summary_output),
        },
        "task_rows": int(len(task_df)),
        "firms": int(len(firm_ids)),
        "days": int(total_days),
        "expected_firm_days": int(len(firm_ids) * int(total_days)),
        "observed_firm_days": int((panel["autoclaw_missing_flag"] == 0).sum()),
        "missing_firm_days": int(panel["autoclaw_missing_flag"].sum()),
        "missing_by_industry": panel.groupby("industry")["autoclaw_missing_flag"].sum().astype(int).to_dict(),
        "task_incident_count": int(task_df["incident_flag"].sum()),
        "task_incident_rate": float(task_df["incident_flag"].mean()),
        "firm_day_incident_count": int(panel["incident_any_flag"].sum()),
        "firm_day_incident_rate": float(panel["incident_any_flag"].mean()),
        "total_loss_sum": float(panel["sum_total_loss"].sum()),
        "firm_day_loss_mean": float(panel["sum_total_loss"].mean()),
        "firm_day_loss_p95": float(panel["sum_total_loss"].quantile(0.95)),
        "firm_day_loss_p99": float(panel["sum_total_loss"].quantile(0.99)),
        "firm_day_loss_max": float(panel["sum_total_loss"].max()),
        "error_rows": int(len(errors)),
        "error_rate_vs_completed_tasks": float(len(errors) / max(len(task_df), 1)),
        "task_type_rates": _rate_table(task_df, "task_type"),
        "task_difficulty_rates": _rate_table(task_df, "task_difficulty"),
        "industry_firm_day_rates": _rate_table(panel, "industry", "incident_any_flag"),
    }
    report_path = output_dir / "data_quality_report.json"
    report_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# AutoCLAW Prepared Panel: {run_tag}",
        "",
        f"- task_rows: {quality['task_rows']}",
        f"- firms: {quality['firms']}",
        f"- days: {quality['days']}",
        f"- expected_firm_days: {quality['expected_firm_days']}",
        f"- missing_firm_days: {quality['missing_firm_days']}",
        f"- task_incident_rate: {quality['task_incident_rate']:.6f}",
        f"- firm_day_incident_rate: {quality['firm_day_incident_rate']:.6f}",
        f"- total_loss_sum: {quality['total_loss_sum']:.6f}",
        f"- error_rows: {quality['error_rows']}",
        f"- missing_policy: {missing_policy}",
        "",
        "## Outputs",
    ]
    for key, path in quality["outputs"].items():
        md_lines.append(f"- {key}: `{path}`")
    (output_dir / "data_quality_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare completed AutoCLAW task output for action_risk_v2 simulation.")
    parser.add_argument("--task-input", required=True, help="Merged task-level firm_daily_action_risk.csv from AutoCLAW.")
    parser.add_argument("--output-dir", required=True, help="Directory where prepared panel files are written.")
    parser.add_argument("--run-tag", default="", help="Run tag for manifest/reporting.")
    parser.add_argument("--error-input", default="", help="Merged benchmark_errors.csv, if available.")
    parser.add_argument("--buyer-population", default="", help="Buyer population JSON used for firm metadata and expected firm grid.")
    parser.add_argument("--selected-firms-input", default="", help="Optional selected_firms.csv used for expected firm order/metadata.")
    parser.add_argument("--days", type=int, default=0, help="Expected number of days. Default infers 0..max(day).")
    parser.add_argument("--missing-policy", default="zero", choices=["zero"], help="How to fill firm-days lost to benchmark execution failures.")
    parser.add_argument("--skip-task-copy", action="store_true", help="Do not copy task-level rows into output-dir.")
    args = parser.parse_args()

    quality = prepare_panel(
        task_input=Path(args.task_input),
        output_dir=Path(args.output_dir),
        run_tag=args.run_tag or Path(args.output_dir).name,
        error_input=Path(args.error_input) if args.error_input else None,
        buyer_population_path=Path(args.buyer_population) if args.buyer_population else None,
        selected_firms_input=Path(args.selected_firms_input) if args.selected_firms_input else None,
        total_days=args.days or None,
        missing_policy=args.missing_policy,
        copy_task_level=not args.skip_task_copy,
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
