#!/usr/bin/env python3
"""Audit formal model traces for backend, parse, and fallback integrity."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = REPO_ROOT / "runs" / "formal"


def parse_seed(name: str) -> int:
    match = re.search(r"seed(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse seed from {name!r}")
    return int(match.group(1))


def audit_trace(run_dir: Path) -> dict[str, int | float | str]:
    trace_path = run_dir / "model_decisions.jsonl"
    if not trace_path.is_file():
        trace_path = run_dir / "model_decisions.jsonl.gz"
    if not trace_path.is_file():
        raise FileNotFoundError(f"Missing model-decision trace under {run_dir}")
    decision_types: Counter[str] = Counter()
    backends: Counter[str] = Counter()
    models: Counter[str] = Counter()
    records = 0
    invalid_json = 0
    missing_parsed = 0
    missing_raw_response = 0
    parse_recovery_records = 0
    other_fallback_records = 0
    non_vllm_backend_records = 0

    opener = gzip.open if trace_path.suffix == ".gz" else io.open
    with opener(trace_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            decision_types[str(row.get("decision_type", ""))] += 1
            backends[str(row.get("backend", ""))] += 1
            models[str(row.get("model", ""))] += 1
            if not isinstance(row.get("parsed"), dict):
                missing_parsed += 1
            if not str(row.get("raw_response", "")).strip():
                missing_raw_response += 1
            fallback_reason = str(row.get("fallback_reason", "")).strip()
            if fallback_reason:
                if str(row.get("reason", "")).startswith("model_parse_fallback:"):
                    parse_recovery_records += 1
                else:
                    other_fallback_records += 1
            if row.get("backend") != "vllm_openai":
                non_vllm_backend_records += 1

    arm = run_dir.name.rsplit("_", 1)[-1]
    return {
        "run": run_dir.name,
        "seed": parse_seed(run_dir.name),
        "arm": arm,
        "records": records,
        "invalid_json": invalid_json,
        "missing_parsed": missing_parsed,
        "missing_raw_response": missing_raw_response,
        "parse_recovery_records": parse_recovery_records,
        "other_fallback_records": other_fallback_records,
        "non_vllm_backend_records": non_vllm_backend_records,
        "vllm_backend_pct": backends["vllm_openai"] / records * 100.0 if records else 0.0,
        "models": ";".join(f"{key}:{value}" for key, value in sorted(models.items())),
        "decision_types": ";".join(f"{key}:{value}" for key, value in sorted(decision_types.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, default=FORMAL_DIR)
    parser.add_argument("--check", action="store_true", help="Fail on malformed trace files or undocumented fallback records.")
    parser.add_argument(
        "--max-parse-recoveries",
        type=int,
        default=0,
        help="Maximum accepted conservative no-change recoveries across the audited traces.",
    )
    args = parser.parse_args()

    formal_dir = args.formal_dir.resolve()
    if not formal_dir.is_dir():
        print("No local formal traces found; trace audit skipped.")
        return
    rows = [audit_trace(run_dir) for run_dir in sorted(formal_dir.glob("seed*_*"))]
    if not rows:
        raise RuntimeError(f"No formal traces found under {formal_dir}")

    output = formal_dir / "trace_audit.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    if args.check:
        failures = [
            row["run"]
            for row in rows
            if row["invalid_json"] or row["other_fallback_records"] or row["non_vllm_backend_records"]
        ]
        if failures:
            raise RuntimeError("Trace audit failed for: " + ", ".join(failures))
        recoveries = sum(int(row["parse_recovery_records"]) for row in rows)
        missing_parsed = sum(int(row["missing_parsed"]) for row in rows)
        missing_raw = sum(int(row["missing_raw_response"]) for row in rows)
        if recoveries > args.max_parse_recoveries or missing_parsed != 0 or missing_raw != recoveries:
            raise RuntimeError(
                "Unexpected parse-recovery profile: "
                f"recoveries={recoveries}, missing_parsed={missing_parsed}, missing_raw_response={missing_raw}"
            )

    print(f"Audited {sum(int(row['records']) for row in rows):,} model decisions across {len(rows)} runs")
    print(f"Conservative parse recoveries: {sum(int(row['parse_recovery_records']) for row in rows)}")
    print(f"Other fallback records: {sum(int(row['other_fallback_records']) for row in rows)}")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
