#!/usr/bin/env python3
"""Recompute compact result tables from locally generated simulation runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAL_DIR = REPO_ROOT / "runs" / "formal"
RULE_DIR = REPO_ROOT / "runs" / "rule_abm"
DAYS_PER_ROUND = 3


def parse_seed(name: str) -> int:
    match = re.search(r"seed(\d+)", name)
    if not match:
        raise ValueError(f"Cannot parse seed from {name!r}")
    return int(match.group(1))


def capital_index(macro: pd.DataFrame) -> float:
    baseline = float(macro.iloc[0]["social_total_capital"])
    return float(macro.iloc[-1]["social_total_capital"]) / baseline * 100.0


def trapezoid(values: pd.Series, coordinates: pd.Series) -> float:
    y = values.to_numpy(dtype=float)
    x = coordinates.to_numpy(dtype=float)
    return float(np.sum((y[:-1] + y[1:]) * np.diff(x) / 2.0))


def summarize_macro(run_dir: Path, arm: str, model: str) -> dict[str, float | int | str]:
    macro = pd.read_csv(run_dir / "macro_daily.csv").sort_values("day")
    final = macro.iloc[-1]
    rounds = int(macro["day"].nunique())
    return {
        "run": run_dir.name,
        "seed": parse_seed(run_dir.name),
        "arm": arm,
        "decision_layer": model,
        "rounds": rounds,
        "calendar_days": rounds * DAYS_PER_ROUND,
        "active_firms": int(final["active_firms"]),
        "bankruptcies": int(final["cumulative_bankruptcies"]),
        "ai_adoption_pct": float(final["ai_penetration"]) * 100.0,
        "insurance_coverage_pct": float(final["insurance_coverage_overall"]) * 100.0,
        "insurance_coverage_ai_adopters_pct": float(final["insurance_coverage_ai_adopters"]) * 100.0,
        "social_capital_index": capital_index(macro),
        "final_mean_panic_x1e3": float(final["avg_panic"]) * 1_000.0,
        "final_p99_panic_x1e3": float(final["panic_p99"]) * 1_000.0,
        "cumulative_premiums": float(macro["total_premiums"].sum()),
        "cumulative_paid_claims": float(macro["total_claims"].sum()),
        "cumulative_unabsorbed_claimable_loss": float(macro["unabsorbed_claimable_loss"].sum()),
        "cumulative_uninsured_claimable_events": int(macro["num_uninsured_claimable_events"].sum()),
    }


def summarize_formal(formal_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    macros: dict[tuple[int, str], pd.DataFrame] = {}
    for run_dir in sorted(formal_dir.glob("seed*_*")):
        arm = run_dir.name.rsplit("_", 1)[-1]
        if arm not in {"on", "off"}:
            continue
        row = summarize_macro(run_dir, arm=arm, model="Qwen3-8B")
        seed = int(row["seed"])
        rows.append(row)
        macros[(seed, arm)] = pd.read_csv(run_dir / "macro_daily.csv").sort_values("day")

    summary = pd.DataFrame(rows).sort_values(["seed", "arm"]).reset_index(drop=True)
    seeds_on = set(summary.loc[summary["arm"].eq("on"), "seed"])
    seeds_off = set(summary.loc[summary["arm"].eq("off"), "seed"])
    if not seeds_on or seeds_on != seeds_off:
        raise RuntimeError(f"Formal runs are not paired: on={sorted(seeds_on)}, off={sorted(seeds_off)}")

    pair_rows = []
    indexed = summary.set_index(["seed", "arm"])
    for seed in sorted(seeds_on):
        on = indexed.loc[(seed, "on")]
        off = indexed.loc[(seed, "off")]
        on_macro = macros[(seed, "on")]
        off_macro = macros[(seed, "off")]
        on_unabsorbed = float(on["cumulative_unabsorbed_claimable_loss"])
        off_unabsorbed = float(off["cumulative_unabsorbed_claimable_loss"])
        pair_rows.append(
            {
                "seed": seed,
                "ai_adoption_gap_pp": float(on["ai_adoption_pct"] - off["ai_adoption_pct"]),
                "mean_daily_ai_gap_pp": float(
                    ((on_macro["ai_penetration"] - off_macro["ai_penetration"]) * 100.0).mean()
                ),
                "ai_auc_gap_pp_rounds": trapezoid(
                    (on_macro["ai_penetration"] - off_macro["ai_penetration"]) * 100.0,
                    on_macro["day"],
                ),
                "social_capital_gap_pp": float(on["social_capital_index"] - off["social_capital_index"]),
                "bankruptcy_reduction": int(off["bankruptcies"] - on["bankruptcies"]),
                "p99_panic_reduction_x1e3": float(
                    off["final_p99_panic_x1e3"] - on["final_p99_panic_x1e3"]
                ),
                "unabsorbed_loss_reduction_pct": (
                    (off_unabsorbed - on_unabsorbed) / off_unabsorbed * 100.0 if off_unabsorbed else np.nan
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    return summary, pairs


def summarize_rule_abm(rule_dir: Path) -> pd.DataFrame:
    rows = [
        summarize_macro(run_dir, arm="on", model="rule_heuristic")
        for run_dir in sorted(rule_dir.glob("seed*"))
        if run_dir.is_dir()
    ]
    if not rows:
        raise RuntimeError(f"No rule-based ABM runs found under {rule_dir}")
    return pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)


def validate_run_integrity(summary: pd.DataFrame, pairs: pd.DataFrame) -> None:
    expected_seeds = {42, 77, 202}
    if set(pairs["seed"]) != expected_seeds:
        raise RuntimeError(f"Expected paired seeds {sorted(expected_seeds)}, found {sorted(pairs['seed'])}")
    numeric_summary = summary.select_dtypes(include=[np.number])
    numeric_pairs = pairs.select_dtypes(include=[np.number])
    pair_values = numeric_pairs.to_numpy(dtype=float)
    finite_pair_values = pair_values[~np.isnan(pair_values)]
    checks = {
        "100 operational rounds per run": summary["rounds"].eq(100).all(),
        "unique seed-arm runs": not summary.duplicated(["seed", "arm"]).any(),
        "finite summary values": np.isfinite(numeric_summary.to_numpy(dtype=float)).all(),
        "finite paired values": np.isfinite(finite_pair_values).all(),
        "valid percentages": summary["ai_adoption_pct"].between(0.0, 100.0).all()
        and summary["insurance_coverage_pct"].between(0.0, 100.0).all(),
        "nonnegative capital index": summary["social_capital_index"].ge(0.0).all(),
    }
    failures = [name for name, passed in checks.items() if not passed]
    if failures:
        raise RuntimeError("Run-integrity validation failed: " + "; ".join(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if pairing, dimensions, or numeric ranges are invalid.")
    parser.add_argument("--formal-dir", type=Path, default=FORMAL_DIR, help="Directory containing seed<seed>_<arm> runs.")
    parser.add_argument("--rule-dir", type=Path, default=RULE_DIR, help="Directory containing seed<seed> rule runs.")
    args = parser.parse_args()

    formal_dir = args.formal_dir.resolve()
    rule_dir = args.rule_dir.resolve()
    if not formal_dir.is_dir() and not rule_dir.is_dir():
        print("No local run outputs found; result-integrity check skipped.")
        return
    summary, pairs = summarize_formal(formal_dir)
    rule = summarize_rule_abm(rule_dir)
    if args.check:
        validate_run_integrity(summary, pairs)

    summary.to_csv(formal_dir / "summary.csv", index=False, float_format="%.8f")
    pairs.to_csv(formal_dir / "paired_summary.csv", index=False, float_format="%.8f")
    rule.to_csv(rule_dir / "summary.csv", index=False, float_format="%.8f")

    means = pairs.mean(numeric_only=True)
    print(f"Formal pairs: {len(pairs)} seeds")
    print(f"Mean AI-adoption gap: {means['ai_adoption_gap_pp']:.2f} pp")
    print(f"Mean social-capital gap: {means['social_capital_gap_pp']:.3f} pp")
    print(f"Mean bankruptcy reduction: {means['bankruptcy_reduction']:.2f} firms")
    print(f"Rule-ABM final adoption: {rule['ai_adoption_pct'].mean():.2f}%")
    print(f"Rule-ABM insurance coverage: {rule['insurance_coverage_pct'].mean():.2f}%")


if __name__ == "__main__":
    main()
