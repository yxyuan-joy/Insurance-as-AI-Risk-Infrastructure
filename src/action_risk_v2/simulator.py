from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import pickle
import random
import shutil
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from .data import ActionRiskPanel
from .decisions import MarketContext, build_decision_policy
from .insurers import InsuranceMarket, load_insurer_profiles, load_vendor_profiles
from .negotiation import NegotiationEngine
from .schema import FirmState, InsurancePolicy, InsuranceQuote, VendorContract, VendorProfile


class StreamingCsv:
    def __init__(self, path: Path, fieldnames: Iterable[str], append: bool = False):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not append or not self.path.exists() or self.path.stat().st_size == 0
        self.file = self.path.open("a" if append else "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames, extrasaction="ignore")
        if write_header:
            self.writer.writeheader()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class JsonlWriter:
    def __init__(self, path: Path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("a" if append else "w", encoding="utf-8")

    def write(self, row: dict) -> None:
        self.file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class ActionRiskSimulator:
    def __init__(
        self,
        config: dict,
        run_dir: Path,
        days: Optional[int] = None,
        firms: Optional[int] = None,
        resume_from: Optional[Path] = None,
    ):
        self.config = dict(config)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.resume_from = Path(resume_from) if resume_from else None
        self.resume_day: Optional[int] = None
        self.cumulative_bankruptcies = 0
        self.prev_capital: Optional[float] = None
        self.insurance_enabled = bool(self.config.get("simulation", {}).get("enable_insurance_market", True))

        self.seed = int(self.config.get("simulation", {}).get("seed", 42))
        self.rng = random.Random(self.seed)
        decision_policy_config = dict(self.config.get("decision_policy", {}) or {})
        if bool(decision_policy_config.get("common_random_numbers", True)):
            decision_policy_config.setdefault("common_random_numbers", True)
            decision_policy_config.setdefault("common_random_seed", self.seed)
        self.config["decision_policy"] = decision_policy_config

        paths = self.config["paths"]
        self.panel = ActionRiskPanel.from_files(
            action_risk_path=Path(paths["action_risk_path"]),
            buyer_population_path=Path(paths["buyer_population_path"]) if paths.get("buyer_population_path") else None,
            selected_firms_path=Path(paths["selected_firms_path"]) if paths.get("selected_firms_path") else None,
            real_firms_path=Path(paths["real_firms_path"]) if paths.get("real_firms_path") else None,
        )

        self.days = self._select_days(days)
        firm_ids = self.panel.firm_ids(limit=firms)
        initial_cash_multiplier = max(
            0.0,
            float(self.config.get("simulation", {}).get("initial_cash_multiplier", 1.0)),
        )
        self.firms: Dict[str, FirmState] = {}
        for fid in firm_ids:
            profile = self.panel.profile_for(fid)
            self.firms[fid] = FirmState(profile=profile, cash=float(profile.cash) * initial_cash_multiplier)
        self.firm_network = self._build_firm_network()

        self.vendors = {v.vendor_id: v for v in load_vendor_profiles(self.config.get("vendors", []))}
        self.vendor_capital = {vid: float(v.subscription_fee) * 100.0 for vid, v in self.vendors.items()}
        self.insurance_market = InsuranceMarket(
            load_insurer_profiles(self.config.get("insurers", [])),
            pricing_config=_insurance_pricing_config(self.config),
        )
        self.policy = build_decision_policy(self.config, self.rng)
        self.negotiator = NegotiationEngine(self.config, self.rng)
        self.recent_claim_flags: List[int] = []
        self.recent_claim_rates: List[float] = []
        self.recent_material_rates: List[float] = []
        self._snapshot_cache = {}

        if self.resume_from is not None:
            self._load_checkpoint(self.resume_from)

        self._write_input_audit()
        self._write_run_metadata()
        append_logs = self.resume_day is not None
        if append_logs:
            self._prepare_resume_logs(int(self.resume_day))
        self.logs = self._open_logs(append=append_logs)

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        run_dir: Path,
        days: Optional[int] = None,
        firms: Optional[int] = None,
        resume_from: Optional[Path] = None,
        decision_mode: Optional[str] = None,
        vllm_base_url: Optional[str] = None,
        vllm_base_urls: Optional[str] = None,
        vllm_model: Optional[str] = None,
        model_fallback_to_rule: Optional[bool] = None,
        insurance_market_enabled: Optional[bool] = None,
        seed: Optional[int] = None,
        path_overrides: Optional[Dict[str, Optional[str]]] = None,
    ):
        config_path = Path(config_path)
        config = _load_yaml_config(config_path)
        root = _find_repository_root(config_path)
        config = _resolve_paths(config, root=root)
        config = _apply_path_overrides(config, root=root, path_overrides=path_overrides)
        config = _apply_decision_overrides(
            config,
            decision_mode=decision_mode,
            vllm_base_url=vllm_base_url,
            vllm_base_urls=vllm_base_urls,
            vllm_model=vllm_model,
            model_fallback_to_rule=model_fallback_to_rule,
            insurance_market_enabled=insurance_market_enabled,
            seed=seed,
        )
        return cls(config=config, run_dir=run_dir, days=days, firms=firms, resume_from=resume_from)

    def _select_days(self, limit: Optional[int]) -> List[int]:
        available = self.panel.days
        max_days = int(self.config.get("simulation", {}).get("days", len(available)))
        if limit is not None:
            max_days = int(limit)
        return available[:max_days]

    def _write_input_audit(self) -> None:
        df = self.panel.records.copy()
        firm_ids = set(self.firms.keys())
        days = set(int(d) for d in self.days)
        df = df[df["firm_id"].isin(firm_ids) & df["day"].isin(days)].copy()

        expected_grid = len(firm_ids) * len(days)
        numeric_cols = [
            "num_tasks",
            "incident_any_flag",
            "incident_task_count",
            "avg_severity",
            "sum_total_loss",
            "avg_risk_score",
            "max_risk_score",
        ]
        quantiles = {}
        for col in numeric_cols:
            if col not in df.columns or df.empty:
                continue
            q = df[col].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).to_dict()
            quantiles[col] = {str(k): float(v) for k, v in q.items()}

        overall = {
            "input_records_used": int(len(df)),
            "expected_firm_day_grid": int(expected_grid),
            "missing_firm_days_filled_at_runtime": int(max(0, expected_grid - len(df))),
            "unique_firms": int(df["firm_id"].nunique()) if not df.empty else 0,
            "unique_days": int(df["day"].nunique()) if not df.empty else 0,
            "incident_any_rate": float(df["incident_any_flag"].mean()) if not df.empty else 0.0,
            "avg_tasks_per_observed_firm_day": float(df["num_tasks"].mean()) if not df.empty else 0.0,
            "quantiles": quantiles,
            "interpretation_note": (
                "incident_any_flag is audited as task-level failure evidence. "
                "The simulator maps it into material and claimable events before contagion and claims."
            ),
        }
        with (self.run_dir / "action_risk_input_audit.json").open("w", encoding="utf-8") as f:
            json.dump(overall, f, ensure_ascii=False, indent=2, sort_keys=True)

        rows = []
        if not df.empty:
            for industry, group in df.groupby("industry", sort=True):
                rows.append(
                    {
                        "industry": str(industry),
                        "rows": int(len(group)),
                        "firms": int(group["firm_id"].nunique()),
                        "incident_any_rate": float(group["incident_any_flag"].mean()),
                        "avg_tasks": float(group["num_tasks"].mean()),
                        "avg_total_loss": float(group["sum_total_loss"].mean()),
                        "p95_total_loss": float(group["sum_total_loss"].quantile(0.95)),
                        "avg_risk_score": float(group["avg_risk_score"].mean()),
                        "p95_max_risk_score": float(group["max_risk_score"].quantile(0.95)),
                    }
                )
        with (self.run_dir / "industry_risk_audit.csv").open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "industry",
                "rows",
                "firms",
                "incident_any_rate",
                "avg_tasks",
                "avg_total_loss",
                "p95_total_loss",
                "avg_risk_score",
                "p95_max_risk_score",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_run_metadata(self) -> None:
        with (self.run_dir / "config_snapshot.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f, allow_unicode=True, sort_keys=False)
        meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "num_firms": len(self.firms),
            "num_days": len(self.days),
            "industries": self.panel.industries,
            "input_records": int(len(self.panel.records)),
            "vendor_ids": sorted(self.vendors),
            "insurer_ids": sorted(self.insurance_market.insurers),
            "insurer_roles": {
                iid: state.profile.market_role
                for iid, state in sorted(self.insurance_market.insurers.items())
            },
            "paths": dict(self.config.get("paths", {})),
            "decision_layer": dict(self.config.get("decision_layer", {})),
            "insurance_market_enabled": bool(self.insurance_enabled),
            "resume": {
                "enabled": self.resume_from is not None,
                "checkpoint_path": str(self.resume_from) if self.resume_from else "",
                "resume_day": int(self.resume_day) if self.resume_day is not None else None,
                "remaining_days": len(self.days),
            },
        }
        with (self.run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _open_logs(self, append: bool = False) -> Dict[str, object]:
        return {
            "macro": StreamingCsv(
                self.run_dir / "macro_daily.csv",
                [
                    "day",
                    "active_firms",
                    "ai_penetration",
                    "insurance_coverage_overall",
                    "insurance_coverage_ai_adopters",
                    "total_claims",
                    "total_premiums",
                    "total_vendor_fees",
                    "total_insurance_refunds",
                    "total_vendor_refunds",
                    "cumulative_bankruptcies",
                    "avg_panic",
                    "panic_p50",
                    "panic_p75",
                    "panic_p90",
                    "panic_p95",
                    "panic_p99",
                    "panic_nonzero_share",
                    "active_firm_cash",
                    "inactive_firm_cash",
                    "all_firm_cash",
                    "total_insurer_capital",
                    "total_vendor_capital",
                    "social_total_capital_active_firms",
                    "social_total_capital_all_firms",
                    "social_total_capital",
                    "delta_social_capital",
                    "max_vendor_share",
                    "num_claim_events",
                    "num_uninsured_claimable_events",
                    "uninsured_claimable_loss",
                    "claim_residual_loss",
                    "unabsorbed_claimable_loss",
                    "indemnity_relief_ratio",
                    "num_task_failures",
                    "num_material_events",
                    "num_claimable_events",
                    "avg_material_event_score",
                    "avg_claimable_event_score",
                ],
                append=append,
            ),
            "firm": StreamingCsv(
                self.run_dir / "firm_daily.csv",
                [
                    "day",
                    "firm_id",
                    "industry",
                    "active",
                    "has_ai",
                    "had_ai_during_day",
                    "vendor_id",
                    "vendor_id_during_day",
                    "vendor_monthly_fee",
                    "vendor_contract_total_fee",
                    "vendor_term_end",
                    "has_insurance",
                    "had_insurance_during_day",
                    "insurer_id",
                    "insurer_id_during_day",
                    "insurance_term_end",
                    "cash_start",
                    "cash_end",
                    "base_pnl",
                    "ai_gain",
                    "operational_loss",
                    "claim_paid",
                    "claim_residual_loss",
                    "unabsorbed_claimable_loss",
                    "claim_paid_flag",
                    "uninsured_claimable_flag",
                    "premium_paid",
                    "vendor_fee_paid",
                    "insurance_refund_received",
                    "vendor_refund_received",
                    "incident_score",
                    "task_failure_rate",
                    "material_event",
                    "claimable_event",
                    "material_event_score",
                    "claimable_event_score",
                    "loss_ratio",
                    "industry_incident_rate",
                    "industry_stress_score",
                    "industry_loss_pressure",
                    "prior_policy",
                    "action_num_tasks",
                    "action_incident_any",
                    "action_incident_task_count",
                    "action_avg_severity",
                    "action_avg_risk_score",
                    "action_max_risk_score",
                    "risk_memory",
                    "loss_memory",
                    "claimable_memory",
                    "panic",
                    "local_adoption_rate",
                    "local_insurance_coverage_rate",
                    "local_avg_panic",
                    "local_recent_claim_rate",
                    "network_neighbor_count",
                    "same_industry_neighbor_share",
                    "state_end",
                ],
                append=append,
            ),
            "insurer": StreamingCsv(
                self.run_dir / "insurer_daily.csv",
                [
                    "day",
                    "insurer_id",
                        "label",
                        "domicile",
                        "market_role",
                        "capital",
                    "capital_ratio",
                    "regime",
                    "underwriting_open",
                    "premiums_today",
                    "claims_today",
                    "refunds_today",
                    "active_policies",
                    "new_policies_today",
                ],
                append=append,
            ),
            "quotes": StreamingCsv(
                self.run_dir / "quotes.csv",
                [
                    "day",
                    "firm_id",
                    "industry",
                    "vendor_id",
                    "insurer_id",
                    "term_days",
                    "premium",
                    "deductible_ratio",
                    "coverage_ratio",
                    "limit_money",
                    "incident_threshold",
                    "expected_loss",
                    "stress_loss",
                    "regime",
                    "market_role",
                    "utility",
                    "selection_reason",
                    "selected",
                ],
                append=append,
            ),
            "decisions": StreamingCsv(
                self.run_dir / "decisions.csv",
                [
                    "day",
                    "firm_id",
                    "decision_type",
                    "action",
                    "score",
                    "threshold",
                    "probability",
                    "draw",
                    "reason",
                    "selected_id",
                    "cash",
                    "risk_need",
                    "decision_selected_vendor_id",
                    "vendor_term_days",
                    "insurance_term_days",
                    "max_rounds",
                    "visible_vendor_ids",
                    "visible_vendor_count",
                    "local_adoption_rate",
                    "local_insurance_coverage_rate",
                    "local_avg_panic",
                    "local_recent_claim_rate",
                    "network_neighbor_count",
                    "same_industry_neighbor_share",
                ],
                append=append,
            ),
            "model_decisions": JsonlWriter(self.run_dir / "model_decisions.jsonl", append=append),
            "interactions": JsonlWriter(self.run_dir / "interactions.jsonl", append=append),
            "events": JsonlWriter(self.run_dir / "events.jsonl", append=append),
        }

    def close(self) -> None:
        for obj in self.logs.values():
            obj.close()

    def run(self) -> None:
        prev_capital = self.prev_capital
        cumulative_bankruptcies = int(self.cumulative_bankruptcies)

        try:
            for day in self.days:
                self._reset_daily_cashflow_markers()
                self.insurance_market.start_day()
                self._expire_contracts(day)
                self._refresh_active_policy_counts()

                context = self._market_context(day)
                fees = self._make_contract_decisions(day, context)
                totals = self._operate_one_day(day, context, fees)
                cumulative_bankruptcies += int(totals["new_bankruptcies"])
                self._update_panic(totals)
                for row in totals.pop("_firm_log_rows", []):
                    self._log_firm_day(**row)

                macro = self._macro_row(day, cumulative_bankruptcies)
                macro["total_vendor_fees"] = fees["vendor_fees"]
                macro["total_premiums"] = fees["premiums"]
                macro["total_vendor_refunds"] = float(getattr(self, "_daily_vendor_refunds", 0.0))
                macro["total_insurance_refunds"] = float(getattr(self, "_daily_insurance_refunds", 0.0))
                macro["total_claims"] = totals["claims_paid"]
                macro["num_claim_events"] = int(totals["claim_events"])
                macro["num_uninsured_claimable_events"] = int(totals["uninsured_claimable_events"])
                macro["uninsured_claimable_loss"] = float(totals["uninsured_claimable_loss"])
                macro["claim_residual_loss"] = float(totals["claim_residual_loss"])
                macro["unabsorbed_claimable_loss"] = float(totals["unabsorbed_claimable_loss"])
                macro["indemnity_relief_ratio"] = float(totals["indemnity_relief_ratio"])
                macro["num_task_failures"] = int(totals["task_failures"])
                macro["num_material_events"] = int(totals["material_events"])
                macro["num_claimable_events"] = int(totals["claimable_events"])
                macro["avg_material_event_score"] = float(totals["avg_material_event_score"])
                macro["avg_claimable_event_score"] = float(totals["avg_claimable_event_score"])
                if prev_capital is None:
                    macro["delta_social_capital"] = 0.0
                else:
                    macro["delta_social_capital"] = float(macro["social_total_capital"] - prev_capital)
                prev_capital = float(macro["social_total_capital"])
                self.prev_capital = prev_capital
                self.cumulative_bankruptcies = cumulative_bankruptcies
                self.logs["macro"].write(macro)

                self._refresh_active_policy_counts()
                for row in self.insurance_market.daily_rows(day):
                    self.logs["insurer"].write(row)

                self._write_checkpoint(day)
        finally:
            self.close()

    def _reset_daily_cashflow_markers(self) -> None:
        self._daily_vendor_refunds = 0.0
        self._daily_insurance_refunds = 0.0
        for firm in self.firms.values():
            firm._premium_paid_today = 0.0
            firm._vendor_fee_paid_today = 0.0
            firm._insurance_refund_today = 0.0
            firm._vendor_refund_today = 0.0
            firm._material_event_score_today = 0.0
            firm._claimable_event_score_today = 0.0
            firm._claim_paid_today_flag = 0.0
            firm._uninsured_claimable_today_flag = 0.0

    def _expire_contracts(self, day: int) -> None:
        for firm in self.firms.values():
            firm._vendor_expired_today = False
            firm._insurance_expired_today = False
            firm._last_vendor_id = ""
            if firm.vendor_contract and int(firm.vendor_contract.end_day) <= int(day):
                firm._vendor_expired_today = True
                expired = firm.vendor_contract
                firm._last_vendor_id = expired.vendor_id
                self.logs["events"].write(
                    {
                        "event_type": "vendor_contract_expired",
                        "day": int(day),
                        "firm_id": firm.profile.firm_id,
                        "vendor_id": expired.vendor_id,
                        "start_day": int(expired.start_day),
                        "end_day": int(expired.end_day),
                        "contract_total_fee": float(expired.price),
                        "monthly_fee": float(getattr(expired, "monthly_fee", 0.0) or 0.0),
                    }
                )
                firm.vendor_contract = None
            if firm.insurance_policy and int(firm.insurance_policy.end_day) <= int(day):
                firm._insurance_expired_today = True
                expired_policy = firm.insurance_policy
                self.logs["events"].write(
                    {
                        "event_type": "insurance_policy_expired",
                        "day": int(day),
                        "firm_id": firm.profile.firm_id,
                        "insurer_id": expired_policy.insurer_id,
                        "start_day": int(expired_policy.start_day),
                        "end_day": int(expired_policy.end_day),
                        "premium": float(expired_policy.premium),
                    }
                )
                firm.insurance_policy = None
            # Do not cancel insurance immediately after a vendor contract expires.
            # The same daily step still gives the firm a renewal opportunity; if it
            # fails to renew, _make_contract_decisions cancels the policy afterward.

    def _contract_lifecycle_config(self) -> dict:
        return dict(self.config.get("contract_lifecycle", {}) or {})

    def _vendor_total_fee(self, monthly_fee: float, term_days: int) -> float:
        base_days = float(self._contract_lifecycle_config().get("vendor_fee_base_days", 30.0))
        base_days = max(1.0, base_days)
        return float(monthly_fee) * max(1.0, float(term_days)) / base_days

    def _cancel_vendor_contract(self, firm: FirmState, day: int, reason: str) -> float:
        contract = firm.vendor_contract
        if contract is None:
            return 0.0

        cfg = self._contract_lifecycle_config().get("vendor_refund", {}) or {}
        max_ratio = _clamp(float(cfg.get("max_refund_ratio", 0.70)), 0.0, 1.0)
        penalty = _clamp(float(cfg.get("refund_penalty_ratio", 0.10)), 0.0, 1.0)
        exponent = max(1.0, float(cfg.get("refund_non_linear", 2.0)))

        term_days = max(1, int(contract.end_day) - int(contract.start_day))
        remaining_days = max(0, int(contract.end_day) - int(day))
        remaining_ratio = _clamp(remaining_days / term_days, 0.0, 1.0)
        refund_ratio = _clamp(max_ratio * (remaining_ratio**exponent) - penalty, 0.0, 1.0)
        refund = max(0.0, float(contract.price) * refund_ratio)

        if refund > 0:
            firm.cash += refund
            self.vendor_capital[contract.vendor_id] = max(
                0.0,
                float(self.vendor_capital.get(contract.vendor_id, 0.0)) - float(refund),
            )
            firm._vendor_refund_today = float(getattr(firm, "_vendor_refund_today", 0.0)) + refund
            self._daily_vendor_refunds += refund

        self.logs["events"].write(
            {
                "event_type": "vendor_contract_cancelled",
                "day": int(day),
                "firm_id": firm.profile.firm_id,
                "vendor_id": contract.vendor_id,
                "reason": str(reason),
                "start_day": int(contract.start_day),
                "end_day": int(contract.end_day),
                "remaining_days": int(remaining_days),
                "contract_total_fee": float(contract.price),
                "monthly_fee": float(getattr(contract, "monthly_fee", 0.0) or 0.0),
                "refund": float(refund),
                "refund_ratio": float(refund_ratio),
            }
        )
        firm.vendor_contract = None
        return float(refund)

    def _cancel_insurance_policy(self, firm: FirmState, day: int, reason: str) -> float:
        policy = firm.insurance_policy
        if policy is None:
            return 0.0
        cfg = self._contract_lifecycle_config().get("insurance_refund", {}) or {}
        penalty = _clamp(float(cfg.get("refund_penalty_ratio", 0.06)), 0.0, 1.0)
        refund = self.insurance_market.cancel_policy(policy, day=day, refund_penalty_ratio=penalty)
        if refund > 0:
            firm.cash += refund
            firm._insurance_refund_today = float(getattr(firm, "_insurance_refund_today", 0.0)) + refund
            self._daily_insurance_refunds += refund
        self.logs["events"].write(
            {
                "event_type": "insurance_policy_cancelled",
                "day": int(day),
                "firm_id": firm.profile.firm_id,
                "insurer_id": policy.insurer_id,
                "vendor_id": str(getattr(policy, "vendor_id", "") or ""),
                "reason": str(reason),
                "start_day": int(policy.start_day),
                "end_day": int(policy.end_day),
                "remaining_days": int(max(0, int(policy.end_day) - int(day))),
                "premium": float(policy.premium),
                "refund": float(refund),
                "refund_penalty_ratio": float(penalty),
            }
        )
        firm.insurance_policy = None
        return float(refund)

    def _terminate_bankrupt_firm_contracts(self, firm: FirmState, day: int) -> None:
        contract = firm.vendor_contract
        policy = firm.insurance_policy
        if contract is not None:
            self.logs["events"].write(
                {
                    "event_type": "vendor_contract_terminated_bankruptcy",
                    "day": int(day),
                    "firm_id": firm.profile.firm_id,
                    "vendor_id": contract.vendor_id,
                    "start_day": int(contract.start_day),
                    "end_day": int(contract.end_day),
                    "contract_total_fee": float(contract.price),
                    "refund": 0.0,
                }
            )
            firm.vendor_contract = None
        if policy is not None:
            self.logs["events"].write(
                {
                    "event_type": "insurance_policy_terminated_bankruptcy",
                    "day": int(day),
                    "firm_id": firm.profile.firm_id,
                    "insurer_id": policy.insurer_id,
                    "vendor_id": str(getattr(policy, "vendor_id", "") or ""),
                    "start_day": int(policy.start_day),
                    "end_day": int(policy.end_day),
                    "premium": float(policy.premium),
                    "refund": 0.0,
                }
            )
            firm.insurance_policy = None

    def _apply_ai_abandon_decision(
        self,
        firm: FirmState,
        day: int,
        reason: str,
        score: float = 0.0,
        threshold: float = 0.0,
    ) -> None:
        contract = firm.vendor_contract
        if contract is None:
            return
        cfg = self._contract_lifecycle_config()
        old_vendor_id = contract.vendor_id
        vendor_refund = self._cancel_vendor_contract(firm, day=day, reason=reason)
        insurance_refund = 0.0
        if firm.insurance_policy is not None:
            insurance_refund = self._cancel_insurance_policy(
                firm=firm,
                day=day,
                reason="cancelled_after_ai_exit",
            )
        same_day_reentry = bool(cfg.get("same_day_reentry_after_abandon", False))
        cooldown = int(cfg.get("reentry_cooldown_days_after_abandon", 0))
        if not same_day_reentry:
            cooldown = max(1, cooldown)
        if cooldown > 0:
            firm.ai_cooldown_until = max(int(firm.ai_cooldown_until), int(day) + cooldown)
        self.logs["events"].write(
            {
                "event_type": "ai_contract_abandoned",
                "day": int(day),
                "firm_id": firm.profile.firm_id,
                "old_vendor_id": old_vendor_id,
                "reason": str(reason),
                "abandon_score": float(score),
                "threshold": float(threshold),
                "vendor_refund": float(vendor_refund),
                "insurance_refund": float(insurance_refund),
                "same_day_reentry_allowed": bool(same_day_reentry),
                "cooldown_until": int(firm.ai_cooldown_until),
            }
        )

    def _maybe_abandon_ai_contract(self, firm: FirmState, day: int, context: MarketContext) -> None:
        if not firm.has_ai:
            return
        cfg = self._contract_lifecycle_config()
        if not bool(cfg.get("allow_midterm_ai_abandon", True)):
            return
        contract = firm.vendor_contract
        if contract is None:
            return
        remaining_days = int(contract.end_day) - int(day)
        if remaining_days <= int(cfg.get("min_remaining_days_for_abandon", 7)):
            return

        score = (
            0.38 * float(firm.loss_memory)
            + 0.26 * float(firm.claimable_memory)
            + 0.22 * float(firm.panic)
            + 0.14 * float(firm.risk_memory)
            - 0.18 * float(firm.profile.risk_tolerance)
        )
        threshold = float(cfg.get("abandon_score_threshold", 0.42))
        has_recent_loss = float(firm.last_operational_loss) > 0
        if not (has_recent_loss and score >= threshold):
            return

        self._apply_ai_abandon_decision(
            firm=firm,
            day=day,
            reason="early_ai_exit_after_bad_experience",
            score=float(score),
            threshold=float(threshold),
        )

    def _market_context(self, day: int) -> MarketContext:
        active = [f for f in self.firms.values() if f.active]
        adopters = [f for f in active if f.has_ai]
        insured = [f for f in active if f.has_insurance]
        avg_panic = sum(f.panic for f in active) / len(active) if active else 0.0
        recent_claim_rate = sum(self.recent_claim_rates[-7:]) / max(1, min(7, len(self.recent_claim_rates)))
        return MarketContext(
            day=int(day),
            adoption_rate=len(adopters) / len(active) if active else 0.0,
            insurance_coverage_rate=len(insured) / len(adopters) if adopters else 0.0,
            avg_panic=float(avg_panic),
            recent_claim_rate=float(recent_claim_rate),
        )

    def _build_firm_network(self) -> Dict[str, List[str]]:
        cfg = dict(self.config.get("network", {}) or {})
        if not bool(cfg.get("enabled", True)):
            return {fid: [] for fid in self.firms}
        same_k = max(0, int(cfg.get("same_industry_neighbors", 8)))
        cross_k = max(0, int(cfg.get("cross_industry_neighbors", 3)))
        min_k = max(0, int(cfg.get("min_neighbors", min(6, same_k + cross_k))))
        seed = int(self.config.get("simulation", {}).get("seed", 42))
        firm_ids = sorted(self.firms)
        out: Dict[str, List[str]] = {}
        for fid in firm_ids:
            industry = self.firms[fid].profile.industry
            same = [x for x in firm_ids if x != fid and self.firms[x].profile.industry == industry]
            cross = [x for x in firm_ids if x != fid and self.firms[x].profile.industry != industry]
            same = sorted(same, key=lambda other: _stable_network_score(seed, fid, other, "same"))
            cross = sorted(cross, key=lambda other: _stable_network_score(seed, fid, other, "cross"))
            neighbors = list(dict.fromkeys(same[:same_k] + cross[:cross_k]))
            if len(neighbors) < min_k:
                all_ranked = sorted(
                    [x for x in firm_ids if x != fid and x not in set(neighbors)],
                    key=lambda other: _stable_network_score(seed, fid, other, "fill"),
                )
                neighbors.extend(all_ranked[: max(0, min_k - len(neighbors))])
            out[fid] = neighbors
        return out

    def _firm_market_context(self, firm: FirmState, base_context: MarketContext) -> MarketContext:
        return self._firm_market_context_from_state(
            firm=firm,
            base_context=base_context,
            state_snapshot=None,
        )

    def _decision_state_snapshot(self) -> Dict[str, dict]:
        return {
            fid: {
                "active": bool(firm.active),
                "has_ai": bool(firm.has_ai),
                "has_insurance": bool(firm.has_insurance),
                "panic": float(firm.panic),
                "last_claim_day": getattr(firm, "last_claim_day", None),
                "industry": firm.profile.industry,
            }
            for fid, firm in self.firms.items()
        }

    def _firm_market_context_from_state(
        self,
        firm: FirmState,
        base_context: MarketContext,
        state_snapshot: Optional[Dict[str, dict]] = None,
    ) -> MarketContext:
        neighbor_ids = [
            fid
            for fid in self.firm_network.get(firm.profile.firm_id, [])
            if fid in self.firms
            and (
                bool(state_snapshot[fid]["active"])
                if state_snapshot is not None and fid in state_snapshot
                else bool(self.firms[fid].active)
            )
        ]
        if not neighbor_ids:
            return MarketContext(
                day=base_context.day,
                adoption_rate=base_context.adoption_rate,
                insurance_coverage_rate=base_context.insurance_coverage_rate,
                avg_panic=base_context.avg_panic,
                recent_claim_rate=base_context.recent_claim_rate,
                local_adoption_rate=base_context.adoption_rate,
                local_insurance_coverage_rate=base_context.insurance_coverage_rate,
                local_avg_panic=base_context.avg_panic,
                local_recent_claim_rate=base_context.recent_claim_rate,
                network_neighbor_count=0,
                same_industry_neighbor_share=0.0,
            )
        if state_snapshot is None:
            adopters = [fid for fid in neighbor_ids if self.firms[fid].has_ai]
            insured = [fid for fid in neighbor_ids if self.firms[fid].has_insurance]
            panic_values = [float(self.firms[fid].panic) for fid in neighbor_ids]
            last_claim_days = {fid: getattr(self.firms[fid], "last_claim_day", None) for fid in neighbor_ids}
            industries = {fid: self.firms[fid].profile.industry for fid in neighbor_ids}
        else:
            adopters = [fid for fid in neighbor_ids if bool(state_snapshot[fid]["has_ai"])]
            insured = [fid for fid in neighbor_ids if bool(state_snapshot[fid]["has_insurance"])]
            panic_values = [float(state_snapshot[fid]["panic"]) for fid in neighbor_ids]
            last_claim_days = {fid: state_snapshot[fid].get("last_claim_day") for fid in neighbor_ids}
            industries = {fid: str(state_snapshot[fid]["industry"]) for fid in neighbor_ids}
        recent_window = int((self.config.get("network", {}) or {}).get("recent_claim_window_days", 7))
        recent_claims = [
            fid for fid in neighbor_ids
            if last_claim_days.get(fid) is not None
            and int(base_context.day) - int(last_claim_days[fid]) < max(1, recent_window)
        ]
        same_industry = [fid for fid in neighbor_ids if industries[fid] == firm.profile.industry]
        insured_set = set(insured)
        return MarketContext(
            day=base_context.day,
            adoption_rate=base_context.adoption_rate,
            insurance_coverage_rate=base_context.insurance_coverage_rate,
            avg_panic=base_context.avg_panic,
            recent_claim_rate=base_context.recent_claim_rate,
            local_adoption_rate=len(adopters) / len(neighbor_ids),
            local_insurance_coverage_rate=len([fid for fid in adopters if fid in insured_set]) / len(adopters) if adopters else 0.0,
            local_avg_panic=sum(panic_values) / len(neighbor_ids),
            local_recent_claim_rate=len(recent_claims) / len(neighbor_ids),
            network_neighbor_count=len(neighbor_ids),
            same_industry_neighbor_share=len(same_industry) / len(neighbor_ids),
        )

    def _industry_snapshot(self, industry: str, day: int):
        key = (str(industry), int(day))
        if key not in self._snapshot_cache:
            self._snapshot_cache[key] = self.panel.industry_snapshot(str(industry), int(day))
        return self._snapshot_cache[key]

    def _pre_operation_risk_signal(
        self,
        firm: FirmState,
        vendor: VendorProfile,
        day: int,
        ai_remaining_days: int,
        snapshot=None,
    ) -> dict:
        """Build the information set visible before the current day's operation.

        Contract, exposure, and insurance-purchase decisions happen before
        `_operate_one_day()`. They must not observe same-day accident realizations
        such as operational_loss or claimable_event_score.
        """
        if snapshot is None:
            snapshot = self._industry_snapshot(firm.profile.industry, day)
        signal = {
            "task_failure_rate": 0.0,
            "risk_excess": 0.0,
            "operational_loss": 0.0,
            "loss_ratio": 0.0,
            "loss_pressure": 0.0,
            "material_event_score": 0.0,
            "claimable_event_score": 0.0,
            "material_event": False,
            "claimable_event": False,
            "raw_incident_any": False,
            "pre_operation_signal": True,
            "same_day_incident_observed": False,
            "risk_memory": _clamp(float(firm.risk_memory), 0.0, 1.0),
            "loss_memory": _clamp(float(firm.loss_memory), 0.0, 1.0),
            "claimable_memory": _clamp(float(firm.claimable_memory), 0.0, 1.0),
            "lagged_material_pressure": _clamp(float(firm.risk_memory), 0.0, 1.0),
            "lagged_loss_pressure": _clamp(float(firm.loss_memory), 0.0, 1.0),
            "lagged_claimable_pressure": _clamp(float(firm.claimable_memory), 0.0, 1.0),
            "last_operational_loss": max(0.0, float(firm.last_operational_loss)),
            "panic": _clamp(float(firm.panic), 0.0, 1.0),
        }
        signal.update(_industry_risk_features(snapshot, vendor, firm, self.config))
        signal["prior_policy"] = 1.0 if bool(getattr(firm, "_insurance_expired_today", False)) else 0.0
        signal["ai_remaining_days"] = int(ai_remaining_days)
        return signal

    def _make_contract_decisions(self, day: int, context: MarketContext) -> dict:
        total_vendor_fees = 0.0
        total_premiums = 0.0
        vendors = list(self.vendors.values())
        decision_state = self._decision_state_snapshot()
        self._decision_context_by_firm = {}

        for firm in self.firms.values():
            if not firm.active:
                continue
            firm._decision_day = int(day)
            firm_context = self._firm_market_context_from_state(firm, context, decision_state)
            self._decision_context_by_firm[firm.profile.firm_id] = firm_context

            premium_paid = 0.0
            vendor_fee_paid = 0.0
            if not _uses_model_decision_layer(self.config):
                self._maybe_abandon_ai_contract(firm, day=day, context=firm_context)
            visible = self.policy.visible_vendors(firm, vendors)
            renewal = bool(getattr(firm, "_vendor_expired_today", False))
            if renewal:
                visible = _with_incumbent_renewal_option(visible, self.vendors, getattr(firm, "_last_vendor_id", ""))

            lifecycle_cfg = self._contract_lifecycle_config()
            if firm.has_ai:
                vendor = self.vendors[firm.vendor_contract.vendor_id]
                ai_remaining_days = max(0, int(firm.vendor_contract.end_day) - int(day))
                risk_signal = self._pre_operation_risk_signal(
                    firm=firm,
                    vendor=vendor,
                    day=day,
                    ai_remaining_days=ai_remaining_days,
                )
                exposure_decision = self.policy.exposure_decision(firm, firm_context, risk_signal)
                self._log_decision(
                    day=day,
                    firm=firm,
                    decision_type="ai_exposure_management",
                    decision=exposure_decision,
                    risk_need=float(exposure_decision.get("score", 0.0)),
                    visible_vendors=visible,
                    context=firm_context,
                )
                if str(exposure_decision.get("vendor_action", "")).lower().strip() == "abandon_ai" or bool(exposure_decision.get("action", False)):
                    self._apply_ai_abandon_decision(
                        firm=firm,
                        day=day,
                        reason="model_exposure_management",
                        score=float(exposure_decision.get("score", 0.0) or 0.0),
                        threshold=float(exposure_decision.get("threshold", 0.0) or 0.0),
                    )

            adopt_decision = dict(self.policy.adoption_decision(firm, firm_context, visible, renewal=renewal))
            selected_vendor = None
            if not firm.has_ai and bool(adopt_decision.get("action", False)):
                selected_vendor = _selected_visible_vendor(adopt_decision, visible)
                if selected_vendor is None:
                    if _is_model_decision(adopt_decision):
                        adopt_decision = self._invalidate_model_decision(
                            day=day,
                            firm=firm,
                            decision=adopt_decision,
                            decision_type="vendor_renewal" if renewal else "ai_adoption",
                            reason="missing_or_non_visible_selected_vendor_id",
                        )
                    else:
                        selected_vendor = self.policy.choose_vendor(firm, visible)
                elif _is_model_decision(adopt_decision) and not _decision_has_positive_term(
                    adopt_decision,
                    keys=("vendor_term_days", "term_days"),
                ):
                    adopt_decision = self._invalidate_model_decision(
                        day=day,
                        firm=firm,
                        decision=adopt_decision,
                        decision_type="vendor_renewal" if renewal else "ai_adoption",
                        reason="missing_positive_vendor_term_days",
                    )
                    selected_vendor = None
            if not firm.has_ai and bool(adopt_decision.get("action", False)) and selected_vendor is not None:
                adopt_decision = _apply_initial_adoption_trial_cap(
                    adopt_decision,
                    renewal=renewal,
                    lifecycle_config=lifecycle_cfg,
                )

            self._log_decision(
                day=day,
                firm=firm,
                decision_type="vendor_renewal" if renewal else "ai_adoption",
                decision=adopt_decision,
                selected_id=selected_vendor.vendor_id if selected_vendor is not None else "",
                visible_vendors=visible,
                context=firm_context,
            )

            if not firm.has_ai and bool(adopt_decision.get("action", False)):
                vendor = selected_vendor
                if vendor is not None:
                    term = _decision_term_days(
                        adopt_decision,
                        keys=("vendor_term_days", "term_days"),
                        fallback=lambda: self.policy.vendor_term_days(firm, firm_context),
                        min_days=int(lifecycle_cfg.get("vendor_min_term_days", 14)),
                        max_days=int(lifecycle_cfg.get("vendor_max_term_days", 120)),
                    )
                    negotiation = self.negotiator.negotiate_vendor(
                        firm=firm,
                        vendor=vendor,
                        context=firm_context,
                        day=day,
                        term_days=term,
                        max_rounds_override=_decision_max_rounds(adopt_decision),
                    )
                    self._log_interactions(negotiation.events)
                    monthly_fee = float(negotiation.final_price)
                    contract_total_fee = self._vendor_total_fee(monthly_fee, term)
                    if negotiation.agreed and firm.cash >= contract_total_fee:
                        firm.vendor_contract = VendorContract(
                            vendor_id=vendor.vendor_id,
                            price=contract_total_fee,
                            start_day=int(day),
                            end_day=int(day) + int(term),
                            monthly_fee=monthly_fee,
                        )
                        firm.cash -= contract_total_fee
                        self.vendor_capital[vendor.vendor_id] = (
                            float(self.vendor_capital.get(vendor.vendor_id, 0.0)) + float(contract_total_fee)
                        )
                        vendor_fee_paid = contract_total_fee
                        firm._vendor_fee_paid_today = vendor_fee_paid
                        total_vendor_fees += contract_total_fee
                        self.logs["events"].write(
                            {
                                "event_type": "vendor_contract_bound",
                                "day": int(day),
                                "firm_id": firm.profile.firm_id,
                                "vendor_id": vendor.vendor_id,
                                "term_days": int(term),
                                "monthly_fee": float(monthly_fee),
                                "price": float(contract_total_fee),
                                "contract_total_fee": float(contract_total_fee),
                                "list_price": float(vendor.subscription_fee),
                                "list_total_fee": float(self._vendor_total_fee(vendor.subscription_fee, term)),
                                "negotiation_rounds": int(negotiation.rounds),
                                "negotiation_outcome": negotiation.outcome,
                                "decision_reason": str(adopt_decision.get("reason", "")),
                            }
                        )
                    else:
                        self.logs["events"].write(
                            {
                                "event_type": "vendor_negotiation_failed",
                                "day": int(day),
                                "firm_id": firm.profile.firm_id,
                                "vendor_id": vendor.vendor_id,
                                "term_days": int(term),
                                "outcome": negotiation.outcome,
                                "cash": float(firm.cash),
                                "monthly_fee": float(monthly_fee),
                                "contract_total_fee": float(contract_total_fee),
                            }
                        )
                        if renewal:
                            failed_renewal_cooldown = int(
                                self._contract_lifecycle_config().get("reentry_cooldown_days_after_failed_renewal", 14)
                            )
                            firm.ai_cooldown_until = max(
                                int(firm.ai_cooldown_until),
                                int(day) + max(0, failed_renewal_cooldown),
                            )
                elif renewal:
                    failed_renewal_cooldown = int(
                        self._contract_lifecycle_config().get("reentry_cooldown_days_after_failed_renewal", 14)
                    )
                    firm.ai_cooldown_until = max(
                        int(firm.ai_cooldown_until),
                        int(day) + max(0, failed_renewal_cooldown),
                    )

            if not firm.has_ai and firm.insurance_policy is not None:
                self._cancel_insurance_policy(
                    firm=firm,
                    day=day,
                    reason="cancelled_no_active_ai_exposure_post_decision",
                )

            if firm.has_ai and firm.insurance_policy is not None:
                policy_vendor_id = str(getattr(firm.insurance_policy, "vendor_id", "") or "")
                current_vendor_id = str(firm.vendor_contract.vendor_id if firm.vendor_contract else "")
                if policy_vendor_id and current_vendor_id and policy_vendor_id != current_vendor_id:
                    self._cancel_insurance_policy(
                        firm=firm,
                        day=day,
                        reason="cancelled_vendor_mismatch_post_decision",
                    )

            if firm.has_ai and not firm.has_insurance:
                vendor = self.vendors[firm.vendor_contract.vendor_id]
                ai_remaining_days = max(0, int(firm.vendor_contract.end_day) - int(day))
                snapshot = self._industry_snapshot(firm.profile.industry, day)
                risk_signal = self._pre_operation_risk_signal(
                    firm=firm,
                    vendor=vendor,
                    day=day,
                    ai_remaining_days=ai_remaining_days,
                    snapshot=snapshot,
                )
                if not self.insurance_enabled:
                    self._log_decision(
                        day=day,
                        firm=firm,
                        decision_type="insurance_purchase",
                        decision={
                            "action": False,
                            "score": 0.0,
                            "threshold": 1.0,
                            "draw": 0.0,
                            "reason": "insurance_market_disabled_counterfactual",
                        },
                        risk_need=0.0,
                        context=firm_context,
                    )
                    continue
                insurance_decision = dict(self.policy.insurance_decision(firm, firm_context, risk_signal))
                if bool(insurance_decision.get("action", False)) and _is_model_decision(insurance_decision):
                    if not _decision_has_positive_term(
                        insurance_decision,
                        keys=("insurance_term_days", "term_days"),
                    ):
                        insurance_decision = self._invalidate_model_decision(
                            day=day,
                            firm=firm,
                            decision=insurance_decision,
                            decision_type="insurance_purchase",
                            reason="missing_positive_insurance_term_days",
                        )
                self._log_decision(
                    day=day,
                    firm=firm,
                    decision_type="insurance_purchase",
                    decision=insurance_decision,
                    risk_need=float(insurance_decision.get("score", 0.0)),
                    context=firm_context,
                )
                if bool(insurance_decision.get("action", False)):
                    risk_need = float(insurance_decision.get("score", 0.0))
                    term = _decision_term_days(
                        insurance_decision,
                        keys=("insurance_term_days", "term_days"),
                        fallback=lambda: self.policy.insurance_term_days(
                            firm,
                            firm_context,
                            risk_signal["claimable_event_score"],
                            risk_signal=risk_signal,
                        ),
                        min_days=int(self._contract_lifecycle_config().get("insurance_min_term_days", 1)),
                        max_days=int(self._contract_lifecycle_config().get("insurance_max_term_days", 90)),
                        max_remaining_days=ai_remaining_days,
                    )
                    if term <= 0:
                        self.logs["events"].write(
                            {
                                "event_type": "insurance_skipped_no_remaining_ai_exposure",
                                "day": int(day),
                                "firm_id": firm.profile.firm_id,
                                "vendor_id": vendor.vendor_id,
                                "ai_remaining_days": int(ai_remaining_days),
                                "risk_need": float(risk_need),
                            }
                        )
                        continue
                    quotes = self.insurance_market.quote_all(
                        firm=firm,
                        vendor=vendor,
                        snapshot=snapshot,
                        day=day,
                        term_days=term,
                        market_panic=float(firm_context.local_avg_panic if firm_context.local_avg_panic is not None else context.avg_panic),
                        recent_claim_rate=float(
                            firm_context.local_recent_claim_rate
                            if firm_context.local_recent_claim_rate is not None
                            else context.recent_claim_rate
                        ),
                        include_backstop=False,
                    )
                    selected, diagnostics = self.policy.choose_quote_with_diagnostics(
                        firm,
                        quotes,
                        risk_need=risk_need,
                        allow_backstop=False,
                    )
                    use_backstop = False
                    backstop_threshold = float(self.config.get("decision_policy", {}).get("backstop_score_threshold", 0.92))
                    backstop_enabled = bool(self.config.get("decision_policy", {}).get("backstop_enabled", True))
                    if selected is None and backstop_enabled and risk_need >= backstop_threshold:
                        backstop_quotes = self.insurance_market.quote_all(
                            firm=firm,
                            vendor=vendor,
                            snapshot=snapshot,
                            day=day,
                            term_days=term,
                            market_panic=float(
                                firm_context.local_avg_panic
                                if firm_context.local_avg_panic is not None
                                else context.avg_panic
                            ),
                            recent_claim_rate=float(
                                firm_context.local_recent_claim_rate
                                if firm_context.local_recent_claim_rate is not None
                                else context.recent_claim_rate
                            ),
                            include_backstop=True,
                            only_backstop=True,
                        )
                        backstop_selected, backstop_diag = self.policy.choose_quote_with_diagnostics(
                            firm,
                            backstop_quotes,
                            risk_need=risk_need,
                            allow_backstop=True,
                        )
                        diagnostics.extend(backstop_diag)
                        selected = backstop_selected
                        use_backstop = selected is not None

                    selected_utility = 0.0
                    for q, utility, reason in diagnostics:
                        is_preselected = bool(selected and selected.insurer_id == q.insurer_id)
                        if is_preselected:
                            selected_utility = float(utility)
                            continue
                        self._log_quote(q, selected=False, utility=utility, selection_reason=reason)

                    negotiated_selected = None
                    negotiation_outcome = ""
                    negotiation_rounds = 0
                    if selected is not None:
                        negotiation = self.negotiator.negotiate_insurance(
                            firm=firm,
                            quote=selected,
                            context=firm_context,
                            risk_need=risk_need,
                            max_rounds_override=_decision_max_rounds(insurance_decision),
                        )
                        self._log_interactions(negotiation.events)
                        negotiation_outcome = negotiation.outcome
                        negotiation_rounds = int(negotiation.rounds)
                        if negotiation.agreed and negotiation.quote is not None:
                            negotiated_selected = negotiation.quote
                        else:
                            self.logs["events"].write(
                                {
                                    "event_type": "insurance_negotiation_failed",
                                    "day": int(day),
                                    "firm_id": firm.profile.firm_id,
                                    "insurer_id": selected.insurer_id,
                                    "vendor_id": selected.vendor_id,
                                    "outcome": negotiation.outcome,
                                    "cash": float(firm.cash),
                                    "risk_need": float(risk_need),
                                }
                            )

                    if negotiated_selected is not None:
                        self._log_quote(
                            negotiated_selected,
                            selected=True,
                            utility=selected_utility,
                            selection_reason=f"negotiated:{negotiation_outcome}",
                        )
                    elif selected is not None:
                        self._log_quote(
                            selected,
                            selected=False,
                            utility=selected_utility,
                            selection_reason=f"negotiation_failed:{negotiation_outcome}",
                        )
                    if negotiated_selected is not None and firm.cash >= negotiated_selected.premium:
                        firm.insurance_policy = self.insurance_market.bind_policy(negotiated_selected, day=day)
                        firm.cash -= negotiated_selected.premium
                        premium_paid = negotiated_selected.premium
                        firm._premium_paid_today = premium_paid
                        total_premiums += negotiated_selected.premium
                        self.logs["events"].write(
                            {
                                "event_type": "insurance_policy_bound",
                                "day": int(day),
                                "firm_id": firm.profile.firm_id,
                                "insurer_id": negotiated_selected.insurer_id,
                                "vendor_id": negotiated_selected.vendor_id,
                                "term_days": int(negotiated_selected.term_days),
                                "ai_remaining_days_at_purchase": int(ai_remaining_days),
                                "premium": float(negotiated_selected.premium),
                                "market_role": negotiated_selected.market_role,
                                "risk_need": float(risk_need),
                                "used_backstop": bool(use_backstop),
                                "negotiation_rounds": int(negotiation_rounds),
                                "negotiation_outcome": str(negotiation_outcome),
                            }
                        )
                    elif negotiated_selected is not None:
                        self.logs["events"].write(
                            {
                                "event_type": "insurance_binding_failed_cash",
                                "day": int(day),
                                "firm_id": firm.profile.firm_id,
                                "insurer_id": negotiated_selected.insurer_id,
                                "vendor_id": negotiated_selected.vendor_id,
                                "premium": float(negotiated_selected.premium),
                                "cash": float(firm.cash),
                                "risk_need": float(risk_need),
                            }
                        )

            firm._premium_paid_today = premium_paid
            firm._vendor_fee_paid_today = vendor_fee_paid

        return {"vendor_fees": float(total_vendor_fees), "premiums": float(total_premiums)}

    def _operate_one_day(self, day: int, context: MarketContext, fees: dict) -> dict:
        claims_paid = 0.0
        claim_events = 0
        task_failures = 0
        material_events = 0
        claimable_events = 0
        uninsured_claimable_events = 0
        uninsured_claimable_loss = 0.0
        claim_residual_loss = 0.0
        unabsorbed_claimable_loss = 0.0
        material_score_sum = 0.0
        claimable_score_sum = 0.0
        new_bankruptcies = 0
        active_count = 0
        firm_log_rows = []

        for firm in self.firms.values():
            if not firm.active:
                continue
            active_count += 1
            cash_start = firm.cash
            had_ai_during_day = bool(firm.has_ai)
            vendor_id_during_day = firm.vendor_contract.vendor_id if firm.vendor_contract else ""
            had_insurance_during_day = bool(firm.has_insurance)
            insurer_id_during_day = firm.insurance_policy.insurer_id if firm.insurance_policy else ""
            record = self.panel.record_for(firm.profile.firm_id, day)
            base_pnl = self._traditional_pnl(firm, day)
            ai_gain = 0.0
            operational_loss = 0.0
            incident_score = 0.0
            claim_paid = 0.0
            risk_signal = _empty_risk_signal(record)
            firm._material_event_score_today = 0.0
            firm._claimable_event_score_today = 0.0
            firm._claim_paid_today_flag = 0.0
            firm._uninsured_claimable_today_flag = 0.0

            if firm.has_ai:
                vendor = self.vendors[firm.vendor_contract.vendor_id]
                risk_signal = _risk_signal(record, vendor, firm, self.config)
                snapshot = self._industry_snapshot(firm.profile.industry, day)
                risk_signal.update(_industry_risk_features(snapshot, vendor, firm, self.config))
                risk_signal["prior_policy"] = 1.0 if bool(getattr(firm, "_insurance_expired_today", False)) else 0.0
                incident_score = risk_signal["claimable_event_score"]
                operational_loss = risk_signal["operational_loss"]
                firm._material_event_score_today = float(risk_signal["material_event_score"]) if risk_signal["material_event"] else 0.0
                firm._claimable_event_score_today = (
                    float(risk_signal["claimable_event_score"]) if risk_signal["claimable_event"] else 0.0
                )
                ai_gain_scale = float(self.config.get("simulation", {}).get("ai_gain_scale", 0.55))
                risk_drag = float(self.config.get("simulation", {}).get("ai_gain_risk_drag", 0.70))
                ai_gain = (
                    firm.profile.asset_value
                    * vendor.productivity_lift
                    * ai_gain_scale
                    * (1.0 - min(0.85, risk_drag * risk_signal["material_event_score"]))
                )
                if record.incident_any:
                    task_failures += 1
                if risk_signal["material_event"]:
                    material_events += 1
                    material_score_sum += risk_signal["material_event_score"]
                    self.logs["events"].write(
                        {
                            "event_type": "ai_material_event",
                            "day": int(day),
                            "firm_id": firm.profile.firm_id,
                            "industry": firm.profile.industry,
                            "vendor_id": vendor.vendor_id,
                            "material_event_score": float(risk_signal["material_event_score"]),
                            "claimable_event_score": float(risk_signal["claimable_event_score"]),
                            "operational_loss": float(operational_loss),
                            "loss_ratio": float(risk_signal["loss_ratio"]),
                            "incident_task_count": int(record.incident_task_count),
                            "task_type_mix": record.task_type_mix,
                        }
                    )
                if risk_signal["claimable_event"]:
                    claimable_events += 1
                    claimable_score_sum += risk_signal["claimable_event_score"]

            firm.cash += base_pnl + ai_gain - operational_loss

            if self.insurance_enabled and firm.has_ai and firm.has_insurance and operational_loss > 0 and risk_signal["claimable_event"]:
                policy_vendor_id = str(getattr(firm.insurance_policy, "vendor_id", "") or "")
                current_vendor_id = str(firm.vendor_contract.vendor_id if firm.vendor_contract else "")
                policy_matches_vendor = not policy_vendor_id or policy_vendor_id == current_vendor_id
                if not policy_matches_vendor:
                    self.logs["events"].write(
                        {
                            "event_type": "claim_skipped_policy_vendor_mismatch",
                            "day": int(day),
                            "firm_id": firm.profile.firm_id,
                            "industry": firm.profile.industry,
                            "insurer_id": firm.insurance_policy.insurer_id,
                            "policy_vendor_id": policy_vendor_id,
                            "current_vendor_id": current_vendor_id,
                            "loss": float(operational_loss),
                            "claimable_event_score": float(risk_signal["claimable_event_score"]),
                        }
                    )
                if policy_matches_vendor:
                    policy_vendor_id = current_vendor_id
                    cooldown_days = max(0, int(self.config.get("claims", {}).get("claim_cooldown_days", 3)))
                    last_claim_day = getattr(firm, "last_claim_day", None)
                    in_cooldown = last_claim_day is not None and (int(day) - int(last_claim_day) < cooldown_days)
                    if in_cooldown:
                        self.logs["events"].write(
                            {
                                "event_type": "claim_skipped_cooldown",
                                "day": int(day),
                                "firm_id": firm.profile.firm_id,
                                "industry": firm.profile.industry,
                                "insurer_id": firm.insurance_policy.insurer_id,
                                "last_claim_day": int(last_claim_day),
                                "cooldown_days": int(cooldown_days),
                                "loss": float(operational_loss),
                                "claimable_event_score": float(risk_signal["claimable_event_score"]),
                            }
                        )
                    else:
                        claim_paid = self.insurance_market.process_claim(
                            policy=firm.insurance_policy,
                            loss_amount=operational_loss,
                            incident_score=risk_signal["claimable_event_score"],
                        )
                        if claim_paid > 0:
                            firm.cash += claim_paid
                            firm.last_claim_day = int(day)
                            firm._claim_paid_today_flag = 1.0
                            claims_paid += claim_paid
                            claim_events += 1
                            self.logs["events"].write(
                                {
                                    "event_type": "claim_paid",
                                    "day": int(day),
                                    "firm_id": firm.profile.firm_id,
                                    "industry": firm.profile.industry,
                                    "insurer_id": firm.insurance_policy.insurer_id,
                                    "vendor_id": str(policy_vendor_id),
                                    "loss": float(operational_loss),
                                    "payout": float(claim_paid),
                                    "claimable_event_score": float(risk_signal["claimable_event_score"]),
                                    "loss_ratio": float(risk_signal["loss_ratio"]),
                                }
                            )

            residual_loss = 0.0
            firm_claim_residual_loss = 0.0
            firm_unabsorbed_claimable_loss = 0.0
            if firm.has_ai and operational_loss > 0 and risk_signal["claimable_event"]:
                residual_loss = max(0.0, float(operational_loss) - float(claim_paid))
                firm_unabsorbed_claimable_loss = residual_loss
                unabsorbed_claimable_loss += residual_loss
                if claim_paid > 0 and residual_loss > 0:
                    firm_claim_residual_loss = residual_loss
                    claim_residual_loss += residual_loss
                    self.logs["events"].write(
                        {
                            "event_type": "claim_residual_loss",
                            "day": int(day),
                            "firm_id": firm.profile.firm_id,
                            "industry": firm.profile.industry,
                            "loss": float(operational_loss),
                            "payout": float(claim_paid),
                            "residual_loss": float(residual_loss),
                            "claimable_event_score": float(risk_signal["claimable_event_score"]),
                            "loss_ratio": float(risk_signal["loss_ratio"]),
                        }
                    )
                if claim_paid <= 0:
                    firm._uninsured_claimable_today_flag = 1.0
                    uninsured_claimable_events += 1
                    uninsured_claimable_loss += operational_loss
                    if self.insurance_enabled and firm.has_insurance:
                        unabsorbed_reason = "policy_no_payout_or_cooldown"
                    elif self.insurance_enabled:
                        unabsorbed_reason = "no_active_policy"
                    else:
                        unabsorbed_reason = "insurance_market_disabled"
                    self.logs["events"].write(
                        {
                            "event_type": "uninsured_claimable_loss",
                            "day": int(day),
                            "firm_id": firm.profile.firm_id,
                            "industry": firm.profile.industry,
                            "loss": float(operational_loss),
                            "claimable_event_score": float(risk_signal["claimable_event_score"]),
                            "loss_ratio": float(risk_signal["loss_ratio"]),
                            "reason": unabsorbed_reason,
                        }
                    )

            firm.last_operational_loss = float(operational_loss)
            firm.last_claim_paid = float(claim_paid)
            self._update_experience_memory(firm, risk_signal)

            if firm.cash < 0:
                self._terminate_bankrupt_firm_contracts(firm=firm, day=day)
                firm.active = False
                new_bankruptcies += 1
                self.logs["events"].write(
                    {
                        "event_type": "firm_bankrupt",
                        "day": int(day),
                        "firm_id": firm.profile.firm_id,
                        "industry": firm.profile.industry,
                        "cash": float(firm.cash),
                    }
                )

            firm_log_rows.append(
                {
                    "day": day,
                    "firm": firm,
                    "cash_start": cash_start,
                    "base_pnl": base_pnl,
                    "ai_gain": ai_gain,
                    "operational_loss": operational_loss,
                    "claim_paid": claim_paid,
                    "claim_residual_loss": firm_claim_residual_loss,
                    "unabsorbed_claimable_loss": firm_unabsorbed_claimable_loss,
                    "had_ai_during_day": had_ai_during_day,
                    "vendor_id_during_day": vendor_id_during_day,
                    "had_insurance_during_day": had_insurance_during_day,
                    "insurer_id_during_day": insurer_id_during_day,
                    "incident_score": incident_score,
                    "record": record,
                    "risk_signal": risk_signal,
                }
            )

        active_den = max(1, active_count)
        self.recent_claim_flags.append(1 if claim_events > 0 else 0)
        self.recent_claim_rates.append(claim_events / active_den)
        self.recent_material_rates.append(material_events / active_den)
        indemnity_den = claims_paid + unabsorbed_claimable_loss
        return {
            "claims_paid": float(claims_paid),
            "claim_events": int(claim_events),
            "uninsured_claimable_events": int(uninsured_claimable_events),
            "uninsured_claimable_loss": float(uninsured_claimable_loss),
            "claim_residual_loss": float(claim_residual_loss),
            "unabsorbed_claimable_loss": float(unabsorbed_claimable_loss),
            "indemnity_relief_ratio": float(claims_paid / indemnity_den) if indemnity_den > 0 else 0.0,
            "task_failures": int(task_failures),
            "material_events": int(material_events),
            "claimable_events": int(claimable_events),
            "avg_material_event_score": float(material_score_sum / max(1, material_events)),
            "avg_claimable_event_score": float(claimable_score_sum / max(1, claimable_events)),
            "active_count": int(active_count),
            "new_bankruptcies": int(new_bankruptcies),
            "_firm_log_rows": firm_log_rows,
        }

    def _traditional_pnl(self, firm: FirmState, day: int) -> float:
        sim = self.config.get("simulation", {})
        mu = float(sim.get("traditional_return_mu", 0.0005))
        sigma = float(sim.get("traditional_return_sigma", 0.006))
        # Keep paired insurance-on/off runs comparable: traditional operating
        # shocks must not depend on how many random draws the treatment arm uses.
        shock_rng = random.Random(_stable_int_seed(self.seed, firm.profile.firm_id, int(day), "traditional_pnl"))
        ret = shock_rng.gauss(mu, sigma)
        ret = max(float(sim.get("traditional_return_cap_down", -0.035)), min(float(sim.get("traditional_return_cap_up", 0.018)), ret))
        return float(firm.profile.asset_value * ret)

    def _update_experience_memory(self, firm: FirmState, risk_signal: dict) -> None:
        memory_cfg = self.config.get("experience_memory", {}) or {}
        risk_decay = _clamp(float(memory_cfg.get("risk_memory_decay", 0.86)), 0.0, 1.0)
        loss_decay = _clamp(float(memory_cfg.get("loss_memory_decay", 0.88)), 0.0, 1.0)
        claimable_decay = _clamp(float(memory_cfg.get("claimable_memory_decay", 0.90)), 0.0, 1.0)

        material_signal = _clamp(float(risk_signal.get("material_event_score", 0.0)), 0.0, 1.0)
        loss_signal = _clamp(float(risk_signal.get("loss_pressure", 0.0)), 0.0, 1.0)
        claimable_signal = _clamp(float(risk_signal.get("claimable_event_score", 0.0)), 0.0, 1.0)
        paid_claim = float(getattr(firm, "_claim_paid_today_flag", 0.0) or 0.0) > 0.0
        uninsured_claimable = float(getattr(firm, "_uninsured_claimable_today_flag", 0.0) or 0.0) > 0.0

        if uninsured_claimable:
            loss_signal = _clamp(
                loss_signal + float(memory_cfg.get("uninsured_loss_memory_boost", 0.16)) * max(claimable_signal, 0.25),
                0.0,
                1.0,
            )
            claimable_signal = _clamp(
                claimable_signal
                + float(memory_cfg.get("uninsured_claimable_memory_boost", 0.10)) * max(claimable_signal, 0.25),
                0.0,
                1.0,
            )
        elif paid_claim:
            # A paid claim softens cash-flow trauma, while the firm still learns
            # that this vendor/task exposure can produce operational incidents.
            loss_signal *= 1.0 - _clamp(float(memory_cfg.get("paid_claim_loss_memory_relief", 0.35)), 0.0, 1.0)
            claimable_signal *= 1.0 - _clamp(
                float(memory_cfg.get("paid_claim_claimable_memory_relief", 0.18)),
                0.0,
                1.0,
            )

        firm.risk_memory = _clamp(
            risk_decay * float(firm.risk_memory) + (1.0 - risk_decay) * material_signal,
            0.0,
            1.0,
        )
        firm.loss_memory = _clamp(
            loss_decay * float(firm.loss_memory) + (1.0 - loss_decay) * loss_signal,
            0.0,
            1.0,
        )
        firm.claimable_memory = _clamp(
            claimable_decay * float(firm.claimable_memory) + (1.0 - claimable_decay) * claimable_signal,
            0.0,
            1.0,
        )

    def _update_panic(self, totals: dict) -> None:
        active = [f for f in self.firms.values() if f.active]
        if not active:
            return
        den = max(1, int(totals.get("active_count", len(active))))
        material_rate = int(totals.get("material_events", 0)) / den
        claimable_rate = int(totals.get("claimable_events", 0)) / den
        claim_rate = int(totals.get("claim_events", 0)) / den
        uninsured_claimable_rate = int(totals.get("uninsured_claimable_events", 0)) / den
        avg_material = float(totals.get("avg_material_event_score", 0.0))

        panic_cfg = self.config.get("panic", {}) or {}
        memory = float(panic_cfg.get("memory", 0.82))
        calm_decay = float(panic_cfg.get("calm_decay", 0.018))
        material_weight = float(panic_cfg.get("material_event_weight", 0.20))
        claimable_weight = float(panic_cfg.get("claimable_event_weight", 0.34))
        claim_weight = float(panic_cfg.get("claim_event_weight", 0.18))
        uninsured_claimable_weight = float(panic_cfg.get("uninsured_claimable_event_weight", 0.44))
        indemnity_relief_weight = float(panic_cfg.get("indemnity_relief_weight", 0.20))
        insurance_reassurance_weight = float(panic_cfg.get("insurance_reassurance_weight", 0.0))
        severity_weight = float(panic_cfg.get("severity_weight", 0.10))
        indemnity_relief_ratio = _clamp(float(totals.get("indemnity_relief_ratio", 0.0)), 0.0, 1.0)
        unabsorbed_claimable_rate = max(
            uninsured_claimable_rate,
            max(0.0, claimable_rate - claim_rate),
        )
        unrelieved_claim_rate = claim_rate * (1.0 - indemnity_relief_ratio)
        public_shock = (
            material_weight * material_rate
            + claimable_weight * unabsorbed_claimable_rate
            + claim_weight * unrelieved_claim_rate
            + uninsured_claimable_weight * uninsured_claimable_rate
            + severity_weight * avg_material
            - indemnity_relief_weight * claim_rate * indemnity_relief_ratio
            - insurance_reassurance_weight * claim_rate * indemnity_relief_ratio
        )
        network_weight = _clamp(float(panic_cfg.get("network_neighbor_weight", 0.35)), 0.0, 1.0)
        own_event_weight = _clamp(float(panic_cfg.get("own_event_weight", 0.30)), 0.0, 1.0)
        if network_weight + own_event_weight > 1.0:
            scale = 1.0 / (network_weight + own_event_weight)
            network_weight *= scale
            own_event_weight *= scale
        public_weight = max(0.0, 1.0 - network_weight - own_event_weight)
        event_shock = {}
        event_source_ids = set()
        for source in self.firms.values():
            shock = self._firm_panic_event_shock(source, panic_cfg)
            if source.active or shock > 0.0:
                event_shock[source.profile.firm_id] = shock
            if shock > 0.0:
                event_source_ids.add(source.profile.firm_id)
        for firm in active:
            sensitivity = firm.profile.contagion_sensitivity
            neighbors = [
                fid
                for fid in self.firm_network.get(firm.profile.firm_id, [])
                if fid in self.firms and (self.firms[fid].active or fid in event_source_ids)
            ]
            if neighbors:
                neighbor_shock = sum(event_shock.get(fid, 0.0) for fid in neighbors) / len(neighbors)
            else:
                neighbor_shock = public_shock
            own_shock = event_shock.get(firm.profile.firm_id, 0.0)
            blended_shock = (
                public_weight * public_shock
                + network_weight * neighbor_shock
                + own_event_weight * own_shock
            )
            blended_shock = max(0.0, blended_shock)
            delta = blended_shock * sensitivity - calm_decay
            firm.panic = max(0.0, min(1.0, firm.panic * memory + delta))

    def _firm_panic_event_shock(self, firm: FirmState, panic_cfg: dict) -> float:
        material_weight = float(panic_cfg.get("material_event_weight", 0.20))
        claimable_weight = float(panic_cfg.get("claimable_event_weight", 0.34))
        claim_weight = float(panic_cfg.get("claim_event_weight", 0.18))
        uninsured_claimable_weight = float(panic_cfg.get("uninsured_claimable_event_weight", 0.44))
        indemnity_relief_weight = float(panic_cfg.get("indemnity_relief_weight", 0.20))
        insurance_reassurance_weight = float(panic_cfg.get("insurance_reassurance_weight", 0.0))
        material_score = float(getattr(firm, "_material_event_score_today", 0.0) or 0.0)
        claimable_score = float(getattr(firm, "_claimable_event_score_today", 0.0) or 0.0)
        paid_claim = float(getattr(firm, "_claim_paid_today_flag", 0.0) or 0.0)
        uninsured_claimable = float(getattr(firm, "_uninsured_claimable_today_flag", 0.0) or 0.0)
        shock = (
            material_weight * material_score
            + claimable_weight * claimable_score * max(uninsured_claimable, 1.0 - paid_claim)
            + claim_weight * claimable_score * paid_claim
            + uninsured_claimable_weight * claimable_score * uninsured_claimable
            - indemnity_relief_weight * claimable_score * paid_claim
            - insurance_reassurance_weight * claimable_score * paid_claim
        )
        return max(0.0, float(shock))

    def _refresh_active_policy_counts(self) -> None:
        for state in self.insurance_market.insurers.values():
            state.active_policies = 0
        for firm in self.firms.values():
            if firm.active and firm.insurance_policy is not None:
                self.insurance_market.mark_active_policy(firm.insurance_policy.insurer_id)

    def _macro_row(self, day: int, cumulative_bankruptcies: int) -> dict:
        all_firms = list(self.firms.values())
        active = [f for f in self.firms.values() if f.active]
        adopters = [f for f in active if f.has_ai]
        insured = [f for f in active if f.has_insurance]
        panic_values = [float(f.panic) for f in active]
        vendor_counts: Dict[str, int] = {}
        for firm in adopters:
            vendor_counts[firm.vendor_contract.vendor_id] = vendor_counts.get(firm.vendor_contract.vendor_id, 0) + 1
        total_insurer_capital = sum(s.capital for s in self.insurance_market.insurers.values())
        active_firm_cash = sum(f.cash for f in active)
        inactive_firm_cash = sum(f.cash for f in all_firms if not f.active)
        all_firm_cash = active_firm_cash + inactive_firm_cash
        total_vendor_stock = sum(float(v) for v in self.vendor_capital.values())
        social_active = active_firm_cash + total_insurer_capital + total_vendor_stock
        social_all = all_firm_cash + total_insurer_capital + total_vendor_stock
        return {
            "day": int(day),
            "active_firms": int(len(active)),
            "ai_penetration": len(adopters) / len(active) if active else 0.0,
            "insurance_coverage_overall": len(insured) / len(active) if active else 0.0,
            "insurance_coverage_ai_adopters": len([f for f in adopters if f.has_insurance]) / len(adopters) if adopters else 0.0,
            "cumulative_bankruptcies": int(cumulative_bankruptcies),
            "avg_panic": sum(panic_values) / len(panic_values) if panic_values else 0.0,
            "panic_p50": _quantile(panic_values, 0.50),
            "panic_p75": _quantile(panic_values, 0.75),
            "panic_p90": _quantile(panic_values, 0.90),
            "panic_p95": _quantile(panic_values, 0.95),
            "panic_p99": _quantile(panic_values, 0.99),
            "panic_nonzero_share": (
                len([value for value in panic_values if value > 0.0]) / len(panic_values)
                if panic_values
                else 0.0
            ),
            "active_firm_cash": float(active_firm_cash),
            "inactive_firm_cash": float(inactive_firm_cash),
            "all_firm_cash": float(all_firm_cash),
            "total_insurer_capital": float(total_insurer_capital),
            "total_vendor_capital": float(total_vendor_stock),
            "social_total_capital_active_firms": float(social_active),
            "social_total_capital_all_firms": float(social_all),
            "social_total_capital": float(social_all),
            "delta_social_capital": 0.0,
            "max_vendor_share": max(vendor_counts.values()) / len(adopters) if adopters else 0.0,
        }

    def _log_firm_day(
        self,
        day: int,
        firm: FirmState,
        cash_start: float,
        base_pnl: float,
        ai_gain: float,
        operational_loss: float,
        claim_paid: float,
        claim_residual_loss: float,
        unabsorbed_claimable_loss: float,
        had_ai_during_day: bool,
        vendor_id_during_day: str,
        had_insurance_during_day: bool,
        insurer_id_during_day: str,
        incident_score: float,
        record,
        risk_signal: dict,
    ) -> None:
        vc = firm.vendor_contract
        pol = firm.insurance_policy
        context = getattr(self, "_decision_context_by_firm", {}).get(firm.profile.firm_id)
        if context is None:
            context = self._firm_market_context(firm, self._market_context(day))
        self.logs["firm"].write(
            {
                "day": int(day),
                "firm_id": firm.profile.firm_id,
                "industry": firm.profile.industry,
                "active": bool(firm.active),
                "has_ai": bool(firm.has_ai),
                "had_ai_during_day": bool(had_ai_during_day),
                "vendor_id": vc.vendor_id if vc else "",
                "vendor_id_during_day": str(vendor_id_during_day),
                "vendor_monthly_fee": float(getattr(vc, "monthly_fee", 0.0) or 0.0) if vc else 0.0,
                "vendor_contract_total_fee": float(getattr(vc, "price", 0.0) or 0.0) if vc else 0.0,
                "vendor_term_end": vc.end_day if vc else "",
                "has_insurance": bool(firm.has_insurance),
                "had_insurance_during_day": bool(had_insurance_during_day),
                "insurer_id": pol.insurer_id if pol else "",
                "insurer_id_during_day": str(insurer_id_during_day),
                "insurance_term_end": pol.end_day if pol else "",
                "cash_start": float(cash_start),
                "cash_end": float(firm.cash),
                "base_pnl": float(base_pnl),
                "ai_gain": float(ai_gain),
                "operational_loss": float(operational_loss),
                "claim_paid": float(claim_paid),
                "claim_residual_loss": float(claim_residual_loss),
                "unabsorbed_claimable_loss": float(unabsorbed_claimable_loss),
                "claim_paid_flag": int(float(getattr(firm, "_claim_paid_today_flag", 0.0) or 0.0) > 0.0),
                "uninsured_claimable_flag": int(float(getattr(firm, "_uninsured_claimable_today_flag", 0.0) or 0.0) > 0.0),
                "premium_paid": float(getattr(firm, "_premium_paid_today", 0.0)),
                "vendor_fee_paid": float(getattr(firm, "_vendor_fee_paid_today", 0.0)),
                "insurance_refund_received": float(getattr(firm, "_insurance_refund_today", 0.0)),
                "vendor_refund_received": float(getattr(firm, "_vendor_refund_today", 0.0)),
                "incident_score": float(incident_score),
                "task_failure_rate": float(risk_signal.get("task_failure_rate", 0.0)),
                "material_event": int(bool(risk_signal.get("material_event", False))),
                "claimable_event": int(bool(risk_signal.get("claimable_event", False))),
                "material_event_score": float(risk_signal.get("material_event_score", 0.0)),
                "claimable_event_score": float(risk_signal.get("claimable_event_score", 0.0)),
                "loss_ratio": float(risk_signal.get("loss_ratio", 0.0)),
                "industry_incident_rate": float(risk_signal.get("industry_incident_rate", 0.0)),
                "industry_stress_score": float(risk_signal.get("industry_stress_score", 0.0)),
                "industry_loss_pressure": float(risk_signal.get("industry_loss_pressure", 0.0)),
                "prior_policy": float(risk_signal.get("prior_policy", 0.0)),
                "action_num_tasks": int(record.num_tasks),
                "action_incident_any": int(record.incident_any),
                "action_incident_task_count": int(record.incident_task_count),
                "action_avg_severity": float(record.avg_severity),
                "action_avg_risk_score": float(record.avg_risk_score),
                "action_max_risk_score": float(record.max_risk_score),
                "risk_memory": float(firm.risk_memory),
                "loss_memory": float(firm.loss_memory),
                "claimable_memory": float(firm.claimable_memory),
                "panic": float(firm.panic),
                "local_adoption_rate": float(context.local_adoption_rate or 0.0),
                "local_insurance_coverage_rate": float(context.local_insurance_coverage_rate or 0.0),
                "local_avg_panic": float(context.local_avg_panic or 0.0),
                "local_recent_claim_rate": float(context.local_recent_claim_rate or 0.0),
                "network_neighbor_count": int(context.network_neighbor_count),
                "same_industry_neighbor_share": float(context.same_industry_neighbor_share),
                "state_end": "Active" if firm.active else "Bankrupt",
                }
            )

    def _log_quote(self, quote: InsuranceQuote, selected: bool, utility: float = 0.0, selection_reason: str = "") -> None:
        row = asdict(quote)
        row["utility"] = float(utility)
        row["selection_reason"] = str(selection_reason)
        row["selected"] = int(selected)
        self.logs["quotes"].write(row)

    def _log_decision(
        self,
        day: int,
        firm: FirmState,
        decision_type: str,
        decision: dict,
        selected_id: str = "",
        risk_need: float = 0.0,
        visible_vendors: Optional[List[VendorProfile]] = None,
        context: Optional[MarketContext] = None,
    ) -> None:
        visible_vendor_ids = ";".join(v.vendor_id for v in (visible_vendors or []))
        local_adoption = context.local_adoption_rate if context and context.local_adoption_rate is not None else ""
        local_insurance = (
            context.local_insurance_coverage_rate
            if context and context.local_insurance_coverage_rate is not None
            else ""
        )
        local_panic = context.local_avg_panic if context and context.local_avg_panic is not None else ""
        local_claim = context.local_recent_claim_rate if context and context.local_recent_claim_rate is not None else ""
        self.logs["decisions"].write(
            {
                "day": int(day),
                "firm_id": firm.profile.firm_id,
                "decision_type": str(decision_type),
                "action": int(bool(decision.get("action", False))),
                "score": float(decision.get("score", decision.get("probability", 0.0)) or 0.0),
                "threshold": float(decision.get("threshold", 0.0) or 0.0),
                "probability": float(decision.get("probability", 0.0) or 0.0),
                "draw": float(decision.get("draw", 0.0) or 0.0),
                "reason": str(decision.get("reason", "")),
                "selected_id": str(selected_id),
                "cash": float(firm.cash),
                "risk_need": float(risk_need),
                "decision_selected_vendor_id": str(decision.get("selected_vendor_id", "")),
                "vendor_term_days": int(decision.get("vendor_term_days", 0) or 0),
                "insurance_term_days": int(decision.get("insurance_term_days", decision.get("term_days", 0)) or 0),
                "max_rounds": int(decision.get("max_rounds", 0) or 0),
                "visible_vendor_ids": visible_vendor_ids,
                "visible_vendor_count": int(len(visible_vendors or [])),
                "local_adoption_rate": local_adoption,
                "local_insurance_coverage_rate": local_insurance,
                "local_avg_panic": local_panic,
                "local_recent_claim_rate": local_claim,
                "network_neighbor_count": int(context.network_neighbor_count) if context else 0,
                "same_industry_neighbor_share": (
                    float(context.same_industry_neighbor_share) if context else 0.0
                ),
            }
        )
        trace = decision.get("model_trace")
        if isinstance(trace, dict):
            row = dict(trace)
            row.update(
                {
                    "day": int(day),
                    "firm_id": firm.profile.firm_id,
                    "decision_type": str(decision_type),
                    "action": bool(decision.get("action", False)),
                    "reason": str(decision.get("reason", "")),
                    "selected_vendor_id": str(decision.get("selected_vendor_id", "")),
                    "vendor_term_days": int(decision.get("vendor_term_days", 0) or 0),
                    "insurance_term_days": int(decision.get("insurance_term_days", decision.get("term_days", 0)) or 0),
                    "max_rounds": int(decision.get("max_rounds", 0) or 0),
                }
            )
            self.logs["model_decisions"].write(row)

    def _invalidate_model_decision(
        self,
        day: int,
        firm: FirmState,
        decision: dict,
        decision_type: str,
        reason: str,
    ) -> dict:
        fixed = dict(decision)
        fixed["action"] = False
        fixed["probability"] = float(fixed.get("probability", fixed.get("score", 0.0)) or 0.0)
        old_reason = str(fixed.get("reason", "model_decision"))
        fixed["reason"] = f"{old_reason}|invalid_model_structure:{reason}"
        self.logs["events"].write(
            {
                "event_type": "model_invalid_decision",
                "day": int(day),
                "firm_id": firm.profile.firm_id,
                "decision_type": str(decision_type),
                "reason": str(reason),
                "original_action": bool(decision.get("action", False)),
                "selected_vendor_id": str(decision.get("selected_vendor_id", "")),
                "vendor_term_days": int(decision.get("vendor_term_days", 0) or 0),
                "insurance_term_days": int(decision.get("insurance_term_days", decision.get("term_days", 0)) or 0),
                "max_rounds": int(decision.get("max_rounds", 0) or 0),
            }
        )
        return fixed

    def _log_interactions(self, rows: Iterable[dict]) -> None:
        for row in rows:
            self.logs["interactions"].write(dict(row))

    def _write_checkpoint(self, day: int) -> None:
        checkpoint = {
            "day": int(day),
            "rng_state": _encode_rng_state(self.rng),
            "cumulative_bankruptcies": int(self.cumulative_bankruptcies),
            "prev_capital": float(self.prev_capital) if self.prev_capital is not None else None,
            "recent_claim_flags": list(self.recent_claim_flags),
            "recent_claim_rates": list(self.recent_claim_rates),
            "recent_material_rates": list(self.recent_material_rates),
            "firms": {
                fid: {
                    "cash": state.cash,
                        "active": state.active,
                        "panic": state.panic,
                        "last_operational_loss": state.last_operational_loss,
                        "last_claim_paid": state.last_claim_paid,
                        "last_claim_day": state.last_claim_day,
                        "risk_memory": state.risk_memory,
                        "loss_memory": state.loss_memory,
                        "claimable_memory": state.claimable_memory,
                        "ai_cooldown_until": state.ai_cooldown_until,
                        "vendor_contract": asdict(state.vendor_contract) if state.vendor_contract else None,
                        "insurance_policy": asdict(state.insurance_policy) if state.insurance_policy else None,
                    }
                for fid, state in self.firms.items()
            },
            "insurers": {
                iid: {
                    "capital": state.capital,
                    "capital_ratio": state.capital_ratio,
                    "regime": state.regime,
                    "active_policies": state.active_policies,
                }
                for iid, state in self.insurance_market.insurers.items()
            },
            "vendor_capital": {vid: float(capital) for vid, capital in self.vendor_capital.items()},
        }
        path = self.run_dir / "checkpoints" / f"day_{int(day):03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2, sort_keys=True)

    def _load_checkpoint(self, path: Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        checkpoint_firms = set((checkpoint.get("firms") or {}).keys())
        current_firms = set(self.firms.keys())
        if checkpoint_firms != current_firms:
            missing = sorted(current_firms - checkpoint_firms)[:5]
            extra = sorted(checkpoint_firms - current_firms)[:5]
            raise ValueError(
                "Resume checkpoint firm set does not match current run. "
                f"missing={missing}, extra={extra}. Use the same --firms setting as the original run."
            )

        checkpoint_insurers = set((checkpoint.get("insurers") or {}).keys())
        current_insurers = set(self.insurance_market.insurers.keys())
        if checkpoint_insurers and checkpoint_insurers != current_insurers:
            raise ValueError("Resume checkpoint insurer set does not match current config.")

        for fid, saved in (checkpoint.get("firms") or {}).items():
            state = self.firms[fid]
            state.cash = float(saved.get("cash", state.cash))
            state.active = bool(saved.get("active", state.active))
            state.panic = float(saved.get("panic", state.panic))
            state.last_operational_loss = float(saved.get("last_operational_loss", 0.0))
            state.last_claim_paid = float(saved.get("last_claim_paid", 0.0))
            saved_claim_day = saved.get("last_claim_day", None)
            state.last_claim_day = int(saved_claim_day) if saved_claim_day is not None else None
            state.risk_memory = float(saved.get("risk_memory", state.risk_memory))
            state.loss_memory = float(saved.get("loss_memory", state.loss_memory))
            state.claimable_memory = float(saved.get("claimable_memory", state.claimable_memory))
            state.ai_cooldown_until = int(saved.get("ai_cooldown_until", state.ai_cooldown_until))
            state.vendor_contract = _restore_dataclass(VendorContract, saved.get("vendor_contract"))
            state.insurance_policy = _restore_dataclass(InsurancePolicy, saved.get("insurance_policy"))

        for iid, saved in (checkpoint.get("insurers") or {}).items():
            if iid in self.insurance_market.insurers:
                state = self.insurance_market.insurers[iid]
                state.capital = float(saved.get("capital", state.capital))
                state.active_policies = int(saved.get("active_policies", state.active_policies))

        for vid, capital in (checkpoint.get("vendor_capital") or {}).items():
            if vid in self.vendor_capital:
                self.vendor_capital[vid] = float(capital)

        if checkpoint.get("rng_state"):
            self.rng.setstate(_decode_rng_state(str(checkpoint["rng_state"])))

        self.recent_claim_flags = [int(x) for x in checkpoint.get("recent_claim_flags", [])]
        self.recent_claim_rates = [float(x) for x in checkpoint.get("recent_claim_rates", [])]
        self.recent_material_rates = [float(x) for x in checkpoint.get("recent_material_rates", [])]
        self.cumulative_bankruptcies = int(
            checkpoint.get("cumulative_bankruptcies", sum(1 for f in self.firms.values() if not f.active))
        )
        self.resume_day = int(checkpoint["day"])
        saved_prev = checkpoint.get("prev_capital")
        self.prev_capital = float(saved_prev) if saved_prev is not None else self._current_social_total_capital()
        self.days = [int(day) for day in self.days if int(day) > self.resume_day]
        self._refresh_active_policy_counts()

    def _current_social_total_capital(self) -> float:
        total_insurer_capital = sum(s.capital for s in self.insurance_market.insurers.values())
        total_cash = sum(f.cash for f in self.firms.values())
        total_vendor_stock = sum(float(v) for v in self.vendor_capital.values())
        return float(total_cash + total_insurer_capital + total_vendor_stock)

    def _prepare_resume_logs(self, resume_day: int) -> None:
        backup_dir = self.run_dir / "resume_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        for filename in [
            "macro_daily.csv",
            "firm_daily.csv",
            "insurer_daily.csv",
            "quotes.csv",
            "decisions.csv",
        ]:
            self._backup_and_prune_csv(self.run_dir / filename, backup_dir, resume_day)
        self._backup_and_prune_jsonl(self.run_dir / "events.jsonl", backup_dir, resume_day)
        self._backup_and_prune_jsonl(self.run_dir / "model_decisions.jsonl", backup_dir, resume_day)
        self._backup_and_prune_jsonl(self.run_dir / "interactions.jsonl", backup_dir, resume_day)

    def _backup_and_prune_csv(self, path: Path, backup_dir: Path, resume_day: int) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if _row_day(row) <= resume_day]
            fieldnames = list(reader.fieldnames or [])
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _backup_and_prune_jsonl(self, path: Path, backup_dir: Path, resume_day: int) -> None:
        if not path.exists() or path.stat().st_size == 0:
            return
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup_dir / path.name)
        kept = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _row_day(row) <= resume_day:
                    kept.append(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        with path.open("w", encoding="utf-8") as f:
            f.writelines(kept)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    q = _clamp(float(q), 0.0, 1.0)
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _stable_network_score(seed: int, firm_id: str, other_firm_id: str, channel: str) -> float:
    payload = f"{int(seed)}|{channel}|{firm_id}|{other_firm_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _stable_int_seed(seed: int, firm_id: str, day: int, channel: str) -> int:
    payload = f"{int(seed)}|{channel}|{firm_id}|{int(day)}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def _empty_risk_signal(record) -> dict:
    return {
        "task_failure_rate": 0.0,
        "risk_excess": 0.0,
        "operational_loss": 0.0,
        "loss_ratio": 0.0,
        "loss_pressure": 0.0,
        "material_event_score": 0.0,
        "claimable_event_score": 0.0,
        "material_event": False,
        "claimable_event": False,
        "industry_incident_rate": 0.0,
        "industry_stress_score": 0.0,
        "industry_loss_pressure": 0.0,
        "prior_policy": 0.0,
        "raw_incident_any": bool(getattr(record, "incident_any", False)),
    }


def _risk_signal(record, vendor: VendorProfile, firm: FirmState, config: dict) -> dict:
    risk_cfg = config.get("risk_mapping", {}) or {}
    sim_cfg = config.get("simulation", {}) or {}

    risk_floor = float(risk_cfg.get("risk_score_floor", 0.02))
    risk_excess = _clamp((float(record.max_risk_score) - risk_floor) / max(1e-9, 1.0 - risk_floor), 0.0, 1.0)
    task_failure_rate = float(record.incident_task_count) / max(1, int(record.num_tasks))
    task_failure_rate = _clamp(task_failure_rate, 0.0, 1.0)
    severity = _clamp(float(record.avg_severity), 0.0, 1.0)

    loss_scale = float(sim_cfg.get("action_loss_scale", 1.0))
    realization_rate = float(risk_cfg.get("loss_realization_rate", 0.62))
    exposure_scale = 0.45 + 0.70 * float(firm.profile.ai_dependency)
    operational_loss = (
        float(record.total_loss)
        * float(vendor.risk_multiplier)
        * exposure_scale
        * loss_scale
        * realization_rate
    )
    tail_threshold = float(risk_cfg.get("catastrophic_tail_threshold", 1.0))
    tail_multiplier = float(risk_cfg.get("catastrophic_tail_loss_multiplier", 0.0))
    if tail_multiplier > 0 and float(record.max_risk_score) > tail_threshold:
        tail_intensity = _clamp(
            (float(record.max_risk_score) - tail_threshold) / max(1e-9, 1.0 - tail_threshold),
            0.0,
            1.0,
        )
        operational_loss *= 1.0 + tail_multiplier * (tail_intensity**2)
    asset = max(float(firm.profile.asset_value), 1.0)
    loss_ratio = max(0.0, float(operational_loss) / asset)

    material_loss_ratio = float(risk_cfg.get("material_loss_ratio_threshold", 0.0015))
    claimable_loss_ratio = float(risk_cfg.get("claimable_loss_ratio_threshold", 0.0060))
    loss_pressure = _clamp(loss_ratio / max(1e-9, material_loss_ratio), 0.0, 1.0)
    claim_loss_pressure = _clamp(loss_ratio / max(1e-9, claimable_loss_ratio), 0.0, 1.0)

    material_score = _clamp(
        0.26 * task_failure_rate
        + 0.26 * risk_excess
        + 0.24 * severity
        + 0.24 * loss_pressure,
        0.0,
        1.0,
    )
    claimable_score = _clamp(
        0.18 * task_failure_rate
        + 0.26 * risk_excess
        + 0.20 * severity
        + 0.36 * claim_loss_pressure,
        0.0,
        1.0,
    )

    material_event = bool(record.incident_any) and (
        material_score >= float(risk_cfg.get("material_event_score_threshold", 0.46))
        or loss_ratio >= material_loss_ratio
    )
    claimable_event = bool(record.incident_any) and (
        claimable_score >= float(risk_cfg.get("claimable_event_score_threshold", 0.66))
        and loss_ratio >= float(risk_cfg.get("claimable_min_loss_ratio", 0.0025))
    )

    return {
        "task_failure_rate": float(task_failure_rate),
        "risk_excess": float(risk_excess),
        "operational_loss": float(operational_loss),
        "loss_ratio": float(loss_ratio),
        "loss_pressure": float(loss_pressure),
        "material_event_score": float(material_score),
        "claimable_event_score": float(claimable_score),
        "material_event": bool(material_event),
        "claimable_event": bool(claimable_event),
        "raw_incident_any": bool(record.incident_any),
    }


def _insurance_pricing_config(config: dict) -> dict:
    pricing = dict((config.get("insurance_pricing", {}) or {}))
    risk_cfg = config.get("risk_mapping", {}) or {}
    pricing.setdefault(
        "catastrophic_tail_loss_multiplier",
        float(risk_cfg.get("catastrophic_tail_loss_multiplier", 0.0)),
    )
    return pricing


def _industry_risk_features(snapshot, vendor: VendorProfile, firm: FirmState, config: dict) -> dict:
    risk_cfg = config.get("risk_mapping", {}) or {}
    sim_cfg = config.get("simulation", {}) or {}

    risk_floor = float(risk_cfg.get("risk_score_floor", 0.02))
    stress_score = _clamp((float(snapshot.stress_risk_score) - risk_floor) / max(1e-9, 1.0 - risk_floor), 0.0, 1.0)

    loss_scale = float(sim_cfg.get("action_loss_scale", 1.0))
    realization_rate = float(risk_cfg.get("loss_realization_rate", 0.62))
    exposure_scale = 0.45 + 0.70 * float(firm.profile.ai_dependency)
    stress_loss = (
        float(snapshot.stress_loss)
        * float(vendor.risk_multiplier)
        * exposure_scale
        * loss_scale
        * realization_rate
    )
    asset = max(float(firm.profile.asset_value), 1.0)
    reference_ratio = float(risk_cfg.get("claimable_loss_ratio_threshold", 0.0060))
    loss_pressure = _clamp((stress_loss / asset) / max(1e-9, reference_ratio), 0.0, 1.0)

    return {
        "industry_incident_rate": _clamp(float(snapshot.incident_rate), 0.0, 1.0),
        "industry_stress_score": float(stress_score),
        "industry_loss_pressure": float(loss_pressure),
    }


def _selected_visible_vendor(decision: dict, visible: List[VendorProfile]) -> Optional[VendorProfile]:
    selected_id = str(
        decision.get("selected_vendor_id")
        or decision.get("selected_vendor")
        or decision.get("vendor_id")
        or ""
    ).strip()
    if not selected_id:
        return None
    return next((vendor for vendor in visible if vendor.vendor_id == selected_id), None)


def _with_incumbent_renewal_option(
    visible: List[VendorProfile],
    vendors_by_id: Dict[str, VendorProfile],
    incumbent_vendor_id: str,
) -> List[VendorProfile]:
    incumbent_vendor_id = str(incumbent_vendor_id or "").strip()
    if not incumbent_vendor_id or incumbent_vendor_id not in vendors_by_id:
        return list(visible)
    if any(v.vendor_id == incumbent_vendor_id for v in visible):
        return list(visible)
    return list(visible) + [vendors_by_id[incumbent_vendor_id]]


def _is_model_decision(decision: dict) -> bool:
    return isinstance(decision.get("model_trace"), dict)


def _decision_has_positive_term(decision: dict, keys: Iterable[str]) -> bool:
    for key in keys:
        try:
            if int(decision.get(key)) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _uses_model_decision_layer(config: dict) -> bool:
    mode = str((config.get("decision_layer") or {}).get("mode", "rule_heuristic"))
    return mode in {"model_mock", "vllm_openai", "openai_compatible"}


def _apply_initial_adoption_trial_cap(decision: dict, renewal: bool, lifecycle_config: dict) -> dict:
    if renewal or not bool(decision.get("action", False)):
        return decision
    cfg = dict(lifecycle_config or {})
    try:
        cap_days = int(cfg.get("initial_adoption_trial_term_cap_days", 0) or 0)
    except (TypeError, ValueError):
        cap_days = 0
    if cap_days <= 0:
        return decision
    try:
        score = float(decision.get("score", decision.get("probability", 0.0)) or 0.0)
        threshold = float(decision.get("threshold", 0.0) or 0.0)
    except (TypeError, ValueError):
        score = 0.0
        threshold = 0.0
    try:
        long_term_margin = float(cfg.get("initial_adoption_long_term_margin", float("inf")))
    except (TypeError, ValueError):
        long_term_margin = float("inf")
    if score - threshold >= long_term_margin:
        return decision

    original_term = 0
    for key in ("vendor_term_days", "term_days"):
        try:
            original_term = int(decision.get(key) or 0)
        except (TypeError, ValueError):
            original_term = 0
        if original_term > 0:
            break
    if original_term <= 0:
        return decision

    capped_term = _bounded_term_days(
        original_term,
        min_days=int(cfg.get("vendor_min_term_days", 14)),
        max_days=cap_days,
    )
    if capped_term >= original_term:
        return decision

    fixed = dict(decision)
    fixed["vendor_term_days"] = int(capped_term)
    fixed["term_days"] = int(capped_term)
    fixed["reason"] = (
        f"{fixed.get('reason', 'decision')}|initial_adoption_trial_term_cap:{int(original_term)}->{int(capped_term)}"
    )
    trace = fixed.get("model_trace")
    if isinstance(trace, dict):
        trace["initial_adoption_trial_term_cap"] = {
            "enabled": True,
            "original_term_days": int(original_term),
            "capped_term_days": int(capped_term),
            "score": float(score),
            "threshold": float(threshold),
            "margin": float(score - threshold),
            "long_term_margin": float(long_term_margin),
        }
    return fixed


def _decision_term_days(
    decision: dict,
    keys: Iterable[str],
    fallback,
    min_days: int = 1,
    max_days: int = 180,
    max_remaining_days: Optional[int] = None,
) -> int:
    for key in keys:
        value = decision.get(key)
        try:
            term = int(value)
        except (TypeError, ValueError):
            continue
        if term > 0:
            return _bounded_term_days(
                term,
                min_days=min_days,
                max_days=max_days,
                max_remaining_days=max_remaining_days,
            )
    return _bounded_term_days(
        fallback(),
        min_days=min_days,
        max_days=max_days,
        max_remaining_days=max_remaining_days,
    )


def _bounded_term_days(
    value,
    min_days: int = 1,
    max_days: int = 180,
    max_remaining_days: Optional[int] = None,
) -> int:
    try:
        term = int(value)
    except (TypeError, ValueError):
        term = int(min_days)
    cap = max(1, int(max_days))
    if max_remaining_days is not None:
        cap = min(cap, max(0, int(max_remaining_days)))
    if cap <= 0:
        return 0
    floor = max(1, int(min_days))
    floor = min(floor, cap)
    return max(floor, min(int(term), cap))


def _decision_max_rounds(decision: dict) -> Optional[int]:
    try:
        rounds = int(decision.get("max_rounds"))
    except (TypeError, ValueError):
        return None
    if rounds <= 0:
        return None
    return max(1, min(int(rounds), 30))


def _resolve_paths(config: dict, root: Path) -> dict:
    out = dict(config)
    paths = dict(out.get("paths", {}))
    for key, value in list(paths.items()):
        if value and not Path(value).is_absolute():
            paths[key] = str((root / value).resolve())
    out["paths"] = paths
    return out


def _find_repository_root(config_path: Path) -> Path:
    """Locate the repository root without depending on config nesting depth."""
    resolved = Path(config_path).resolve()
    for candidate in (resolved.parent, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "action_risk_v2").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the repository root from config path: {resolved}. "
        "Expected pyproject.toml and src/action_risk_v2 in a parent directory."
    )


def _load_yaml_config(config_path: Path) -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    extends = config.pop("extends", None)
    if not extends:
        return config
    base_path = Path(extends)
    if not base_path.is_absolute():
        base_path = Path(config_path).resolve().parent / base_path
    base_config = _load_yaml_config(base_path)
    return _deep_merge(base_config, config)


def _deep_merge(base: dict, overrides: dict) -> dict:
    merged = dict(base or {})
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _is_disabled_path(value: Optional[str]) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"", "none", "null", "false", "off", "-"}


def _apply_path_overrides(config: dict, root: Path, path_overrides: Optional[Dict[str, Optional[str]]] = None) -> dict:
    if not path_overrides:
        return config
    out = dict(config)
    paths = dict(out.get("paths", {}) or {})
    for key, raw_value in path_overrides.items():
        if raw_value is None:
            continue
        if _is_disabled_path(raw_value):
            paths[key] = ""
            continue
        value = Path(str(raw_value)).expanduser()
        if not value.is_absolute():
            value = (root / value).resolve()
        paths[key] = str(value)
    out["paths"] = paths
    return out


def _apply_decision_overrides(
    config: dict,
    decision_mode: Optional[str] = None,
    vllm_base_url: Optional[str] = None,
    vllm_base_urls: Optional[str] = None,
    vllm_model: Optional[str] = None,
    model_fallback_to_rule: Optional[bool] = None,
    insurance_market_enabled: Optional[bool] = None,
    seed: Optional[int] = None,
) -> dict:
    out = dict(config)
    layer = dict(out.get("decision_layer", {}) or {})
    if decision_mode:
        layer["mode"] = str(decision_mode)
        layer["llm_or_vllm_enabled"] = str(decision_mode) != "rule_heuristic"
    if vllm_base_url:
        layer["base_url"] = str(vllm_base_url)
    if vllm_base_urls:
        layer["base_urls"] = str(vllm_base_urls)
        first_url = [part.strip() for part in str(vllm_base_urls).replace(";", ",").split(",") if part.strip()]
        if first_url:
            layer["base_url"] = first_url[0]
    if vllm_model:
        layer["model"] = str(vllm_model)
    if model_fallback_to_rule is not None:
        layer["fallback_to_rule"] = bool(model_fallback_to_rule)
    out["decision_layer"] = layer
    if insurance_market_enabled is not None:
        sim = dict(out.get("simulation", {}) or {})
        sim["enable_insurance_market"] = bool(insurance_market_enabled)
        out["simulation"] = sim
    if seed is not None:
        sim = dict(out.get("simulation", {}) or {})
        sim["seed"] = int(seed)
        out["simulation"] = sim
    return out


def _encode_rng_state(rng: random.Random) -> str:
    return base64.b64encode(pickle.dumps(rng.getstate())).decode("ascii")


def _decode_rng_state(value: str):
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def _restore_dataclass(cls, value: Optional[dict]):
    if value is None:
        return None
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: val for key, val in dict(value).items() if key in allowed})


def _row_day(row: dict) -> int:
    try:
        return int(row.get("day", -1))
    except (TypeError, ValueError):
        return -1


def latest_checkpoint_path(run_dir: Path) -> Path:
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("day_*.json"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {checkpoint_dir}")
    return checkpoints[-1]


def prepare_run_dir(base_dir: Path, run_name: str, overwrite: bool = False, resume: bool = False) -> Path:
    run_dir = Path(base_dir) / run_name
    if resume and overwrite:
        raise ValueError("--resume and --overwrite cannot be used together.")
    if resume and not run_dir.exists():
        raise FileNotFoundError(f"Cannot resume because run directory does not exist: {run_dir}")
    if run_dir.exists() and overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
