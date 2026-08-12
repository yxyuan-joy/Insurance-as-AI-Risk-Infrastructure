from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
BENCHMARK_ENGINE = ROOT / "autoclaw" / "engine"
if str(BENCHMARK_ENGINE) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ENGINE))

from tests.build_fixture import build_fixture

TEST_DATA_DIR = ROOT / ".test-data"
TEST_CONFIG = ROOT / "tests" / "configs" / "base.yaml"
TEST_STRESS_CONFIG = ROOT / "tests" / "configs" / "stress_tail.yaml"
build_fixture(TEST_DATA_DIR)

from action_risk_v2.simulator import (
    ActionRiskSimulator,
    latest_checkpoint_path,
    _apply_initial_adoption_trial_cap,
    _bounded_term_days,
    _load_yaml_config,
    _resolve_paths,
)
from action_risk_v2.decisions import (
    HeuristicDecisionPolicy,
    MarketContext,
    ModelDecisionPolicy,
    _adoption_diffusion_state,
    _model_adoption_threshold,
)
from action_risk_v2.data import ActionRiskPanel
from action_risk_v2.insurers import InsuranceMarket, load_insurer_profiles
from action_risk_v2.negotiation import NegotiationEngine
from action_risk_v2.schema import ActionRiskRecord, FirmProfile, FirmState, IndustryRiskSnapshot, InsurancePolicy, InsuranceQuote, VendorContract, VendorProfile
from autoclaw_runner import AutoClawRunner
from prepare_autoclaw_panel import prepare_panel


def _resolved_test_config() -> dict:
    return _resolve_paths(_load_yaml_config(TEST_CONFIG), root=ROOT)


def _resolved_test_config() -> dict:
    return _resolve_paths(_load_yaml_config(TEST_CONFIG), root=ROOT)


class ActionRiskV2SmokeTest(unittest.TestCase):
    def test_limited_firm_selection_is_industry_stratified(self):
        selected = [f"comm_{i}" for i in range(20)] + [f"fin_{i}" for i in range(5)] + [f"it_{i}" for i in range(5)]
        profiles = {}
        rows = []
        for fid in selected:
            industry = "communication_services" if fid.startswith("comm_") else ("financials" if fid.startswith("fin_") else "information_technology")
            profiles[fid] = FirmProfile(firm_id=fid, name=fid, industry=industry, cash=100000.0, asset_value=100000.0)
            rows.append(
                {
                    "firm_id": fid,
                    "day": 0,
                    "industry": industry,
                    "num_tasks": 1,
                    "incident_any_flag": 0,
                    "incident_task_count": 0,
                    "avg_severity": 0.0,
                    "sum_total_loss": 0.0,
                    "avg_risk_score": 0.0,
                    "max_risk_score": 0.0,
                }
            )

        panel = ActionRiskPanel(records=pd.DataFrame(rows), profiles=profiles, selected_firms=selected)
        limited = panel.firm_ids(limit=9)
        industries = {profiles[fid].industry for fid in limited}

        self.assertEqual(len(limited), 9)
        self.assertGreaterEqual(len(industries), 3)
        self.assertNotEqual(limited, selected[:9])

    def test_industry_snapshot_uses_only_prior_days(self):
        profiles = {
            "firm_a": FirmProfile(firm_id="firm_a", name="firm_a", industry="industrials", cash=100000.0, asset_value=100000.0),
            "firm_b": FirmProfile(firm_id="firm_b", name="firm_b", industry="industrials", cash=100000.0, asset_value=100000.0),
        }
        records = pd.DataFrame(
            [
                {
                    "firm_id": "firm_a",
                    "day": 0,
                    "industry": "industrials",
                    "num_tasks": 1,
                    "incident_any_flag": 1,
                    "incident_task_count": 1,
                    "avg_severity": 0.9,
                    "sum_total_loss": 1000.0,
                    "avg_risk_score": 0.8,
                    "max_risk_score": 0.9,
                },
                {
                    "firm_id": "firm_b",
                    "day": 1,
                    "industry": "industrials",
                    "num_tasks": 1,
                    "incident_any_flag": 1,
                    "incident_task_count": 1,
                    "avg_severity": 0.4,
                    "sum_total_loss": 5000.0,
                    "avg_risk_score": 0.4,
                    "max_risk_score": 0.5,
                },
            ]
        )
        panel = ActionRiskPanel(records=records, profiles=profiles, selected_firms=profiles)

        day0 = panel.industry_snapshot("industrials", day=0)
        self.assertEqual(day0.observations, 0)
        self.assertEqual(day0.avg_loss, 0.0)
        self.assertEqual(day0.stress_loss, 0.0)

        day1 = panel.industry_snapshot("industrials", day=1)
        self.assertEqual(day1.observations, 1)
        self.assertAlmostEqual(day1.avg_loss, 1000.0)
        self.assertAlmostEqual(day1.stress_loss, 1000.0)

    def test_quote_selection_penalizes_overpriced_high_coverage_policy(self):
        policy = HeuristicDecisionPolicy(
            {
                "max_premium_cash_share": 0.035,
                "target_premium_cash_share": 0.012,
                "quote_price_sensitivity": 0.46,
                "quote_deductible_weight": 0.30,
                "min_quote_utility": -0.04,
            },
            random.Random(7),
        )
        firm = FirmState(
            FirmProfile(
                firm_id="firm_quote_mid",
                name="firm_quote_mid",
                industry="industrials",
                cash=100_000.0,
                asset_value=100_000.0,
                risk_tolerance=0.70,
                ai_dependency=0.45,
            ),
            cash=100_000.0,
        )
        quotes = [
            InsuranceQuote(
                insurer_id="Insurer_Mutual_Commercial",
                firm_id=firm.profile.firm_id,
                vendor_id="Vendor_Beta",
                industry=firm.profile.industry,
                day=0,
                term_days=30,
                premium=700.0,
                deductible_ratio=0.30,
                coverage_ratio=0.59,
                limit_money=10_000.0,
                incident_threshold=0.36,
                expected_loss=300.0,
                stress_loss=1_000.0,
                regime="NORMAL",
            ),
            InsuranceQuote(
                insurer_id="Insurer_Specialty_Tech",
                firm_id=firm.profile.firm_id,
                vendor_id="Vendor_Beta",
                industry=firm.profile.industry,
                day=0,
                term_days=30,
                premium=1_250.0,
                deductible_ratio=0.34,
                coverage_ratio=0.74,
                limit_money=15_000.0,
                incident_threshold=0.34,
                expected_loss=300.0,
                stress_loss=1_000.0,
                regime="NORMAL",
            ),
        ]

        selected, diagnostics = policy.choose_quote_with_diagnostics(firm, quotes, risk_need=0.55)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.insurer_id, "Insurer_Mutual_Commercial")
        self.assertTrue(all(reason in {"candidate", "low_net_utility"} for _, _, reason in diagnostics))

    def test_quote_selection_can_choose_specialty_when_tail_need_and_price_are_close(self):
        policy = HeuristicDecisionPolicy(
            {
                "max_premium_cash_share": 0.035,
                "target_premium_cash_share": 0.012,
                "quote_price_sensitivity": 0.46,
                "quote_deductible_weight": 0.30,
                "min_quote_utility": -0.04,
            },
            random.Random(8),
        )
        firm = FirmState(
            FirmProfile(
                firm_id="firm_quote_high",
                name="firm_quote_high",
                industry="information_technology",
                cash=120_000.0,
                asset_value=100_000.0,
                risk_tolerance=0.18,
                ai_dependency=0.92,
            ),
            cash=120_000.0,
        )
        quotes = [
            InsuranceQuote(
                insurer_id="Insurer_Digital_CN",
                firm_id=firm.profile.firm_id,
                vendor_id="Vendor_Gamma",
                industry=firm.profile.industry,
                day=0,
                term_days=30,
                premium=760.0,
                deductible_ratio=0.30,
                coverage_ratio=0.56,
                limit_money=8_000.0,
                incident_threshold=0.36,
                expected_loss=520.0,
                stress_loss=1_700.0,
                regime="NORMAL",
            ),
            InsuranceQuote(
                insurer_id="Insurer_Specialty_Tech",
                firm_id=firm.profile.firm_id,
                vendor_id="Vendor_Gamma",
                industry=firm.profile.industry,
                day=0,
                term_days=30,
                premium=850.0,
                deductible_ratio=0.34,
                coverage_ratio=0.74,
                limit_money=15_000.0,
                incident_threshold=0.34,
                expected_loss=520.0,
                stress_loss=1_700.0,
                regime="NORMAL",
            ),
        ]

        selected, _ = policy.choose_quote_with_diagnostics(firm, quotes, risk_need=0.92)

        self.assertIsNotNone(selected)
        self.assertEqual(selected.insurer_id, "Insurer_Specialty_Tech")

    def test_insurance_pricing_uses_deductible_in_expected_payout(self):
        firm = FirmState(
            FirmProfile(
                firm_id="firm_pricing",
                name="firm_pricing",
                industry="financials",
                cash=120_000.0,
                asset_value=600_000.0,
                ai_dependency=0.80,
            ),
            cash=120_000.0,
        )
        vendor = VendorProfile("Vendor_Alpha", "Stable", 2_800.0, 0.011, 0.62, 0.78, 1.05)
        snapshot = IndustryRiskSnapshot(
            industry="financials",
            day=0,
            observations=20,
            incident_rate=0.10,
            avg_severity=0.70,
            avg_loss=4_000.0,
            avg_risk_score=0.65,
            stress_loss=18_000.0,
            stress_risk_score=0.90,
        )
        low_deductible, high_deductible = load_insurer_profiles(
            [
                {
                    "id": "low",
                    "initial_capital": 1_000_000.0,
                    "deductible_ratio": 0.10,
                    "coverage_ratio": 0.70,
                    "limit_ratio": 0.12,
                },
                {
                    "id": "high",
                    "initial_capital": 1_000_000.0,
                    "deductible_ratio": 0.60,
                    "coverage_ratio": 0.70,
                    "limit_ratio": 0.12,
                },
            ]
        )
        market = InsuranceMarket(
            [low_deductible, high_deductible],
            pricing_config={"global_multiplier": 1.0, "premium_cap_asset_share": 1.0},
        )

        quotes = {
            q.insurer_id: q
            for q in market.quote_all(
                firm=firm,
                vendor=vendor,
                snapshot=snapshot,
                day=0,
                term_days=30,
                market_panic=0.0,
                recent_claim_rate=0.0,
            )
        }

        self.assertLess(quotes["high"].premium, quotes["low"].premium)

    def test_insurance_product_adjustments_change_quote_terms(self):
        firm = FirmState(
            FirmProfile(
                firm_id="firm_product_design",
                name="firm_product_design",
                industry="information_technology",
                cash=150_000.0,
                asset_value=500_000.0,
                ai_dependency=0.85,
            ),
            cash=150_000.0,
        )
        vendor = VendorProfile("Vendor_Beta", "Efficient", 2_200.0, 0.013, 0.82, 0.68, 0.95)
        snapshot = IndustryRiskSnapshot(
            industry="information_technology",
            day=0,
            observations=20,
            incident_rate=0.12,
            avg_severity=0.72,
            avg_loss=5_000.0,
            avg_risk_score=0.70,
            stress_loss=20_000.0,
            stress_risk_score=0.95,
        )
        profile = load_insurer_profiles(
            [
                {
                    "id": "product",
                    "initial_capital": 1_000_000.0,
                    "deductible_ratio": 0.30,
                    "coverage_ratio": 0.60,
                    "limit_ratio": 0.10,
                }
            ]
        )[0]
        market = InsuranceMarket(
            [profile],
            pricing_config={
                "deductible_ratio_delta": -0.08,
                "coverage_ratio_delta": 0.12,
                "limit_ratio_multiplier": 1.75,
                "incident_threshold_delta": -0.04,
                "premium_cap_asset_share": 1.0,
            },
        )
        quote = market.quote_all(firm, vendor, snapshot, day=0, term_days=30, market_panic=0.0, recent_claim_rate=0.0)[0]

        self.assertAlmostEqual(quote.deductible_ratio, 0.22)
        self.assertAlmostEqual(quote.coverage_ratio, 0.72)
        self.assertAlmostEqual(quote.limit_money, firm.profile.asset_value * 0.10 * 1.75)
        self.assertLess(quote.incident_threshold, 0.36)

    def test_model_exposure_management_respects_abandon_threshold(self):
        class FixedResponsePolicy(ModelDecisionPolicy):
            def _complete(self, prompt: str, payload: dict) -> str:
                return json.dumps(
                    {
                        "abandon_ai": True,
                        "abandon_score": 0.01,
                        "vendor_action": "abandon_ai",
                        "reason": "wants to switch despite low realized exposure",
                    }
                )

        policy = FixedResponsePolicy(
            {"abandon_score_threshold": 0.62},
            random.Random(11),
            {"mode": "vllm_openai", "fallback_to_rule": False},
        )
        firm = FirmState(
            FirmProfile(
                firm_id="firm_exposure_threshold",
                name="firm_exposure_threshold",
                industry="industrials",
                cash=100_000.0,
                asset_value=100_000.0,
            ),
            cash=100_000.0,
            vendor_contract=VendorContract(vendor_id="Vendor_Beta", price=1_000.0, monthly_fee=1_000.0, start_day=0, end_day=60),
        )
        decision = policy.exposure_decision(
            firm,
            MarketContext(day=10, adoption_rate=0.30, insurance_coverage_rate=0.40, avg_panic=0.03, recent_claim_rate=0.01),
            {
                "material_event_score": 0.0,
                "claimable_event_score": 0.0,
                "ai_remaining_days": 50,
            },
        )

        self.assertFalse(decision["action"])
        self.assertEqual(decision["vendor_action"], "keep_vendor")
        self.assertIn("below_decision_threshold", decision["reason"])

    def test_model_exposure_management_forces_high_abandon_score_consistency(self):
        class InconsistentResponsePolicy(ModelDecisionPolicy):
            def _complete(self, prompt: str, payload: dict) -> str:
                return json.dumps(
                    {
                        "abandon_ai": False,
                        "abandon_score": 0.91,
                        "vendor_action": "keep_vendor",
                        "reason": "high risk score but verbally wants to keep vendor",
                    }
                )

        policy = InconsistentResponsePolicy(
            {"abandon_score_threshold": 0.62},
            random.Random(12),
            {"mode": "vllm_openai", "fallback_to_rule": False},
        )
        firm = FirmState(
            FirmProfile(
                firm_id="firm_exposure_inconsistent",
                name="firm_exposure_inconsistent",
                industry="industrials",
                cash=100_000.0,
                asset_value=100_000.0,
            ),
            cash=100_000.0,
            vendor_contract=VendorContract(vendor_id="Vendor_Beta", price=1_000.0, monthly_fee=1_000.0, start_day=0, end_day=60),
        )
        decision = policy.exposure_decision(
            firm,
            MarketContext(day=10, adoption_rate=0.30, insurance_coverage_rate=0.40, avg_panic=0.03, recent_claim_rate=0.01),
            {
                "material_event_score": 0.80,
                "claimable_event_score": 0.90,
                "ai_remaining_days": 50,
            },
        )

        self.assertTrue(decision["action"])
        self.assertEqual(decision["vendor_action"], "abandon_ai")
        self.assertIn("score_threshold_forced_abandon", decision["reason"])

    def test_model_adoption_guard_blocks_optimistic_llm_adoption(self):
        class OptimisticAdoptionPolicy(ModelDecisionPolicy):
            def _complete(self, prompt: str, payload: dict) -> str:
                return json.dumps(
                    {
                        "adopt_ai": True,
                        "adoption_score": 0.95,
                        "selected_vendor_id": "Vendor_Beta",
                        "vendor_term_days": 60,
                        "max_rounds": 10,
                        "reason": "optimistic local adoption story",
                    }
                )

        policy = OptimisticAdoptionPolicy(
            {
                "insurance_market_enabled": True,
                "model_adoption_guard_enabled": True,
                "model_adoption_guard_slack": 0.0,
                "model_adoption_base_threshold": 0.82,
                "model_adoption_min_threshold": 0.78,
                "model_adoption_max_threshold": 0.95,
                "model_adoption_maturity_days": 120,
                "model_adoption_early_friction": 0.20,
                "model_adoption_local_evidence_reference": 0.60,
                "model_adoption_local_evidence_floor": 0.12,
                "model_adoption_local_evidence_lag_days": 45,
                "model_adoption_implementation_uncertainty": 0.08,
                "model_adoption_local_evidence_relief": 0.05,
                "model_adoption_insurance_confidence_discount": 0.02,
            },
            random.Random(13),
            {"mode": "vllm_openai", "fallback_to_rule": False},
        )
        firm = FirmState(
            FirmProfile(
                firm_id="firm_guarded_adoption",
                name="firm_guarded_adoption",
                industry="industrials",
                cash=100_000.0,
                asset_value=100_000.0,
                tech_urgency=0.10,
                innovativeness=0.10,
                ai_dependency=0.10,
                risk_tolerance=0.20,
                inertia=0.90,
            ),
            cash=100_000.0,
        )
        visible = [VendorProfile("Vendor_Beta", "Efficient", 2_200.0, 0.013, 0.82, 0.68, 0.95)]

        decision = policy.adoption_decision(
            firm,
            MarketContext(
                day=0,
                adoption_rate=0.0,
                insurance_coverage_rate=0.0,
                avg_panic=0.0,
                recent_claim_rate=0.0,
                local_adoption_rate=0.0,
                local_insurance_coverage_rate=0.0,
                local_avg_panic=0.0,
                local_recent_claim_rate=0.0,
            ),
            visible,
        )

        self.assertFalse(decision["action"])
        self.assertEqual(decision["selected_vendor_id"], "")
        self.assertEqual(decision["vendor_term_days"], 0)
        self.assertIn("adoption_guard_blocked", decision["reason"])
        self.assertFalse(decision["model_trace"]["adoption_guard"]["passed"])

    def test_model_adoption_required_margin_blocks_barely_positive_adoption(self):
        class BarelyPositiveAdoptionPolicy(ModelDecisionPolicy):
            def _complete(self, prompt: str, payload: dict) -> str:
                threshold = float(payload["decision_threshold"])
                return json.dumps(
                    {
                        "adopt_ai": True,
                        "adoption_score": threshold + 0.004,
                        "selected_vendor_id": "Vendor_Beta",
                        "vendor_term_days": 90,
                        "max_rounds": 10,
                        "reason": "barely above threshold",
                    }
                )

        policy = BarelyPositiveAdoptionPolicy(
            {
                "insurance_market_enabled": True,
                "model_adoption_required_margin": 0.010,
                "model_adoption_base_threshold": 0.62,
                "model_adoption_min_threshold": 0.50,
                "model_adoption_max_threshold": 0.95,
                "model_adoption_maturity_days": 100,
                "model_adoption_early_friction": 0.10,
                "model_adoption_local_evidence_reference": 0.55,
                "model_adoption_local_evidence_floor": 0.10,
                "model_adoption_local_evidence_lag_days": 60,
                "model_adoption_insurance_confidence_discount": 0.02,
            },
            random.Random(14),
            {"mode": "vllm_openai", "fallback_to_rule": False},
        )
        firm = FirmState(
            FirmProfile(
                firm_id="firm_margin_adoption",
                name="firm_margin_adoption",
                industry="industrials",
                cash=100_000.0,
                asset_value=100_000.0,
                tech_urgency=0.70,
                innovativeness=0.60,
                ai_dependency=0.60,
                risk_tolerance=0.45,
                inertia=0.35,
            ),
            cash=100_000.0,
        )
        visible = [VendorProfile("Vendor_Beta", "Efficient", 2_200.0, 0.013, 0.82, 0.68, 0.95)]

        decision = policy.adoption_decision(
            firm,
            MarketContext(
                day=55,
                adoption_rate=0.50,
                insurance_coverage_rate=0.60,
                avg_panic=0.0,
                recent_claim_rate=0.0,
                local_adoption_rate=0.50,
                local_insurance_coverage_rate=0.60,
                local_avg_panic=0.0,
                local_recent_claim_rate=0.0,
            ),
            visible,
        )

        self.assertFalse(decision["action"])
        self.assertEqual(decision["selected_vendor_id"], "")
        self.assertEqual(decision["vendor_term_days"], 0)
        self.assertEqual(decision["max_rounds"], 0)
        self.assertIn("below_required_decision_margin", decision["reason"])

    def test_initial_adoption_trial_cap_only_limits_low_margin_new_adoption(self):
        lifecycle = {
            "vendor_min_term_days": 14,
            "initial_adoption_trial_term_cap_days": 60,
            "initial_adoption_long_term_margin": 0.05,
        }
        low_margin = {
            "action": True,
            "score": 0.624,
            "threshold": 0.620,
            "vendor_term_days": 120,
            "term_days": 120,
            "reason": "model:low_margin",
        }
        capped = _apply_initial_adoption_trial_cap(low_margin, renewal=False, lifecycle_config=lifecycle)
        self.assertEqual(capped["vendor_term_days"], 60)
        self.assertEqual(capped["term_days"], 60)
        self.assertIn("initial_adoption_trial_term_cap:120->60", capped["reason"])

        high_margin = {
            **low_margin,
            "score": 0.700,
            "reason": "model:high_margin",
        }
        uncapped = _apply_initial_adoption_trial_cap(high_margin, renewal=False, lifecycle_config=lifecycle)
        self.assertEqual(uncapped["vendor_term_days"], 120)

        renewal = _apply_initial_adoption_trial_cap(low_margin, renewal=True, lifecycle_config=lifecycle)
        self.assertEqual(renewal["vendor_term_days"], 120)

    def test_no_insurance_model_payload_marks_market_unavailable(self):
        class CapturingPolicy(ModelDecisionPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.last_payload = None
                self.last_prompt = None

            def _complete(self, prompt: str, payload: dict) -> str:
                self.last_payload = payload
                self.last_prompt = prompt
                return json.dumps({"abandon_ai": False, "abandon_score": 0.0, "vendor_action": "keep_vendor"})

        policy = CapturingPolicy(
            {"insurance_market_enabled": False},
            random.Random(12),
            {"mode": "vllm_openai", "fallback_to_rule": False},
        )
        firm = FirmState(
            FirmProfile(
                firm_id="firm_no_ins_prompt",
                name="firm_no_ins_prompt",
                industry="industrials",
                cash=100_000.0,
                asset_value=100_000.0,
            ),
            cash=100_000.0,
            vendor_contract=VendorContract(vendor_id="Vendor_Beta", price=1_000.0, monthly_fee=1_000.0, start_day=0, end_day=60),
        )

        policy.exposure_decision(
            firm,
            MarketContext(day=10, adoption_rate=0.30, insurance_coverage_rate=0.0, avg_panic=0.03, recent_claim_rate=0.0),
            {
                "material_event_score": 0.0,
                "claimable_event_score": 0.0,
                "ai_remaining_days": 50,
                "pre_operation_signal": True,
            },
        )

        self.assertIsNotNone(policy.last_payload)
        self.assertEqual(policy.last_payload["firm"]["insurance_market_available"], 0.0)
        self.assertIn("decision_threshold", policy.last_payload)
        self.assertIn("abandon_score is the pressure to exit current AI exposure", policy.last_prompt)
        self.assertIn("same-day incidents, losses, and claimable events are not observed yet", policy.last_prompt)

    def test_autoclaw_runner_marks_zero_change_success_as_no_op(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = AutoClawRunner(command="/usr/bin/true", model="ignored", timeout_sec=5)
            result = runner.run_episode("ignored prompt", tmp)

            self.assertFalse(result["success"])
            self.assertTrue(result["no_op_execution"])
            self.assertFalse(result["workspace_changed"])
            self.assertEqual(result["changed_path_count"], 0)

    def test_autoclaw_runner_accepts_successful_workspace_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            script_path = Path(tmp) / "write_result.sh"
            script_path.write_text("#!/usr/bin/env bash\nprintf ok > result.txt\n", encoding="utf-8")
            script_path.chmod(0o755)

            runner = AutoClawRunner(command=str(script_path), model="ignored", timeout_sec=5)
            result = runner.run_episode("ignored prompt", tmp)

            self.assertTrue(result["success"])
            self.assertFalse(result["no_op_execution"])
            self.assertTrue(result["workspace_changed"])
            self.assertIn("result.txt", result["changed_paths"])

    def test_small_run_writes_clean_action_risk_outputs(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "smoke"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=4, firms=25)
            sim.run()

            with (run_dir / "firm_daily.csv").open(newline="", encoding="utf-8") as f:
                firm_rows = list(csv.DictReader(f))
            self.assertEqual(len(firm_rows), 4 * 25)

            forbidden = {"mse", "baseline_mse", "vendor_mse", "y_true", "y_pred"}
            self.assertTrue(forbidden.isdisjoint(set(firm_rows[0].keys())))
            self.assertIn("vendor_monthly_fee", firm_rows[0])
            self.assertIn("vendor_contract_total_fee", firm_rows[0])
            self.assertIn("had_ai_during_day", firm_rows[0])
            self.assertIn("vendor_id_during_day", firm_rows[0])
            self.assertIn("had_insurance_during_day", firm_rows[0])
            self.assertIn("insurer_id_during_day", firm_rows[0])
            self.assertIn("insurance_refund_received", firm_rows[0])
            self.assertIn("vendor_refund_received", firm_rows[0])

            with (run_dir / "macro_daily.csv").open(newline="", encoding="utf-8") as f:
                macro_rows = list(csv.DictReader(f))
            self.assertEqual(len(macro_rows), 4)
            self.assertIn("insurance_coverage_ai_adopters", macro_rows[0])
            self.assertIn("active_firm_cash", macro_rows[0])
            self.assertIn("inactive_firm_cash", macro_rows[0])
            self.assertIn("social_total_capital_active_firms", macro_rows[0])
            self.assertIn("social_total_capital_all_firms", macro_rows[0])
            self.assertIn("panic_p95", macro_rows[0])
            self.assertIn("panic_p99", macro_rows[0])
            self.assertIn("panic_nonzero_share", macro_rows[0])
            self.assertIn("claim_residual_loss", macro_rows[0])
            self.assertIn("unabsorbed_claimable_loss", macro_rows[0])
            self.assertEqual(
                float(macro_rows[0]["social_total_capital"]),
                float(macro_rows[0]["social_total_capital_all_firms"]),
            )

            with (run_dir / "insurer_daily.csv").open(newline="", encoding="utf-8") as f:
                insurer_rows = list(csv.DictReader(f))
            self.assertEqual(len(insurer_rows), 4 * 5)
            self.assertIn("insurer_id", insurer_rows[0])
            self.assertIn("refunds_today", insurer_rows[0])

            with (run_dir / "quotes.csv").open(newline="", encoding="utf-8") as f:
                quote_rows = list(csv.DictReader(f))
            self.assertIn("insurer_id", quote_rows[0] if quote_rows else {"insurer_id": ""})
            self.assertIn("selection_reason", quote_rows[0] if quote_rows else {"selection_reason": ""})

            self.assertTrue((run_dir / "action_risk_input_audit.json").exists())
            self.assertTrue((run_dir / "industry_risk_audit.csv").exists())
            self.assertTrue((run_dir / "decisions.csv").exists())
            self.assertIn("claim_paid_flag", firm_rows[0])
            self.assertIn("claim_residual_loss", firm_rows[0])
            self.assertIn("unabsorbed_claimable_loss", firm_rows[0])
            self.assertIn("uninsured_claimable_flag", firm_rows[0])
            with (run_dir / "decisions.csv").open(newline="", encoding="utf-8") as f:
                decision_rows = list(csv.DictReader(f))
            self.assertIn("visible_vendor_ids", decision_rows[0])
            self.assertIn("visible_vendor_count", decision_rows[0])
            vendor_rows = [row for row in decision_rows if row["decision_type"] in {"ai_adoption", "vendor_renewal"}]
            self.assertTrue(vendor_rows)
            self.assertTrue(all(int(row["visible_vendor_count"]) <= 3 for row in vendor_rows))

    def test_social_capital_reports_all_firms_and_active_firm_view(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "capital_metrics", days=1, firms=5)
            try:
                firms = list(sim.firms.values())
                firm = firms[0]
                firm.cash = -1234.0
                firm.active = False
                for idx, state in enumerate(firms[1:], start=1):
                    state.panic = 0.10 * idx

                row = sim._macro_row(day=0, cumulative_bankruptcies=1)
            finally:
                sim.close()

        self.assertLess(float(row["inactive_firm_cash"]), 0.0)
        self.assertAlmostEqual(
            float(row["social_total_capital"]),
            float(row["social_total_capital_all_firms"]),
        )
        self.assertAlmostEqual(
            float(row["social_total_capital_active_firms"]) + float(row["inactive_firm_cash"]),
            float(row["social_total_capital_all_firms"]),
        )
        self.assertGreater(float(row["panic_p95"]), float(row["avg_panic"]))
        self.assertGreater(float(row["panic_p99"]), float(row["panic_p95"]))
        self.assertAlmostEqual(float(row["panic_nonzero_share"]), 1.0)

    def test_traditional_pnl_is_stable_by_firm_day_not_rng_position(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "stable_pnl", days=1, firms=5)
            try:
                firm = next(iter(sim.firms.values()))
                first = sim._traditional_pnl(firm, day=0)
                for _ in range(100):
                    sim.rng.random()
                second = sim._traditional_pnl(firm, day=0)
                next_day = sim._traditional_pnl(firm, day=1)
            finally:
                sim.close()

        self.assertEqual(first, second)
        self.assertNotEqual(first, next_day)

    def test_common_random_decision_channels_are_not_shifted_by_insurance_draws(self):
        profile = FirmProfile(
            firm_id="F_common_rng",
            name="Common RNG adoption firm",
            industry="financials",
            cash=200_000.0,
            asset_value=700_000.0,
            risk_tolerance=0.45,
            tech_urgency=0.70,
            ai_dependency=0.80,
            inertia=0.50,
            innovativeness=0.65,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        insured_profile = FirmProfile(
            firm_id="F_common_rng_insured",
            name="Common RNG insurance firm",
            industry="financials",
            cash=200_000.0,
            asset_value=700_000.0,
            risk_tolerance=0.35,
            tech_urgency=0.80,
            ai_dependency=0.85,
            inertia=0.40,
            innovativeness=0.60,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=200_000.0)
        insurance_firm = FirmState(
            profile=insured_profile,
            cash=200_000.0,
            vendor_contract=VendorContract(
                vendor_id="Vendor_Alpha",
                price=6_000.0,
                monthly_fee=2_000.0,
                start_day=0,
                end_day=45,
            ),
        )
        vendors = [
            VendorProfile("Vendor_Alpha", "Alpha", 2_000.0, 0.010, 0.90, 0.70, 1.00, ("ALL",)),
            VendorProfile("Vendor_Beta", "Beta", 2_200.0, 0.012, 1.10, 0.60, 0.85, ("financials",)),
            VendorProfile("Vendor_Gamma", "Gamma", 1_800.0, 0.009, 0.80, 0.80, 0.70, ("ALL",)),
        ]
        context = MarketContext(day=7, adoption_rate=0.20, insurance_coverage_rate=0.15, avg_panic=0.10, recent_claim_rate=0.02)
        policy = HeuristicDecisionPolicy(
            config={
                "common_random_numbers": True,
                "common_random_seed": 20260606,
                "ads_bandwidth": 2,
                "insurance_buy_threshold": 0.10,
                "insurance_max_term_days": 90,
            },
            rng=random.Random(1),
        )

        baseline_visible = [v.vendor_id for v in policy.visible_vendors(firm, vendors)]
        baseline_decision = policy.adoption_decision(firm, context, vendors)
        policy.insurance_decision(
            insurance_firm,
            context,
            {
                "claimable_event_score": 0.80,
                "material_event_score": 0.70,
                "industry_stress_score": 0.75,
                "ai_remaining_days": 38,
            },
        )
        policy.insurance_term_days(
            insurance_firm,
            context,
            0.80,
            {
                "claimable_event_score": 0.80,
                "material_event_score": 0.70,
                "industry_stress_score": 0.75,
                "ai_remaining_days": 38,
            },
        )
        after_insurance_visible = [v.vendor_id for v in policy.visible_vendors(firm, vendors)]
        after_insurance_decision = policy.adoption_decision(firm, context, vendors)

        self.assertEqual(baseline_visible, after_insurance_visible)
        self.assertEqual(baseline_decision["draw"], after_insurance_decision["draw"])

    def test_common_random_model_endpoint_routing_is_not_shifted_by_insurance_calls(self):
        policy = ModelDecisionPolicy(
            config={
                "common_random_numbers": True,
                "common_random_seed": 20260606,
            },
            rng=random.Random(1),
            layer_config={
                "mode": "vllm_openai",
                "base_urls": "http://endpoint-a/v1,http://endpoint-b/v1,http://endpoint-c/v1",
            },
        )
        adoption_payload = {
            "decision_type": "ai_adoption",
            "day": 12,
            "firm": {"firm_id": "F_endpoint_route"},
        }
        insurance_payload = {
            "decision_type": "insurance_purchase",
            "day": 12,
            "firm": {"firm_id": "F_endpoint_route"},
        }

        baseline = policy._base_url_for_payload(adoption_payload, attempt=0)
        policy._base_url_for_payload(insurance_payload, attempt=0)
        policy._base_url_for_payload(insurance_payload, attempt=1)
        after_insurance = policy._base_url_for_payload(adoption_payload, attempt=0)

        self.assertEqual(baseline, after_insurance)
        self.assertIn(baseline, policy.base_urls)

    def test_common_random_negotiation_endpoint_routing_is_not_shifted_by_insurance_calls(self):
        engine = NegotiationEngine(
            config={
                "decision_layer": {
                    "mode": "vllm_openai",
                    "base_urls": "http://endpoint-a/v1,http://endpoint-b/v1,http://endpoint-c/v1",
                },
                "decision_policy": {
                    "common_random_numbers": True,
                    "common_random_seed": 20260606,
                },
                "negotiation": {},
            },
            rng=random.Random(1),
        )
        vendor_payload = {
            "negotiation_type": "vendor_contract",
            "side": "vendor",
            "day": 12,
            "round": 1,
            "firm": {"firm_id": "F_endpoint_route"},
            "vendor": {"vendor_id": "Vendor_Test"},
        }
        insurance_payload = {
            "negotiation_type": "insurance_policy",
            "side": "insurer",
            "day": 12,
            "round": 1,
            "firm": {"firm_id": "F_endpoint_route"},
            "quote": {"insurer_id": "Insurer_Test", "vendor_id": "Vendor_Test"},
        }

        baseline = engine._base_url_for_payload(vendor_payload, attempt=0)
        engine._base_url_for_payload(insurance_payload, attempt=0)
        engine._base_url_for_payload(insurance_payload, attempt=1)
        after_insurance = engine._base_url_for_payload(vendor_payload, attempt=0)

        self.assertEqual(baseline, after_insurance)
        self.assertIn(baseline, engine.base_urls)

    def test_uninsured_claimable_loss_builds_more_experience_memory_than_paid_claim(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "experience_memory", days=1, firms=5)
            try:
                firms = list(sim.firms.values())
                paid = firms[0]
                uninsured = firms[1]
                signal = {
                    "material_event_score": 0.70,
                    "loss_pressure": 0.60,
                    "claimable_event_score": 0.80,
                }

                paid._claim_paid_today_flag = 1.0
                paid._uninsured_claimable_today_flag = 0.0
                uninsured._claim_paid_today_flag = 0.0
                uninsured._uninsured_claimable_today_flag = 1.0

                sim._update_experience_memory(paid, signal)
                sim._update_experience_memory(uninsured, signal)
            finally:
                sim.close()

        self.assertGreater(uninsured.loss_memory, paid.loss_memory)
        self.assertGreater(uninsured.claimable_memory, paid.claimable_memory)
        self.assertAlmostEqual(uninsured.risk_memory, paid.risk_memory)

    def test_vendor_refund_lifecycle_is_non_linear_and_logged(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "lifecycle"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=5)
            try:
                sim._reset_daily_cashflow_markers()
                firm = next(iter(sim.firms.values()))
                firm.cash = 10_000.0
                firm.vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=6_000.0,
                    monthly_fee=3_000.0,
                    start_day=0,
                    end_day=60,
                )
                sim.vendor_capital["Vendor_Alpha"] += 6_000.0
                vendor_capital_before = sim.vendor_capital["Vendor_Alpha"]

                self.assertAlmostEqual(sim._vendor_total_fee(3_000.0, 60), 6_000.0)
                refund = sim._cancel_vendor_contract(firm=firm, day=30, reason="unit_test")
                self.assertAlmostEqual(refund, 450.0)
                self.assertIsNone(firm.vendor_contract)
                self.assertAlmostEqual(firm.cash, 10_450.0)
                self.assertAlmostEqual(sim.vendor_capital["Vendor_Alpha"], vendor_capital_before - 450.0)
                self.assertAlmostEqual(getattr(firm, "_vendor_refund_today", 0.0), 450.0)
            finally:
                sim.close()

            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row.get("event_type") == "vendor_contract_cancelled" for row in events))

    def test_bound_insurance_policy_preserves_quote_vendor_id(self):
        market = InsuranceMarket(
            load_insurer_profiles(
                [
                    {
                        "id": "Insurer_Test",
                        "label": "Test insurer",
                        "domicile": "US",
                        "initial_capital": 1_000_000.0,
                        "base_margin": 0.15,
                        "risk_appetite": 0.50,
                        "expense_load": 0.05,
                        "capital_load": 0.10,
                        "deductible_ratio": 0.20,
                        "coverage_ratio": 0.70,
                        "limit_ratio": 0.10,
                        "max_active_policies": 100,
                        "solvency_floor_ratio": 0.20,
                        "soft_threshold_ratio": 0.45,
                        "hard_threshold_ratio": 0.30,
                    }
                ]
            )
        )
        quote = InsuranceQuote(
            insurer_id="Insurer_Test",
            firm_id="F001",
            vendor_id="Vendor_Beta",
            industry="financials",
            day=3,
            term_days=20,
            premium=500.0,
            deductible_ratio=0.20,
            coverage_ratio=0.70,
            limit_money=25_000.0,
            incident_threshold=0.30,
            expected_loss=200.0,
            stress_loss=1_000.0,
            regime="NORMAL",
            market_role="private",
        )

        policy = market.bind_policy(quote, day=3)

        self.assertEqual(policy.vendor_id, "Vendor_Beta")
        self.assertEqual(policy.end_day, 23)

    def test_vendor_mismatched_insurance_policy_is_cancelled_before_reuse(self):
        class KeepAiSkipInsurancePolicy(HeuristicDecisionPolicy):
            def visible_vendors(self, firm, vendors):
                return []

            def adoption_decision(self, firm, context, visible, renewal=False):
                return {"action": False, "score": 0.0, "reason": "no_adoption"}

            def exposure_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "keep_vendor", "vendor_action": "keep_vendor"}

            def insurance_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "skip"}

        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "policy_vendor_mismatch"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=5)
            try:
                sim._reset_daily_cashflow_markers()
                sim.insurance_market.start_day()
                firm = next(iter(sim.firms.values()))
                firm.vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=4_000.0,
                    monthly_fee=2_000.0,
                    start_day=0,
                    end_day=30,
                )
                firm.insurance_policy = InsurancePolicy(
                    insurer_id="Insurer_Apex_Global",
                    premium=0.0,
                    deductible_ratio=0.10,
                    coverage_ratio=0.80,
                    limit_money=50_000.0,
                    incident_threshold=0.0,
                    start_day=0,
                    end_day=30,
                    vendor_id="Vendor_Beta",
                )
                sim.policy = KeepAiSkipInsurancePolicy(sim.config.get("decision_policy", {}), sim.rng)

                sim._make_contract_decisions(day=5, context=sim._market_context(day=5))

                self.assertIsNone(firm.insurance_policy)
            finally:
                sim.close()

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(
                any(
                    row.get("event_type") == "insurance_policy_cancelled"
                    and row.get("reason") == "cancelled_vendor_mismatch_post_decision"
                    and row.get("vendor_id") == "Vendor_Beta"
                    for row in events
                )
            )

    def test_vendor_expiry_does_not_cancel_insurance_before_renewal_attempt(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "expiry"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=5)
            try:
                firm = next(iter(sim.firms.values()))
                firm.vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=3_000.0,
                    monthly_fee=3_000.0,
                    start_day=0,
                    end_day=1,
                )
                firm.insurance_policy = InsurancePolicy(
                    insurer_id="Apex_AI_Assurance",
                    premium=500.0,
                    deductible_ratio=0.30,
                    coverage_ratio=0.60,
                    limit_money=10_000.0,
                    incident_threshold=0.35,
                    start_day=0,
                    end_day=10,
                )

                sim._reset_daily_cashflow_markers()
                sim.insurance_market.start_day()
                sim._expire_contracts(day=1)

                self.assertIsNone(firm.vendor_contract)
                self.assertTrue(getattr(firm, "_vendor_expired_today", False))
                self.assertIsNotNone(firm.insurance_policy)
                self.assertEqual(firm.insurance_policy.insurer_id, "Apex_AI_Assurance")
            finally:
                sim.close()

            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(row.get("event_type") == "vendor_contract_expired" for row in events))
            self.assertFalse(any(row.get("event_type") == "insurance_policy_cancelled" for row in events))

    def test_vendor_renewal_can_use_incumbent_even_if_not_in_new_ads(self):
        class IncumbentRenewalPolicy(HeuristicDecisionPolicy):
            def visible_vendors(self, firm, vendors):
                return [v for v in vendors if v.vendor_id != "Vendor_Alpha"][:3]

            def adoption_decision(self, firm, context, visible, renewal=False):
                if renewal and not firm.has_ai:
                    return {
                        "action": True,
                        "score": 0.80,
                        "probability": 0.80,
                        "reason": "renew_incumbent_vendor",
                        "selected_vendor_id": "Vendor_Alpha",
                        "vendor_term_days": 30,
                        "max_rounds": 10,
                    }
                return {"action": False, "score": 0.0, "reason": "no_new_adoption"}

            def exposure_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "reason": "keep_vendor", "vendor_action": "keep_vendor"}

            def insurance_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "skip"}

        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "incumbent_renewal"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=6)
            firm = next(iter(sim.firms.values()))
            firm.cash = 100_000.0
            firm.vendor_contract = VendorContract(
                vendor_id="Vendor_Alpha",
                price=4_200.0,
                monthly_fee=2_100.0,
                start_day=-30,
                end_day=0,
            )
            sim.policy = IncumbentRenewalPolicy(sim.config.get("decision_policy", {}), sim.rng)
            sim.run()

            with (run_dir / "events.jsonl").open(encoding="utf-8") as f:
                events = [json.loads(line) for line in f if line.strip()]
            bound = [
                row
                for row in events
                if row.get("event_type") == "vendor_contract_bound"
                and row.get("firm_id") == firm.profile.firm_id
            ]
            self.assertTrue(bound)
            self.assertEqual(bound[0]["vendor_id"], "Vendor_Alpha")

            with (run_dir / "decisions.csv").open(newline="", encoding="utf-8") as f:
                decisions = list(csv.DictReader(f))
            renewal_rows = [
                row
                for row in decisions
                if row["firm_id"] == firm.profile.firm_id and row["decision_type"] == "vendor_renewal"
            ]
            self.assertTrue(renewal_rows)
            self.assertIn("Vendor_Alpha", renewal_rows[0]["visible_vendor_ids"].split(";"))
            self.assertGreaterEqual(int(renewal_rows[0]["visible_vendor_count"]), 4)

    def test_resume_continues_from_latest_complete_checkpoint(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "resumed"
            first = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=2, firms=25)
            first.run()

            partial_extra = {"event_type": "partial_day_noise", "day": 3, "firm_id": "partial"}
            with (run_dir / "events.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(partial_extra) + "\n")
            with (run_dir / "macro_daily.csv").open("a", encoding="utf-8") as f:
                f.write("3,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial,partial\n")

            resumed = ActionRiskSimulator.from_yaml(
                config_path=config,
                run_dir=run_dir,
                days=4,
                firms=25,
                resume_from=latest_checkpoint_path(run_dir),
            )
            resumed.run()

            with (run_dir / "macro_daily.csv").open(newline="", encoding="utf-8") as f:
                macro_rows = list(csv.DictReader(f))
            self.assertEqual([int(row["day"]) for row in macro_rows], [0, 1, 2, 3])

            with (run_dir / "firm_daily.csv").open(newline="", encoding="utf-8") as f:
                firm_rows = list(csv.DictReader(f))
            self.assertEqual(len(firm_rows), 4 * 25)

            backups = list((run_dir / "resume_backups").glob("*/macro_daily.csv"))
            self.assertTrue(backups)

    def test_resume_matches_uninterrupted_model_mock_run(self):
        config = TEST_CONFIG
        compare_cols = [
            "day",
            "firm_id",
            "active",
            "has_ai",
            "has_insurance",
            "cash_end",
            "base_pnl",
            "ai_gain",
            "operational_loss",
            "claim_paid",
            "panic",
            "risk_memory",
            "loss_memory",
            "claimable_memory",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            full_dir = Path(tmp) / "full"
            full = ActionRiskSimulator.from_yaml(
                config_path=config,
                run_dir=full_dir,
                days=6,
                firms=30,
                decision_mode="model_mock",
                seed=123,
            )
            full.run()

            resumed_dir = Path(tmp) / "resumed"
            first = ActionRiskSimulator.from_yaml(
                config_path=config,
                run_dir=resumed_dir,
                days=2,
                firms=30,
                decision_mode="model_mock",
                seed=123,
            )
            first.run()
            resumed = ActionRiskSimulator.from_yaml(
                config_path=config,
                run_dir=resumed_dir,
                days=6,
                firms=30,
                decision_mode="model_mock",
                seed=123,
                resume_from=latest_checkpoint_path(resumed_dir),
            )
            resumed.run()

            full_macro = pd.read_csv(full_dir / "macro_daily.csv")
            resumed_macro = pd.read_csv(resumed_dir / "macro_daily.csv")
            full_firms = pd.read_csv(full_dir / "firm_daily.csv")[compare_cols]
            resumed_firms = pd.read_csv(resumed_dir / "firm_daily.csv")[compare_cols]

        pd.testing.assert_frame_equal(full_macro, resumed_macro)
        pd.testing.assert_frame_equal(full_firms, resumed_firms)

    def test_model_mock_writes_decision_traces(self):
        config = _resolved_test_config()
        config["decision_layer"]["mode"] = "model_mock"
        config["decision_layer"]["llm_or_vllm_enabled"] = True

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "model_mock"
            sim = ActionRiskSimulator(config=config, run_dir=run_dir, days=2, firms=15)
            sim.run()

            trace_path = run_dir / "model_decisions.jsonl"
            self.assertTrue(trace_path.exists())
            traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertTrue(traces)
            self.assertIn("prompt", traces[0])
            self.assertIn("raw_response", traces[0])
            self.assertEqual(traces[0]["backend"], "model_mock")

    def test_model_adoption_cannot_fallback_to_rule_vendor_selection(self):
        class MissingVendorModelPolicy(HeuristicDecisionPolicy):
            def adoption_decision(self, firm, context, visible, renewal=False):
                if firm.has_ai or not visible:
                    return {"action": False, "score": 0.0, "reason": "not_eligible"}
                return {
                    "action": True,
                    "score": 0.90,
                    "probability": 0.90,
                    "reason": "model_wants_ai_but_omits_vendor",
                    "vendor_term_days": 45,
                    "max_rounds": 10,
                    "model_trace": {
                        "backend": "unit_model",
                        "prompt": "unit",
                        "raw_response": "{}",
                        "parsed": {},
                    },
                }

            def choose_vendor(self, firm, visible):
                raise AssertionError("model-mode adoption must not silently fall back to rule vendor choice")

            def exposure_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "reason": "keep_vendor", "vendor_action": "keep_vendor"}

            def insurance_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "skip"}

        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "invalid_model_vendor"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=6)
            sim.policy = MissingVendorModelPolicy(sim.config.get("decision_policy", {}), sim.rng)
            sim.run()

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            invalid = [row for row in events if row.get("event_type") == "model_invalid_decision"]
            self.assertTrue(invalid)
            self.assertTrue(
                all(row.get("reason") == "missing_or_non_visible_selected_vendor_id" for row in invalid)
            )
            self.assertFalse(any(row.get("event_type") == "vendor_contract_bound" for row in events))

            with (run_dir / "decisions.csv").open(newline="", encoding="utf-8") as f:
                decisions = list(csv.DictReader(f))
            adoption_rows = [row for row in decisions if row["decision_type"] == "ai_adoption"]
            self.assertTrue(adoption_rows)
            self.assertTrue(all(row["action"] == "0" for row in adoption_rows))
            self.assertTrue(
                any("invalid_model_structure:missing_or_non_visible_selected_vendor_id" in row["reason"] for row in adoption_rows)
            )

    def test_model_insurance_cannot_fallback_to_rule_term_selection(self):
        class MissingInsuranceTermPolicy(HeuristicDecisionPolicy):
            def adoption_decision(self, firm, context, visible, renewal=False):
                return {"action": False, "score": 0.0, "reason": "no_new_adoption"}

            def exposure_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "reason": "keep_vendor", "vendor_action": "keep_vendor"}

            def insurance_decision(self, firm, context, risk_signal):
                if not firm.has_ai:
                    return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "not_eligible"}
                return {
                    "action": True,
                    "score": 0.95,
                    "threshold": 0.35,
                    "reason": "model_wants_insurance_but_omits_term",
                    "max_rounds": 10,
                    "model_trace": {
                        "backend": "unit_model",
                        "prompt": "unit",
                        "raw_response": "{}",
                        "parsed": {},
                    },
                }

            def insurance_term_days(self, firm, context, incident_score, risk_signal=None):
                raise AssertionError("model-mode insurance must not silently fall back to rule term choice")

        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "invalid_model_insurance_term"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=6)
            firm = next(iter(sim.firms.values()))
            firm.vendor_contract = VendorContract(
                vendor_id="Vendor_Alpha",
                price=4_200.0,
                monthly_fee=2_100.0,
                start_day=0,
                end_day=40,
            )
            sim.policy = MissingInsuranceTermPolicy(sim.config.get("decision_policy", {}), sim.rng)
            sim.run()

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            invalid = [row for row in events if row.get("event_type") == "model_invalid_decision"]
            self.assertTrue(invalid)
            self.assertTrue(all(row.get("reason") == "missing_positive_insurance_term_days" for row in invalid))
            self.assertFalse(any(row.get("event_type") == "insurance_policy_bound" for row in events))

            with (run_dir / "decisions.csv").open(newline="", encoding="utf-8") as f:
                decisions = list(csv.DictReader(f))
            insurance_rows = [row for row in decisions if row["decision_type"] == "insurance_purchase"]
            self.assertTrue(insurance_rows)
            self.assertTrue(all(row["action"] == "0" for row in insurance_rows))
            self.assertTrue(
                any("invalid_model_structure:missing_positive_insurance_term_days" in row["reason"] for row in insurance_rows)
            )

    def test_model_exposure_decision_is_not_preempted_by_rule_abandon(self):
        class ModelKeepsExposurePolicy(HeuristicDecisionPolicy):
            def adoption_decision(self, firm, context, visible, renewal=False):
                return {"action": False, "score": 0.0, "reason": "no_new_adoption"}

            def exposure_decision(self, firm, context, risk_signal):
                return {
                    "action": False,
                    "score": 0.95,
                    "threshold": 0.40,
                    "reason": "model_keeps_existing_contract",
                    "vendor_action": "keep_vendor",
                    "model_trace": {
                        "backend": "unit_model",
                        "prompt": "unit",
                        "raw_response": "{}",
                        "parsed": {},
                    },
                }

            def insurance_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "skip"}

        config = _resolved_test_config()
        config["decision_layer"]["mode"] = "model_mock"
        config["decision_layer"]["llm_or_vllm_enabled"] = True

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "model_exposure_no_rule_preempt"
            sim = ActionRiskSimulator(config=config, run_dir=run_dir, days=1, firms=6)
            firm = next(iter(sim.firms.values()))
            firm.vendor_contract = VendorContract(
                vendor_id="Vendor_Alpha",
                price=4_200.0,
                monthly_fee=2_100.0,
                start_day=0,
                end_day=60,
            )
            firm.last_operational_loss = 20_000.0
            firm.loss_memory = 1.0
            firm.claimable_memory = 1.0
            firm.risk_memory = 1.0
            firm.panic = 1.0
            sim.policy = ModelKeepsExposurePolicy(sim.config.get("decision_policy", {}), sim.rng)
            sim.run()

            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertFalse(any(row.get("event_type") == "ai_contract_abandoned" for row in events))

            with (run_dir / "firm_daily.csv").open(newline="", encoding="utf-8") as f:
                firm_rows = list(csv.DictReader(f))
            target = [row for row in firm_rows if row["firm_id"] == firm.profile.firm_id][0]
            self.assertEqual(target["has_ai"], "True")
            self.assertEqual(target["vendor_id"], "Vendor_Alpha")

    def test_pre_operation_contract_decisions_do_not_observe_same_day_incidents(self):
        class CapturingPolicy(HeuristicDecisionPolicy):
            def __init__(self, config, rng):
                super().__init__(config, rng)
                self.exposure_signals = []
                self.insurance_signals = []

            def adoption_decision(self, firm, context, visible, renewal=False):
                return {"action": False, "score": 0.0, "reason": "no_new_adoption"}

            def exposure_decision(self, firm, context, risk_signal):
                self.exposure_signals.append(dict(risk_signal))
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "keep_vendor", "vendor_action": "keep_vendor"}

            def insurance_decision(self, firm, context, risk_signal):
                self.insurance_signals.append(dict(risk_signal))
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "skip"}

        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "pre_operation_no_same_day_leakage"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=6)
            try:
                sim.config["decision_layer"]["mode"] = "model_mock"
                sim.config["decision_layer"]["llm_or_vllm_enabled"] = True
                firm = next(iter(sim.firms.values()))
                firm.vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=4_200.0,
                    monthly_fee=2_100.0,
                    start_day=0,
                    end_day=60,
                )
                firm.risk_memory = 0.70
                firm.loss_memory = 0.60
                firm.claimable_memory = 0.50
                firm.last_operational_loss = 12_345.0
                firm.panic = 0.40

                def fail_if_current_day_record_is_requested(firm_id, day):
                    raise AssertionError("pre-operation decisions must not read same-day firm records")

                sim.panel.record_for = fail_if_current_day_record_is_requested
                policy = CapturingPolicy(sim.config.get("decision_policy", {}), sim.rng)
                sim.policy = policy

                context = sim._market_context(day=0)
                sim._make_contract_decisions(day=0, context=context)

                self.assertEqual(len(policy.exposure_signals), 1)
                self.assertEqual(len(policy.insurance_signals), 1)
                for signal in policy.exposure_signals + policy.insurance_signals:
                    self.assertTrue(signal["pre_operation_signal"])
                    self.assertFalse(signal["same_day_incident_observed"])
                    self.assertFalse(signal["raw_incident_any"])
                    self.assertFalse(signal["material_event"])
                    self.assertFalse(signal["claimable_event"])
                    self.assertEqual(signal["operational_loss"], 0.0)
                    self.assertEqual(signal["loss_ratio"], 0.0)
                    self.assertEqual(signal["material_event_score"], 0.0)
                    self.assertEqual(signal["claimable_event_score"], 0.0)
                    self.assertAlmostEqual(signal["risk_memory"], 0.70)
                    self.assertAlmostEqual(signal["loss_memory"], 0.60)
                    self.assertAlmostEqual(signal["claimable_memory"], 0.50)
                    self.assertAlmostEqual(signal["last_operational_loss"], 12_345.0)
                    self.assertIn("industry_incident_rate", signal)
                    self.assertIn("industry_stress_score", signal)
                    self.assertIn("industry_loss_pressure", signal)
            finally:
                sim.close()

    def test_daily_peer_context_is_frozen_at_day_start(self):
        class FirstFirmAdoptsPolicy(HeuristicDecisionPolicy):
            def __init__(self, config, rng, first_firm_id, second_firm_id):
                super().__init__(config, rng)
                self.first_firm_id = first_firm_id
                self.second_firm_id = second_firm_id
                self.second_context = None

            def visible_vendors(self, firm, vendors):
                return [v for v in vendors if v.vendor_id == "Vendor_Alpha"]

            def adoption_decision(self, firm, context, visible, renewal=False):
                if firm.profile.firm_id == self.second_firm_id:
                    self.second_context = context
                if firm.profile.firm_id == self.first_firm_id:
                    return {
                        "action": True,
                        "score": 1.0,
                        "probability": 1.0,
                        "reason": "first_firm_adopts",
                        "selected_vendor_id": "Vendor_Alpha",
                        "vendor_term_days": 30,
                        "max_rounds": 1,
                    }
                return {"action": False, "score": 0.0, "probability": 0.0, "reason": "wait"}

            def exposure_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "keep_vendor", "vendor_action": "keep_vendor"}

            def insurance_decision(self, firm, context, risk_signal):
                return {"action": False, "score": 0.0, "threshold": 1.0, "reason": "skip"}

        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "day_start_peer_context"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=6)
            try:
                sim.config["negotiation"]["enabled"] = False
                first, second = list(sim.firms.values())[:2]
                sim.firm_network = {fid: [] for fid in sim.firms}
                sim.firm_network[second.profile.firm_id] = [first.profile.firm_id]
                policy = FirstFirmAdoptsPolicy(
                    sim.config.get("decision_policy", {}),
                    sim.rng,
                    first.profile.firm_id,
                    second.profile.firm_id,
                )
                sim.policy = policy

                context = sim._market_context(day=0)
                sim._make_contract_decisions(day=0, context=context)

                self.assertTrue(first.has_ai)
                self.assertIsNotNone(policy.second_context)
                self.assertEqual(policy.second_context.network_neighbor_count, 1)
                self.assertEqual(policy.second_context.local_adoption_rate, 0.0)
                self.assertEqual(policy.second_context.local_insurance_coverage_rate, 0.0)
            finally:
                sim.close()

    def test_no_insurance_counterfactual_disables_policies_quotes_and_claims(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "no_insurance"
            sim = ActionRiskSimulator.from_yaml(
                config_path=config,
                run_dir=run_dir,
                days=12,
                firms=35,
                insurance_market_enabled=False,
            )
            sim.run()

            with (run_dir / "firm_daily.csv").open(newline="", encoding="utf-8") as f:
                firm_rows = list(csv.DictReader(f))
            self.assertTrue(firm_rows)
            self.assertFalse(any(row["has_insurance"] == "True" for row in firm_rows))

            with (run_dir / "quotes.csv").open(newline="", encoding="utf-8") as f:
                quote_rows = list(csv.DictReader(f))
            self.assertEqual(quote_rows, [])

            with (run_dir / "macro_daily.csv").open(newline="", encoding="utf-8") as f:
                macro_rows = list(csv.DictReader(f))
            self.assertTrue(macro_rows)
            self.assertTrue(all(float(row["insurance_coverage_overall"]) == 0.0 for row in macro_rows))
            self.assertTrue(all(float(row["total_premiums"]) == 0.0 for row in macro_rows))
            self.assertTrue(all(float(row["total_claims"]) == 0.0 for row in macro_rows))

            events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
            event_types = {row.get("event_type") for row in events}
            self.assertNotIn("insurance_policy_bound", event_types)
            self.assertNotIn("claim_paid", event_types)
            event_vendor_fees = sum(
                float(row.get("contract_total_fee", 0.0) or 0.0)
                for row in events
                if row.get("event_type") == "vendor_contract_bound"
            )
            firm_daily_vendor_fees = sum(float(row["vendor_fee_paid"]) for row in firm_rows)
            self.assertAlmostEqual(firm_daily_vendor_fees, event_vendor_fees)

    def test_claim_is_paid_before_bankruptcy_check(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "claim_before_bankruptcy"
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=run_dir, days=1, firms=5)
            try:
                firm = next(iter(sim.firms.values()))
                firm.cash = 100.0
                firm.vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=1_000.0,
                    monthly_fee=1_000.0,
                    start_day=0,
                    end_day=30,
                )
                firm.insurance_policy = InsurancePolicy(
                    insurer_id="Insurer_Apex_Global",
                    premium=200.0,
                    deductible_ratio=0.0,
                    coverage_ratio=1.0,
                    limit_money=60_000.0,
                    incident_threshold=0.0,
                    start_day=0,
                    end_day=30,
                )

                record = ActionRiskRecord(
                    firm_id=firm.profile.firm_id,
                    day=0,
                    industry=firm.profile.industry,
                    num_tasks=1,
                    incident_any=True,
                    incident_task_count=1,
                    avg_severity=1.0,
                    total_loss=8_000.0,
                    avg_risk_score=1.0,
                    max_risk_score=1.0,
                )
                sim.panel.record_for = lambda firm_id, day: record
                sim._traditional_pnl = lambda active_firm, day: 0.0
                sim.config["simulation"]["ai_gain_scale"] = 0.0
                sim.config["simulation"]["action_loss_scale"] = 1.0
                sim.config["risk_mapping"]["claimable_event_score_threshold"] = 0.0
                sim.config["risk_mapping"]["claimable_min_loss_ratio"] = 0.0
                sim.config["claims"]["claim_cooldown_days"] = 0

                totals = sim._operate_one_day(
                    day=0,
                    context=MarketContext(day=0, adoption_rate=0.0, insurance_coverage_rate=1.0, avg_panic=0.0, recent_claim_rate=0.0),
                    fees={},
                )
            finally:
                sim.close()

            self.assertEqual(totals["new_bankruptcies"], 0)
            self.assertGreater(totals["claims_paid"], 0.0)
            self.assertTrue(firm.active)
            self.assertGreater(firm.cash, 0.0)

    def test_partial_claim_residual_is_tracked_without_relabeling_as_uninsured(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "partial_claim", days=1, firms=1)
            try:
                firm = next(iter(sim.firms.values()))
                firm.cash = 1_000_000.0
                firm.vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=1_000.0,
                    monthly_fee=1_000.0,
                    start_day=0,
                    end_day=30,
                )
                firm.insurance_policy = InsurancePolicy(
                    insurer_id="Insurer_Apex_Global",
                    premium=200.0,
                    deductible_ratio=0.0,
                    coverage_ratio=1.0,
                    limit_money=1_000.0,
                    incident_threshold=0.0,
                    start_day=0,
                    end_day=30,
                )
                record = ActionRiskRecord(
                    firm_id=firm.profile.firm_id,
                    day=0,
                    industry=firm.profile.industry,
                    num_tasks=1,
                    incident_any=True,
                    incident_task_count=1,
                    avg_severity=1.0,
                    total_loss=10_000.0,
                    avg_risk_score=1.0,
                    max_risk_score=1.0,
                )
                sim.panel.record_for = lambda firm_id, day: record
                sim._traditional_pnl = lambda active_firm, day: 0.0
                sim.config["simulation"]["ai_gain_scale"] = 0.0
                sim.config["simulation"]["action_loss_scale"] = 1.0
                sim.config["risk_mapping"]["loss_realization_rate"] = 1.0
                sim.config["risk_mapping"]["catastrophic_tail_loss_multiplier"] = 0.0
                sim.config["risk_mapping"]["claimable_event_score_threshold"] = 0.0
                sim.config["risk_mapping"]["claimable_min_loss_ratio"] = 0.0
                sim.config["claims"]["claim_cooldown_days"] = 0

                totals = sim._operate_one_day(
                    day=0,
                    context=MarketContext(day=0, adoption_rate=0.0, insurance_coverage_rate=1.0, avg_panic=0.0, recent_claim_rate=0.0),
                    fees={},
                )
            finally:
                sim.close()

        self.assertGreater(totals["claims_paid"], 0.0)
        self.assertGreater(totals["claim_residual_loss"], 0.0)
        self.assertAlmostEqual(totals["unabsorbed_claimable_loss"], totals["claim_residual_loss"])
        self.assertEqual(totals["uninsured_claimable_events"], 0)
        self.assertEqual(totals["uninsured_claimable_loss"], 0.0)
        self.assertLess(totals["indemnity_relief_ratio"], 1.0)
        self.assertGreater(totals["indemnity_relief_ratio"], 0.0)

    def test_uninsured_claimable_firm_log_is_not_claim_residual_after_payout(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "uninsured_claimable", days=1, firms=1)
            try:
                firm = next(iter(sim.firms.values()))
                firm.cash = 1_000_000.0
                firm.vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=1_000.0,
                    monthly_fee=1_000.0,
                    start_day=0,
                    end_day=30,
                )
                record = ActionRiskRecord(
                    firm_id=firm.profile.firm_id,
                    day=0,
                    industry=firm.profile.industry,
                    num_tasks=1,
                    incident_any=True,
                    incident_task_count=1,
                    avg_severity=1.0,
                    total_loss=10_000.0,
                    avg_risk_score=1.0,
                    max_risk_score=1.0,
                )
                sim.panel.record_for = lambda firm_id, day: record
                sim._traditional_pnl = lambda active_firm, day: 0.0
                sim.config["simulation"]["ai_gain_scale"] = 0.0
                sim.config["simulation"]["action_loss_scale"] = 1.0
                sim.config["risk_mapping"]["loss_realization_rate"] = 1.0
                sim.config["risk_mapping"]["catastrophic_tail_loss_multiplier"] = 0.0
                sim.config["risk_mapping"]["claimable_event_score_threshold"] = 0.0
                sim.config["risk_mapping"]["claimable_min_loss_ratio"] = 0.0

                totals = sim._operate_one_day(
                    day=0,
                    context=MarketContext(day=0, adoption_rate=0.0, insurance_coverage_rate=0.0, avg_panic=0.0, recent_claim_rate=0.0),
                    fees={},
                )
            finally:
                sim.close()

        firm_row = totals["_firm_log_rows"][0]
        self.assertEqual(totals["claim_residual_loss"], 0.0)
        self.assertGreater(totals["unabsorbed_claimable_loss"], 0.0)
        self.assertEqual(firm_row["claim_residual_loss"], 0.0)
        self.assertGreater(firm_row["unabsorbed_claimable_loss"], 0.0)

    def test_bankruptcy_firm_log_preserves_during_day_ai_and_insurance_state(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "insured_bankruptcy_log", days=1, firms=1)
            try:
                firm = next(iter(sim.firms.values()))
                firm.cash = 100.0
                firm.vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=1_000.0,
                    monthly_fee=1_000.0,
                    start_day=0,
                    end_day=30,
                )
                firm.insurance_policy = InsurancePolicy(
                    insurer_id="Insurer_Apex_Global",
                    premium=200.0,
                    deductible_ratio=0.0,
                    coverage_ratio=1.0,
                    limit_money=100.0,
                    incident_threshold=0.0,
                    start_day=0,
                    end_day=30,
                )
                record = ActionRiskRecord(
                    firm_id=firm.profile.firm_id,
                    day=0,
                    industry=firm.profile.industry,
                    num_tasks=1,
                    incident_any=True,
                    incident_task_count=1,
                    avg_severity=1.0,
                    total_loss=20_000.0,
                    avg_risk_score=1.0,
                    max_risk_score=1.0,
                )
                sim.panel.record_for = lambda firm_id, day: record
                sim._traditional_pnl = lambda active_firm, day: 0.0
                sim.config["simulation"]["ai_gain_scale"] = 0.0
                sim.config["simulation"]["action_loss_scale"] = 1.0
                sim.config["risk_mapping"]["loss_realization_rate"] = 1.0
                sim.config["risk_mapping"]["catastrophic_tail_loss_multiplier"] = 0.0
                sim.config["risk_mapping"]["claimable_event_score_threshold"] = 0.0
                sim.config["risk_mapping"]["claimable_min_loss_ratio"] = 0.0
                sim.config["claims"]["claim_cooldown_days"] = 0

                totals = sim._operate_one_day(
                    day=0,
                    context=MarketContext(day=0, adoption_rate=0.0, insurance_coverage_rate=1.0, avg_panic=0.0, recent_claim_rate=0.0),
                    fees={},
                )
            finally:
                sim.close()

        row = totals["_firm_log_rows"][0]
        self.assertEqual(totals["new_bankruptcies"], 1)
        self.assertFalse(firm.active)
        self.assertTrue(row["had_ai_during_day"])
        self.assertEqual(row["vendor_id_during_day"], "Vendor_Alpha")
        self.assertTrue(row["had_insurance_during_day"])
        self.assertEqual(row["insurer_id_during_day"], "Insurer_Apex_Global")

    def test_fresh_autoclaw_episode_csv_can_override_legacy_panel(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            action_risk_path = tmp_path / "fresh_firm_daily_action_risk.csv"
            rows = [
                {
                    "industry": "financials",
                    "firm_id": "fresh_firm_001",
                    "day": 0,
                    "task_type": "file_cleanup",
                    "incident_flag": 1,
                    "severity": 0.72,
                    "total_loss": 1200.0,
                    "risk_score": 0.68,
                },
                {
                    "industry": "financials",
                    "firm_id": "fresh_firm_001",
                    "day": 1,
                    "task_type": "report_generation",
                    "incident_flag": 0,
                    "severity": 0.0,
                    "total_loss": 0.0,
                    "risk_score": 0.08,
                },
                {
                    "industry": "health_care",
                    "firm_id": "fresh_firm_002",
                    "day": 0,
                    "task_type": "record_update",
                    "incident_flag": 0,
                    "severity": 0.0,
                    "total_loss": 0.0,
                    "risk_score": 0.12,
                },
                {
                    "industry": "health_care",
                    "firm_id": "fresh_firm_002",
                    "day": 1,
                    "task_type": "summary_with_constraints",
                    "incident_flag": 1,
                    "severity": 0.55,
                    "total_loss": 600.0,
                    "risk_score": 0.51,
                },
            ]
            with action_risk_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            run_dir = tmp_path / "fresh_input"
            sim = ActionRiskSimulator.from_yaml(
                config_path=config,
                run_dir=run_dir,
                days=2,
                firms=2,
                path_overrides={
                    "action_risk_path": str(action_risk_path),
                    "selected_firms_path": "NONE",
                },
            )
            sim.run()

            with (run_dir / "firm_daily.csv").open(newline="", encoding="utf-8") as f:
                firm_rows = list(csv.DictReader(f))
            self.assertEqual(len(firm_rows), 4)
            self.assertEqual({row["firm_id"] for row in firm_rows}, {"fresh_firm_001", "fresh_firm_002"})

            meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["paths"]["action_risk_path"], str(action_risk_path))
            self.assertEqual(meta["paths"]["selected_firms_path"], "")

            audit = json.loads((run_dir / "action_risk_input_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["input_records_used"], 4)
            self.assertAlmostEqual(audit["incident_any_rate"], 0.5)

    def test_prepare_autoclaw_panel_fills_missing_firm_days(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            task_path = tmp_path / "firm_daily_action_risk.csv"
            rows = [
                {
                    "firm_id": "F001",
                    "day": 0,
                    "industry": "financials",
                    "task_id": "F001_d0_t0",
                    "task_type": "file_cleanup",
                    "incident_flag": 1,
                    "severity": 0.4,
                    "direct_loss_base": 100.0,
                    "total_loss": 200.0,
                    "risk_score": 0.6,
                },
                {
                    "firm_id": "F001",
                    "day": 0,
                    "industry": "financials",
                    "task_id": "F001_d0_t1",
                    "task_type": "record_update",
                    "incident_flag": 0,
                    "severity": 0.0,
                    "direct_loss_base": 0.0,
                    "total_loss": 0.0,
                    "risk_score": 0.1,
                },
                {
                    "firm_id": "F002",
                    "day": 1,
                    "industry": "health_care",
                    "task_id": "F002_d1_t0",
                    "task_type": "report_generation",
                    "incident_flag": 0,
                    "severity": 0.0,
                    "direct_loss_base": 0.0,
                    "total_loss": 0.0,
                    "risk_score": 0.05,
                },
            ]
            with task_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            buyer_path = tmp_path / "buyers.json"
            buyer_path.write_text(
                json.dumps(
                    [
                        {"id": "F001", "name": "Firm 1", "industry_code": "financials", "cash": 1000.0},
                        {"id": "F002", "name": "Firm 2", "industry_code": "health_care", "cash": 2000.0},
                    ]
                ),
                encoding="utf-8",
            )
            error_path = tmp_path / "benchmark_errors.csv"
            with error_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["firm_id", "day", "task_type", "task_difficulty", "error_type"])
                writer.writeheader()
                writer.writerow(
                    {
                        "firm_id": "F002",
                        "day": 0,
                        "task_type": "file_cleanup",
                        "task_difficulty": "stress",
                        "error_type": "AutoCLAWNoOpExecution",
                    }
                )

            out = tmp_path / "prepared"
            report = prepare_panel(
                task_input=task_path,
                output_dir=out,
                run_tag="unit",
                error_input=error_path,
                buyer_population_path=buyer_path,
                total_days=2,
            )

            with (out / "firm_daily_action_risk.csv").open(newline="", encoding="utf-8") as f:
                panel = list(csv.DictReader(f))
            self.assertEqual(len(panel), 4)
            missing = [row for row in panel if row["autoclaw_missing_flag"] == "1"]
            self.assertEqual(len(missing), 2)
            f001_d0 = [row for row in panel if row["firm_id"] == "F001" and row["day"] == "0"][0]
            self.assertEqual(f001_d0["num_tasks"], "2")
            self.assertEqual(f001_d0["incident_any_flag"], "1")
            self.assertEqual(f001_d0["incident_task_count"], "1")
            self.assertIn("file_cleanup:1", f001_d0["task_type_mix"])
            self.assertEqual(report["expected_firm_days"], 4)
            self.assertEqual(report["missing_firm_days"], 2)
            self.assertEqual(report["error_rows"], 1)

    def test_insurance_market_confidence_lifts_adoption_probability(self):
        profile = FirmProfile(
            firm_id="F001",
            name="Risk sensitive firm",
            industry="financials",
            cash=100_000.0,
            asset_value=500_000.0,
            risk_tolerance=0.25,
            tech_urgency=0.70,
            ai_dependency=0.65,
            inertia=0.20,
            innovativeness=0.60,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=100_000.0)
        visible = [
            VendorProfile(
                vendor_id="Vendor_Test",
                label="Test vendor",
                subscription_fee=1000.0,
                productivity_lift=0.012,
                risk_multiplier=0.80,
                reputation=0.75,
                marketing_weight=1.0,
            )
        ]
        context = MarketContext(day=5, adoption_rate=0.20, insurance_coverage_rate=0.50, avg_panic=0.20, recent_claim_rate=0.02)
        base_cfg = {
            "adoption_base_probability": 0.008,
            "adoption_tech_weight": 0.050,
            "adoption_innov_weight": 0.035,
            "adoption_peer_weight": 0.035,
            "adoption_inertia_weight": 0.040,
            "adoption_panic_weight": 0.055,
            "adoption_insurance_confidence_weight": 0.020,
            "adoption_saturation_weight": 0.55,
            "adoption_saturation_floor": 0.32,
            "adoption_min_probability": 0.002,
            "adoption_max_probability": 0.075,
        }
        enabled = HeuristicDecisionPolicy({**base_cfg, "insurance_market_enabled": True}, random.Random(1))
        disabled = HeuristicDecisionPolicy({**base_cfg, "insurance_market_enabled": False}, random.Random(1))

        enabled_decision = enabled.adoption_decision(firm, context, visible)
        disabled_decision = disabled.adoption_decision(firm, context, visible)

        self.assertGreater(enabled_decision["probability"], disabled_decision["probability"])

    def test_model_adoption_threshold_has_peer_diffusion_shape(self):
        laggard = FirmState(
            profile=FirmProfile(
                firm_id="F001",
                name="Wait-and-see firm",
                industry="industrials",
                cash=100_000.0,
                asset_value=500_000.0,
                risk_tolerance=0.45,
                tech_urgency=0.45,
                ai_dependency=0.40,
                inertia=0.70,
                innovativeness=0.35,
                contagion_sensitivity=0.55,
                size_label="medium",
            ),
            cash=100_000.0,
        )
        pioneer = FirmState(
            profile=FirmProfile(
                firm_id="F002",
                name="Pioneer firm",
                industry="financials",
                cash=150_000.0,
                asset_value=600_000.0,
                risk_tolerance=0.35,
                tech_urgency=0.95,
                ai_dependency=0.80,
                inertia=0.05,
                innovativeness=0.90,
                contagion_sensitivity=0.55,
                size_label="medium",
            ),
            cash=150_000.0,
        )
        visible = [VendorProfile("Vendor_Alpha", "Stable", 2_800.0, 0.011, 0.62, 0.78, 1.05)]
        cfg = {
            "insurance_market_enabled": True,
            "model_adoption_base_threshold": 0.70,
            "model_adoption_maturity_days": 90,
            "model_adoption_early_friction": 0.20,
            "model_adoption_peer_reference": 0.45,
            "model_adoption_peer_floor": 0.08,
            "model_adoption_low_peer_friction": 0.12,
            "model_adoption_peer_evidence_lag_days": 45,
            "model_adoption_implementation_uncertainty": 0.12,
            "model_adoption_local_evidence_relief": 0.14,
            "model_adoption_local_evidence_curve_power": 2.0,
            "model_adoption_pioneer_readiness_discount": 0.06,
            "model_adoption_inertia_penalty": 0.05,
            "model_adoption_max_threshold": 0.98,
        }
        policy = ModelDecisionPolicy(config=cfg, rng=random.Random(7), layer_config={"mode": "model_mock"})

        early_laggard = policy.adoption_decision(
            laggard,
            MarketContext(day=0, adoption_rate=0.02, insurance_coverage_rate=0.20, avg_panic=0.10, recent_claim_rate=0.0),
            visible,
        )
        early_pioneer = policy.adoption_decision(
            pioneer,
            MarketContext(day=0, adoption_rate=0.02, insurance_coverage_rate=0.20, avg_panic=0.10, recent_claim_rate=0.0),
            visible,
        )
        later_laggard = policy.adoption_decision(
            laggard,
            MarketContext(day=60, adoption_rate=0.55, insurance_coverage_rate=0.65, avg_panic=0.10, recent_claim_rate=0.0),
            visible,
        )
        early_high_peer_laggard = policy.adoption_decision(
            laggard,
            MarketContext(day=5, adoption_rate=0.55, insurance_coverage_rate=0.65, avg_panic=0.10, recent_claim_rate=0.0),
            visible,
        )

        self.assertGreater(early_laggard["threshold"], early_pioneer["threshold"])
        self.assertGreater(early_laggard["threshold"], later_laggard["threshold"] + 0.15)
        self.assertGreater(early_high_peer_laggard["threshold"], later_laggard["threshold"] + 0.08)

    def test_model_adoption_threshold_offset_is_auditable(self):
        firm = FirmState(
            profile=FirmProfile(
                firm_id="F001",
                name="Calibration firm",
                industry="industrials",
                cash=100_000.0,
                asset_value=500_000.0,
                risk_tolerance=0.45,
                tech_urgency=0.55,
                ai_dependency=0.50,
                inertia=0.45,
                innovativeness=0.50,
                contagion_sensitivity=0.55,
                size_label="medium",
            ),
            cash=100_000.0,
        )
        context = MarketContext(
            day=35,
            adoption_rate=0.35,
            insurance_coverage_rate=0.45,
            avg_panic=0.05,
            recent_claim_rate=0.01,
            local_adoption_rate=0.35,
            local_insurance_coverage_rate=0.45,
            local_avg_panic=0.05,
            local_recent_claim_rate=0.01,
        )
        cfg = {
            "insurance_market_enabled": True,
            "model_adoption_base_threshold": 0.70,
            "model_adoption_min_threshold": 0.50,
            "model_adoption_max_threshold": 0.98,
            "model_adoption_maturity_days": 100,
            "model_adoption_early_friction": 0.12,
            "model_adoption_local_evidence_reference": 0.55,
            "model_adoption_local_evidence_floor": 0.10,
            "model_adoption_local_evidence_lag_days": 40,
            "model_adoption_insurance_confidence_discount": 0.03,
            "model_renewal_min_threshold": 0.30,
            "model_renewal_max_threshold": 0.80,
        }

        base = _model_adoption_threshold(firm, context, config=cfg)
        shifted = _model_adoption_threshold(
            firm,
            context,
            config={**cfg, "model_adoption_threshold_offset": 0.03},
        )
        renewal_base = _model_adoption_threshold(firm, context, renewal=True, config=cfg)
        renewal_shifted = _model_adoption_threshold(
            firm,
            context,
            renewal=True,
            config={**cfg, "model_renewal_threshold_offset": 0.04},
        )

        self.assertAlmostEqual(shifted - base, 0.03, places=10)
        self.assertAlmostEqual(renewal_shifted - renewal_base, 0.04, places=10)

    def test_no_insurance_counterfactual_ignores_old_tail_penalty_keys(self):
        profile = FirmProfile(
            firm_id="F001",
            name="AI-dependent buyer",
            industry="financials",
            cash=100_000.0,
            asset_value=500_000.0,
            risk_tolerance=0.25,
            tech_urgency=0.75,
            ai_dependency=0.85,
            inertia=0.35,
            innovativeness=0.60,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=100_000.0)
        context = MarketContext(
            day=20,
            adoption_rate=0.25,
            insurance_coverage_rate=0.55,
            avg_panic=0.20,
            recent_claim_rate=0.08,
            local_adoption_rate=0.25,
            local_insurance_coverage_rate=0.55,
            local_avg_panic=0.20,
            local_recent_claim_rate=0.08,
            network_neighbor_count=12,
        )
        cfg = {
            "model_adoption_base_threshold": 0.66,
            "model_adoption_maturity_days": 90,
            "model_adoption_early_friction": 0.14,
            "model_adoption_local_evidence_reference": 0.55,
            "model_adoption_local_evidence_floor": 0.10,
            "model_adoption_low_local_evidence_friction": 0.08,
            "model_adoption_local_evidence_lag_days": 30,
            "model_adoption_insurance_confidence_discount": 0.016,
            # These deprecated keys are intentionally ignored. The paired
            # counterfactual must differ only by insurance availability, not
            # by no-insurance-only adoption penalties.
            "model_adoption_uninsured_tail_penalty": 0.10,
            "model_adoption_uninsured_cash_scarcity_weight": 0.80,
            "model_adoption_min_threshold": 0.52,
            "model_adoption_max_threshold": 0.98,
        }

        insured_threshold = _model_adoption_threshold(
            firm,
            context,
            config={**cfg, "insurance_market_enabled": True},
        )
        uninsured_threshold = _model_adoption_threshold(
            firm,
            context,
            config={**cfg, "insurance_market_enabled": False},
        )
        firm.cash = 450_000.0
        uninsured_with_cash_buffer = _model_adoption_threshold(
            firm,
            context,
            config={**cfg, "insurance_market_enabled": False},
        )

        self.assertGreater(uninsured_threshold, insured_threshold)
        self.assertAlmostEqual(uninsured_threshold, uninsured_with_cash_buffer, places=10)

    def test_prior_policy_makes_insurance_renewal_sticky_but_bounded(self):
        profile = FirmProfile(
            firm_id="F001",
            name="AI dependent firm",
            industry="financials",
            cash=120_000.0,
            asset_value=550_000.0,
            risk_tolerance=0.40,
            tech_urgency=0.70,
            ai_dependency=0.72,
            inertia=0.30,
            innovativeness=0.50,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=120_000.0)
        firm.vendor_contract = VendorContract(
            vendor_id="Vendor_Alpha",
            price=5_600.0,
            monthly_fee=2_800.0,
            start_day=0,
            end_day=60,
        )
        context = MarketContext(day=60, adoption_rate=0.60, insurance_coverage_rate=0.35, avg_panic=0.20, recent_claim_rate=0.01)
        cfg = {
            "insurance_buy_threshold": 0.40,
            "insurance_min_threshold": 0.28,
            "insurance_renewal_bonus": 0.18,
            "insurance_renewal_threshold_discount": 0.06,
            "insurance_medium_dependency_threshold": 0.68,
            "insurance_medium_risk_tolerance_threshold": 0.45,
        }
        policy = HeuristicDecisionPolicy(cfg, random.Random(4))

        fresh = policy.insurance_decision(
            firm,
            context,
            {"claimable_event_score": 0.18, "material_event_score": 0.20, "prior_policy": 0.0},
        )
        renewal = policy.insurance_decision(
            firm,
            context,
            {"claimable_event_score": 0.18, "material_event_score": 0.20, "prior_policy": 1.0},
        )

        self.assertLess(renewal["threshold"], fresh["threshold"])
        self.assertGreater(renewal["score"], fresh["score"])

    def test_model_insurance_decision_applies_renewal_threshold_discount(self):
        class FixedInsurancePolicy(ModelDecisionPolicy):
            def _complete(self, prompt, payload):
                return json.dumps(
                    {
                        "buy_insurance": True,
                        "insurance_score": 0.35,
                        "insurance_term_days": 30,
                        "term_days": 30,
                        "max_rounds": 3,
                        "reason": "fixed_score",
                    }
                )

        profile = FirmProfile(
            firm_id="F001",
            name="AI dependent firm",
            industry="financials",
            cash=120_000.0,
            asset_value=550_000.0,
            risk_tolerance=0.40,
            tech_urgency=0.70,
            ai_dependency=0.72,
            inertia=0.30,
            innovativeness=0.50,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=120_000.0)
        firm.vendor_contract = VendorContract(
            vendor_id="Vendor_Alpha",
            price=5_600.0,
            monthly_fee=2_800.0,
            start_day=0,
            end_day=60,
        )
        context = MarketContext(day=60, adoption_rate=0.60, insurance_coverage_rate=0.35, avg_panic=0.20, recent_claim_rate=0.01)
        policy = FixedInsurancePolicy(
            {
                "insurance_buy_threshold": 0.40,
                "insurance_min_threshold": 0.28,
                "insurance_renewal_threshold_discount": 0.06,
            },
            random.Random(4),
            {"mode": "vllm_openai", "fallback_to_rule": False},
        )

        fresh = policy.insurance_decision(
            firm,
            context,
            {"claimable_event_score": 0.18, "material_event_score": 0.20, "prior_policy": 0.0, "ai_remaining_days": 30},
        )
        renewal = policy.insurance_decision(
            firm,
            context,
            {"claimable_event_score": 0.18, "material_event_score": 0.20, "prior_policy": 1.0, "ai_remaining_days": 30},
        )

        self.assertFalse(fresh["action"])
        self.assertTrue(renewal["action"])
        self.assertAlmostEqual(fresh["threshold"], 0.40)
        self.assertAlmostEqual(renewal["threshold"], 0.34)
        term = policy.insurance_term_days(
            firm,
            context,
            0.18,
            {"prior_policy": 1.0, "ai_remaining_days": 37},
        )
        self.assertGreaterEqual(term, 1)
        self.assertLessEqual(term, 37)

    def test_model_mock_returns_rich_vendor_and_insurance_terms(self):
        profile = FirmProfile(
            firm_id="F001",
            name="High urgency firm",
            industry="financials",
            cash=150_000.0,
            asset_value=600_000.0,
            risk_tolerance=0.25,
            tech_urgency=0.95,
            ai_dependency=0.80,
            inertia=0.05,
            innovativeness=0.90,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=150_000.0)
        visible = [
            VendorProfile("Vendor_Alpha", "Stable", 2_800.0, 0.011, 0.62, 0.78, 1.05),
            VendorProfile("Vendor_Beta", "Efficient", 2_200.0, 0.013, 0.82, 0.68, 0.95),
        ]
        context = MarketContext(day=60, adoption_rate=0.55, insurance_coverage_rate=0.45, avg_panic=0.15, recent_claim_rate=0.03)
        policy = ModelDecisionPolicy(
            config={"insurance_market_enabled": True, "insurance_buy_threshold": 0.58},
            rng=random.Random(5),
            layer_config={"mode": "model_mock"},
        )

        adoption = policy.adoption_decision(firm, context, visible)
        self.assertTrue(adoption["action"])
        self.assertIn(adoption["selected_vendor_id"], {"Vendor_Alpha", "Vendor_Beta"})
        self.assertGreaterEqual(adoption["vendor_term_days"], 5)
        self.assertLessEqual(adoption["max_rounds"], 30)

        firm.vendor_contract = VendorContract(
            vendor_id=adoption["selected_vendor_id"],
            price=5_000.0,
            monthly_fee=2_500.0,
            start_day=0,
            end_day=60,
        )
        insurance = policy.insurance_decision(
            firm,
            context,
            {
                "claimable_event_score": 0.80,
                "material_event_score": 0.75,
                "industry_incident_rate": 0.20,
                "industry_stress_score": 0.70,
                "industry_loss_pressure": 0.60,
                "prior_policy": 0.0,
            },
        )
        self.assertTrue(insurance["action"])
        self.assertGreaterEqual(insurance["insurance_term_days"], 5)
        self.assertEqual(insurance["term_days"], insurance["insurance_term_days"])
        self.assertLessEqual(insurance["max_rounds"], 30)

    def test_model_decision_rounds_respect_minimum_negotiation_limits(self):
        profile = FirmProfile(
            firm_id="F001",
            name="Round floor firm",
            industry="financials",
            cash=150_000.0,
            asset_value=600_000.0,
            risk_tolerance=0.25,
            tech_urgency=0.95,
            ai_dependency=0.80,
            inertia=0.05,
            innovativeness=0.90,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=150_000.0)
        visible = [VendorProfile("Vendor_Alpha", "Stable", 2_800.0, 0.011, 0.62, 0.78, 1.05)]
        context = MarketContext(day=60, adoption_rate=0.55, insurance_coverage_rate=0.45, avg_panic=0.15, recent_claim_rate=0.03)
        policy = ModelDecisionPolicy(
            config={
                "insurance_market_enabled": True,
                "vendor_min_rounds": 3,
                "insurance_min_rounds": 4,
            },
            rng=random.Random(5),
            layer_config={"mode": "model_mock"},
        )

        adoption = policy.adoption_decision(firm, context, visible)
        self.assertTrue(adoption["action"])
        self.assertGreaterEqual(adoption["max_rounds"], 3)

        firm.vendor_contract = VendorContract(
            vendor_id=adoption["selected_vendor_id"],
            price=5_000.0,
            monthly_fee=2_500.0,
            start_day=0,
            end_day=60,
        )
        insurance = policy.insurance_decision(
            firm,
            context,
            {
                "claimable_event_score": 0.80,
                "material_event_score": 0.75,
                "industry_stress_score": 0.70,
                "prior_policy": 0.0,
                "ai_remaining_days": 57,
            },
        )
        self.assertTrue(insurance["action"])
        self.assertGreaterEqual(insurance["max_rounds"], 4)

    def test_insurer_capital_load_increases_quoted_premium(self):
        firm = FirmState(
            profile=FirmProfile(
                firm_id="F001",
                name="Pricing firm",
                industry="financials",
                cash=150_000.0,
                asset_value=600_000.0,
                risk_tolerance=0.35,
                tech_urgency=0.85,
                ai_dependency=0.80,
                inertia=0.20,
                innovativeness=0.75,
                contagion_sensitivity=0.55,
                size_label="medium",
            ),
            cash=150_000.0,
        )
        vendor = VendorProfile("Vendor_Test", "Test vendor", 2_000.0, 0.012, 0.90, 0.70, 1.0)
        snapshot = IndustryRiskSnapshot(
            industry="financials",
            day=0,
            observations=20,
            incident_rate=0.10,
            avg_severity=0.40,
            avg_loss=1_200.0,
            avg_risk_score=0.35,
            stress_loss=3_000.0,
            stress_risk_score=0.60,
        )
        base_row = {
            "id": "Insurer_Test",
            "initial_capital": 1_000_000,
            "base_margin": 0.25,
            "risk_appetite": 0.55,
            "expense_load": 0.04,
            "deductible_ratio": 0.30,
            "coverage_ratio": 0.60,
            "limit_ratio": 0.10,
            "max_active_policies": 100,
            "solvency_floor_ratio": 0.20,
            "soft_threshold_ratio": 0.70,
            "hard_threshold_ratio": 0.45,
            "target_sectors": ["ALL"],
        }
        low_market = InsuranceMarket(load_insurer_profiles([dict(base_row, capital_load=0.0)]))
        loaded_market = InsuranceMarket(load_insurer_profiles([dict(base_row, capital_load=0.25)]))

        low_quote = low_market.quote_all(firm, vendor, snapshot, day=0, term_days=30, market_panic=0.10, recent_claim_rate=0.01)[0]
        loaded_quote = loaded_market.quote_all(firm, vendor, snapshot, day=0, term_days=30, market_panic=0.10, recent_claim_rate=0.01)[0]
        self.assertGreater(loaded_quote.premium, low_quote.premium * 1.20)

    def test_formal_model_run_can_disable_rule_fallback(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(
                config_path=config,
                run_dir=Path(tmp) / "strict_model",
                days=1,
                firms=2,
                decision_mode="vllm_openai",
                vllm_base_urls="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1",
                vllm_model="qwen3-agent-local",
                model_fallback_to_rule=False,
            )

            layer = sim.config["decision_layer"]
            self.assertEqual(layer["mode"], "vllm_openai")
            self.assertFalse(layer["fallback_to_rule"])
            self.assertEqual(layer["model"], "qwen3-agent-local")
            self.assertIn("8001", str(layer["base_urls"]))
            sim.close()

    def test_contract_terms_are_variable_and_insurance_capped_by_ai_remaining_days(self):
        self.assertEqual(
            _bounded_term_days(90, min_days=1, max_days=120, max_remaining_days=17),
            17,
        )
        self.assertEqual(
            _bounded_term_days(0, min_days=1, max_days=120, max_remaining_days=0),
            0,
        )

        profile = FirmProfile(
            firm_id="F002",
            name="Term test firm",
            industry="financials",
            cash=200_000.0,
            asset_value=700_000.0,
            risk_tolerance=0.45,
            tech_urgency=0.70,
            ai_dependency=0.80,
            inertia=0.50,
            innovativeness=0.65,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=200_000.0)
        policy = HeuristicDecisionPolicy(
            config={
                "vendor_min_term_days": 14,
                "vendor_max_term_days": 120,
                "insurance_min_term_days": 1,
                "insurance_max_term_days": 90,
            },
            rng=random.Random(13),
        )
        context = MarketContext(day=10, adoption_rate=0.30, insurance_coverage_rate=0.20, avg_panic=0.20, recent_claim_rate=0.04)
        vendor_terms = [policy.vendor_term_days(firm, context) for _ in range(12)]
        self.assertTrue(all(14 <= term <= 120 for term in vendor_terms))
        self.assertGreater(len(set(vendor_terms)), 1)

        firm.vendor_contract = VendorContract(
            vendor_id="Vendor_Alpha",
            price=6_000.0,
            monthly_fee=2_000.0,
            start_day=0,
            end_day=27,
        )
        insurance_term = policy.insurance_term_days(
            firm,
            context,
            0.80,
            {
                "claimable_event_score": 0.80,
                "material_event_score": 0.70,
                "industry_stress_score": 0.75,
                "ai_remaining_days": 17,
            },
        )
        self.assertGreaterEqual(insurance_term, 1)
        self.assertLessEqual(insurance_term, 17)

    def test_panic_treats_uninsured_claimable_loss_as_stronger_than_paid_claim(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "panic", days=1, firms=2)
            try:
                for firm in sim.firms.values():
                    firm.panic = 0.0

                paid_totals = {
                    "active_count": 2,
                    "material_events": 1,
                    "claimable_events": 1,
                    "claim_events": 1,
                    "uninsured_claimable_events": 0,
                    "avg_material_event_score": 0.70,
                    "indemnity_relief_ratio": 1.0,
                }
                sim._update_panic(paid_totals)
                paid_panic = sum(f.panic for f in sim.firms.values()) / len(sim.firms)

                for firm in sim.firms.values():
                    firm.panic = 0.0

                uninsured_totals = {
                    "active_count": 2,
                    "material_events": 1,
                    "claimable_events": 1,
                    "claim_events": 0,
                    "uninsured_claimable_events": 1,
                    "avg_material_event_score": 0.70,
                    "indemnity_relief_ratio": 0.0,
                }
                sim._update_panic(uninsured_totals)
                uninsured_panic = sum(f.panic for f in sim.firms.values()) / len(sim.firms)
            finally:
                sim.close()

        self.assertGreater(uninsured_panic, paid_panic)

    def test_paid_claim_without_unabsorbed_loss_can_calm_market_panic(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "panic_reassurance", days=1, firms=2)
            try:
                for firm in sim.firms.values():
                    firm.panic = 0.10

                paid_totals = {
                    "active_count": 2,
                    "material_events": 0,
                    "claimable_events": 1,
                    "claim_events": 1,
                    "uninsured_claimable_events": 0,
                    "avg_material_event_score": 0.0,
                    "indemnity_relief_ratio": 1.0,
                }
                sim._update_panic(paid_totals)
                paid_panic = sum(f.panic for f in sim.firms.values()) / len(sim.firms)
            finally:
                sim.close()

        self.assertLess(paid_panic, 0.10)

    def test_stress_tail_config_extends_base_without_dropping_market_structure(self):
        config = TEST_STRESS_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "stress_tail", days=1, firms=5)
            try:
                self.assertEqual(len(sim.vendors), 4)
                self.assertEqual(len(sim.insurance_market.insurers), 5)
                self.assertAlmostEqual(sim.config["simulation"]["action_loss_scale"], 2.40)
                self.assertAlmostEqual(sim.config["simulation"]["ai_gain_scale"], 0.05)
                self.assertLess(sim.config["risk_mapping"]["catastrophic_tail_threshold"], 0.30)
            finally:
                sim.close()

    def test_negotiation_engine_keeps_round_level_vendor_and_insurance_traces(self):
        config = {
            "decision_layer": {"mode": "rule_heuristic"},
            "negotiation": {
                "vendor_max_rounds": 3,
                "insurance_max_rounds": 3,
                "vendor_max_cash_share": 0.20,
                "insurance_max_cash_share": 0.08,
                "insurance_floor_ratio": 0.88,
            },
        }
        rng = random.Random(7)
        engine = NegotiationEngine(config=config, rng=rng)
        profile = FirmProfile(
            firm_id="F001",
            name="Firm 001",
            industry="financials",
            cash=100_000.0,
            asset_value=500_000.0,
            risk_tolerance=0.35,
            tech_urgency=0.90,
            ai_dependency=0.75,
            inertia=0.10,
            innovativeness=0.80,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=100_000.0)
        context = MarketContext(day=0, adoption_rate=0.20, insurance_coverage_rate=0.10, avg_panic=0.15, recent_claim_rate=0.05)
        vendor = VendorProfile(
            vendor_id="Vendor_Test",
            label="Test vendor",
            subscription_fee=1000.0,
            productivity_lift=0.012,
            risk_multiplier=0.80,
            reputation=0.75,
            marketing_weight=1.0,
        )

        vendor_result = engine.negotiate_vendor(firm=firm, vendor=vendor, context=context, day=0, term_days=60)
        self.assertTrue(vendor_result.agreed)
        self.assertGreaterEqual(len(vendor_result.events), 2)
        self.assertEqual({row["event_type"] for row in vendor_result.events}, {"vendor_negotiation_round"})
        self.assertTrue(any(row.get("side") == "vendor" for row in vendor_result.events))
        self.assertTrue(any(row.get("side") == "buyer" for row in vendor_result.events))

        quote = InsuranceQuote(
            insurer_id="Insurer_Test",
            firm_id="F001",
            vendor_id="Vendor_Test",
            industry="financials",
            day=0,
            term_days=60,
            premium=600.0,
            deductible_ratio=0.34,
            coverage_ratio=0.64,
            limit_money=50_000.0,
            incident_threshold=0.42,
            expected_loss=280.0,
            stress_loss=900.0,
            regime="NORMAL",
            market_role="private",
        )
        insurance_result = engine.negotiate_insurance(
            firm=firm,
            quote=quote,
            context=context,
            risk_need=0.90,
        )
        self.assertTrue(insurance_result.agreed)
        self.assertIsNotNone(insurance_result.quote)
        self.assertGreaterEqual(len(insurance_result.events), 2)
        self.assertTrue(any(row.get("event_type") == "insurance_negotiation_round" for row in insurance_result.events))
        self.assertTrue(any(row.get("side") == "insurer" for row in insurance_result.events))
        self.assertTrue(any(row.get("side") == "buyer" for row in insurance_result.events))

        tight_cash_firm = FirmState(profile=profile, cash=10_000.0)
        long_contract_result = engine.negotiate_vendor(
            firm=tight_cash_firm,
            vendor=vendor,
            context=context,
            day=0,
            term_days=120,
        )
        self.assertFalse(long_contract_result.agreed)

        model_engine = NegotiationEngine(
            config={
                "decision_layer": {"mode": "model_mock", "fallback_to_rule": True},
                "negotiation": config["negotiation"],
            },
            rng=random.Random(8),
        )
        model_vendor_result = model_engine.negotiate_vendor(
            firm=firm,
            vendor=vendor,
            context=context,
            day=0,
            term_days=60,
        )
        self.assertTrue(model_vendor_result.agreed)
        self.assertTrue(any(isinstance(row.get("model_trace"), dict) for row in model_vendor_result.events))

    def test_model_decision_retries_malformed_json_without_rule_fallback(self):
        class RetryPolicy(ModelDecisionPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = 0

            def _complete(self, prompt, payload):
                self.calls += 1
                if self.calls == 1:
                    return "I need to think about this first."
                return json.dumps(
                    {
                        "adopt_ai": True,
                        "adoption_score": 0.95,
                        "selected_vendor_id": "Vendor_Test",
                        "vendor_term_days": 45,
                        "max_rounds": 3,
                        "reason": "retry_json_ok",
                    }
                )

        firm = FirmState(
            profile=FirmProfile(
                firm_id="F001",
                name="Firm 001",
                industry="financials",
                cash=100_000.0,
                asset_value=500_000.0,
                risk_tolerance=0.35,
                tech_urgency=0.90,
                ai_dependency=0.75,
                inertia=0.10,
                innovativeness=0.80,
                contagion_sensitivity=0.55,
                size_label="medium",
            ),
            cash=100_000.0,
        )
        vendor = VendorProfile("Vendor_Test", "Test vendor", 1000.0, 0.012, 0.80, 0.75, 1.0)
        policy = RetryPolicy(
            config={"insurance_market_enabled": True},
            rng=random.Random(10),
            layer_config={"mode": "vllm_openai", "fallback_to_rule": False, "json_retries": 1},
        )
        decision = policy.adoption_decision(
            firm,
            MarketContext(day=0, adoption_rate=0.2, insurance_coverage_rate=0.1, avg_panic=0.1, recent_claim_rate=0.0),
            [vendor],
        )
        self.assertTrue(decision["action"])
        self.assertEqual(policy.calls, 2)
        self.assertEqual(len(decision["model_trace"]["raw_responses"]), 2)

    def test_negotiation_retries_malformed_json_without_rule_fallback(self):
        engine = NegotiationEngine(
            config={
                "decision_layer": {"mode": "vllm_openai", "fallback_to_rule": False, "json_retries": 1},
                "negotiation": {"vendor_max_rounds": 1, "max_rounds_cap": 1},
            },
            rng=random.Random(11),
        )
        calls = {"n": 0}

        def fake_complete(prompt):
            calls["n"] += 1
            if calls["n"] == 1:
                return "not json"
            return json.dumps({"decision": "accept", "offer_monthly_fee": 1000.0, "message": "Accepted."})

        engine._complete = fake_complete
        firm = FirmState(
            profile=FirmProfile(
                firm_id="F001",
                name="Firm 001",
                industry="financials",
                cash=100_000.0,
                asset_value=500_000.0,
                risk_tolerance=0.35,
                tech_urgency=0.90,
                ai_dependency=0.75,
                inertia=0.10,
                innovativeness=0.80,
                contagion_sensitivity=0.55,
                size_label="medium",
            ),
            cash=100_000.0,
        )
        vendor = VendorProfile("Vendor_Test", "Test vendor", 1000.0, 0.012, 0.80, 0.75, 1.0)
        result = engine.negotiate_vendor(
            firm=firm,
            vendor=vendor,
            context=MarketContext(day=0, adoption_rate=0.2, insurance_coverage_rate=0.1, avg_panic=0.1, recent_claim_rate=0.0),
            day=0,
            term_days=45,
        )
        self.assertTrue(result.agreed)
        self.assertEqual(calls["n"], 2)

    def test_firm_context_uses_local_network_adoption_not_global_average(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "network_context", days=1, firms=10)
            fids = list(sim.firms)
            target = sim.firms[fids[0]]
            neighbor_ids = fids[1:4]
            sim.firm_network = {fid: [] for fid in fids}
            sim.firm_network[target.profile.firm_id] = neighbor_ids

            for fid in fids[4:]:
                sim.firms[fid].vendor_contract = VendorContract(
                    vendor_id="Vendor_Alpha",
                    price=1_000.0,
                    monthly_fee=1_000.0,
                    start_day=0,
                    end_day=30,
                )
            for fid in neighbor_ids:
                sim.firms[fid].vendor_contract = None

            base_context = sim._market_context(day=0)
            local_context = sim._firm_market_context(target, base_context)

            self.assertGreater(base_context.adoption_rate, 0.5)
            self.assertEqual(local_context.network_neighbor_count, len(neighbor_ids))
            self.assertEqual(local_context.local_adoption_rate, 0.0)
            sim.close()

    def test_model_adoption_threshold_uses_local_network_evidence(self):
        firm = FirmState(
            profile=FirmProfile(
                firm_id="F001",
                name="Network-aware firm",
                industry="industrials",
                cash=100_000.0,
                asset_value=500_000.0,
                risk_tolerance=0.45,
                tech_urgency=0.55,
                ai_dependency=0.50,
                inertia=0.55,
                innovativeness=0.45,
                contagion_sensitivity=0.55,
                size_label="medium",
            ),
            cash=100_000.0,
        )
        visible = [VendorProfile("Vendor_Alpha", "Stable", 2_800.0, 0.011, 0.62, 0.78, 1.05)]
        cfg = {
            "insurance_market_enabled": True,
            "model_adoption_base_threshold": 0.70,
            "model_adoption_maturity_days": 90,
            "model_adoption_early_friction": 0.18,
            "model_adoption_peer_reference": 0.50,
            "model_adoption_peer_floor": 0.08,
            "model_adoption_low_peer_friction": 0.12,
            "model_adoption_peer_evidence_lag_days": 45,
            "model_adoption_local_evidence_relief": 0.14,
            "model_adoption_local_evidence_curve_power": 2.0,
        }
        policy = ModelDecisionPolicy(config=cfg, rng=random.Random(17), layer_config={"mode": "model_mock"})
        global_high_local_low = MarketContext(
            day=60,
            adoption_rate=0.80,
            insurance_coverage_rate=0.60,
            avg_panic=0.10,
            recent_claim_rate=0.0,
            local_adoption_rate=0.0,
            local_insurance_coverage_rate=0.20,
            local_avg_panic=0.10,
            network_neighbor_count=8,
        )
        global_low_local_high = MarketContext(
            day=60,
            adoption_rate=0.10,
            insurance_coverage_rate=0.60,
            avg_panic=0.10,
            recent_claim_rate=0.0,
            local_adoption_rate=0.80,
            local_insurance_coverage_rate=0.60,
            local_avg_panic=0.10,
            network_neighbor_count=8,
        )

        low_local = policy.adoption_decision(firm, global_high_local_low, visible)
        high_local = policy.adoption_decision(firm, global_low_local_high, visible)

        self.assertGreater(low_local["threshold"], high_local["threshold"] + 0.05)

    def test_local_panic_dampens_peer_evidence_without_hard_ban(self):
        firm = FirmState(
            profile=FirmProfile(
                firm_id="F001",
                name="Network-aware cautious firm",
                industry="industrials",
                cash=100_000.0,
                asset_value=500_000.0,
                risk_tolerance=0.40,
                tech_urgency=0.55,
                ai_dependency=0.55,
                inertia=0.50,
                innovativeness=0.45,
                contagion_sensitivity=0.65,
                size_label="medium",
            ),
            cash=100_000.0,
        )
        visible = [VendorProfile("Vendor_Alpha", "Stable", 2_800.0, 0.011, 0.62, 0.78, 1.05)]
        cfg = {
            "insurance_market_enabled": True,
            "model_adoption_base_threshold": 0.70,
            "model_adoption_maturity_days": 90,
            "model_adoption_early_friction": 0.18,
            "model_adoption_peer_reference": 0.50,
            "model_adoption_peer_floor": 0.08,
            "model_adoption_low_peer_friction": 0.12,
            "model_adoption_peer_evidence_lag_days": 45,
            "model_adoption_local_evidence_relief": 0.14,
            "model_adoption_local_evidence_curve_power": 2.0,
            "model_adoption_local_evidence_panic_dampening": 0.80,
            "model_adoption_panic_threshold_penalty": 0.06,
        }
        policy = ModelDecisionPolicy(config=cfg, rng=random.Random(19), layer_config={"mode": "model_mock"})
        low_panic_context = MarketContext(
            day=60,
            adoption_rate=0.20,
            insurance_coverage_rate=0.60,
            avg_panic=0.05,
            recent_claim_rate=0.0,
            local_adoption_rate=0.80,
            local_insurance_coverage_rate=0.60,
            local_avg_panic=0.05,
            network_neighbor_count=8,
        )
        high_panic_context = MarketContext(
            day=60,
            adoption_rate=0.20,
            insurance_coverage_rate=0.60,
            avg_panic=0.05,
            recent_claim_rate=0.0,
            local_adoption_rate=0.80,
            local_insurance_coverage_rate=0.60,
            local_avg_panic=0.75,
            network_neighbor_count=8,
        )

        low_panic = policy.adoption_decision(firm, low_panic_context, visible)
        high_panic = policy.adoption_decision(firm, high_panic_context, visible)

        self.assertGreater(high_panic["threshold"], low_panic["threshold"] + 0.04)
        self.assertGreater(high_panic["score"], 0.0)

    def test_risk_transfer_evidence_gates_model_peer_adoption_signal(self):
        profile = FirmProfile(
            firm_id="F001",
            name="Evidence-sensitive firm",
            industry="industrials",
            cash=100_000.0,
            asset_value=500_000.0,
            risk_tolerance=0.40,
            tech_urgency=0.55,
            ai_dependency=0.55,
            inertia=0.50,
            innovativeness=0.45,
            contagion_sensitivity=0.65,
            size_label="medium",
        )
        firm = FirmState(profile=profile, cash=100_000.0)
        cfg = {
            "insurance_market_enabled": True,
            "model_adoption_base_threshold": 0.70,
            "model_adoption_min_threshold": 0.40,
            "model_adoption_maturity_days": 90,
            "model_adoption_early_friction": 0.18,
            "model_adoption_peer_reference": 0.50,
            "model_adoption_peer_floor": 0.08,
            "model_adoption_low_peer_friction": 0.12,
            "model_adoption_peer_evidence_lag_days": 45,
            "model_adoption_local_evidence_relief": 0.16,
            "model_adoption_local_evidence_curve_power": 2.0,
            "model_adoption_risk_transfer_evidence_gate": 0.70,
            "model_adoption_insurance_availability_floor": 0.15,
            "model_adoption_peer_coverage_weight": 0.65,
            "model_adoption_paid_claim_weight": 0.20,
            "model_adoption_paid_claim_reference": 0.015,
        }
        weak_evidence = MarketContext(
            day=70,
            adoption_rate=0.30,
            insurance_coverage_rate=0.10,
            avg_panic=0.05,
            recent_claim_rate=0.0,
            local_adoption_rate=0.80,
            local_insurance_coverage_rate=0.0,
            local_avg_panic=0.05,
            local_recent_claim_rate=0.0,
            network_neighbor_count=8,
        )
        strong_evidence = MarketContext(
            day=70,
            adoption_rate=0.30,
            insurance_coverage_rate=0.70,
            avg_panic=0.05,
            recent_claim_rate=0.02,
            local_adoption_rate=0.80,
            local_insurance_coverage_rate=0.75,
            local_avg_panic=0.05,
            local_recent_claim_rate=0.02,
            network_neighbor_count=8,
        )

        weak_threshold = _model_adoption_threshold(firm, weak_evidence, config=cfg)
        strong_threshold = _model_adoption_threshold(firm, strong_evidence, config=cfg)
        weak_state = _adoption_diffusion_state(firm, weak_evidence, cfg)
        strong_state = _adoption_diffusion_state(firm, strong_evidence, cfg)

        self.assertGreater(weak_threshold, strong_threshold + 0.06)
        self.assertLess(weak_state["risk_transfer_adjusted_peer_adoption_rate"], weak_state["panic_adjusted_peer_adoption_rate"])
        self.assertGreater(strong_state["risk_transfer_evidence"], weak_state["risk_transfer_evidence"] + 0.60)

    def test_bad_experience_memory_raises_model_adoption_threshold(self):
        profile = FirmProfile(
            firm_id="F001",
            name="Loss-aware firm",
            industry="industrials",
            cash=100_000.0,
            asset_value=500_000.0,
            risk_tolerance=0.35,
            tech_urgency=0.55,
            ai_dependency=0.55,
            inertia=0.45,
            innovativeness=0.45,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        calm = FirmState(profile=profile, cash=100_000.0)
        burned = FirmState(profile=profile, cash=100_000.0)
        burned.loss_memory = 0.80
        burned.claimable_memory = 0.75
        cfg = {
            "insurance_market_enabled": True,
            "model_adoption_base_threshold": 0.70,
            "model_adoption_maturity_days": 90,
            "model_adoption_early_friction": 0.18,
            "model_adoption_peer_reference": 0.50,
            "model_adoption_peer_floor": 0.08,
            "model_adoption_low_peer_friction": 0.12,
            "model_adoption_peer_evidence_lag_days": 45,
            "model_adoption_local_evidence_relief": 0.12,
            "model_adoption_local_evidence_curve_power": 2.0,
            "model_adoption_local_evidence_panic_dampening": 0.70,
            "model_adoption_loss_memory_threshold_penalty": 0.08,
            "model_adoption_claimable_memory_threshold_penalty": 0.07,
        }
        context = MarketContext(
            day=70,
            adoption_rate=0.20,
            insurance_coverage_rate=0.50,
            avg_panic=0.05,
            recent_claim_rate=0.0,
            local_adoption_rate=0.80,
            local_insurance_coverage_rate=0.50,
            local_avg_panic=0.15,
            network_neighbor_count=8,
        )

        calm_threshold = _model_adoption_threshold(calm, context, config=cfg)
        burned_threshold = _model_adoption_threshold(burned, context, config=cfg)

        self.assertGreater(burned_threshold, calm_threshold + 0.08)

    def test_model_renewal_threshold_uses_panic_and_insurance_confidence(self):
        profile = FirmProfile(
            firm_id="F001",
            name="Renewal firm",
            industry="industrials",
            cash=100_000.0,
            asset_value=500_000.0,
            risk_tolerance=0.35,
            tech_urgency=0.55,
            ai_dependency=0.55,
            inertia=0.45,
            innovativeness=0.45,
            contagion_sensitivity=0.55,
            size_label="medium",
        )
        protected = FirmState(profile=profile, cash=100_000.0)
        protected.insurance_policy = InsurancePolicy(
            insurer_id="Insurer_Test",
            premium=1_000.0,
            deductible_ratio=0.30,
            coverage_ratio=0.60,
            limit_money=20_000.0,
            incident_threshold=0.50,
            start_day=0,
            end_day=30,
        )
        exposed = FirmState(profile=profile, cash=100_000.0)
        exposed.loss_memory = 0.80
        exposed.claimable_memory = 0.75
        cfg = {
            "insurance_market_enabled": True,
            "model_adoption_insurance_confidence_discount": 0.04,
            "model_renewal_insurance_confidence_discount": 0.04,
            "model_renewal_loss_memory_threshold_penalty": 0.08,
            "model_renewal_claimable_memory_threshold_penalty": 0.06,
            "model_renewal_panic_threshold_penalty": 0.08,
        }
        calm_context = MarketContext(
            day=60,
            adoption_rate=0.50,
            insurance_coverage_rate=0.70,
            avg_panic=0.02,
            recent_claim_rate=0.0,
            local_adoption_rate=0.70,
            local_insurance_coverage_rate=0.80,
            local_avg_panic=0.02,
            network_neighbor_count=8,
        )
        panic_context = MarketContext(
            day=60,
            adoption_rate=0.50,
            insurance_coverage_rate=0.10,
            avg_panic=0.60,
            recent_claim_rate=0.0,
            local_adoption_rate=0.70,
            local_insurance_coverage_rate=0.00,
            local_avg_panic=0.60,
            network_neighbor_count=8,
        )

        protected_threshold = _model_adoption_threshold(protected, calm_context, renewal=True, config=cfg)
        exposed_threshold = _model_adoption_threshold(exposed, panic_context, renewal=True, config=cfg)

        self.assertGreater(exposed_threshold, protected_threshold + 0.08)

    def test_vendor_negotiation_uses_local_network_context(self):
        firm = FirmState(
            profile=FirmProfile(
                firm_id="F001",
                name="Buyer",
                industry="industrials",
                cash=100_000.0,
                asset_value=500_000.0,
                risk_tolerance=0.45,
                tech_urgency=0.55,
                ai_dependency=0.50,
                inertia=0.40,
                innovativeness=0.45,
                contagion_sensitivity=0.55,
                size_label="medium",
            ),
            cash=100_000.0,
        )
        engine = NegotiationEngine(config={"negotiation": {}}, rng=random.Random(9))
        local_low = MarketContext(
            day=10,
            adoption_rate=0.90,
            insurance_coverage_rate=0.50,
            avg_panic=0.05,
            recent_claim_rate=0.0,
            local_adoption_rate=0.0,
            local_avg_panic=0.05,
            network_neighbor_count=8,
        )
        local_high = MarketContext(
            day=10,
            adoption_rate=0.10,
            insurance_coverage_rate=0.50,
            avg_panic=0.05,
            recent_claim_rate=0.0,
            local_adoption_rate=0.90,
            local_avg_panic=0.05,
            network_neighbor_count=8,
        )

        self.assertGreater(
            engine._vendor_opening_ratio(firm, local_high),
            engine._vendor_opening_ratio(firm, local_low),
        )

    def test_neighbor_uninsured_loss_propagates_panic_locally(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "network_panic", days=1, firms=4)
            fids = list(sim.firms)
            target = sim.firms[fids[0]]
            source = sim.firms[fids[1]]
            isolated = sim.firms[fids[2]]
            sim.firm_network = {
                target.profile.firm_id: [source.profile.firm_id],
                source.profile.firm_id: [],
                isolated.profile.firm_id: [],
                fids[3]: [],
            }
            sim.config["panic"]["network_neighbor_weight"] = 0.70
            sim.config["panic"]["own_event_weight"] = 0.0
            sim.config["panic"]["calm_decay"] = 0.0
            source._claimable_event_score_today = 1.0
            source._uninsured_claimable_today_flag = 1.0

            sim._update_panic(
                {
                    "active_count": 4,
                    "material_events": 0,
                    "claimable_events": 0,
                    "claim_events": 0,
                    "uninsured_claimable_events": 0,
                    "avg_material_event_score": 0.0,
                    "indemnity_relief_ratio": 0.0,
                }
            )

            self.assertGreater(target.panic, isolated.panic)
            self.assertGreater(target.panic, 0.0)
            sim.close()

    def test_bankrupt_uninsured_event_source_still_propagates_same_day_panic(self):
        config = TEST_CONFIG
        with tempfile.TemporaryDirectory() as tmp:
            sim = ActionRiskSimulator.from_yaml(config_path=config, run_dir=Path(tmp) / "bankrupt_source_panic", days=1, firms=4)
            try:
                fids = list(sim.firms)
                target = sim.firms[fids[0]]
                source = sim.firms[fids[1]]
                isolated = sim.firms[fids[2]]
                sim.firm_network = {
                    target.profile.firm_id: [source.profile.firm_id],
                    source.profile.firm_id: [],
                    isolated.profile.firm_id: [],
                    fids[3]: [],
                }
                sim.config["panic"]["network_neighbor_weight"] = 0.70
                sim.config["panic"]["own_event_weight"] = 0.0
                sim.config["panic"]["calm_decay"] = 0.0
                source.active = False
                source._claimable_event_score_today = 1.0
                source._uninsured_claimable_today_flag = 1.0

                sim._update_panic(
                    {
                        "active_count": 3,
                        "material_events": 0,
                        "claimable_events": 1,
                        "claim_events": 0,
                        "uninsured_claimable_events": 1,
                        "avg_material_event_score": 0.0,
                        "indemnity_relief_ratio": 0.0,
                    }
                )
            finally:
                sim.close()

        self.assertGreater(target.panic, isolated.panic)
        self.assertGreater(target.panic, 0.0)


if __name__ == "__main__":
    unittest.main()
