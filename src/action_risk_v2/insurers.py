from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

from .schema import (
    FirmState,
    IndustryRiskSnapshot,
    InsurancePolicy,
    InsuranceQuote,
    InsurerProfile,
    InsurerState,
    VendorProfile,
)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def load_insurer_profiles(config_rows: Iterable[dict]) -> List[InsurerProfile]:
    profiles: List[InsurerProfile] = []
    for row in config_rows:
        profiles.append(
            InsurerProfile(
                insurer_id=str(row["id"]),
                label=str(row.get("label", row["id"])),
                domicile=str(row.get("domicile", "unknown")),
                initial_capital=float(row.get("initial_capital", 5_000_000.0)),
                base_margin=float(row.get("base_margin", 0.25)),
                risk_appetite=float(row.get("risk_appetite", 0.5)),
                expense_load=float(row.get("expense_load", 0.04)),
                capital_load=float(row.get("capital_load", 0.0)),
                deductible_ratio=float(row.get("deductible_ratio", 0.30)),
                coverage_ratio=float(row.get("coverage_ratio", 0.65)),
                limit_ratio=float(row.get("limit_ratio", 0.12)),
                max_active_policies=int(row.get("max_active_policies", 1_000_000_000)),
                solvency_floor_ratio=float(row.get("solvency_floor_ratio", 0.22)),
                soft_threshold_ratio=float(row.get("soft_threshold_ratio", 0.70)),
                hard_threshold_ratio=float(row.get("hard_threshold_ratio", 0.45)),
                target_sectors=tuple(row.get("target_sectors", ["ALL"])),
                market_role=str(row.get("market_role", "private")),
            )
        )
    return profiles


def load_vendor_profiles(config_rows: Iterable[dict]) -> List[VendorProfile]:
    profiles: List[VendorProfile] = []
    for row in config_rows:
        profiles.append(
            VendorProfile(
                vendor_id=str(row["id"]),
                label=str(row.get("label", row["id"])),
                subscription_fee=float(row.get("subscription_fee", 2000.0)),
                productivity_lift=float(row.get("productivity_lift", 0.010)),
                risk_multiplier=float(row.get("risk_multiplier", 1.0)),
                reputation=float(row.get("reputation", 0.6)),
                marketing_weight=float(row.get("marketing_weight", 1.0)),
                target_sectors=tuple(row.get("target_sectors", ["ALL"])),
            )
        )
    return profiles


class InsuranceMarket:
    """Competitive insurance market with insurer capital and solvency regimes."""

    REGIME_PREMIUM_MULT = {
        "NORMAL": 1.00,
        "SOFT": 1.20,
        "HARD": 1.75,
        "RUNOFF": math.inf,
    }

    def __init__(self, profiles: Iterable[InsurerProfile], pricing_config: Optional[dict] = None):
        self.insurers: Dict[str, InsurerState] = {
            p.insurer_id: InsurerState(profile=p, capital=float(p.initial_capital))
            for p in profiles
        }
        self.pricing_config = dict(pricing_config or {})

    def start_day(self) -> None:
        for state in self.insurers.values():
            state.premiums_today = 0.0
            state.claims_today = 0.0
            state.refunds_today = 0.0
            state.new_policies_today = 0

    def mark_active_policy(self, insurer_id: Optional[str]) -> None:
        if insurer_id and insurer_id in self.insurers:
            self.insurers[insurer_id].active_policies += 1

    def quote_all(
        self,
        firm: FirmState,
        vendor: VendorProfile,
        snapshot: IndustryRiskSnapshot,
        day: int,
        term_days: int,
        market_panic: float,
        recent_claim_rate: float,
        include_backstop: bool = False,
        only_backstop: bool = False,
    ) -> List[InsuranceQuote]:
        quotes = []
        for state in self.insurers.values():
            is_backstop = state.profile.market_role == "backstop"
            if only_backstop and not is_backstop:
                continue
            if is_backstop and not include_backstop:
                continue
            q = self._quote_one(
                state=state,
                firm=firm,
                vendor=vendor,
                snapshot=snapshot,
                day=day,
                term_days=term_days,
                market_panic=market_panic,
                recent_claim_rate=recent_claim_rate,
            )
            if q is not None:
                quotes.append(q)
        return sorted(quotes, key=lambda x: (x.premium, -x.coverage_ratio, x.deductible_ratio))

    def _quote_one(
        self,
        state: InsurerState,
        firm: FirmState,
        vendor: VendorProfile,
        snapshot: IndustryRiskSnapshot,
        day: int,
        term_days: int,
        market_panic: float,
        recent_claim_rate: float,
    ) -> Optional[InsuranceQuote]:
        if not state.underwriting_open:
            return None
        if state.active_policies >= state.profile.max_active_policies:
            return None

        profile = state.profile
        exposure_scale = 0.55 + float(firm.profile.ai_dependency)
        term_scale = max(1.0, float(term_days)) / 30.0
        expected_loss = max(0.0, snapshot.avg_loss * vendor.risk_multiplier * exposure_scale * term_scale)
        stress_loss = max(0.0, snapshot.stress_loss * vendor.risk_multiplier * exposure_scale * math.sqrt(term_scale))

        regime_mult = self.REGIME_PREMIUM_MULT[state.regime]
        sector_mult = profile.sector_affinity(firm.profile.industry)
        appetite_mult = 1.0 + 0.55 * (1.0 - _clamp(profile.risk_appetite, 0.0, 1.0))
        sentiment_mult = 1.0 + 0.35 * _clamp(market_panic, 0.0, 1.0) + 0.80 * _clamp(recent_claim_rate, 0.0, 1.0)
        solvency_mult = 1.0 + max(0.0, profile.soft_threshold_ratio - state.capital_ratio)
        portfolio_load = state.active_policies / max(float(profile.max_active_policies), 1.0)
        portfolio_mult = 1.0 + 0.70 * portfolio_load + 1.20 * max(0.0, portfolio_load - 0.65)

        deductible_ratio = _clamp(
            profile.deductible_ratio + float(self.pricing_config.get("deductible_ratio_delta", 0.0)),
            0.05,
            0.90,
        )
        coverage_ratio = _clamp(
            profile.coverage_ratio + float(self.pricing_config.get("coverage_ratio_delta", 0.0)),
            0.15,
            0.95,
        )
        limit_ratio_multiplier = max(0.10, float(self.pricing_config.get("limit_ratio_multiplier", 1.0)))
        limit_money = float(firm.profile.asset_value * profile.limit_ratio * limit_ratio_multiplier)

        pure_risk_mult = float(self.pricing_config.get("pure_loss_multiplier", 1.0))
        stress_loss_mult = float(self.pricing_config.get("stress_loss_multiplier", 1.0))
        expense_mult = float(self.pricing_config.get("expense_multiplier", 1.0))
        global_mult = float(self.pricing_config.get("global_multiplier", 1.0))
        tail_multiplier = float(self.pricing_config.get("catastrophic_tail_loss_multiplier", 0.0))
        tail_pricing_share = float(self.pricing_config.get("catastrophic_tail_pricing_share", 0.0))
        tail_pricing_mult = 1.0 + max(0.0, tail_multiplier) * max(0.0, tail_pricing_share)

        covered_loss_share = (1.0 - deductible_ratio) * coverage_ratio
        pure_risk = expected_loss * covered_loss_share * pure_risk_mult
        tail_load = stress_loss * covered_loss_share * profile.base_margin * appetite_mult * stress_loss_mult * tail_pricing_mult
        expense = firm.profile.asset_value * profile.expense_load * 0.01 * term_scale * expense_mult
        capital_load_mult = 1.0 + _clamp(profile.capital_load, 0.0, 2.0)
        premium = (
            pure_risk
            + tail_load
            + expense
        ) * capital_load_mult * regime_mult * sector_mult * sentiment_mult * solvency_mult * portfolio_mult * global_mult
        premium_cap_asset_share = float(self.pricing_config.get("premium_cap_asset_share", 0.18))
        premium = max(50.0, min(premium, firm.profile.asset_value * premium_cap_asset_share))

        incident_threshold = _clamp(
            0.25
            + 0.45 * deductible_ratio
            - 0.20 * coverage_ratio
            + 0.18 * (1.0 - profile.risk_appetite),
            0.18,
            0.90,
        )
        incident_threshold = _clamp(
            incident_threshold + float(self.pricing_config.get("incident_threshold_delta", 0.0)),
            0.05,
            0.95,
        )

        return InsuranceQuote(
            insurer_id=profile.insurer_id,
            firm_id=firm.profile.firm_id,
            vendor_id=vendor.vendor_id,
            industry=firm.profile.industry,
            day=int(day),
            term_days=int(term_days),
            premium=float(premium),
            deductible_ratio=float(deductible_ratio),
            coverage_ratio=float(coverage_ratio),
            limit_money=float(limit_money),
            incident_threshold=float(incident_threshold),
            expected_loss=float(expected_loss),
            stress_loss=float(stress_loss),
            regime=state.regime,
            market_role=profile.market_role,
        )

    def bind_policy(self, quote: InsuranceQuote, day: int) -> InsurancePolicy:
        state = self.insurers[quote.insurer_id]
        state.capital += quote.premium
        state.premiums_today += quote.premium
        state.new_policies_today += 1
        state.active_policies += 1
        return InsurancePolicy(
            insurer_id=quote.insurer_id,
            premium=quote.premium,
            deductible_ratio=quote.deductible_ratio,
            coverage_ratio=quote.coverage_ratio,
            limit_money=quote.limit_money,
            incident_threshold=quote.incident_threshold,
            start_day=int(day),
            end_day=int(day) + int(quote.term_days),
            vendor_id=quote.vendor_id,
        )

    def cancel_policy(
        self,
        policy: InsurancePolicy,
        day: int,
        refund_penalty_ratio: float = 0.06,
    ) -> float:
        state = self.insurers.get(policy.insurer_id)
        if state is None:
            return 0.0

        term_days = max(1, int(policy.end_day) - int(policy.start_day))
        remaining_days = max(0, int(policy.end_day) - int(day))
        unearned = float(policy.premium) * min(1.0, remaining_days / term_days)
        refund = unearned * max(0.0, 1.0 - float(refund_penalty_ratio))

        floor_capital = state.profile.initial_capital * state.profile.solvency_floor_ratio
        max_refundable = max(0.0, state.capital - floor_capital)
        refund = min(float(refund), max_refundable)
        if refund > 0:
            state.capital -= refund
            state.refunds_today += refund
        state.active_policies = max(0, int(state.active_policies) - 1)
        return float(refund)

    def process_claim(self, policy: InsurancePolicy, loss_amount: float, incident_score: float) -> float:
        state = self.insurers.get(policy.insurer_id)
        if state is None:
            return 0.0
        if incident_score < policy.incident_threshold:
            return 0.0

        loss_amount = max(0.0, float(loss_amount))
        covered = max(0.0, loss_amount * (1.0 - policy.deductible_ratio))
        payout = min(policy.limit_money, covered * policy.coverage_ratio)

        floor_capital = state.profile.initial_capital * state.profile.solvency_floor_ratio
        max_payable = max(0.0, state.capital - floor_capital)
        payout = min(payout, max_payable)

        if payout > 0:
            state.capital -= payout
            state.claims_today += payout
        return float(payout)

    def daily_rows(self, day: int) -> List[dict]:
        rows = []
        for state in self.insurers.values():
            rows.append(
                {
                    "day": int(day),
                        "insurer_id": state.profile.insurer_id,
                        "label": state.profile.label,
                        "domicile": state.profile.domicile,
                        "market_role": state.profile.market_role,
                        "capital": float(state.capital),
                    "capital_ratio": float(state.capital_ratio),
                    "regime": state.regime,
                    "underwriting_open": bool(state.underwriting_open),
                    "premiums_today": float(state.premiums_today),
                    "claims_today": float(state.claims_today),
                    "refunds_today": float(state.refunds_today),
                    "active_policies": int(state.active_policies),
                    "new_policies_today": int(state.new_policies_today),
                }
            )
        return rows
