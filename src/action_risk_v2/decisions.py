from __future__ import annotations

import json
import hashlib
import math
import random
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
from urllib import request
from urllib.error import URLError

from .schema import FirmState, InsuranceQuote, VendorProfile


@dataclass(frozen=True)
class MarketContext:
    day: int
    adoption_rate: float
    insurance_coverage_rate: float
    avg_panic: float
    recent_claim_rate: float
    local_adoption_rate: Optional[float] = None
    local_insurance_coverage_rate: Optional[float] = None
    local_avg_panic: Optional[float] = None
    local_recent_claim_rate: Optional[float] = None
    network_neighbor_count: int = 0
    same_industry_neighbor_share: float = 0.0


class HeuristicDecisionPolicy:
    """Local deterministic-enough policy used before the vLLM decision layer is added."""

    def __init__(self, config: dict, rng: random.Random):
        self.config = dict(config or {})
        self.rng = rng

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(value)))

    def _rng_for(self, channel: str, firm: Optional[FirmState] = None, context: Optional[MarketContext] = None) -> random.Random:
        seed = self.config.get("common_random_seed", None)
        if seed is None or not bool(self.config.get("common_random_numbers", True)):
            return self.rng
        firm_id = str(firm.profile.firm_id if firm is not None else "")
        day = int(context.day if context is not None else getattr(firm, "_decision_day", 0) or 0)
        payload = f"{int(seed)}|decision|{channel}|{firm_id}|{day}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    def visible_vendors(self, firm: FirmState, vendors: Iterable[VendorProfile]) -> List[VendorProfile]:
        vendors = list(vendors)
        if not vendors:
            return []

        bandwidth = int(self.config.get("ads_bandwidth", 3))
        bandwidth = max(1, min(len(vendors), bandwidth))

        # Always include one exploration slot, then sample the rest by marketing/sector fit.
        rng = self._rng_for("visible_vendors", firm=firm)
        visible = [rng.choice(vendors)]
        remaining = [v for v in vendors if v.vendor_id != visible[0].vendor_id]

        while len(visible) < bandwidth and remaining:
            weights = [
                max(
                    0.01,
                    v.marketing_weight
                    * v.sector_affinity(firm.profile.industry)
                    * (0.5 + v.reputation),
                )
                for v in remaining
            ]
            pick = rng.choices(remaining, weights=weights, k=1)[0]
            visible.append(pick)
            remaining = [v for v in remaining if v.vendor_id != pick.vendor_id]
        return visible

    def adoption_decision(
        self,
        firm: FirmState,
        context: MarketContext,
        visible: List[VendorProfile],
        renewal: bool = False,
    ) -> Dict[str, object]:
        if firm.has_ai or not firm.active or not visible:
            return {"action": False, "probability": 0.0, "draw": 1.0, "score": 0.0, "reason": "not_eligible"}

        if renewal:
            p = float(self.config.get("renewal_base_probability", 0.58))
            p += 0.12 * firm.profile.tech_urgency
            p += 0.06 * firm.profile.innovativeness
            if bool(self.config.get("insurance_market_enabled", True)):
                market_confidence = 0.35 + 0.65 * _context_peer_insurance_coverage_rate(context)
                own_cover = 1.0 if firm.has_insurance else 0.0
                p += (
                    float(self.config.get("renewal_insurance_confidence_weight", 0.055))
                    * (0.55 * own_cover + 0.45 * market_confidence)
                    * (1.0 - 0.45 * firm.profile.risk_tolerance)
                )
            p -= 0.22 * firm.profile.inertia
            p -= float(self.config.get("renewal_risk_memory_weight", 0.34)) * firm.risk_memory * (
                1.0 - 0.35 * firm.profile.risk_tolerance
            )
            p -= float(self.config.get("renewal_loss_memory_weight", 0.18)) * firm.loss_memory
            p -= float(self.config.get("renewal_claimable_memory_weight", 0.10)) * firm.claimable_memory
            p -= float(self.config.get("renewal_panic_weight", 0.18)) * _context_panic_rate(context) * (
                1.0 - firm.profile.risk_tolerance
            )
            p = self._clamp(
                p,
                float(self.config.get("renewal_min_probability", 0.12)),
                float(self.config.get("renewal_max_probability", 0.82)),
            )
            draw = self._rng_for("vendor_renewal_decision", firm=firm, context=context).random()
            return {
                "action": draw < p,
                "probability": float(p),
                "draw": float(draw),
                "score": float(p),
                "reason": "renew_contract" if draw < p else "renewal_friction",
            }

        if int(context.day) < int(getattr(firm, "ai_cooldown_until", 0)):
            return {
                "action": False,
                "probability": 0.0,
                "draw": 1.0,
                "score": 0.0,
                "reason": "cooldown_after_bad_experience",
            }

        p = float(self.config.get("adoption_base_probability", 0.008))
        p += float(self.config.get("adoption_tech_weight", 0.050)) * firm.profile.tech_urgency
        p += float(self.config.get("adoption_innov_weight", 0.035)) * firm.profile.innovativeness
        peer_rate = _context_peer_adoption_rate(context)
        peer_insurance_coverage = _context_peer_insurance_coverage_rate(context)
        panic_rate = _context_panic_rate(context)
        peer_rate = _risk_transfer_adjusted_peer_rate(self.config, context, peer_rate, prefix="adoption")
        p += float(self.config.get("adoption_peer_weight", 0.035)) * peer_rate
        if bool(self.config.get("insurance_market_enabled", True)):
            market_confidence = 0.35 + 0.65 * peer_insurance_coverage
            p += (
                float(self.config.get("adoption_insurance_confidence_weight", 0.020))
                * market_confidence
                * (1.0 - 0.35 * firm.profile.risk_tolerance)
            )
        p -= float(self.config.get("adoption_inertia_weight", 0.040)) * firm.profile.inertia
        p -= float(self.config.get("adoption_panic_weight", 0.055)) * panic_rate * (1.0 - firm.profile.risk_tolerance)
        p -= float(self.config.get("adoption_loss_memory_weight", 0.018)) * firm.loss_memory * (
            1.0 - 0.25 * firm.profile.risk_tolerance
        )
        p -= float(self.config.get("adoption_claimable_memory_weight", 0.014)) * firm.claimable_memory * (
            1.0 - 0.25 * firm.profile.risk_tolerance
        )

        saturation = 1.0 - float(self.config.get("adoption_saturation_weight", 0.55)) * context.adoption_rate
        saturation = self._clamp(saturation, float(self.config.get("adoption_saturation_floor", 0.32)), 1.0)
        p *= saturation
        p = self._clamp(
            p,
            float(self.config.get("adoption_min_probability", 0.002)),
            float(self.config.get("adoption_max_probability", 0.075)),
        )
        draw = self._rng_for("ai_adoption_decision", firm=firm, context=context).random()
        return {
            "action": draw < p,
            "probability": float(p),
            "draw": float(draw),
            "score": float(p),
            "reason": "new_adoption" if draw < p else "wait_and_see",
        }

    def choose_vendor(self, firm: FirmState, visible: List[VendorProfile]) -> Optional[VendorProfile]:
        if not visible:
            return None
        cash = max(1.0, firm.cash)

        def score(v: VendorProfile) -> float:
            affordability_penalty = min(1.0, v.subscription_fee / (cash * 0.045))
            risk_penalty = v.risk_multiplier * (1.0 - firm.profile.risk_tolerance)
            sector_bonus = 0.10 * v.sector_affinity(firm.profile.industry)
            return (
                1.15 * v.productivity_lift
                + 0.32 * v.reputation
                + sector_bonus
                - 0.24 * affordability_penalty
                - 0.18 * risk_penalty
            )

        affordable = [v for v in visible if v.subscription_fee <= cash * 0.06]
        candidates = affordable or visible
        return max(candidates, key=score)

    def vendor_term_days(self, firm: FirmState, context: MarketContext) -> int:
        lo = int(self.config.get("vendor_min_term_days", 14))
        hi = int(self.config.get("vendor_max_term_days", 120))
        panic_rate = _context_panic_rate(context)
        if panic_rate > 0.45 or firm.profile.risk_tolerance < 0.30:
            base = 28
            spread = 10
        elif firm.profile.inertia > 0.62:
            base = 92
            spread = 18
        else:
            base = 58 + int(round(18 * firm.profile.tech_urgency)) - int(round(12 * panic_rate))
            spread = 15
        rng = self._rng_for("vendor_term_days", firm=firm, context=context)
        return _bounded_int(base + rng.randint(-spread, spread), default=60, lo=lo, hi=hi)

    def insurance_score(self, firm: FirmState, context: MarketContext, risk_signal: Dict[str, float]) -> float:
        score = float(self.config.get("insurance_base_score", 0.06))
        score += float(self.config.get("insurance_claimable_weight", 0.28)) * float(risk_signal.get("claimable_event_score", 0.0))
        score += float(self.config.get("insurance_material_weight", 0.18)) * float(risk_signal.get("material_event_score", 0.0))
        score += float(self.config.get("insurance_memory_weight", 0.16)) * firm.risk_memory
        score += float(self.config.get("insurance_claimable_memory_weight", 0.12)) * firm.claimable_memory
        score += float(self.config.get("insurance_industry_incident_weight", 0.08)) * float(risk_signal.get("industry_incident_rate", 0.0))
        score += float(self.config.get("insurance_industry_stress_weight", 0.10)) * float(risk_signal.get("industry_stress_score", 0.0))
        score += float(self.config.get("insurance_industry_loss_weight", 0.06)) * float(risk_signal.get("industry_loss_pressure", 0.0))
        score += float(self.config.get("insurance_panic_weight", 0.10)) * _context_panic_rate(context)
        score += float(self.config.get("insurance_recent_claim_weight", 0.15)) * _context_recent_claim_rate(context)
        score += float(self.config.get("insurance_peer_coverage_weight", 0.03)) * _context_peer_insurance_coverage_rate(context)
        score += float(self.config.get("insurance_risk_aversion_weight", 0.14)) * (1.0 - firm.profile.risk_tolerance)
        score += float(self.config.get("insurance_ai_dependency_weight", 0.11)) * firm.profile.ai_dependency
        score += float(self.config.get("insurance_renewal_bonus", 0.06)) * float(risk_signal.get("prior_policy", 0.0))
        score -= float(self.config.get("insurance_innovativeness_discount", 0.07)) * firm.profile.innovativeness
        return float(score)

    def insurance_decision(self, firm: FirmState, context: MarketContext, risk_signal: Dict[str, float]) -> Dict[str, object]:
        if not firm.has_ai or firm.has_insurance or not firm.active:
            return {"action": False, "score": 0.0, "threshold": 1.0, "draw": 0.0, "reason": "not_eligible"}

        score = self.insurance_score(firm, context, risk_signal)
        threshold = float(self.config.get("insurance_buy_threshold", 0.70))
        if float(risk_signal.get("prior_policy", 0.0)) > 0.0:
            threshold -= float(self.config.get("insurance_renewal_threshold_discount", 0.0))
        threshold = self._clamp(threshold, float(self.config.get("insurance_min_threshold", 0.0)), 1.0)
        noise = self._rng_for("insurance_purchase_decision", firm=firm, context=context).uniform(-0.05, 0.05)
        action = score + noise >= threshold
        return {
            "action": bool(action),
            "score": float(score),
            "threshold": float(threshold),
            "draw": float(noise),
            "reason": (
                "renew_risk_transfer_policy"
                if action and float(risk_signal.get("prior_policy", 0.0)) > 0.0
                else ("risk_transfer_demand" if action else "self_insure_for_now")
            ),
        }

    def insurance_term_days(
        self,
        firm: FirmState,
        context: MarketContext,
        incident_score: float,
        risk_signal: Optional[Dict[str, float]] = None,
    ) -> int:
        signal = risk_signal or {}
        lo = int(self.config.get("insurance_min_term_days", 1))
        hi = int(self.config.get("insurance_max_term_days", 90))
        ai_remaining = int(signal.get("ai_remaining_days", 0) or 0)
        if ai_remaining > 0:
            hi = min(hi, ai_remaining)
        if hi <= 0:
            return 0
        if firm.cash > 0:
            premium_pressure = float(signal.get("indicative_premium", 0.0)) / max(float(firm.cash), 1.0)
            if premium_pressure > float(self.config.get("insurance_cash_tight_share", 0.045)):
                rng = self._rng_for("insurance_term_days", firm=firm, context=context)
                return _bounded_int(14 + rng.randint(-5, 7), default=14, lo=lo, hi=hi)
        medium_threshold = float(self.config.get("insurance_medium_risk_term_threshold", 0.42))
        high_threshold = float(self.config.get("insurance_high_risk_term_threshold", 0.70))
        medium_dependency = float(self.config.get("insurance_medium_dependency_threshold", 0.68))
        medium_risk_tolerance = float(self.config.get("insurance_medium_risk_tolerance_threshold", 0.45))
        score_proxy = max(
            float(incident_score),
            float(signal.get("claimable_event_score", 0.0)),
            float(signal.get("material_event_score", 0.0)),
            float(signal.get("industry_stress_score", 0.0)),
        )
        if (
            score_proxy > high_threshold
            or _context_panic_rate(context) > 0.65
            or float(signal.get("claimable_memory", firm.claimable_memory)) > 0.55
        ):
            rng = self._rng_for("insurance_term_days", firm=firm, context=context)
            return _bounded_int(78 + rng.randint(-12, 12), default=78, lo=lo, hi=hi)
        if (
            float(signal.get("prior_policy", 0.0)) > 0.0
            or score_proxy > medium_threshold
            or firm.profile.risk_tolerance < medium_risk_tolerance
            or firm.profile.ai_dependency > medium_dependency
            or _context_recent_claim_rate(context) > 0.015
        ):
            rng = self._rng_for("insurance_term_days", firm=firm, context=context)
            return _bounded_int(48 + rng.randint(-10, 12), default=48, lo=lo, hi=hi)
        rng = self._rng_for("insurance_term_days", firm=firm, context=context)
        return _bounded_int(21 + rng.randint(-7, 9), default=21, lo=lo, hi=hi)

    def exposure_decision(self, firm: FirmState, context: MarketContext, risk_signal: Dict[str, float]) -> Dict[str, object]:
        if not firm.has_ai or not firm.active:
            return {"action": False, "score": 0.0, "threshold": 1.0, "draw": 0.0, "reason": "not_eligible", "vendor_action": "keep_vendor"}
        return {
            "action": False,
            "score": 0.0,
            "threshold": float(self.config.get("abandon_score_threshold", 0.52)),
            "draw": 0.0,
            "reason": "rule_keep_existing_ai_exposure",
            "vendor_action": "keep_vendor",
        }

    def quote_utilities(
        self,
        firm: FirmState,
        quotes: List[InsuranceQuote],
        risk_need: float,
        allow_backstop: bool = False,
    ) -> List[Tuple[InsuranceQuote, float, str]]:
        cash = max(1.0, firm.cash)
        asset_value = max(firm.profile.asset_value, 1.0)
        max_cash_share = float(self.config.get("max_premium_cash_share", 0.035))
        target_cash_share = float(self.config.get("target_premium_cash_share", 0.010))
        price_sensitivity = float(self.config.get("quote_price_sensitivity", 0.28))
        deductible_weight = float(self.config.get("quote_deductible_weight", 0.42))
        min_utility = float(self.config.get("min_quote_utility", 0.02))
        cash_ratio = cash / asset_value
        cash_tightness = self._clamp((0.18 - cash_ratio) / 0.18, 0.0, 1.0)
        protection_preference = self._clamp(
            0.70
            + 0.28 * (1.0 - firm.profile.risk_tolerance)
            + 0.24 * firm.profile.ai_dependency
            + 0.22 * float(risk_need),
            0.70,
            1.45,
        )
        cost_preference = self._clamp(
            price_sensitivity
            * (
                0.65
                + 0.35 * firm.profile.risk_tolerance
                + 0.40 * cash_tightness
                + 0.20 * (1.0 - float(risk_need))
            ),
            0.18,
            1.35,
        )
        out: List[Tuple[InsuranceQuote, float, str]] = []

        for q in quotes:
            if q.market_role == "backstop" and not allow_backstop:
                out.append((q, -999.0, "backstop_reserved"))
                continue
            if q.premium > cash * max_cash_share:
                out.append((q, -999.0, "over_cash_budget"))
                continue

            premium_burden = q.premium / max(cash * target_cash_share, 1.0)
            limit_ratio = q.limit_money / asset_value
            coverage_effect = 1.0 - math.exp(-2.10 * max(0.0, q.coverage_ratio))
            limit_effect = 1.0 - math.exp(-8.0 * max(0.0, limit_ratio))
            deductible_benefit = 1.0 - q.deductible_ratio
            actuarial_value = (
                q.expected_loss * q.coverage_ratio
                + 0.35 * q.stress_loss * q.coverage_ratio
            ) / max(q.premium, 1.0)
            value_score = self._clamp(actuarial_value / 1.35, 0.0, 1.20)
            protection = (
                0.34 * coverage_effect
                + 0.18 * limit_effect
                + 0.18 * deductible_benefit
                + 0.30 * value_score
            )
            tail_need = self._clamp((float(risk_need) - 0.55) / 0.45, 0.0, 1.0) * self._clamp(
                firm.profile.ai_dependency, 0.0, 1.0
            )
            friction = cost_preference * premium_burden + deductible_weight * q.deductible_ratio
            role_penalty = float(self.config.get("backstop_utility_penalty", 0.16)) if q.market_role == "backstop" else 0.0
            utility = float(risk_need) * protection_preference * protection + 0.08 * tail_need * coverage_effect - friction - role_penalty
            reason = "candidate" if utility >= min_utility else "low_net_utility"
            out.append((q, float(utility), reason))
        return out

    def choose_quote_with_diagnostics(
        self,
        firm: FirmState,
        quotes: List[InsuranceQuote],
        risk_need: float,
        allow_backstop: bool = False,
    ) -> Tuple[Optional[InsuranceQuote], List[Tuple[InsuranceQuote, float, str]]]:
        if not quotes:
            return None, []
        diagnostics = self.quote_utilities(firm, quotes, risk_need, allow_backstop=allow_backstop)
        candidates = [(q, u, r) for q, u, r in diagnostics if r == "candidate"]
        if not candidates:
            return None, diagnostics
        best = max(candidates, key=lambda item: (item[1], item[0].coverage_ratio, -item[0].premium))
        return best[0], diagnostics

class ModelDecisionPolicy(HeuristicDecisionPolicy):
    """OpenAI-compatible decision policy with a local mock mode for offline tests."""

    def __init__(self, config: dict, rng: random.Random, layer_config: dict):
        super().__init__(config, rng)
        self.layer_config = dict(layer_config or {})
        self.mode = str(self.layer_config.get("mode", "model_mock"))
        self.base_urls = _parse_base_urls(self.layer_config)
        self.base_url = self.base_urls[0]
        self._endpoint_lock = threading.Lock()
        self._endpoint_index = 0
        self.model = str(self.layer_config.get("model", "qwen3-local"))
        self.api_key = str(self.layer_config.get("api_key", "EMPTY"))
        self.timeout_seconds = float(self.layer_config.get("timeout_seconds", 30))
        self.temperature = float(self.layer_config.get("temperature", 0.0))
        self.max_tokens = int(self.layer_config.get("max_tokens", 220))
        self.fallback_to_rule = bool(self.layer_config.get("fallback_to_rule", True))
        self.json_retries = max(0, int(self.layer_config.get("json_retries", 2)))

    def adoption_decision(
        self,
        firm: FirmState,
        context: MarketContext,
        visible: List[VendorProfile],
        renewal: bool = False,
    ) -> Dict[str, object]:
        if firm.has_ai or not firm.active or not visible or int(context.day) < int(getattr(firm, "ai_cooldown_until", 0)):
            return super().adoption_decision(firm, context, visible, renewal=renewal)

        decision_threshold = _model_adoption_threshold(firm, context, renewal=renewal, config=self.config)
        payload = {
            "decision_type": "vendor_renewal" if renewal else "ai_adoption",
            "day": int(context.day),
            "firm": _firm_payload(
                firm,
                insurance_market_enabled=bool(self.config.get("insurance_market_enabled", True)),
            ),
            "market_context": _context_payload(context),
            "visible_vendors": [_vendor_payload(v) for v in visible],
            "renewal": bool(renewal),
            "decision_threshold": float(decision_threshold),
            "adoption_diffusion_calibration": _adoption_diffusion_payload(self.config),
            "adoption_diffusion_state": _adoption_diffusion_state(firm, context, self.config),
            "term_limits": {
                "vendor_min_days": int(self.config.get("vendor_min_term_days", 14)),
                "vendor_max_days": int(self.config.get("vendor_max_term_days", 120)),
            },
            "negotiation_limits": {
                "min_rounds": int(self.config.get("vendor_min_rounds", 3)),
                "max_rounds": int(self.config.get("max_rounds_cap", 30)),
            },
        }
        payload["decision_rubric"] = {
            "economic_logic": (
                "AI adoption is a productive investment under uncertainty, not a yes/no safety certification. "
                "High technology urgency, innovativeness, AI dependency, local network adoption, and available risk-transfer capacity "
                "support adoption or renewal; high inertia, cash stress, recent losses, and market panic support waiting."
            ),
            "diffusion_stage": (
                "Early in the market, firms face procurement, implementation, "
                "budget, training, and board-approval friction. When local neighbor adoption is low, most firms wait unless they "
                "have strong urgency, innovativeness, dependency, and vendor fit. Neighbor purchases do not instantly prove "
                "success; observed neighbor success and implementation learning become persuasive only after enough time has "
                "passed for implementation evidence to accumulate. Later, persistent local adoption can reduce uncertainty "
                "and make adoption more attractive."
            ),
            "renewal_logic": (
                "For vendor_renewal, the previous vendor contract has expired and the firm is choosing among visible_vendors again. "
                "Contract expiry is a neutral review window, not evidence for or against renewal. "
                "firm.last_vendor_id identifies the expired incumbent vendor. The firm may select the incumbent or any other visible vendor, "
                "or choose not to renew if procurement readiness does not exceed the threshold."
            ),
            "vendor_choice": (
                "Choose selected_vendor_id from visible_vendors. Safer/reputable vendors are more attractive for risk-averse "
                "or high-dependency firms; cheaper/high-productivity vendors are more attractive for cash-tight or urgent firms."
            ),
            "term_choice": (
                "Choose vendor_term_days within term_limits. Shorter terms fit uncertain or cash-tight adoption; longer terms "
                "fit high dependency, stable vendor fit, and low panic."
            ),
        }
        fallback = super().adoption_decision
        return self._decide_bool(
            payload=payload,
            bool_key="adopt_ai",
            score_key="adoption_score",
            threshold=float(decision_threshold),
            fallback=lambda: fallback(firm, context, visible, renewal=renewal),
        )

    def insurance_decision(self, firm: FirmState, context: MarketContext, risk_signal: Dict[str, float]) -> Dict[str, object]:
        if not firm.has_ai or firm.has_insurance or not firm.active:
            return super().insurance_decision(firm, context, risk_signal)

        risk_need = self.insurance_score(firm, context, risk_signal)
        ai_remaining = int(risk_signal.get("ai_remaining_days", 0) or 0)
        insurance_max = int(self.config.get("insurance_max_term_days", 90))
        if ai_remaining > 0:
            insurance_max = min(insurance_max, ai_remaining)
        threshold = float(self.config.get("insurance_buy_threshold", 0.70))
        if float(risk_signal.get("prior_policy", 0.0)) > 0.0:
            threshold -= float(self.config.get("insurance_renewal_threshold_discount", 0.0))
        threshold = self._clamp(threshold, float(self.config.get("insurance_min_threshold", 0.0)), 1.0)
        payload = {
            "decision_type": "insurance_purchase",
            "day": int(context.day),
            "firm": _firm_payload(
                firm,
                insurance_market_enabled=bool(self.config.get("insurance_market_enabled", True)),
            ),
            "market_context": _context_payload(context),
            "risk_signal": _rounded_dict(risk_signal),
            "rule_risk_need_reference": float(risk_need),
            "decision_threshold": float(threshold),
            "term_limits": {
                "insurance_min_days": int(self.config.get("insurance_min_term_days", 1)),
                "insurance_max_days": int(max(0, insurance_max)),
            },
            "negotiation_limits": {
                "min_rounds": int(self.config.get("insurance_min_rounds", 4)),
                "max_rounds": int(self.config.get("max_rounds_cap", 30)),
            },
            "decision_rubric": {
                "eligibility": "AI-risk insurance can only be bought when the firm has active AI exposure and no active policy.",
                "renewal": (
                    "If risk_signal.prior_policy is 1, the previous policy expired today while AI exposure continues. "
                    "Treat this as a renewal decision with lower search friction and a bias toward continuity unless "
                    "risk is clearly negligible or liquidity is very tight."
                ),
                "term": "Choose an integer policy term within term_limits: shorter for exploratory or cash-tight cover, longer for high dependency or high risk.",
                "term_cap": "The insurance policy term must not exceed risk_signal.ai_remaining_days.",
            },
        }
        return self._decide_bool(
            payload=payload,
            bool_key="buy_insurance",
            score_key="insurance_score",
            threshold=threshold,
            fallback=lambda: super(ModelDecisionPolicy, self).insurance_decision(firm, context, risk_signal),
        )

    def exposure_decision(self, firm: FirmState, context: MarketContext, risk_signal: Dict[str, float]) -> Dict[str, object]:
        if not firm.has_ai or not firm.active:
            return super().exposure_decision(firm, context, risk_signal)

        threshold = float(self.config.get("abandon_score_threshold", 0.52))
        score_ref = (
            0.24 * float(risk_signal.get("material_event_score", 0.0))
            + 0.22 * float(risk_signal.get("claimable_event_score", 0.0))
            + 0.18 * float(firm.loss_memory)
            + 0.16 * float(firm.claimable_memory)
            + 0.14 * float(firm.panic)
            - 0.12 * float(firm.profile.risk_tolerance)
        )
        payload = {
            "decision_type": "ai_exposure_management",
            "day": int(context.day),
            "firm": _firm_payload(
                firm,
                insurance_market_enabled=bool(self.config.get("insurance_market_enabled", True)),
            ),
            "market_context": _context_payload(context),
            "risk_signal": _rounded_dict(risk_signal),
            "rule_abandon_reference": float(score_ref),
            "decision_threshold": float(threshold),
            "allowed_vendor_actions": ["keep_vendor", "abandon_ai"],
            "contract_note": (
                "Mid-term switch_vendor is represented as abandon_ai plus a fresh adoption negotiation. "
                "Exiting cancels current AI exposure and any attached insurance under refund rules; same-day re-entry may be allowed by lifecycle config."
            ),
        }
        return self._decide_bool(
            payload=payload,
            bool_key="abandon_ai",
            score_key="abandon_score",
            threshold=threshold,
            fallback=lambda: super(ModelDecisionPolicy, self).exposure_decision(firm, context, risk_signal),
        )

    def _decide_bool(self, payload: dict, bool_key: str, score_key: str, threshold: float, fallback) -> Dict[str, object]:
        prompt = _decision_prompt(payload, bool_key=bool_key, score_key=score_key)
        trace = {
            "backend": self.mode,
            "model": self.model,
            "prompt": prompt,
            "payload": payload,
            "raw_response": "",
            "parsed": {},
            "fallback_reason": "",
        }
        try:
            required_keys = [bool_key, score_key, "reason"]
            raw_responses: List[str] = []
            parsed = None
            raw = ""
            last_exc: Optional[Exception] = None
            for attempt in range(self.json_retries + 1):
                active_prompt = (
                    prompt
                    if attempt == 0
                    else _json_retry_prompt(prompt, raw_responses[-1] if raw_responses else "", required_keys)
                )
                raw = self._complete_with_attempt(prompt=active_prompt, payload=payload, attempt=attempt)
                raw_responses.append(raw)
                try:
                    parsed = _extract_json(raw)
                    break
                except Exception as exc:
                    last_exc = exc
            if parsed is None:
                assert last_exc is not None
                raise last_exc
            action = _as_bool(parsed.get(bool_key, parsed.get("action", False)))
            score = float(parsed.get(score_key, parsed.get("score", 1.0 if action else 0.0)) or 0.0)
            reason = str(parsed.get("reason", "model_decision"))
            trace["raw_response"] = raw
            trace["raw_responses"] = raw_responses
            trace["parsed"] = parsed
            decision_type = str(payload.get("decision_type", ""))
            if action and score < float(threshold):
                action = False
                reason = f"{reason}|below_decision_threshold"
            if action and decision_type in {"ai_adoption", "vendor_renewal"}:
                margin_key = (
                    "model_renewal_required_margin"
                    if decision_type == "vendor_renewal"
                    else "model_adoption_required_margin"
                )
                required_margin = max(
                    0.0,
                    float(self.config.get(margin_key, self.config.get("model_adoption_required_margin", 0.0))),
                )
                if required_margin > 0.0 and score < float(threshold) + required_margin:
                    action = False
                    reason = f"{reason}|below_required_decision_margin"
            decision = {
                "action": bool(action),
                "score": float(score),
                "threshold": float(threshold),
                "draw": 0.0,
                "probability": float(score),
                "reason": f"model:{reason}",
                "model_trace": trace,
            }
            decision.update(_structured_decision_fields(parsed, payload))
            if decision_type in {"ai_adoption", "vendor_renewal"} and not bool(decision.get("action", False)):
                decision["selected_vendor_id"] = ""
                decision["decision_selected_vendor_id"] = ""
                decision["vendor_term_days"] = 0
                decision["term_days"] = 0
                decision["max_rounds"] = 0
            if payload.get("decision_type") in {"ai_adoption", "vendor_renewal"}:
                self._apply_adoption_guard(decision=decision, payload=payload, threshold=float(threshold))
            if payload.get("decision_type") == "ai_exposure_management":
                if float(score) >= float(threshold):
                    if not action or decision.get("vendor_action") != "abandon_ai":
                        decision["reason"] = f"{decision['reason']}|score_threshold_forced_abandon"
                    decision["action"] = True
                    decision["vendor_action"] = "abandon_ai"
                else:
                    decision["action"] = False
                    decision["vendor_action"] = "keep_vendor"
            return decision
        except Exception as exc:
            trace["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            recoverable_model_response_error = isinstance(exc, ValueError)
            if not self.fallback_to_rule and not recoverable_model_response_error:
                raise
            decision = dict(fallback())
            fallback_prefix = "model_parse_fallback" if recoverable_model_response_error else "model_fallback"
            decision["reason"] = f"{fallback_prefix}:{decision.get('reason', '')}"
            decision["model_trace"] = trace
            return decision

    def _apply_adoption_guard(self, decision: Dict[str, object], payload: dict, threshold: float) -> None:
        if not bool(self.config.get("model_adoption_guard_enabled", False)):
            return
        if not bool(decision.get("action", False)):
            return

        guard = _mock_model_response(payload)
        guard_score = float(guard.get("adoption_score", guard.get("score", 0.0)) or 0.0)
        slack = float(self.config.get("model_adoption_guard_slack", 0.0))
        if payload.get("decision_type") == "vendor_renewal":
            slack = float(self.config.get("model_renewal_guard_slack", slack))
        guard_threshold = float(threshold) - max(0.0, slack)

        trace = decision.get("model_trace")
        if isinstance(trace, dict):
            trace["adoption_guard"] = {
                "enabled": True,
                "score": float(guard_score),
                "threshold": float(guard_threshold),
                "raw_threshold": float(threshold),
                "slack": float(slack),
                "passed": bool(guard_score >= guard_threshold),
            }
        if guard_score < guard_threshold:
            decision["action"] = False
            decision["selected_vendor_id"] = ""
            decision["decision_selected_vendor_id"] = ""
            decision["vendor_term_days"] = 0
            decision["term_days"] = 0
            decision["max_rounds"] = 0
            decision["reason"] = f"{decision.get('reason', 'model_decision')}|adoption_guard_blocked"

    def _complete_with_attempt(self, prompt: str, payload: dict, attempt: int = 0) -> str:
        try:
            return self._complete(prompt=prompt, payload=payload, attempt=attempt)
        except TypeError as exc:
            if "unexpected keyword argument 'attempt'" not in str(exc):
                raise
            return self._complete(prompt=prompt, payload=payload)

    def _complete(self, prompt: str, payload: dict, attempt: int = 0) -> str:
        if self.mode == "model_mock":
            return json.dumps(_mock_model_response(payload), ensure_ascii=False, sort_keys=True)
        if self.mode not in {"vllm_openai", "openai_compatible"}:
            raise ValueError(f"Unsupported model decision mode: {self.mode}")

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only compact JSON. No markdown, reasoning, or <think> tags."},
                {"role": "user", "content": f"{prompt}\n/no_think"},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(body).encode("utf-8")
        base_url = self._base_url_for_payload(payload=payload, attempt=attempt)
        req = request.Request(
            f"{base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(f"vLLM endpoint request failed at {base_url}: {exc}") from exc
        return str(result["choices"][0]["message"]["content"])

    def _base_url_for_payload(self, payload: dict, attempt: int = 0) -> str:
        if len(self.base_urls) <= 1:
            return self.base_urls[0]
        seed = self.config.get("common_random_seed", None)
        if seed is None or not bool(self.config.get("common_random_numbers", True)):
            return self._next_base_url()
        firm = payload.get("firm") or {}
        key = (
            f"{int(seed)}|endpoint|{payload.get('decision_type', '')}|"
            f"{firm.get('firm_id', '')}|{int(payload.get('day', 0) or 0)}|{int(attempt)}"
        ).encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return self.base_urls[int.from_bytes(digest[:8], "big") % len(self.base_urls)]

    def _next_base_url(self) -> str:
        with self._endpoint_lock:
            value = self.base_urls[self._endpoint_index % len(self.base_urls)]
            self._endpoint_index += 1
            return value


def build_decision_policy(config: dict, rng: random.Random) -> HeuristicDecisionPolicy:
    layer = dict((config.get("decision_layer") or {}))
    mode = str(layer.get("mode", "rule_heuristic"))
    decision_config = dict(config.get("decision_policy", {}) or {})
    decision_config["insurance_market_enabled"] = bool(config.get("simulation", {}).get("enable_insurance_market", True))
    if bool(decision_config.get("common_random_numbers", True)):
        decision_config.setdefault("common_random_seed", int(config.get("simulation", {}).get("seed", 42)))
    if mode in {"model_mock", "vllm_openai", "openai_compatible"}:
        return ModelDecisionPolicy(decision_config, rng, layer)
    return HeuristicDecisionPolicy(decision_config, rng)


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _context_peer_adoption_rate(context: MarketContext) -> float:
    value = context.local_adoption_rate
    if value is None:
        value = context.adoption_rate
    return _clamp(float(value), 0.0, 1.0)


def _context_peer_insurance_coverage_rate(context: MarketContext) -> float:
    value = context.local_insurance_coverage_rate
    if value is None:
        value = context.insurance_coverage_rate
    return _clamp(float(value), 0.0, 1.0)


def _risk_transfer_evidence(config: dict, context: MarketContext, prefix: str = "model_adoption") -> float:
    cfg = dict(config or {})
    if not bool(cfg.get("insurance_market_enabled", True)):
        return 0.0
    floor = float(
        cfg.get(
            f"{prefix}_insurance_availability_floor",
            cfg.get("adoption_insurance_availability_floor", 0.15),
        )
    )
    coverage_weight = float(
        cfg.get(
            f"{prefix}_peer_coverage_weight",
            cfg.get("adoption_peer_coverage_weight", 0.65),
        )
    )
    paid_claim_weight = float(
        cfg.get(
            f"{prefix}_paid_claim_weight",
            cfg.get("adoption_paid_claim_weight", 0.20),
        )
    )
    paid_claim_reference = max(
        1e-9,
        float(
            cfg.get(
                f"{prefix}_paid_claim_reference",
                cfg.get("adoption_paid_claim_reference", 0.015),
            )
        ),
    )
    paid_claim_signal = _clamp(_context_recent_claim_rate(context) / paid_claim_reference, 0.0, 1.0)
    return _clamp(
        floor
        + coverage_weight * _context_peer_insurance_coverage_rate(context)
        + paid_claim_weight * paid_claim_signal,
        0.0,
        1.0,
    )


def _risk_transfer_gate(config: dict, prefix: str = "model_adoption") -> float:
    cfg = dict(config or {})
    return _clamp(
        float(
            cfg.get(
                f"{prefix}_risk_transfer_evidence_gate",
                cfg.get(
                    f"{prefix}_peer_evidence_gate",
                    cfg.get("adoption_peer_evidence_gate", 0.0),
                ),
            )
        ),
        0.0,
        1.0,
    )


def _risk_transfer_adjusted_peer_rate(
    config: dict,
    context: MarketContext,
    peer_rate: float,
    prefix: str = "model_adoption",
) -> float:
    evidence = _risk_transfer_evidence(config, context, prefix=prefix)
    gate = _risk_transfer_gate(config, prefix=prefix)
    return _clamp(float(peer_rate) * (1.0 - gate * (1.0 - evidence)), 0.0, 1.0)


def _adoption_cfg_float(cfg: dict, local_key: str, legacy_key: str, default: float) -> float:
    if local_key in cfg:
        return float(cfg.get(local_key, default))
    return float(cfg.get(legacy_key, default))


def _context_panic_rate(context: MarketContext) -> float:
    value = context.local_avg_panic
    if value is None:
        value = context.avg_panic
    return _clamp(float(value), 0.0, 1.0)


def _context_recent_claim_rate(context: MarketContext) -> float:
    value = context.local_recent_claim_rate
    if value is None:
        value = context.recent_claim_rate
    return _clamp(float(value), 0.0, 1.0)


def _decision_prompt(payload: dict, bool_key: str, score_key: str) -> str:
    decision_type = str(payload.get("decision_type", ""))
    threshold_note = ""
    if "decision_threshold" in payload:
        threshold_note = (
            " Set the boolean action to true only when your score is at least decision_threshold; "
            "otherwise return false even if the action is directionally attractive."
        )
    pre_operation_note = ""
    risk_signal = payload.get("risk_signal") or {}
    if bool(risk_signal.get("pre_operation_signal", False)):
        pre_operation_note = (
            " This is a pre-operation decision: same-day incidents, losses, and claimable events are not observed yet. "
            "Use lagged firm memory, panic, contract state, and prior industry/network stress; do not invent current-day "
            "accident realizations."
        )
    if decision_type in {"ai_adoption", "vendor_renewal"}:
        extra = (
            " Also include selected_vendor_id from visible_vendors when adopting/renewing, "
            "vendor_term_days as a positive integer within term_limits, and max_rounds as an integer within negotiation_limits. "
            "For vendor_renewal, visible_vendors lists the available vendor choices; firm.last_vendor_id identifies the expired incumbent vendor. "
            "Treat the expired contract as a neutral review point, not a positive renewal signal. "
            "If not adopting/renewing, use selected_vendor_id=\"\", vendor_term_days=0, max_rounds=0. "
            "For ai_adoption, treat the score as procurement readiness, not generic enthusiasm. Local network "
            "evidence is a gradual accelerator rather than a hard veto: below the local evidence floor, only "
            "unusually ready pioneer firms should score above threshold, but high internal readiness, elapsed "
            "evidence maturity, and credible risk-transfer conditions can still justify adoption. Neighbor adoption "
            "should raise readiness gradually, because early same-period purchases are noisy signals rather than "
            "proof that internal procurement is ready."
        )
    elif decision_type == "ai_exposure_management":
        extra = (
            " Also include vendor_action, exactly one of keep_vendor or abandon_ai. "
            "For ai_exposure_management, abandon_score is the pressure to exit current AI exposure, not a score for "
            "continuing to use AI. If abandon_score is at least decision_threshold, return abandon_ai=true and "
            "vendor_action=\"abandon_ai\". If abandon_score is below decision_threshold, return abandon_ai=false "
            "and vendor_action=\"keep_vendor\". If the firm wants to switch providers, choose abandon_ai; the "
            "simulator can then run a fresh adoption negotiation."
        )
    elif decision_type == "insurance_purchase":
        extra = (
            " Also include insurance_term_days and term_days as positive integers within term_limits when buying, "
            "and never longer than risk_signal.ai_remaining_days, "
            "plus max_rounds as an integer within negotiation_limits. If not buying, use term_days=0, "
            "insurance_term_days=0, max_rounds=0. If risk_signal.prior_policy is 1, treat the decision "
            "as an insurance decision after a prior policy expired; decide using current risk, cash constraints, and term limits."
        )
    else:
        extra = ""
    return (
        "You are an economic decision agent in an AI adoption and insurance simulation. "
        "Use the provided JSON state and return exactly one compact JSON object with keys "
        f"{bool_key} (boolean), {score_key} (number between 0 and 1), and reason (short string). "
        + threshold_note
        + pre_operation_note
        + extra
        + " "
        "State JSON: "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _mock_model_response(payload: dict) -> dict:
    firm = payload.get("firm", {}) or {}
    context = payload.get("market_context", {}) or {}
    peer_rate = float(context.get("local_adoption_rate", context.get("adoption_rate", 0.0)) or 0.0)
    peer_insurance_coverage = float(
        context.get("local_insurance_coverage_rate", context.get("insurance_coverage_rate", 0.0)) or 0.0
    )
    panic_rate = float(context.get("local_avg_panic", context.get("avg_panic", 0.0)) or 0.0)
    recent_claim_rate = float(context.get("local_recent_claim_rate", context.get("recent_claim_rate", 0.0)) or 0.0)
    if payload.get("decision_type") in {"ai_adoption", "vendor_renewal"}:
        vendors = payload.get("visible_vendors", []) or []
        selected = ""
        if vendors:
            def vendor_score(v: dict) -> float:
                targets = {str(x).lower() for x in (v.get("target_sectors") or [])}
                industry = str(firm.get("industry", "")).lower()
                if industry in targets:
                    sector_fit = 1.0
                elif "all" in targets:
                    sector_fit = 0.78
                else:
                    sector_fit = 0.36
                productivity_norm = float(v.get("productivity_lift", 0.0)) / 0.013
                price_burden = float(v.get("subscription_fee", 0.0)) / 2800.0
                risk_aversion = 1.0 - float(firm.get("risk_tolerance", 0.5))
                return (
                    0.30 * productivity_norm
                    + 0.24 * float(v.get("reputation", 0.0))
                    + 0.22 * sector_fit
                    - 0.18 * float(v.get("risk_multiplier", 1.0)) * risk_aversion
                    - 0.15 * price_burden
                )

            selected = str(max(vendors, key=vendor_score).get("vendor_id", ""))
        diffusion = payload.get("adoption_diffusion_calibration", {}) or {}
        peer_floor = float(diffusion.get("local_evidence_floor", diffusion.get("peer_floor", 0.08)))
        peer_reference = max(
            peer_floor + 1e-6,
            float(diffusion.get("local_evidence_reference", diffusion.get("peer_reference", 0.45))),
        )
        evidence_panic_dampening = _clamp(float(diffusion.get("local_evidence_panic_dampening", 0.0)), 0.0, 1.0)
        effective_peer_rate = peer_rate * (1.0 - evidence_panic_dampening * panic_rate)
        insurance_available = float(firm.get("insurance_market_available", 1.0))
        paid_claim_reference = max(1e-9, float(diffusion.get("risk_transfer_paid_claim_reference", 0.015)))
        paid_claim_signal = _clamp(recent_claim_rate / paid_claim_reference, 0.0, 1.0)
        risk_transfer_evidence = _clamp(
            insurance_available
            * (
                float(diffusion.get("risk_transfer_insurance_availability_floor", 0.15))
                + float(diffusion.get("risk_transfer_peer_coverage_weight", 0.65)) * peer_insurance_coverage
                + float(diffusion.get("risk_transfer_paid_claim_weight", 0.20)) * paid_claim_signal
            ),
            0.0,
            1.0,
        )
        risk_transfer_gate = _clamp(float(diffusion.get("risk_transfer_evidence_gate", 0.0)), 0.0, 1.0)
        effective_peer_rate *= 1.0 - risk_transfer_gate * (1.0 - risk_transfer_evidence)
        peer_evidence_progress = _clamp(
            (effective_peer_rate - peer_floor) / (peer_reference - peer_floor),
            0.0,
            1.0,
        )
        evidence_lag_days = max(
            1.0,
            float(diffusion.get("local_evidence_lag_days", diffusion.get("peer_evidence_lag_days", 45.0))),
        )
        peer_evidence_maturity = _clamp(float(context.get("day", 0.0)) / evidence_lag_days, 0.0, 1.0)
        readiness_strength = (
            0.40 * float(firm.get("tech_urgency", 0.5))
            + 0.35 * float(firm.get("innovativeness", 0.5))
            + 0.25 * float(firm.get("ai_dependency", 0.5))
        )
        pioneer_readiness = _clamp((readiness_strength - 0.70) / 0.30, 0.0, 1.0)
        score = (
            0.25
            + 0.34 * float(firm.get("tech_urgency", 0.5))
            + 0.22 * float(firm.get("innovativeness", 0.5))
            + 0.05 * float(firm.get("ai_dependency", 0.5))
            + 0.12 * effective_peer_rate * peer_evidence_maturity
            + 0.12 * pioneer_readiness
            + 0.05 * peer_evidence_progress * peer_evidence_maturity
            + 0.05
            * insurance_available
            * (0.35 + 0.65 * peer_insurance_coverage)
            * (1.0 - 0.35 * float(firm.get("risk_tolerance", 0.5)))
            - 0.22 * float(firm.get("inertia", 0.5))
            - 0.10 * (1.0 - peer_evidence_maturity) * (1.0 - pioneer_readiness)
            - 0.18 * panic_rate * (1.0 - float(firm.get("risk_tolerance", 0.5)))
        )
        score = _clamp(score, 0.0, 1.0)
        lo = int((payload.get("term_limits") or {}).get("vendor_min_days", 14))
        hi = int((payload.get("term_limits") or {}).get("vendor_max_days", 120))
        term_days = int(round(22 + 64 * score + 18 * float(firm.get("inertia", 0.5)) - 12 * panic_rate))
        term_days = _bounded_int(term_days, default=60, lo=lo, hi=hi)
        if float(firm.get("inertia", 0.5)) > 0.62 and panic_rate < 0.45:
            term_days = _bounded_int(term_days + 18, default=90, lo=lo, hi=hi)
        elif panic_rate < 0.30 and float(firm.get("tech_urgency", 0.5)) > 0.70:
            term_days = _bounded_int(term_days + 8, default=60, lo=lo, hi=hi)
        threshold = float(payload.get("decision_threshold", 0.50))
        return {
            "adopt_ai": bool(score >= threshold),
            "adoption_score": float(score),
            "selected_vendor_id": selected if score >= threshold else "",
            "vendor_term_days": int(term_days) if score >= threshold else 0,
            "max_rounds": 10 if score >= threshold else 0,
            "reason": "mock_adoption_decision",
        }

    if payload.get("decision_type") == "ai_exposure_management":
        risk = payload.get("risk_signal", {}) or {}
        score = (
            0.12
            + 0.26 * float(risk.get("material_event_score", 0.0))
            + 0.24 * float(risk.get("claimable_event_score", 0.0))
            + 0.18 * float(firm.get("loss_memory", 0.0))
            + 0.16 * float(firm.get("claimable_memory", 0.0))
            + 0.14 * panic_rate
            - 0.18 * float(firm.get("risk_tolerance", 0.5))
        )
        score = _clamp(score, 0.0, 1.0)
        threshold = float(payload.get("decision_threshold", 0.62))
        abandon = bool(score >= threshold)
        return {
            "abandon_ai": abandon,
            "abandon_score": float(score),
            "vendor_action": "abandon_ai" if abandon else "keep_vendor",
            "reason": "mock_exposure_management",
        }

    risk = payload.get("risk_signal", {}) or {}
    prior_policy = float(risk.get("prior_policy", 0.0))
    score = (
        0.10
        + 0.30 * float(risk.get("claimable_event_score", 0.0))
        + 0.20 * float(risk.get("material_event_score", 0.0))
        + 0.15 * float(firm.get("ai_dependency", 0.5))
        + 0.12 * (1.0 - float(firm.get("risk_tolerance", 0.5)))
        + 0.12 * panic_rate
        + 0.10 * recent_claim_rate
        + 0.14 * prior_policy
    )
    score = _clamp(score, 0.0, 1.0)
    limits = payload.get("term_limits") or {}
    lo = int(limits.get("insurance_min_days", 1))
    hi = int(limits.get("insurance_max_days", 90))
    if hi <= 0:
        return {
            "buy_insurance": False,
            "insurance_score": float(score),
            "term_days": 0,
            "insurance_term_days": 0,
            "max_rounds": 0,
            "reason": "mock_no_remaining_ai_exposure_to_insure",
        }
    term_days = _bounded_int(
        12 + int(round(60 * score)) + (12 if prior_policy > 0.0 else 0),
        default=30,
        lo=max(1, min(lo, max(1, hi))),
        hi=max(1, hi),
    )
    if panic_rate > 0.65 or score > 0.78:
        term_days = _bounded_int(term_days + 18, default=90, lo=max(1, min(lo, max(1, hi))), hi=max(1, hi))
    elif prior_policy > 0.0 or float(firm.get("risk_tolerance", 0.5)) < 0.45 or score > 0.66:
        term_days = _bounded_int(term_days + 8, default=60, lo=max(1, min(lo, max(1, hi))), hi=max(1, hi))
    threshold = 0.34 if prior_policy > 0.0 else 0.38
    return {
        "buy_insurance": bool(score >= threshold),
        "insurance_score": float(score),
        "term_days": int(term_days) if score >= threshold else 0,
        "insurance_term_days": int(term_days) if score >= threshold else 0,
        "max_rounds": 10 if score >= threshold else 0,
        "reason": "mock_insurance_decision",
    }


def _model_adoption_threshold(
    firm: FirmState,
    context: MarketContext,
    renewal: bool = False,
    config: Optional[dict] = None,
) -> float:
    cfg = dict(config or {})
    if renewal:
        base = float(cfg.get("model_renewal_base_threshold", 0.46))
        panic_rate = _context_panic_rate(context)
        insurance_market_enabled = bool(cfg.get("insurance_market_enabled", True))
        insurance_confidence = 0.35 + 0.65 * _context_peer_insurance_coverage_rate(context)
        own_cover = 1.0 if bool(firm.has_insurance) else 0.0
        risk_memory_penalty = (
            float(cfg.get("model_renewal_loss_memory_threshold_penalty", 0.08)) * float(firm.loss_memory)
            + float(cfg.get("model_renewal_claimable_memory_threshold_penalty", 0.06)) * float(firm.claimable_memory)
        )
        panic_penalty = (
            float(cfg.get("model_renewal_panic_threshold_penalty", 0.06))
            * panic_rate
            * (1.0 - float(firm.profile.risk_tolerance))
        )
        insurance_adjustment = (
            -float(
                cfg.get(
                    "model_renewal_insurance_confidence_discount",
                    cfg.get("model_adoption_insurance_confidence_discount", 0.020),
                )
            )
            * (0.55 * own_cover + 0.45 * insurance_confidence)
            if insurance_market_enabled
            else 0.0
        )
        continuity_bonus = 0.04 * float(firm.profile.inertia)
        threshold = (
            base
            + risk_memory_penalty
            + panic_penalty
            + insurance_adjustment
            - continuity_bonus
            + float(cfg.get("model_renewal_threshold_offset", cfg.get("model_adoption_threshold_offset", 0.0)))
        )
        return _clamp(
            threshold,
            float(cfg.get("model_renewal_min_threshold", 0.40)),
            float(cfg.get("model_renewal_max_threshold", 0.68)),
        )

    day = max(0, int(context.day))
    maturity_days = max(1.0, float(cfg.get("model_adoption_maturity_days", 90.0)))
    maturity = _clamp(day / maturity_days, 0.0, 1.0)
    raw_peer_rate = _context_peer_adoption_rate(context)
    panic_rate = _context_panic_rate(context)
    local_evidence_panic_dampening = _clamp(
        float(cfg.get("model_adoption_local_evidence_panic_dampening", 0.0)),
        0.0,
        1.0,
    )
    peer_rate = raw_peer_rate * (1.0 - local_evidence_panic_dampening * panic_rate)
    peer_rate = _risk_transfer_adjusted_peer_rate(cfg, context, peer_rate, prefix="model_adoption")
    early_friction = float(cfg.get("model_adoption_early_friction", 0.20)) * (1.0 - maturity)
    peer_reference = max(
        1e-6,
        _adoption_cfg_float(cfg, "model_adoption_local_evidence_reference", "model_adoption_peer_reference", 0.45),
    )
    peer_floor = _clamp(
        _adoption_cfg_float(cfg, "model_adoption_local_evidence_floor", "model_adoption_peer_floor", 0.08),
        0.0,
        min(0.99, peer_reference),
    )
    peer_evidence_lag_days = max(
        1.0,
        _adoption_cfg_float(
            cfg,
            "model_adoption_local_evidence_lag_days",
            "model_adoption_peer_evidence_lag_days",
            45.0,
        ),
    )
    peer_evidence_maturity = _clamp(day / peer_evidence_lag_days, 0.0, 1.0)
    low_peer_friction = _adoption_cfg_float(
        cfg,
        "model_adoption_low_local_evidence_friction",
        "model_adoption_low_peer_friction",
        0.12,
    ) * _clamp(
        (peer_reference - peer_rate) / peer_reference,
        0.0,
        1.0,
    ) * (1.0 - 0.65 * maturity)
    peer_evidence_progress = _clamp((peer_rate - peer_floor) / max(1e-6, peer_reference - peer_floor), 0.0, 1.0)
    # Local neighbor adoption is delayed implementation evidence, not a target
    # adoption rate.
    local_evidence_relief_strength = float(cfg.get("model_adoption_local_evidence_relief", 0.0))
    local_evidence_curve_power = max(
        0.10,
        float(cfg.get("model_adoption_local_evidence_curve_power", 1.0)),
    )
    local_evidence_relief = local_evidence_relief_strength * (
        peer_evidence_progress ** local_evidence_curve_power
    ) * peer_evidence_maturity
    insurance_market_enabled = bool(cfg.get("insurance_market_enabled", True))
    insurance_confidence = 0.35 + 0.65 * _context_peer_insurance_coverage_rate(context)
    insurance_adjustment = (
        -float(cfg.get("model_adoption_insurance_confidence_discount", 0.020)) * insurance_confidence
        if insurance_market_enabled
        else 0.0
    )
    firm_readiness = (
        float(cfg.get("model_adoption_tech_discount", 0.04)) * float(firm.profile.tech_urgency)
        + float(cfg.get("model_adoption_innov_discount", 0.03)) * float(firm.profile.innovativeness)
        + float(cfg.get("model_adoption_dependency_discount", 0.03)) * float(firm.profile.ai_dependency)
    )
    readiness_strength = (
        0.40 * float(firm.profile.tech_urgency)
        + 0.35 * float(firm.profile.innovativeness)
        + 0.25 * float(firm.profile.ai_dependency)
    )
    pioneer_discount = (
        float(cfg.get("model_adoption_pioneer_readiness_discount", 0.0))
        * _clamp((readiness_strength - 0.70) / 0.30, 0.0, 1.0)
        * (1.0 - maturity)
    )
    pioneer_readiness = _clamp((readiness_strength - 0.70) / 0.30, 0.0, 1.0)
    implementation_uncertainty = (
        float(cfg.get("model_adoption_implementation_uncertainty", 0.0))
        * (1.0 - peer_evidence_maturity)
        * (1.0 - pioneer_readiness)
    )
    inertia_penalty = float(cfg.get("model_adoption_inertia_penalty", 0.0)) * float(firm.profile.inertia) * (
        0.35 + 0.65 * (1.0 - maturity)
    )
    panic_penalty = (
        float(cfg.get("model_adoption_panic_threshold_penalty", 0.04))
        * panic_rate
        * (1.0 - float(firm.profile.risk_tolerance))
    )
    bad_experience_penalty = (
        float(cfg.get("model_adoption_loss_memory_threshold_penalty", 0.06))
        * float(firm.loss_memory)
        * (1.0 - 0.25 * float(firm.profile.risk_tolerance))
        + float(cfg.get("model_adoption_claimable_memory_threshold_penalty", 0.05))
        * float(firm.claimable_memory)
        * (1.0 - 0.25 * float(firm.profile.risk_tolerance))
    )
    threshold = (
        float(cfg.get("model_adoption_base_threshold", 0.70))
        + early_friction
        + low_peer_friction
        + implementation_uncertainty
        + inertia_penalty
        + panic_penalty
        + bad_experience_penalty
        + insurance_adjustment
        - firm_readiness
        - pioneer_discount
        - local_evidence_relief
        + float(cfg.get("model_adoption_threshold_offset", 0.0))
    )
    return _clamp(
        threshold,
        float(cfg.get("model_adoption_min_threshold", 0.58)),
        float(cfg.get("model_adoption_max_threshold", 0.94)),
    )


def _adoption_diffusion_payload(config: dict) -> dict:
    cfg = dict(config or {})
    local_floor = _adoption_cfg_float(cfg, "model_adoption_local_evidence_floor", "model_adoption_peer_floor", 0.08)
    local_reference = _adoption_cfg_float(
        cfg,
        "model_adoption_local_evidence_reference",
        "model_adoption_peer_reference",
        0.45,
    )
    local_lag_days = _adoption_cfg_float(
        cfg,
        "model_adoption_local_evidence_lag_days",
        "model_adoption_peer_evidence_lag_days",
        45.0,
    )
    return {
        "local_evidence_floor": float(local_floor),
        "local_evidence_reference": float(local_reference),
        "maturity_days": float(cfg.get("model_adoption_maturity_days", 90.0)),
        "local_evidence_lag_days": float(local_lag_days),
        "risk_transfer_evidence_gate": float(_risk_transfer_gate(cfg, prefix="model_adoption")),
        "risk_transfer_insurance_availability_floor": float(
            cfg.get(
                "model_adoption_insurance_availability_floor",
                cfg.get("adoption_insurance_availability_floor", 0.15),
            )
        ),
        "risk_transfer_peer_coverage_weight": float(
            cfg.get("model_adoption_peer_coverage_weight", cfg.get("adoption_peer_coverage_weight", 0.65))
        ),
        "risk_transfer_paid_claim_weight": float(
            cfg.get("model_adoption_paid_claim_weight", cfg.get("adoption_paid_claim_weight", 0.20))
        ),
        "risk_transfer_paid_claim_reference": float(
            cfg.get("model_adoption_paid_claim_reference", cfg.get("adoption_paid_claim_reference", 0.015))
        ),
        "logic": (
            "Early adoption requires pioneer-level internal readiness. Local network adoption provides delayed "
            "implementation evidence that can gradually reduce procurement uncertainty as neighbor adoption moves "
            "from local_evidence_floor toward local_evidence_reference, but same-period neighbor purchases are still "
            "noisy signals. Local panic and weak risk-transfer evidence dampen the persuasive value of neighbor "
            "adoption, so unabsorbed failures can offset social proof without imposing a hard purchase ban."
        ),
        "local_evidence_panic_dampening": float(cfg.get("model_adoption_local_evidence_panic_dampening", 0.0)),
    }


def _adoption_diffusion_state(firm: FirmState, context: MarketContext, config: dict) -> dict:
    cfg = dict(config or {})
    readiness_strength = (
        0.40 * float(firm.profile.tech_urgency)
        + 0.35 * float(firm.profile.innovativeness)
        + 0.25 * float(firm.profile.ai_dependency)
    )
    peer_floor = _clamp(
        _adoption_cfg_float(cfg, "model_adoption_local_evidence_floor", "model_adoption_peer_floor", 0.08),
        0.0,
        1.0,
    )
    peer_reference = max(
        peer_floor + 1e-6,
        _adoption_cfg_float(cfg, "model_adoption_local_evidence_reference", "model_adoption_peer_reference", 0.45),
    )
    peer_evidence_lag_days = max(
        1.0,
        _adoption_cfg_float(
            cfg,
            "model_adoption_local_evidence_lag_days",
            "model_adoption_peer_evidence_lag_days",
            45.0,
        ),
    )
    local_evidence_panic_dampening = _clamp(
        float(cfg.get("model_adoption_local_evidence_panic_dampening", 0.0)),
        0.0,
        1.0,
    )
    raw_peer_rate = _context_peer_adoption_rate(context)
    panic_adjusted_peer_rate = raw_peer_rate * (1.0 - local_evidence_panic_dampening * _context_panic_rate(context))
    effective_peer_rate = _risk_transfer_adjusted_peer_rate(
        cfg,
        context,
        panic_adjusted_peer_rate,
        prefix="model_adoption",
    )
    return {
        "readiness_strength": round(float(readiness_strength), 4),
        "pioneer_readiness": round(_clamp((readiness_strength - 0.70) / 0.30, 0.0, 1.0), 4),
        "peer_progress": round(
            _clamp((effective_peer_rate - peer_floor) / max(1e-6, peer_reference - peer_floor), 0.0, 1.0),
            4,
        ),
        "raw_peer_adoption_rate": round(raw_peer_rate, 4),
        "panic_adjusted_peer_adoption_rate": round(panic_adjusted_peer_rate, 4),
        "risk_transfer_evidence": round(_risk_transfer_evidence(cfg, context, prefix="model_adoption"), 4),
        "risk_transfer_adjusted_peer_adoption_rate": round(effective_peer_rate, 4),
        "peer_evidence_maturity": round(_clamp(float(context.day) / peer_evidence_lag_days, 0.0, 1.0), 4),
        "local_evidence_panic_dampening": round(
            local_evidence_panic_dampening,
            4,
        ),
    }


def _structured_decision_fields(parsed: dict, payload: dict) -> dict:
    decision_type = str(payload.get("decision_type", ""))
    out: Dict[str, object] = {}
    if decision_type in {"ai_adoption", "vendor_renewal"}:
        limits = payload.get("term_limits") or {}
        lo = int(limits.get("vendor_min_days", 1))
        hi = int(limits.get("vendor_max_days", 180))
        visible_ids = {
            str(v.get("vendor_id", "")).strip()
            for v in (payload.get("visible_vendors", []) or [])
            if str(v.get("vendor_id", "")).strip()
        }
        selected = str(
            parsed.get("selected_vendor_id")
            or parsed.get("selected_vendor")
            or parsed.get("vendor_id")
            or ""
        ).strip()
        if selected not in visible_ids:
            selected = ""
        out["selected_vendor_id"] = selected
        out["vendor_term_days"] = _bounded_int(
            parsed.get("vendor_term_days", parsed.get("term_days")),
            default=0,
            lo=lo,
            hi=hi,
            allow_zero=True,
        )
        negotiation_limits = payload.get("negotiation_limits") or {}
        min_rounds = int(negotiation_limits.get("min_rounds", 1))
        max_rounds = int(negotiation_limits.get("max_rounds", 30))
        out["max_rounds"] = _bounded_int(parsed.get("max_rounds"), default=0, lo=min_rounds, hi=max_rounds, allow_zero=True)
    elif decision_type == "insurance_purchase":
        limits = payload.get("term_limits") or {}
        lo = int(limits.get("insurance_min_days", 1))
        hi = int(limits.get("insurance_max_days", 180))
        hi = max(0, hi)
        out["insurance_term_days"] = _bounded_int(
            parsed.get("insurance_term_days", parsed.get("term_days")),
            default=0,
            lo=max(1, min(lo, max(1, hi))),
            hi=max(1, hi),
            allow_zero=True,
        )
        if hi <= 0:
            out["insurance_term_days"] = 0
        out["term_days"] = out["insurance_term_days"]
        negotiation_limits = payload.get("negotiation_limits") or {}
        min_rounds = int(negotiation_limits.get("min_rounds", 1))
        max_rounds = int(negotiation_limits.get("max_rounds", 30))
        out["max_rounds"] = _bounded_int(parsed.get("max_rounds"), default=0, lo=min_rounds, hi=max_rounds, allow_zero=True)
        if "insurance_offer_money" in parsed:
            try:
                out["insurance_offer_money"] = float(parsed.get("insurance_offer_money"))
            except (TypeError, ValueError):
                pass
    elif decision_type == "ai_exposure_management":
        vendor_action = str(parsed.get("vendor_action", "")).strip().lower()
        if _as_bool(parsed.get("abandon_ai", False)):
            vendor_action = "abandon_ai"
        if vendor_action not in {"keep_vendor", "abandon_ai"}:
            vendor_action = "keep_vendor"
        out["vendor_action"] = vendor_action
    return out


def _bounded_int(value, default: int, lo: int, hi: int, allow_zero: bool = False) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = int(default)
    if allow_zero and result <= 0:
        return 0
    if int(hi) < int(lo):
        hi = lo
    return max(int(lo), min(int(result), int(hi)))


def _extract_json(text: str) -> dict:
    text = str(text or "").strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        value = json.loads(text[start : end + 1])
        if isinstance(value, dict):
            return value
    raise ValueError("model response did not contain a JSON object")


def _json_retry_prompt(original_prompt: str, raw_response: str, required_keys: List[str]) -> str:
    excerpt = str(raw_response or "").strip().replace("\n", " ")[:1200]
    keys = ", ".join(str(key) for key in required_keys)
    return (
        f"{original_prompt}\n\n"
        "Your previous response was invalid because it was not exactly one JSON object. "
        f"Previous response excerpt: {json.dumps(excerpt, ensure_ascii=False)}. "
        f"Return exactly one compact JSON object now, with these keys: {keys}. "
        "No markdown, no prose, no <think>, no code block."
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return False


def _firm_payload(firm: FirmState, insurance_market_enabled: Optional[bool] = None) -> dict:
    p = firm.profile
    vendor_contract = firm.vendor_contract
    insurance_policy = firm.insurance_policy
    current_day = int(getattr(firm, "_decision_day", 0) or 0)
    ai_remaining_days = (
        max(0, int(vendor_contract.end_day) - current_day)
        if vendor_contract is not None
        else 0
    )
    insurance_remaining_days = (
        max(0, int(insurance_policy.end_day) - current_day)
        if insurance_policy is not None
        else 0
    )
    return {
        "firm_id": p.firm_id,
        "industry": p.industry,
        "cash": round(float(firm.cash), 4),
        "asset_value": round(float(p.asset_value), 4),
        "risk_tolerance": round(float(p.risk_tolerance), 4),
        "tech_urgency": round(float(p.tech_urgency), 4),
        "ai_dependency": round(float(p.ai_dependency), 4),
        "inertia": round(float(p.inertia), 4),
        "innovativeness": round(float(p.innovativeness), 4),
        "panic": round(float(firm.panic), 4),
        "risk_memory": round(float(firm.risk_memory), 4),
        "loss_memory": round(float(firm.loss_memory), 4),
        "claimable_memory": round(float(firm.claimable_memory), 4),
        "has_ai": bool(firm.has_ai),
        "has_insurance": bool(firm.has_insurance),
        "last_vendor_id": str(getattr(firm, "_last_vendor_id", "") or ""),
        "vendor_expired_today": bool(getattr(firm, "_vendor_expired_today", False)),
        "ai_remaining_days": int(ai_remaining_days),
        "insurance_remaining_days": int(insurance_remaining_days),
        "insurance_market_available": 1.0
        if bool(
            getattr(
                firm,
                "_insurance_market_enabled",
                True if insurance_market_enabled is None else bool(insurance_market_enabled),
            )
        )
        else 0.0,
    }


def _context_payload(context: MarketContext) -> dict:
    return {
        "day": int(context.day),
        "adoption_rate": round(float(context.adoption_rate), 4),
        "insurance_coverage_rate": round(float(context.insurance_coverage_rate), 4),
        "avg_panic": round(float(context.avg_panic), 4),
        "recent_claim_rate": round(float(context.recent_claim_rate), 4),
        "local_adoption_rate": round(_context_peer_adoption_rate(context), 4),
        "local_insurance_coverage_rate": round(_context_peer_insurance_coverage_rate(context), 4),
        "local_avg_panic": round(_context_panic_rate(context), 4),
        "local_recent_claim_rate": round(_context_recent_claim_rate(context), 4),
        "network_neighbor_count": int(context.network_neighbor_count),
        "same_industry_neighbor_share": round(float(context.same_industry_neighbor_share), 4),
    }


def _vendor_payload(vendor: VendorProfile) -> dict:
    return {
        "vendor_id": vendor.vendor_id,
        "label": vendor.label,
        "subscription_fee": float(vendor.subscription_fee),
        "productivity_lift": float(vendor.productivity_lift),
        "risk_multiplier": float(vendor.risk_multiplier),
        "reputation": float(vendor.reputation),
        "target_sectors": list(vendor.target_sectors),
    }


def _rounded_dict(value: dict) -> dict:
    out = {}
    for key, item in value.items():
        if isinstance(item, float):
            out[key] = round(float(item), 4)
        elif isinstance(item, (str, int, bool)) or item is None:
            out[key] = item
    return out


def _parse_base_urls(layer_config: dict) -> List[str]:
    raw = layer_config.get("base_urls") or layer_config.get("base_url") or "http://127.0.0.1:8000/v1"
    if isinstance(raw, (list, tuple)):
        values = [str(x).strip().rstrip("/") for x in raw if str(x).strip()]
    else:
        text = str(raw)
        for sep in ["\n", ";"]:
            text = text.replace(sep, ",")
        values = [x.strip().rstrip("/") for x in text.split(",") if x.strip()]
    return values or ["http://127.0.0.1:8000/v1"]
