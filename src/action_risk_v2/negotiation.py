from __future__ import annotations

import hashlib
import json
import random
import threading
from dataclasses import dataclass, replace
from typing import Dict, List, Optional
from urllib import request
from urllib.error import URLError

from .decisions import (
    MarketContext,
    _context_panic_rate,
    _context_peer_adoption_rate,
    _context_peer_insurance_coverage_rate,
    _context_recent_claim_rate,
    _extract_json,
    _json_retry_prompt,
)
from .schema import FirmState, InsuranceQuote, VendorProfile


@dataclass(frozen=True)
class NegotiationResult:
    agreed: bool
    final_price: float
    rounds: int
    outcome: str
    events: List[dict]
    quote: Optional[InsuranceQuote] = None


class NegotiationEngine:
    """Two-sided vendor and insurance negotiation used after demand is formed."""

    def __init__(self, config: dict, rng: random.Random):
        self.config = dict(config or {})
        self.rng = rng
        self.layer = dict(self.config.get("decision_layer", {}) or {})
        self.negotiation = dict(self.config.get("negotiation", {}) or {})
        self.mode = str(self.layer.get("mode", "rule_heuristic"))
        self.base_urls = _parse_base_urls(self.layer)
        self.base_url = self.base_urls[0]
        self._endpoint_lock = threading.Lock()
        self._endpoint_index = 0
        self.model = str(self.layer.get("model", "qwen3-local"))
        self.api_key = str(self.layer.get("api_key", "EMPTY"))
        self.timeout_seconds = float(self.layer.get("timeout_seconds", 30))
        self.temperature = float(self.layer.get("temperature", 0.0))
        self.max_tokens = int(self.layer.get("negotiation_max_tokens", self.layer.get("max_tokens", 260)))
        self.fallback_to_rule = bool(self.layer.get("fallback_to_rule", True))
        self.json_retries = max(0, int(self.layer.get("json_retries", 2)))
        decision_policy = dict(self.config.get("decision_policy", {}) or {})
        self.common_random_numbers = bool(decision_policy.get("common_random_numbers", True))
        seed = decision_policy.get("common_random_seed", self.config.get("simulation", {}).get("seed", None))
        self.common_random_seed = int(seed) if seed is not None else None

    @property
    def model_enabled(self) -> bool:
        return self.mode in {"model_mock", "vllm_openai", "openai_compatible"}

    def negotiate_vendor(
        self,
        firm: FirmState,
        vendor: VendorProfile,
        context: MarketContext,
        day: int,
        term_days: int,
        max_rounds_override: Optional[int] = None,
    ) -> NegotiationResult:
        if not bool(self.negotiation.get("enabled", True)):
            return NegotiationResult(True, float(vendor.subscription_fee), 0, "NEGOTIATION_DISABLED", [])

        ask = float(vendor.subscription_fee)
        floor = ask * self._vendor_floor_ratio(vendor)
        max_rounds = _bounded_rounds(
            max_rounds_override,
            fallback=int(self.negotiation.get("vendor_max_rounds", 10)),
            cap=int(self.negotiation.get("max_rounds_cap", 30)),
            min_rounds=int(self.negotiation.get("vendor_min_rounds", 1)),
        )
        min_rounds = min(max_rounds, max(1, int(self.negotiation.get("vendor_min_rounds", 1))))
        buyer_ceiling = self._vendor_buyer_ceiling(firm, vendor, context, term_days=term_days)
        buyer_offer = min(ask, buyer_ceiling, max(0.0, ask * self._vendor_opening_ratio(firm, context)))
        events: List[dict] = []

        for round_idx in range(1, max(1, max_rounds) + 1):
            seller = self._seller_vendor_response(
                firm=firm,
                vendor=vendor,
                context=context,
                day=day,
                round_idx=round_idx,
                max_rounds=max_rounds,
                ask=ask,
                floor=floor,
                buyer_offer=buyer_offer,
            )
            seller_decision = str(seller.get("decision", "counter")).lower().strip()
            seller_counter = _as_float(seller.get("offer_monthly_fee", ask), ask)
            seller_counter = max(float(floor), min(float(seller_counter), ask * 1.30))
            if seller_decision == "accept" and round_idx < min_rounds:
                seller_decision = "counter"
                seller_counter = min(float(ask), max(float(seller_counter), float(buyer_offer)))

            seller_event = {
                "event_type": "vendor_negotiation_round",
                "day": int(day),
                "round": int(round_idx),
                "side": "vendor",
                "firm_id": firm.profile.firm_id,
                "industry": firm.profile.industry,
                "vendor_id": vendor.vendor_id,
                "term_days": int(term_days),
                "buyer_offer": float(buyer_offer),
                "vendor_ask": float(ask),
                "vendor_floor": float(floor),
                "seller_decision": seller_decision,
                "seller_counter": float(seller_counter),
                "message": str(seller.get("message", "")),
                "backend": self.mode,
            }
            _attach_trace(seller_event, seller)
            events.append(seller_event)

            if (
                seller_decision == "accept"
                and buyer_offer >= floor
                and buyer_offer <= buyer_ceiling
                and self._vendor_monthly_price_affordable(firm, buyer_offer, term_days)
            ):
                return NegotiationResult(True, float(buyer_offer), round_idx, "AGREED_SELLER_ACCEPT", events)
            if seller_decision == "reject":
                return NegotiationResult(False, 0.0, round_idx, "REJECTED_BY_VENDOR", events)

            ask = float(seller_counter)
            buyer = self._buyer_vendor_response(
                firm=firm,
                vendor=vendor,
                context=context,
                day=day,
                round_idx=round_idx,
                max_rounds=max_rounds,
                ask=ask,
                floor=floor,
                last_offer=buyer_offer,
                last_counter=seller_counter,
                buyer_ceiling=buyer_ceiling,
            )
            buyer_decision = str(buyer.get("decision", "counter")).lower().strip()
            new_offer = _as_float(buyer.get("offer_monthly_fee", buyer_offer), buyer_offer)
            new_offer = max(0.0, min(float(new_offer), max(float(buyer_ceiling), 0.0), ask * 1.15))
            if buyer_decision == "accept" and round_idx < min_rounds:
                buyer_decision = "counter"
                new_offer = min(float(ask), max(float(new_offer), float(buyer_offer)))

            buyer_event = {
                "event_type": "vendor_negotiation_round",
                "day": int(day),
                "round": int(round_idx),
                "side": "buyer",
                "firm_id": firm.profile.firm_id,
                "industry": firm.profile.industry,
                "vendor_id": vendor.vendor_id,
                "term_days": int(term_days),
                "vendor_ask": float(ask),
                "vendor_floor": float(floor),
                "buyer_ceiling": float(buyer_ceiling),
                "last_offer": float(buyer_offer),
                "seller_counter": float(seller_counter),
                "buyer_decision": buyer_decision,
                "buyer_offer": float(new_offer),
                "message": str(buyer.get("message", "")),
                "backend": self.mode,
            }
            _attach_trace(buyer_event, buyer)
            events.append(buyer_event)

            if (
                buyer_decision == "accept"
                and ask <= buyer_ceiling
                and self._vendor_monthly_price_affordable(firm, ask, term_days)
            ):
                return NegotiationResult(True, float(ask), round_idx, "AGREED_BUYER_ACCEPT", events)
            if buyer_decision == "quit":
                return NegotiationResult(False, 0.0, round_idx, "QUIT_BY_BUYER", events)
            if (
                new_offer >= ask
                and ask <= buyer_ceiling
                and self._vendor_monthly_price_affordable(firm, ask, term_days)
            ):
                return NegotiationResult(True, float(ask), round_idx, "AGREED_COUNTER_REACHED_ASK", events)
            buyer_offer = float(new_offer)

        events.append(
            {
                "event_type": "vendor_negotiation_end",
                "day": int(day),
                "firm_id": firm.profile.firm_id,
                "industry": firm.profile.industry,
                "vendor_id": vendor.vendor_id,
                "term_days": int(term_days),
                "outcome": "MAX_ROUNDS_NO_AGREEMENT",
                "vendor_floor": float(floor),
                "buyer_ceiling": float(buyer_ceiling),
                "backend": self.mode,
            }
        )
        return NegotiationResult(False, 0.0, max_rounds, "MAX_ROUNDS_NO_AGREEMENT", events)

    def negotiate_insurance(
        self,
        firm: FirmState,
        quote: InsuranceQuote,
        context: MarketContext,
        risk_need: float,
        max_rounds_override: Optional[int] = None,
    ) -> NegotiationResult:
        if not bool(self.negotiation.get("enabled", True)):
            return NegotiationResult(True, float(quote.premium), 0, "NEGOTIATION_DISABLED", [], quote=quote)

        max_rounds = _bounded_rounds(
            max_rounds_override,
            fallback=int(self.negotiation.get("insurance_max_rounds", 10)),
            cap=int(self.negotiation.get("max_rounds_cap", 30)),
            min_rounds=int(self.negotiation.get("insurance_min_rounds", 1)),
        )
        min_rounds = min(max_rounds, max(1, int(self.negotiation.get("insurance_min_rounds", 1))))
        menu = self._insurance_tier_menu(quote)
        current_tier = "STANDARD"
        current_quote = menu[current_tier]
        floor_ratio = float(self.negotiation.get("insurance_floor_ratio", 0.88))
        ask = float(current_quote.premium)
        floor = max(50.0, ask * floor_ratio)
        buyer_offer = min(ask, max(0.0, ask * self._insurance_opening_ratio(firm, context, risk_need)))
        buyer_ceiling = self._insurance_buyer_ceiling(firm, quote, context, risk_need)
        events: List[dict] = []

        for round_idx in range(1, max(1, max_rounds) + 1):
            insurer = self._insurer_response(
                firm=firm,
                quote=current_quote,
                context=context,
                risk_need=risk_need,
                day=quote.day,
                round_idx=round_idx,
                max_rounds=max_rounds,
                ask=ask,
                floor=floor,
                buyer_offer=buyer_offer,
                current_tier=current_tier,
                menu=menu,
            )
            insurer_decision = str(insurer.get("decision", "counter")).lower().strip()
            requested_tier = str(insurer.get("tier", current_tier)).upper().strip()
            if requested_tier in menu and requested_tier != current_tier:
                current_tier = requested_tier
                current_quote = menu[current_tier]
                ask = max(float(current_quote.premium), buyer_offer)
                floor = max(50.0, float(current_quote.premium) * floor_ratio)
            insurer_counter = _as_float(insurer.get("premium_money", ask), ask)
            insurer_counter = max(float(floor), min(float(insurer_counter), ask * 1.35))
            if insurer_decision == "accept" and round_idx < min_rounds:
                insurer_decision = "counter"
                insurer_counter = min(float(ask), max(float(insurer_counter), float(buyer_offer)))

            insurer_event = {
                "event_type": "insurance_negotiation_round",
                "day": int(quote.day),
                "round": int(round_idx),
                "side": "insurer",
                "firm_id": firm.profile.firm_id,
                "industry": firm.profile.industry,
                "vendor_id": quote.vendor_id,
                "insurer_id": quote.insurer_id,
                "term_days": int(quote.term_days),
                "tier": current_tier,
                "buyer_offer": float(buyer_offer),
                "ask_premium": float(ask),
                "floor_premium": float(floor),
                "insurer_decision": insurer_decision,
                "insurer_counter": float(insurer_counter),
                "risk_need": float(risk_need),
                "message": str(insurer.get("message", "")),
                "backend": self.mode,
            }
            _attach_trace(insurer_event, insurer)
            events.append(insurer_event)

            if insurer_decision == "accept" and buyer_offer >= floor and buyer_offer <= firm.cash:
                final_quote = replace(current_quote, premium=float(buyer_offer))
                return NegotiationResult(True, float(buyer_offer), round_idx, "AGREED_INSURER_ACCEPT", events, quote=final_quote)
            if insurer_decision == "reject":
                return NegotiationResult(False, 0.0, round_idx, "REJECTED_BY_INSURER", events)

            ask = float(insurer_counter)
            buyer = self._buyer_insurance_response(
                firm=firm,
                quote=current_quote,
                context=context,
                risk_need=risk_need,
                day=quote.day,
                round_idx=round_idx,
                max_rounds=max_rounds,
                ask=ask,
                floor=floor,
                last_offer=buyer_offer,
                last_counter=insurer_counter,
                buyer_ceiling=buyer_ceiling,
                current_tier=current_tier,
                menu=menu,
            )
            buyer_decision = str(buyer.get("decision", "counter")).lower().strip()
            requested_tier = str(buyer.get("tier_request", buyer.get("tier", current_tier))).upper().strip()
            if requested_tier in menu and requested_tier != current_tier:
                current_tier = requested_tier
                current_quote = menu[current_tier]
                ask = max(float(current_quote.premium), min(buyer_offer, float(current_quote.premium)))
                floor = max(50.0, float(current_quote.premium) * floor_ratio)
            new_offer = _as_float(buyer.get("premium_money", buyer.get("offer_money", buyer_offer)), buyer_offer)
            new_offer = max(0.0, min(float(new_offer), max(float(buyer_ceiling), 0.0), ask * 1.20))
            if buyer_decision == "accept" and round_idx < min_rounds:
                buyer_decision = "counter"
                new_offer = min(float(ask), max(float(new_offer), float(buyer_offer)))

            buyer_event = {
                "event_type": "insurance_negotiation_round",
                "day": int(quote.day),
                "round": int(round_idx),
                "side": "buyer",
                "firm_id": firm.profile.firm_id,
                "industry": firm.profile.industry,
                "vendor_id": quote.vendor_id,
                "insurer_id": quote.insurer_id,
                "term_days": int(quote.term_days),
                "tier": current_tier,
                "ask_premium": float(ask),
                "floor_premium": float(floor),
                "buyer_ceiling": float(buyer_ceiling),
                "last_offer": float(buyer_offer),
                "insurer_counter": float(insurer_counter),
                "buyer_decision": buyer_decision,
                "buyer_offer": float(new_offer),
                "risk_need": float(risk_need),
                "message": str(buyer.get("message", "")),
                "backend": self.mode,
            }
            _attach_trace(buyer_event, buyer)
            events.append(buyer_event)

            if buyer_decision == "accept" and ask <= firm.cash:
                final_quote = replace(current_quote, premium=float(ask))
                return NegotiationResult(True, float(ask), round_idx, "AGREED_BUYER_ACCEPT", events, quote=final_quote)
            if buyer_decision == "quit":
                return NegotiationResult(False, 0.0, round_idx, "QUIT_BY_BUYER", events)
            if new_offer >= ask and ask <= firm.cash:
                final_quote = replace(current_quote, premium=float(ask))
                return NegotiationResult(True, float(ask), round_idx, "AGREED_COUNTER_REACHED_ASK", events, quote=final_quote)
            buyer_offer = float(new_offer)

        events.append(
            {
                "event_type": "insurance_negotiation_end",
                "day": int(quote.day),
                "firm_id": firm.profile.firm_id,
                "industry": firm.profile.industry,
                "vendor_id": quote.vendor_id,
                "insurer_id": quote.insurer_id,
                "term_days": int(quote.term_days),
                "outcome": "MAX_ROUNDS_NO_AGREEMENT",
                "floor_premium": float(floor),
                "buyer_ceiling": float(buyer_ceiling),
                "risk_need": float(risk_need),
                "backend": self.mode,
            }
        )
        return NegotiationResult(False, 0.0, max_rounds, "MAX_ROUNDS_NO_AGREEMENT", events)

    def _seller_vendor_response(self, **kwargs) -> dict:
        if self.model_enabled:
            payload = self._vendor_payload(side="vendor", **kwargs)
            return self._model_or_rule(payload, lambda: self._rule_seller_vendor(**kwargs))
        return self._rule_seller_vendor(**kwargs)

    def _buyer_vendor_response(self, **kwargs) -> dict:
        if self.model_enabled:
            payload = self._vendor_payload(side="buyer", **kwargs)
            return self._model_or_rule(payload, lambda: self._rule_buyer_vendor(**kwargs))
        return self._rule_buyer_vendor(**kwargs)

    def _insurer_response(self, **kwargs) -> dict:
        if self.model_enabled:
            payload = self._insurance_payload(side="insurer", **kwargs)
            return self._model_or_rule(payload, lambda: self._rule_insurer(**kwargs))
        return self._rule_insurer(**kwargs)

    def _buyer_insurance_response(self, **kwargs) -> dict:
        if self.model_enabled:
            payload = self._insurance_payload(side="buyer", **kwargs)
            return self._model_or_rule(payload, lambda: self._rule_buyer_insurance(**kwargs))
        return self._rule_buyer_insurance(**kwargs)

    def _rule_seller_vendor(
        self,
        firm: FirmState,
        vendor: VendorProfile,
        context: MarketContext,
        day: int,
        round_idx: int,
        max_rounds: int,
        ask: float,
        floor: float,
        buyer_offer: float,
        **_: object,
    ) -> dict:
        if buyer_offer >= ask:
            return {"decision": "accept", "offer_monthly_fee": buyer_offer, "message": "Buyer offer meets current ask."}
        if buyer_offer < floor * 0.72 and round_idx >= max_rounds:
            return {"decision": "reject", "offer_monthly_fee": ask, "message": "Offer remains below sustainable service cost."}
        progress = round_idx / max(1, max_rounds)
        concession = (ask - floor) * (0.28 + 0.48 * progress + 0.10 * (1.0 - vendor.reputation))
        counter = max(floor, ask - concession)
        if buyer_offer >= counter * (1.0 - float(self.negotiation.get("vendor_accept_slack", 0.015))):
            return {"decision": "accept", "offer_monthly_fee": buyer_offer, "message": "Close enough to settle the contract."}
        return {"decision": "counter", "offer_monthly_fee": counter, "message": "Countering within the vendor floor."}

    def _rule_buyer_vendor(
        self,
        firm: FirmState,
        vendor: VendorProfile,
        context: MarketContext,
        day: int,
        round_idx: int,
        max_rounds: int,
        ask: float,
        floor: float,
        last_offer: float,
        last_counter: float,
        buyer_ceiling: float,
        **_: object,
    ) -> dict:
        if ask <= buyer_ceiling:
            return {"decision": "accept", "offer_monthly_fee": ask, "message": "Accepted after vendor concession."}
        if round_idx >= max_rounds or buyer_ceiling < floor:
            return {"decision": "quit", "offer_monthly_fee": last_offer, "message": "Price exceeds budget discipline."}
        urgency = 0.30 + 0.32 * firm.profile.tech_urgency + 0.16 * _context_panic_rate(context)
        offer = last_offer + (min(last_counter, buyer_ceiling) - last_offer) * _clamp(urgency, 0.20, 0.82)
        return {"decision": "counter", "offer_monthly_fee": offer, "message": "Raising offer without revealing ceiling."}

    def _rule_insurer(
        self,
        firm: FirmState,
        quote: InsuranceQuote,
        context: MarketContext,
        risk_need: float,
        day: int,
        round_idx: int,
        max_rounds: int,
        ask: float,
        floor: float,
        buyer_offer: float,
        current_tier: str,
        menu: Dict[str, InsuranceQuote],
        **_: object,
    ) -> dict:
        if buyer_offer >= ask:
            return {"decision": "accept", "premium_money": buyer_offer, "tier": current_tier, "message": "Buyer offer meets quoted premium."}
        if buyer_offer < floor * 0.74 and round_idx >= max_rounds:
            return {"decision": "reject", "premium_money": ask, "tier": current_tier, "message": "Offer is below actuarial floor."}
        progress = round_idx / max(1, max_rounds)
        concession = (ask - floor) * (0.18 + 0.44 * progress)
        counter = max(floor, ask - concession)
        return {"decision": "counter", "premium_money": counter, "tier": current_tier, "message": "Countering above actuarial floor."}

    def _rule_buyer_insurance(
        self,
        firm: FirmState,
        quote: InsuranceQuote,
        context: MarketContext,
        risk_need: float,
        day: int,
        round_idx: int,
        max_rounds: int,
        ask: float,
        floor: float,
        last_offer: float,
        last_counter: float,
        buyer_ceiling: float,
        current_tier: str,
        menu: Dict[str, InsuranceQuote],
        **_: object,
    ) -> dict:
        if ask <= min(buyer_ceiling, firm.cash):
            return {"decision": "accept", "premium_money": ask, "tier_request": current_tier, "message": "Coverage terms are acceptable."}
        if round_idx == 1 and "BASIC" in menu and buyer_ceiling < ask and risk_need < 0.95:
            basic = menu["BASIC"]
            if basic.premium <= firm.cash:
                return {"decision": "counter", "premium_money": min(buyer_ceiling, basic.premium), "tier_request": "BASIC", "message": "Requesting the lower-premium tier."}
        if round_idx >= max_rounds or buyer_ceiling < floor:
            return {"decision": "quit", "premium_money": last_offer, "tier_request": current_tier, "message": "Premium exceeds hedging budget."}
        pressure = 0.24 + 0.30 * _clamp(risk_need, 0.0, 1.0) + 0.20 * _context_panic_rate(context)
        offer = last_offer + (min(last_counter, buyer_ceiling) - last_offer) * _clamp(pressure, 0.20, 0.84)
        return {"decision": "counter", "premium_money": offer, "tier_request": current_tier, "message": "Improving offer for risk transfer."}

    def _model_or_rule(self, payload: dict, rule_fn) -> dict:
        prompt = _negotiation_prompt(payload)
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
            if self.mode == "model_mock":
                parsed = _mock_negotiation_response(payload)
                raw = json.dumps(parsed, ensure_ascii=False, sort_keys=True)
            else:
                raw_responses: List[str] = []
                parsed = None
                raw = ""
                last_exc: Optional[Exception] = None
                decision_field = "premium_money" if payload.get("negotiation_type") == "insurance_policy" else "offer_monthly_fee"
                required_keys = ["decision", decision_field, "message"]
                for attempt in range(self.json_retries + 1):
                    active_prompt = (
                        prompt
                        if attempt == 0
                        else _json_retry_prompt(prompt, raw_responses[-1] if raw_responses else "", required_keys)
                    )
                    raw = self._complete_with_attempt(active_prompt, payload=payload, attempt=attempt)
                    raw_responses.append(raw)
                    try:
                        parsed = _extract_json(raw)
                        _validate_negotiation_response(parsed, payload)
                        break
                    except Exception as exc:
                        last_exc = exc
                if parsed is None:
                    assert last_exc is not None
                    raise last_exc
                trace["raw_responses"] = raw_responses
            trace["raw_response"] = raw
            trace["parsed"] = parsed
            parsed = dict(parsed)
            _validate_negotiation_response(parsed, payload)
            parsed["model_trace"] = trace
            return parsed
        except Exception as exc:
            trace["fallback_reason"] = f"{type(exc).__name__}: {exc}"
            recoverable_model_response_error = isinstance(exc, ValueError)
            if not self.fallback_to_rule and not recoverable_model_response_error:
                raise
            result = dict(rule_fn())
            result["model_trace"] = trace
            if recoverable_model_response_error:
                result["fallback_reason"] = trace["fallback_reason"]
            result["message"] = str(result.get("message", "model fallback"))
            return result

    def _complete_with_attempt(self, prompt: str, payload: dict, attempt: int = 0) -> str:
        try:
            return self._complete(prompt=prompt, payload=payload, attempt=attempt)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return self._complete(prompt)

    def _complete(self, prompt: str, payload: Optional[dict] = None, attempt: int = 0) -> str:
        base_url = self._base_url_for_payload(payload=payload or {}, attempt=attempt)
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
        req = request.Request(
            f"{base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
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
        if self.common_random_seed is None or not self.common_random_numbers:
            return self._next_base_url()
        firm = payload.get("firm") or {}
        quote = payload.get("quote") or {}
        vendor = payload.get("vendor") or {}
        counterparty = (
            quote.get("insurer_id")
            or quote.get("vendor_id")
            or vendor.get("vendor_id")
            or ""
        )
        key = (
            f"{int(self.common_random_seed)}|negotiation_endpoint|"
            f"{payload.get('negotiation_type', '')}|{payload.get('side', '')}|"
            f"{firm.get('firm_id', '')}|{int(payload.get('day', 0) or 0)}|"
            f"{int(payload.get('round', 0) or 0)}|{counterparty}|{int(attempt)}"
        ).encode("utf-8")
        digest = hashlib.sha256(key).digest()
        return self.base_urls[int.from_bytes(digest[:8], "big") % len(self.base_urls)]

    def _next_base_url(self) -> str:
        with self._endpoint_lock:
            value = self.base_urls[self._endpoint_index % len(self.base_urls)]
            self._endpoint_index += 1
            return value

    def _vendor_floor_ratio(self, vendor: VendorProfile) -> float:
        ratio = (
            0.70
            + 0.16 * _clamp(vendor.reputation, 0.0, 1.0)
            + 0.08 * _clamp(vendor.productivity_lift / 0.015, 0.0, 1.0)
            - 0.08 * _clamp((vendor.risk_multiplier - 0.8) / 0.8, 0.0, 1.0)
        )
        return _clamp(ratio, 0.70, 0.96)

    def _vendor_opening_ratio(self, firm: FirmState, context: MarketContext) -> float:
        ratio = (
            0.78
            + 0.08 * firm.profile.tech_urgency
            + 0.05 * firm.profile.innovativeness
            + 0.06 * _context_peer_adoption_rate(context)
            - 0.05 * firm.profile.inertia
        )
        return _clamp(ratio, 0.72, 0.96)

    def _vendor_fee_base_days(self) -> float:
        lifecycle = dict(self.config.get("contract_lifecycle", {}) or {})
        return max(1.0, float(lifecycle.get("vendor_fee_base_days", 30.0)))

    def _vendor_contract_total_fee(self, monthly_fee: float, term_days: int) -> float:
        return float(monthly_fee) * max(1.0, float(term_days)) / self._vendor_fee_base_days()

    def _vendor_monthly_price_affordable(self, firm: FirmState, monthly_fee: float, term_days: int) -> bool:
        return self._vendor_contract_total_fee(monthly_fee, term_days) <= float(firm.cash) + 1e-9

    def _vendor_buyer_ceiling(
        self,
        firm: FirmState,
        vendor: VendorProfile,
        context: MarketContext,
        term_days: int,
    ) -> float:
        # Vendor negotiation is quoted as a monthly fee, but contracts are prepaid
        # for the selected term in the simulator. Convert cash constraints to a
        # monthly-equivalent ceiling so negotiation and binding use the same budget.
        term_to_base = self._vendor_fee_base_days() / max(1.0, float(term_days))
        cash_budget = firm.cash * float(self.negotiation.get("vendor_max_cash_share", 0.055)) * term_to_base
        cash_available = firm.cash * term_to_base
        strategic_value = vendor.subscription_fee * (
            0.88
            + 0.30 * firm.profile.tech_urgency
            + 0.18 * firm.profile.innovativeness
            + 0.10 * _context_peer_adoption_rate(context)
            - 0.22 * firm.profile.inertia
            - 0.10 * _context_panic_rate(context) * (1.0 - firm.profile.risk_tolerance)
        )
        return max(0.0, min(float(cash_budget), float(strategic_value), float(cash_available)))

    def _insurance_opening_ratio(self, firm: FirmState, context: MarketContext, risk_need: float) -> float:
        ratio = (
            0.76
            + 0.10 * _clamp(risk_need, 0.0, 1.0)
            + 0.06 * _context_panic_rate(context)
            + 0.06 * (1.0 - firm.profile.risk_tolerance)
        )
        return _clamp(ratio, 0.72, 0.96)

    def _insurance_buyer_ceiling(self, firm: FirmState, quote: InsuranceQuote, context: MarketContext, risk_need: float) -> float:
        cash_budget = firm.cash * float(self.negotiation.get("insurance_max_cash_share", 0.035))
        hedge_value = quote.premium * (
            0.82
            + 0.36 * _clamp(risk_need, 0.0, 1.0)
            + 0.16 * _context_panic_rate(context)
            + 0.10 * (1.0 - firm.profile.risk_tolerance)
        )
        return max(0.0, min(float(cash_budget), float(hedge_value), firm.cash))

    def _insurance_tier_menu(self, quote: InsuranceQuote) -> Dict[str, InsuranceQuote]:
        def q(mult: float, deductible_delta: float, coverage_delta: float, limit_mult: float, threshold_delta: float) -> InsuranceQuote:
            return replace(
                quote,
                premium=max(50.0, float(quote.premium) * mult),
                deductible_ratio=_clamp(float(quote.deductible_ratio) + deductible_delta, 0.05, 0.90),
                coverage_ratio=_clamp(float(quote.coverage_ratio) + coverage_delta, 0.15, 0.92),
                limit_money=max(1.0, float(quote.limit_money) * limit_mult),
                incident_threshold=_clamp(float(quote.incident_threshold) + threshold_delta, 0.05, 0.95),
            )

        return {
            "BASIC": q(0.82, 0.10, -0.10, 0.75, 0.08),
            "STANDARD": quote,
            "PLUS": q(1.25, -0.08, 0.10, 1.25, -0.05),
        }

    def _vendor_payload(self, side: str, **kwargs) -> dict:
        firm: FirmState = kwargs["firm"]
        vendor: VendorProfile = kwargs["vendor"]
        context: MarketContext = kwargs["context"]
        return {
            "negotiation_type": "vendor_subscription",
            "side": side,
            "day": int(kwargs["day"]),
            "round": int(kwargs["round_idx"]),
            "max_rounds": int(kwargs["max_rounds"]),
            "min_rounds": int(self.negotiation.get("vendor_min_rounds", 1)),
            "firm": _firm_payload(firm),
            "vendor": {
                "vendor_id": vendor.vendor_id,
                "subscription_fee": float(vendor.subscription_fee),
                "productivity_lift": float(vendor.productivity_lift),
                "risk_multiplier": float(vendor.risk_multiplier),
                "reputation": float(vendor.reputation),
            },
            "context": _context_payload(context),
            "ask": float(kwargs["ask"]),
            "floor": float(kwargs["floor"]),
            "buyer_offer": float(kwargs.get("buyer_offer", kwargs.get("last_offer", 0.0))),
            "last_counter": float(kwargs.get("last_counter", kwargs.get("ask", 0.0))),
            "buyer_ceiling_reference": float(kwargs.get("buyer_ceiling", 0.0)),
            "allowed_decisions": ["accept", "counter", "reject"] if side == "vendor" else ["accept", "counter", "quit"],
        }

    def _insurance_payload(self, side: str, **kwargs) -> dict:
        firm: FirmState = kwargs["firm"]
        quote: InsuranceQuote = kwargs["quote"]
        context: MarketContext = kwargs["context"]
        menu = kwargs.get("menu", {}) or {}
        return {
            "negotiation_type": "insurance_policy",
            "side": side,
            "day": int(kwargs["day"]),
            "round": int(kwargs["round_idx"]),
            "max_rounds": int(kwargs["max_rounds"]),
            "min_rounds": int(self.negotiation.get("insurance_min_rounds", 1)),
            "firm": _firm_payload(firm),
            "quote": _quote_payload(quote),
            "context": _context_payload(context),
            "risk_need": float(kwargs["risk_need"]),
            "ask": float(kwargs["ask"]),
            "floor": float(kwargs["floor"]),
            "buyer_offer": float(kwargs.get("buyer_offer", kwargs.get("last_offer", 0.0))),
            "last_counter": float(kwargs.get("last_counter", kwargs.get("ask", 0.0))),
            "buyer_ceiling_reference": float(kwargs.get("buyer_ceiling", 0.0)),
            "current_tier": str(kwargs.get("current_tier", "STANDARD")),
            "policy_menu": {name: _quote_payload(q) for name, q in menu.items()},
            "allowed_decisions": ["accept", "counter", "reject"] if side == "insurer" else ["accept", "counter", "quit"],
        }


def _negotiation_prompt(payload: dict) -> str:
    decision_field = "premium_money" if payload.get("negotiation_type") == "insurance_policy" else "offer_monthly_fee"
    return (
        "You are one side of a two-party negotiation inside an AI adoption insurance simulation. "
        "Use the JSON state, respect the floor, min_rounds, max_rounds, and allowed decisions, and return exactly one compact JSON object. "
        "Before min_rounds, prefer counter over accept unless the other side already fully meets the current ask. "
        f"Required keys: decision, {decision_field}, message. "
        "For insurance, optionally include tier or tier_request from the policy_menu only. "
        "Do not reveal private ceilings or floors in the message. State JSON: "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True)
    )


def _validate_negotiation_response(parsed: dict, payload: dict) -> None:
    if not isinstance(parsed, dict):
        raise ValueError("negotiation response is not a JSON object")
    allowed = {
        str(value).strip().lower()
        for value in (payload.get("allowed_decisions") or [])
        if str(value).strip()
    }
    decision = str(parsed.get("decision", "")).strip().lower()
    if decision not in allowed:
        raise ValueError(f"invalid negotiation decision: {decision!r}")
    price_key = "premium_money" if payload.get("negotiation_type") == "insurance_policy" else "offer_monthly_fee"
    if price_key not in parsed:
        raise ValueError(f"missing required negotiation price field: {price_key}")
    _as_float(parsed.get(price_key), None)
    if payload.get("negotiation_type") == "insurance_policy":
        menu = set((payload.get("policy_menu") or {}).keys())
        for key in ("tier", "tier_request"):
            if key in parsed and str(parsed.get(key, "")).strip():
                tier = str(parsed.get(key)).upper().strip()
                if menu and tier not in menu:
                    raise ValueError(f"invalid insurance tier: {tier!r}")


def _mock_negotiation_response(payload: dict) -> dict:
    side = str(payload.get("side", "buyer"))
    ask = float(payload.get("ask", 0.0))
    floor = float(payload.get("floor", 0.0))
    buyer_offer = float(payload.get("buyer_offer", 0.0))
    ceiling = float(payload.get("buyer_ceiling_reference", 0.0))
    max_rounds = max(1, int(payload.get("max_rounds", 1)))
    min_rounds = min(max_rounds, max(1, int(payload.get("min_rounds", 1))))
    round_idx = max(1, int(payload.get("round", 1)))
    progress = round_idx / max_rounds
    key = "premium_money" if payload.get("negotiation_type") == "insurance_policy" else "offer_monthly_fee"

    if side in {"vendor", "insurer"}:
        if buyer_offer >= ask and round_idx >= min_rounds:
            return {"decision": "accept", key: buyer_offer, "message": "Accepted at current terms."}
        counter = max(floor, ask - (ask - floor) * (0.25 + 0.45 * progress))
        return {"decision": "counter", key: counter, "message": "Countering within sustainable bounds."}

    if ask <= ceiling and round_idx >= min_rounds:
        return {"decision": "accept", key: ask, "message": "Accepted after concessions."}
    if round_idx >= max_rounds or ceiling < floor:
        return {"decision": "quit", key: buyer_offer, "message": "Not viable within budget."}
    offer = buyer_offer + (min(ask, ceiling) - buyer_offer) * (0.35 + 0.30 * progress)
    return {"decision": "counter", key: offer, "message": "Improving offer without revealing ceiling."}


def _attach_trace(event: dict, response: dict) -> None:
    trace = response.get("model_trace")
    if isinstance(trace, dict):
        event["model_trace"] = trace


def _firm_payload(firm: FirmState) -> dict:
    p = firm.profile
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


def _quote_payload(quote: InsuranceQuote) -> dict:
    return {
        "insurer_id": quote.insurer_id,
        "vendor_id": quote.vendor_id,
        "term_days": int(quote.term_days),
        "premium": round(float(quote.premium), 4),
        "deductible_ratio": round(float(quote.deductible_ratio), 4),
        "coverage_ratio": round(float(quote.coverage_ratio), 4),
        "limit_money": round(float(quote.limit_money), 4),
        "incident_threshold": round(float(quote.incident_threshold), 4),
        "expected_loss": round(float(quote.expected_loss), 4),
        "stress_loss": round(float(quote.stress_loss), 4),
        "regime": quote.regime,
        "market_role": quote.market_role,
    }


def _as_float(value, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _bounded_rounds(value, fallback: int, cap: int = 30, min_rounds: int = 1) -> int:
    try:
        rounds = int(value)
    except (TypeError, ValueError):
        rounds = int(fallback)
    floor = max(1, int(min_rounds))
    return max(floor, min(int(rounds), max(floor, int(cap))))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


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
