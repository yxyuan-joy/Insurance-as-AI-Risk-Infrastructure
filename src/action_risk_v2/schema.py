from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass(frozen=True)
class FirmProfile:
    firm_id: str
    name: str
    industry: str
    cash: float
    asset_value: float
    risk_tolerance: float = 0.5
    tech_urgency: float = 0.5
    ai_dependency: float = 0.5
    inertia: float = 0.5
    innovativeness: float = 0.5
    contagion_sensitivity: float = 0.5
    size_label: str = "unknown"


@dataclass(frozen=True)
class ActionRiskRecord:
    firm_id: str
    day: int
    industry: str
    num_tasks: int
    incident_any: bool
    incident_task_count: int
    avg_severity: float
    total_loss: float
    avg_risk_score: float
    max_risk_score: float
    task_type_mix: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class IndustryRiskSnapshot:
    industry: str
    day: int
    observations: int
    incident_rate: float
    avg_severity: float
    avg_loss: float
    avg_risk_score: float
    stress_loss: float
    stress_risk_score: float


@dataclass(frozen=True)
class VendorProfile:
    vendor_id: str
    label: str
    subscription_fee: float
    productivity_lift: float
    risk_multiplier: float
    reputation: float
    marketing_weight: float
    target_sectors: tuple[str, ...] = ("ALL",)

    def sector_affinity(self, industry: str) -> float:
        targets = {s.lower() for s in self.target_sectors}
        if industry.lower() in targets:
            return 1.0
        if "all" in targets:
            return 0.75
        return 0.35


@dataclass(frozen=True)
class InsurerProfile:
    insurer_id: str
    label: str
    domicile: str
    initial_capital: float
    base_margin: float
    risk_appetite: float
    expense_load: float
    capital_load: float
    deductible_ratio: float
    coverage_ratio: float
    limit_ratio: float
    max_active_policies: int
    solvency_floor_ratio: float
    soft_threshold_ratio: float
    hard_threshold_ratio: float
    target_sectors: tuple[str, ...] = ("ALL",)
    market_role: str = "private"

    def sector_affinity(self, industry: str) -> float:
        targets = {s.lower() for s in self.target_sectors}
        if "all" in targets or industry.lower() in targets:
            return 1.0
        return 1.45


@dataclass
class VendorContract:
    vendor_id: str
    price: float
    start_day: int
    end_day: int
    monthly_fee: float = 0.0


@dataclass
class InsurancePolicy:
    insurer_id: str
    premium: float
    deductible_ratio: float
    coverage_ratio: float
    limit_money: float
    incident_threshold: float
    start_day: int
    end_day: int
    vendor_id: str = ""


@dataclass
class FirmState:
    profile: FirmProfile
    cash: float
    active: bool = True
    vendor_contract: Optional[VendorContract] = None
    insurance_policy: Optional[InsurancePolicy] = None
    panic: float = 0.0
    last_operational_loss: float = 0.0
    last_claim_paid: float = 0.0
    last_claim_day: Optional[int] = None
    risk_memory: float = 0.0
    loss_memory: float = 0.0
    claimable_memory: float = 0.0
    ai_cooldown_until: int = 0

    @property
    def has_ai(self) -> bool:
        return self.vendor_contract is not None and self.active

    @property
    def has_insurance(self) -> bool:
        return self.insurance_policy is not None and self.active


@dataclass
class InsurerState:
    profile: InsurerProfile
    capital: float
    premiums_today: float = 0.0
    claims_today: float = 0.0
    refunds_today: float = 0.0
    active_policies: int = 0
    new_policies_today: int = 0

    @property
    def capital_ratio(self) -> float:
        if self.profile.initial_capital <= 0:
            return 0.0
        return self.capital / self.profile.initial_capital

    @property
    def regime(self) -> str:
        r = self.capital_ratio
        if r <= self.profile.solvency_floor_ratio:
            return "RUNOFF"
        if r <= self.profile.hard_threshold_ratio:
            return "HARD"
        if r <= self.profile.soft_threshold_ratio:
            return "SOFT"
        return "NORMAL"

    @property
    def underwriting_open(self) -> bool:
        return self.regime != "RUNOFF"


@dataclass(frozen=True)
class InsuranceQuote:
    insurer_id: str
    firm_id: str
    vendor_id: str
    industry: str
    day: int
    term_days: int
    premium: float
    deductible_ratio: float
    coverage_ratio: float
    limit_money: float
    incident_threshold: float
    expected_loss: float
    stress_loss: float
    regime: str
    market_role: str = "private"
