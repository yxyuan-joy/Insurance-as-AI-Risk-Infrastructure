from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .schema import ActionRiskRecord, FirmProfile, IndustryRiskSnapshot


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _as_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def parse_task_mix(raw: str) -> Dict[str, int]:
    if not raw:
        return {}
    out: Dict[str, int] = {}
    for part in str(raw).split(";"):
        if not part.strip() or ":" not in part:
            continue
        key, val = part.split(":", 1)
        key = key.strip()
        if not key:
            continue
        out[key] = out.get(key, 0) + _as_int(val, 0)
    return out


class ActionRiskPanel:
    """First-class loader for AutoCLAW action-risk observations."""

    REQUIRED_FIRM_DAY = {
        "industry",
        "firm_id",
        "day",
        "num_tasks",
        "incident_any_flag",
        "incident_task_count",
        "avg_severity",
        "sum_total_loss",
        "avg_risk_score",
        "max_risk_score",
    }

    REQUIRED_EPISODE = {
        "industry",
        "firm_id",
        "day",
        "task_type",
        "incident_flag",
        "severity",
        "total_loss",
        "risk_score",
    }

    def __init__(
        self,
        records: pd.DataFrame,
        profiles: Dict[str, FirmProfile],
        selected_firms: Optional[Iterable[str]] = None,
    ):
        self.records = self._normalize_records(records)
        self.profiles = dict(profiles)
        self.selected_firms = list(selected_firms or sorted(self.records["firm_id"].unique()))
        self._record_map: Dict[tuple[str, int], ActionRiskRecord] = {}
        self._build_record_map()

    @classmethod
    def from_files(
        cls,
        action_risk_path: Path,
        buyer_population_path: Optional[Path] = None,
        selected_firms_path: Optional[Path] = None,
        real_firms_path: Optional[Path] = None,
    ) -> "ActionRiskPanel":
        action_risk_path = Path(action_risk_path)
        raw = pd.read_csv(action_risk_path)

        cols = set(raw.columns)
        if cls.REQUIRED_FIRM_DAY.issubset(cols):
            records = raw.copy()
        elif cls.REQUIRED_EPISODE.issubset(cols):
            records = cls._aggregate_episodes(raw)
        else:
            missing_firm_day = sorted(cls.REQUIRED_FIRM_DAY - cols)
            missing_episode = sorted(cls.REQUIRED_EPISODE - cols)
            raise ValueError(
                f"Unsupported action-risk schema at {action_risk_path}. "
                f"Missing firm-day fields: {missing_firm_day}; "
                f"missing episode fields: {missing_episode}"
            )

        selected_ids: Optional[List[str]] = None
        selected_meta: Dict[str, dict] = {}
        if selected_firms_path and Path(selected_firms_path).exists():
            selected_df = pd.read_csv(selected_firms_path)
            selected_ids = [str(x) for x in selected_df["firm_id"].tolist()]
            for _, row in selected_df.iterrows():
                selected_meta[str(row["firm_id"])] = {
                    "id": str(row["firm_id"]),
                    "name": row.get("name", str(row["firm_id"])),
                    "industry_code": row.get("industry", row.get("industry_code", "unknown")),
                    "cash": _as_float(row.get("cash"), 100000.0),
                }

        profiles = load_firm_profiles(
            buyer_population_path=buyer_population_path,
            real_firms_path=real_firms_path,
            selected_meta=selected_meta,
            fallback_records=records,
        )
        return cls(records=records, profiles=profiles, selected_firms=selected_ids)

    @staticmethod
    def _aggregate_episodes(df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        work["incident_flag"] = work["incident_flag"].map(_as_int)
        work["severity"] = work["severity"].map(_as_float)
        work["total_loss"] = work["total_loss"].map(_as_float)
        work["risk_score"] = work["risk_score"].map(_as_float)

        rows = []
        for (firm_id, day, industry), group in work.groupby(["firm_id", "day", "industry"], sort=True):
            task_counts = group["task_type"].value_counts().to_dict()
            rows.append(
                {
                    "firm_id": str(firm_id),
                    "day": int(day),
                    "industry": str(industry),
                    "num_tasks": int(len(group)),
                    "incident_any_flag": int(group["incident_flag"].max()),
                    "incident_task_count": int(group["incident_flag"].sum()),
                    "avg_severity": float(group["severity"].mean()),
                    "sum_direct_loss_base": float(group.get("direct_loss_base", pd.Series(dtype=float)).map(_as_float).sum())
                    if "direct_loss_base" in group
                    else 0.0,
                    "sum_total_loss": float(group["total_loss"].sum()),
                    "avg_risk_score": float(group["risk_score"].mean()),
                    "max_risk_score": float(group["risk_score"].max()),
                    "task_type_mix": ";".join(f"{k}:{int(v)}" for k, v in sorted(task_counts.items())),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _normalize_records(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["firm_id"] = out["firm_id"].astype(str)
        out["industry"] = out["industry"].astype(str)
        out["day"] = out["day"].map(_as_int)
        for col in [
            "num_tasks",
            "incident_any_flag",
            "incident_task_count",
        ]:
            out[col] = out[col].map(_as_int)
        for col in [
            "avg_severity",
            "sum_total_loss",
            "avg_risk_score",
            "max_risk_score",
        ]:
            out[col] = out[col].map(_as_float)
        if "task_type_mix" not in out.columns:
            out["task_type_mix"] = ""
        return out.sort_values(["day", "industry", "firm_id"]).reset_index(drop=True)

    def _build_record_map(self) -> None:
        self._record_map.clear()
        for _, row in self.records.iterrows():
            record = ActionRiskRecord(
                firm_id=str(row["firm_id"]),
                day=int(row["day"]),
                industry=str(row["industry"]),
                num_tasks=_as_int(row.get("num_tasks")),
                incident_any=bool(_as_int(row.get("incident_any_flag"))),
                incident_task_count=_as_int(row.get("incident_task_count")),
                avg_severity=_as_float(row.get("avg_severity")),
                total_loss=_as_float(row.get("sum_total_loss")),
                avg_risk_score=_as_float(row.get("avg_risk_score")),
                max_risk_score=_as_float(row.get("max_risk_score")),
                task_type_mix=parse_task_mix(row.get("task_type_mix", "")),
            )
            self._record_map[(record.firm_id, record.day)] = record

    @property
    def days(self) -> List[int]:
        return sorted(int(x) for x in self.records["day"].unique())

    @property
    def industries(self) -> List[str]:
        return sorted(str(x) for x in self.records["industry"].unique())

    def firm_ids(self, limit: Optional[int] = None) -> List[str]:
        ids = [fid for fid in self.selected_firms if fid in self.profiles]
        if limit is not None:
            ids = _stratified_limit(ids, int(limit), self.profiles)
        return ids

    def profile_for(self, firm_id: str) -> FirmProfile:
        if firm_id in self.profiles:
            return self.profiles[firm_id]
        row = self.records[self.records["firm_id"] == firm_id].head(1)
        industry = str(row.iloc[0]["industry"]) if not row.empty else "unknown"
        return FirmProfile(
            firm_id=firm_id,
            name=firm_id,
            industry=industry,
            cash=100000.0,
            asset_value=100000.0,
        )

    def record_for(self, firm_id: str, day: int) -> ActionRiskRecord:
        record = self._record_map.get((str(firm_id), int(day)))
        if record is not None:
            return record
        profile = self.profile_for(str(firm_id))
        return ActionRiskRecord(
            firm_id=str(firm_id),
            day=int(day),
            industry=profile.industry,
            num_tasks=0,
            incident_any=False,
            incident_task_count=0,
            avg_severity=0.0,
            total_loss=0.0,
            avg_risk_score=0.0,
            max_risk_score=0.0,
            task_type_mix={},
        )

    def industry_snapshot(self, industry: str, day: int, trailing_window: int = 14) -> IndustryRiskSnapshot:
        industry = str(industry)
        day = int(day)
        prior = self.records[(self.records["industry"] == industry) & (self.records["day"] < day)]
        if trailing_window > 0 and not prior.empty:
            min_day = max(int(prior["day"].max()) - trailing_window + 1, int(prior["day"].min()))
            prior = prior[prior["day"] >= min_day]

        sample = prior
        if sample.empty:
            return IndustryRiskSnapshot(
                industry=industry,
                day=day,
                observations=0,
                incident_rate=0.0,
                avg_severity=0.0,
                avg_loss=0.0,
                avg_risk_score=0.0,
                stress_loss=0.0,
                stress_risk_score=0.0,
            )

        return IndustryRiskSnapshot(
            industry=industry,
            day=day,
            observations=int(len(sample)),
            incident_rate=float(sample["incident_any_flag"].mean()),
            avg_severity=float(sample["avg_severity"].mean()),
            avg_loss=float(sample["sum_total_loss"].mean()),
            avg_risk_score=float(sample["avg_risk_score"].mean()),
            stress_loss=float(sample["sum_total_loss"].quantile(0.95)),
            stress_risk_score=float(sample["max_risk_score"].quantile(0.95)),
        )


def _stratified_limit(ids: List[str], limit: int, profiles: Dict[str, FirmProfile]) -> List[str]:
    if limit <= 0:
        return []
    if limit >= len(ids):
        return list(ids)

    groups: Dict[str, List[str]] = {}
    for fid in ids:
        industry = str(profiles[fid].industry if fid in profiles else "unknown")
        groups.setdefault(industry, []).append(fid)
    if len(groups) <= 1:
        return ids[:limit]

    total = sum(len(group) for group in groups.values())
    quotas: Dict[str, int] = {}
    remainders = []
    for industry, group in groups.items():
        exact = limit * len(group) / max(1, total)
        quota = int(exact)
        quotas[industry] = quota
        remainders.append((exact - quota, len(group), industry))

    if limit >= len(groups):
        for industry in groups:
            if quotas[industry] == 0:
                quotas[industry] = 1

    while sum(quotas.values()) > limit:
        candidates = sorted(
            (quotas[industry], len(groups[industry]), industry)
            for industry in groups
            if quotas[industry] > 0
        )
        _, _, industry = candidates[0]
        quotas[industry] -= 1

    remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
    cursor = 0
    while sum(quotas.values()) < limit and remainders:
        _, _, industry = remainders[cursor % len(remainders)]
        if quotas[industry] < len(groups[industry]):
            quotas[industry] += 1
        cursor += 1

    order = sorted(groups, key=lambda industry: (-len(groups[industry]), industry))
    offsets = {industry: 0 for industry in groups}
    selected: List[str] = []
    while len(selected) < limit and any(quotas[industry] > 0 for industry in order):
        for industry in order:
            if quotas[industry] <= 0:
                continue
            idx = offsets[industry]
            if idx >= len(groups[industry]):
                quotas[industry] = 0
                continue
            selected.append(groups[industry][idx])
            offsets[industry] = idx + 1
            quotas[industry] -= 1
            if len(selected) >= limit:
                break
    return selected


def load_firm_profiles(
    buyer_population_path: Optional[Path],
    real_firms_path: Optional[Path],
    selected_meta: Optional[Dict[str, dict]],
    fallback_records: pd.DataFrame,
) -> Dict[str, FirmProfile]:
    raw_by_id: Dict[str, dict] = {}

    for path in [real_firms_path, buyer_population_path]:
        if path and Path(path).exists():
            with Path(path).open("r", encoding="utf-8") as f:
                data = json.load(f)
            for row in data:
                raw_by_id[str(row.get("id"))] = dict(row)

    for firm_id, row in (selected_meta or {}).items():
        raw_by_id.setdefault(str(firm_id), {}).update(row)

    industries = {
        str(row["firm_id"]): str(row["industry"])
        for _, row in fallback_records[["firm_id", "industry"]].drop_duplicates().iterrows()
    }

    profiles: Dict[str, FirmProfile] = {}
    for firm_id, industry in industries.items():
        row = raw_by_id.get(firm_id, {})
        cash = _as_float(row.get("cash"), 100000.0)
        asset_value = _as_float(row.get("asset_value"), cash)
        if asset_value <= 0:
            asset_value = max(cash, 100000.0)
        profiles[firm_id] = FirmProfile(
            firm_id=firm_id,
            name=str(row.get("name", firm_id)),
            industry=str(row.get("industry_code", row.get("industry", industry))),
            cash=cash,
            asset_value=asset_value,
            risk_tolerance=min(1.0, max(0.0, _as_float(row.get("risk_tolerance"), 0.5))),
            tech_urgency=min(1.0, max(0.0, _as_float(row.get("tech_urgency"), 0.5))),
            ai_dependency=min(1.0, max(0.0, _as_float(row.get("ai_dependency"), 0.5))),
            inertia=min(1.0, max(0.0, _as_float(row.get("inertia"), 0.5))),
            innovativeness=min(1.0, max(0.0, _as_float(row.get("innovativeness"), 0.5))),
            contagion_sensitivity=min(1.0, max(0.0, _as_float(row.get("contagion_sensitivity"), 0.5))),
            size_label=str(row.get("size_label", "unknown")),
        )
    return profiles
