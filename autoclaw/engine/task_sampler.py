import hashlib
import random
import re
import yaml


class TaskSampler:
    def __init__(self, config_path: str):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

    def _rng_for_firm_day(self, buyer, day_idx: int, industry: str, base_seed: int | None):
        if base_seed is None:
            return random
        seed_material = f"{base_seed}|{buyer.id}|{day_idx}|{industry}"
        seed_hash = hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
        return random.Random(int(seed_hash[:16], 16))

    @staticmethod
    def _weighted_dist_to_count(dist: dict, rng) -> int:
        if not dist:
            return 1
        counts = []
        weights = []
        for raw_count, raw_weight in dist.items():
            count = int(raw_count)
            weight = float(raw_weight)
            if count > 0 and weight > 0:
                counts.append(count)
                weights.append(weight)
        if not counts:
            return 1
        return int(rng.choices(counts, weights=weights, k=1)[0])

    @staticmethod
    def _weighted_dist_to_label(dist: dict, rng, default: str) -> str:
        if not dist:
            return default
        labels = []
        weights = []
        for raw_label, raw_weight in dist.items():
            label = str(raw_label).strip().lower()
            weight = float(raw_weight)
            if label and weight > 0:
                labels.append(label)
                weights.append(weight)
        if not labels:
            return default
        return str(rng.choices(labels, weights=weights, k=1)[0])

    def _sample_task_difficulty(self, requested: str, sector_cfg: dict, rng) -> str:
        requested = str(requested or "easy").strip().lower()
        if requested != "mixed":
            return requested
        mix = sector_cfg.get("difficulty_mix", self.cfg.get("default_difficulty_mix", {}))
        sampled = self._weighted_dist_to_label(mix, rng, "calibrated")
        if sampled not in {"easy", "calibrated", "stress"}:
            return "calibrated"
        return sampled

    @staticmethod
    def _safe_id(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_") or "task"

    @staticmethod
    def _weighted_without_replacement(names: list[str], weights: list[float], k: int, rng) -> list[str]:
        pool = [(name, float(weight)) for name, weight in zip(names, weights) if float(weight) > 0]
        chosen = []
        for _ in range(min(k, len(pool))):
            total = sum(weight for _, weight in pool)
            if total <= 0:
                idx = rng.randrange(len(pool))
            else:
                threshold = rng.random() * total
                acc = 0.0
                idx = 0
                for idx, (_, weight) in enumerate(pool):
                    acc += weight
                    if acc >= threshold:
                        break
            chosen.append(pool[idx][0])
            pool.pop(idx)
        return chosen

    def sample_task_count_for_firm_day(
        self,
        buyer,
        day_idx: int,
        base_seed: int | None = None,
        task_count_override: int | None = None,
        max_tasks_per_firm_day: int | None = None,
    ) -> int:
        industry = buyer.profile.get("industry_code", "default")
        profiles = self.cfg.get("task_profiles", {})
        sector_cfg = profiles.get(industry, profiles.get("default", {}))
        rng = self._rng_for_firm_day(buyer, day_idx, industry, base_seed)

        if task_count_override is not None:
            count = int(task_count_override)
        else:
            count_cfg = sector_cfg.get("daily_task_count", profiles.get("default", {}).get("daily_task_count", {}))
            count = self._weighted_dist_to_count(count_cfg.get("probabilities", {}), rng)

        if max_tasks_per_firm_day is not None:
            count = min(count, int(max_tasks_per_firm_day))
        return max(1, int(count))

    def sample_tasks_for_firm_day(
        self,
        buyer,
        day_idx: int,
        base_seed: int | None = None,
        difficulty: str = "easy",
        task_count_override: int | None = None,
        max_tasks_per_firm_day: int | None = None,
        force_task_type: str = "",
    ):
        industry = buyer.profile.get("industry_code", "default")
        profiles = self.cfg.get("task_profiles", {})
        sector_cfg = profiles.get(industry, profiles.get("default", {}))
        task_probs = sector_cfg.get("main_tasks", {"report_generation": 1.0})
        task_names = list(task_probs.keys())
        task_weights = list(task_probs.values())
        rng = self._rng_for_firm_day(buyer, day_idx, industry, base_seed)
        task_count = self.sample_task_count_for_firm_day(
            buyer,
            day_idx,
            base_seed=base_seed,
            task_count_override=task_count_override,
            max_tasks_per_firm_day=max_tasks_per_firm_day,
        )

        if force_task_type:
            sampled_types = [force_task_type] * task_count
        elif task_count <= len(task_names):
            sampled_types = self._weighted_without_replacement(task_names, task_weights, task_count, rng)
        else:
            sampled_types = self._weighted_without_replacement(task_names, task_weights, len(task_names), rng)
            sampled_types.extend(rng.choices(task_names, weights=task_weights, k=task_count - len(task_names)))

        bundles = []
        safe_firm_id = self._safe_id(buyer.id)
        for task_index, task_type in enumerate(sampled_types):
            task_difficulty = self._sample_task_difficulty(difficulty, sector_cfg, rng)
            bundles.append(
                {
                    "firm_id": str(buyer.id),
                    "day": int(day_idx),
                    "industry": industry,
                    "task_type": task_type,
                    "difficulty": task_difficulty,
                    "difficulty_policy": difficulty,
                    "task_index": int(task_index),
                    "firm_day_task_count": int(task_count),
                    "task_id": f"{safe_firm_id}_d{int(day_idx):03d}_t{int(task_index):02d}_{self._safe_id(task_type)}",
                }
            )
        return bundles
