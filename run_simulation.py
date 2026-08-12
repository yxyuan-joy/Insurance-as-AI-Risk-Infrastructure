#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from datetime import datetime
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
SRC_DIR = THIS_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from action_risk_v2.simulator import ActionRiskSimulator, latest_checkpoint_path, prepare_run_dir


TRACKED_ENV_KEYS = [
    "PROJECT_DIR",
    "RUNS_DIR",
    "RUN_NAME",
    "CONFIG_PATH",
    "PROFILE",
    "EXPERIMENT_TAG",
    "AUTORISK_RUN_TAG",
    "DAYS",
    "FIRMS",
    "SIM_SEED",
    "GPU_COUNT",
    "MODEL_ID",
    "SERVED_MODEL_NAME",
    "DECISION_MODE",
    "MODEL_FALLBACK_TO_RULE",
    "AUTO_RESUME",
    "VLLM_BASE_URL",
    "VLLM_BASE_URLS",
    "VLLM_MODEL",
    "VLLM_READY_TIMEOUT_SEC",
    "GPU_MEMORY_UTILIZATION",
    "MAX_MODEL_LEN",
    "VLLM_MAX_NUM_SEQS",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "HF_ENDPOINT",
    "PIP_INDEX_URL",
    "SKIP_PIP_INSTALL",
]

SENSITIVE_ENV_FRAGMENTS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "PRIVATE", "CREDENTIAL")


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _redact_if_sensitive(key: str, value: str) -> str:
    upper = key.upper()
    if any(fragment in upper for fragment in SENSITIVE_ENV_FRAGMENTS):
        return "<redacted>"
    return value


def _env_snapshot() -> dict:
    snapshot = {}
    for key in TRACKED_ENV_KEYS:
        if key in os.environ:
            snapshot[key] = _redact_if_sensitive(key, os.environ[key])
    return snapshot


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def _write_invocation_meta(run_dir: Path, args: argparse.Namespace, resume_from: Path | None) -> None:
    payload = {
        "created_at": _now_iso(),
        "argv": list(sys.argv),
        "cwd": str(Path.cwd()),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "args": vars(args),
        "resume_from_resolved": str(resume_from) if resume_from else "",
        "tracked_environment": _env_snapshot(),
    }
    _write_json(run_dir / "invocation_meta.json", payload)


def _write_run_status(run_dir: Path, status: str, **extra: object) -> None:
    payload = {"status": status, "timestamp": _now_iso()}
    payload.update(extra)
    _write_json(run_dir / "run_status.json", payload)


def _count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _output_manifest(run_dir: Path) -> dict:
    files = {}
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        entry = {
            "size_bytes": int(path.stat().st_size),
            "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
        }
        if path.suffix in {".csv", ".jsonl"}:
            lines = _count_lines(path)
            entry["line_count"] = int(lines)
            if path.suffix == ".csv":
                entry["data_rows_estimate"] = max(0, int(lines) - 1)
        files[path.name] = entry

    checkpoints = sorted((run_dir / "checkpoints").glob("day_*.json"))
    return {
        "created_at": _now_iso(),
        "run_dir": str(run_dir),
        "files": files,
        "checkpoints": {
            "count": len(checkpoints),
            "latest": checkpoints[-1].name if checkpoints else "",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the insurance-mediated AI risk simulation.")
    parser.add_argument("--config", default=str(THIS_DIR / "configs" / "formal.yaml"))
    parser.add_argument("--run-name", default="local_run")
    parser.add_argument("--runs-dir", default=str(THIS_DIR / "runs"))
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--firms", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Override simulation.seed for paired robustness runs.")
    parser.add_argument(
        "--action-risk-path",
        default=None,
        help=(
            "Override paths.action_risk_path. Use this for freshly generated AutoCLAW "
            "firm_daily_action_risk.csv outputs instead of the legacy local smoke-test panel."
        ),
    )
    parser.add_argument(
        "--selected-firms-path",
        default=None,
        help=(
            "Override paths.selected_firms_path. Pass NONE to disable the legacy selected-firm "
            "filter when using a fresh AutoCLAW CSV."
        ),
    )
    parser.add_argument(
        "--buyer-population-path",
        default=None,
        help="Optional override for paths.buyer_population_path.",
    )
    parser.add_argument(
        "--real-firms-path",
        default=None,
        help="Optional override for paths.real_firms_path.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--decision-mode",
        choices=["rule_heuristic", "model_mock", "vllm_openai", "openai_compatible"],
        default=None,
        help="Override decision_layer.mode without editing the YAML config.",
    )
    parser.add_argument("--vllm-base-url", default=None, help="Override decision_layer.base_url for vLLM/OpenAI-compatible serving.")
    parser.add_argument(
        "--vllm-base-urls",
        default=None,
        help="Comma-separated OpenAI-compatible vLLM endpoints for model-agent decisions and negotiations.",
    )
    parser.add_argument("--vllm-model", default=None, help="Override decision_layer.model for vLLM/OpenAI-compatible serving.")
    parser.add_argument(
        "--model-fallback-to-rule",
        choices=["0", "1", "false", "true", "False", "True"],
        default=None,
        help=(
            "Override decision_layer.fallback_to_rule. Use 0/false for formal model-agent runs to disable "
            "general backend fallback. Exhausted JSON parse errors still use the logged conservative "
            "no-change recovery defined by the decision and negotiation layers."
        ),
    )
    parser.add_argument(
        "--disable-insurance-market",
        action="store_true",
        help="Run the matched counterfactual society with AI-risk insurance purchasing and claims disabled.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing run from the latest checkpoint in its run directory.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume from a specific checkpoint JSON path instead of the latest checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_enabled = bool(args.resume or args.resume_from)
    run_dir = prepare_run_dir(Path(args.runs_dir), args.run_name, overwrite=args.overwrite, resume=resume_enabled)
    resume_from = Path(args.resume_from) if args.resume_from else None
    if args.resume and resume_from is None:
        resume_from = latest_checkpoint_path(run_dir)
    _write_invocation_meta(run_dir, args, resume_from)
    _write_run_status(
        run_dir,
        "started",
        run_name=args.run_name,
        runs_dir=str(Path(args.runs_dir)),
        config=str(Path(args.config)),
        resume_enabled=resume_enabled,
        resume_from=str(resume_from) if resume_from else "",
    )
    path_overrides = {
        key: value
        for key, value in {
            "action_risk_path": args.action_risk_path,
            "selected_firms_path": args.selected_firms_path,
            "buyer_population_path": args.buyer_population_path,
            "real_firms_path": args.real_firms_path,
        }.items()
        if value is not None
    }
    sim = ActionRiskSimulator.from_yaml(
        config_path=Path(args.config),
        run_dir=run_dir,
        days=args.days,
        firms=args.firms,
        resume_from=resume_from,
        decision_mode=args.decision_mode,
        vllm_base_url=args.vllm_base_url,
        vllm_base_urls=args.vllm_base_urls,
        vllm_model=args.vllm_model,
        model_fallback_to_rule=(
            None
            if args.model_fallback_to_rule is None
            else str(args.model_fallback_to_rule).lower() in {"1", "true"}
        ),
        insurance_market_enabled=False if args.disable_insurance_market else None,
        seed=args.seed,
        path_overrides=path_overrides,
    )
    try:
        sim.run()
    except Exception as exc:
        _write_run_status(
            run_dir,
            "failed",
            run_name=args.run_name,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    _write_json(run_dir / "output_manifest.json", _output_manifest(run_dir))
    _write_run_status(
        run_dir,
        "completed",
        run_name=args.run_name,
        output_manifest=str(run_dir / "output_manifest.json"),
    )
    print(f"Action-risk v2 run complete: {run_dir}")


if __name__ == "__main__":
    main()
