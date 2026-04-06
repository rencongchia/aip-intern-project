"""CLI entry point for the Phase 3 fault injection sweep.

Usage:
    python scripts/run_phase3.py --config config/phase3.yaml
    python scripts/run_phase3.py --config config/phase3.yaml --dry-run
    python scripts/run_phase3.py --config config/phase3.yaml --fault timeout
    python scripts/run_phase3.py --config config/phase3.yaml --fault timeout --dry-run

With --fault FAULT, only that fault type runs (useful for targeted re-runs).
Without --fault, all four fault types run sequentially.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()  # Load .env before config resolves ${ENV_VAR} placeholders

import yaml

from aip_intern.baseline.runner import RunConfig
from aip_intern.core.config import load_config
from aip_intern.phase3.runner import FaultRunConfig, run


def _parse_fault_types(raw: dict) -> list[FaultRunConfig]:
    """Build FaultRunConfig list from config YAML fault_types section."""
    fault_list = raw.get("fault_types", [])
    configs = []
    for ft in fault_list:
        name = ft["name"]
        configs.append(FaultRunConfig(
            fault_name=name,
            after_seconds=float(ft.get("after_seconds", 3.0)),
            fail_on_call=int(ft.get("fail_on_call", 1)),
            token_limit=int(ft.get("token_limit", 500)),
        ))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3 fault injection sweep")
    parser.add_argument("--config", default="config/phase3.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run 1 iteration per fault type for smoke test",
    )
    parser.add_argument(
        "--fault",
        choices=["timeout", "malformed_json", "checkpoint_loss", "context_overflow"],
        default=None,
        help="Run only this fault type (default: all four)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    # Load raw YAML for fault_types section, then use load_config for everything else
    raw = yaml.safe_load(config_path.read_text())
    fault_types = _parse_fault_types(raw)

    app_cfg = load_config(config_path)
    run_cfg = RunConfig(
        run_id_prefix=app_cfg.run.run_id_prefix,
        n_runs=1 if args.dry_run else app_cfg.run.n_runs,
        config_path=config_path,
        llm_model=app_cfg.llm.model,
        llm_base_url=app_cfg.llm.base_url,
        llm_api_key=app_cfg.llm.api_key,
        llm_temperature=app_cfg.llm.temperature,
        llm_max_tokens=app_cfg.llm.max_tokens,
        llm_request_timeout=app_cfg.llm.request_timeout,
        workspace_root=Path(app_cfg.mcp.workspace_root),
        artifacts_dir=Path(app_cfg.artifacts.output_dir),
    )

    # Filter to specific fault type if requested
    if args.fault:
        fault_types = [f for f in fault_types if f.fault_name == args.fault]
        if not fault_types:
            print(f"Fault type {args.fault!r} not found in config")
            sys.exit(1)

    if not fault_types:
        print("No fault types configured in config/phase3.yaml")
        sys.exit(1)

    n_label = "dry-run" if args.dry_run else f"{run_cfg.n_runs}-run"
    fault_label = args.fault or "all 4 fault types"
    total = run_cfg.n_runs * len(fault_types)
    print(f"Starting {n_label} Phase 3 sweep: {fault_label} ({total} total runs)...")

    results = asyncio.run(run(run_cfg, fault_types))
    passed = sum(r.success for r in results)

    # Summary table per fault type
    print(f"\nComplete: {passed}/{len(results)} runs succeeded.")
    print(f"Artifacts: {run_cfg.artifacts_dir}/")
    print("\nRecovery summary:")
    print(f"  {'Fault Type':<25} {'Runs':>5} {'Auto':>6} {'Manual':>7} {'Avg recovery_time_s':>20}")
    print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*7} {'-'*20}")
    by_fault: dict[str, list] = {}
    for r in results:
        ft = r.metrics.get("failure_type", "unknown")
        by_fault.setdefault(ft, []).append(r)
    for ft, rs in by_fault.items():
        auto = sum(1 for r in rs if r.metrics.get("recovery_mode") == "automatic")
        manual = sum(1 for r in rs if r.metrics.get("recovery_mode") == "manual")
        times = [r.metrics.get("recovery_time_s", 0) for r in rs]
        avg_t = sum(times) / len(times) if times else 0.0
        print(f"  {ft:<25} {len(rs):>5} {auto:>6} {manual:>7} {avg_t:>20.2f}s")


if __name__ == "__main__":
    main()
