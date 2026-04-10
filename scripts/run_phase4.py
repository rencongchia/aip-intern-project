"""CLI entry point for the Phase 4 GCC-constrained fault injection sweep.

Usage:
    python scripts/run_phase4.py --config config/phase4.yaml
    python scripts/run_phase4.py --config config/phase4.yaml --dry-run
    python scripts/run_phase4.py --config config/phase4.yaml --fault timeout
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

import yaml

from aip_intern.baseline.runner import RunConfig
from aip_intern.core.config import load_config
from aip_intern.phase3.runner import FaultRunConfig
from aip_intern.phase4.runner import GCCRunConfig, run


def _parse_fault_types(raw: dict) -> list[FaultRunConfig]:
    """Build FaultRunConfig list from config YAML fault_types section."""
    fault_list = raw.get("fault_types", [])
    configs = []
    for ft in fault_list:
        configs.append(FaultRunConfig(
            fault_name=ft["name"],
            after_seconds=float(ft.get("after_seconds", 3.0)),
            fail_on_call=int(ft.get("fail_on_call", 1)),
            token_limit=int(ft.get("token_limit", 500)),
        ))
    return configs


def _parse_gcc(raw: dict) -> GCCRunConfig:
    """Build GCCRunConfig from config YAML gcc section."""
    gcc = raw.get("gcc", {})
    return GCCRunConfig(
        latency_ms=int(gcc.get("latency_ms", 200)),
        egress_jitter_ms=int(gcc.get("egress_jitter_ms", 50)),
        tpm_limit=int(gcc.get("tpm_limit", 60_000)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Phase 4 GCC-constrained fault injection sweep"
    )
    parser.add_argument("--config", default="config/phase4.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fault",
        choices=["timeout", "malformed_json", "checkpoint_loss", "context_overflow"],
        default=None,
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    raw = yaml.safe_load(config_path.read_text())
    fault_types = _parse_fault_types(raw)
    gcc_cfg = _parse_gcc(raw)

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
    run_cfg.inter_run_delay_s = app_cfg.run.inter_run_delay_s  # type: ignore[attr-defined]

    if args.fault:
        fault_types = [f for f in fault_types if f.fault_name == args.fault]
        if not fault_types:
            print(f"Fault type {args.fault!r} not found in config")
            sys.exit(1)

    if not fault_types:
        print("No fault types configured")
        sys.exit(1)

    n_label = "dry-run" if args.dry_run else f"{run_cfg.n_runs}-run"
    fault_label = args.fault or "all 4 fault types"
    total = run_cfg.n_runs * len(fault_types)
    print(
        f"Starting {n_label} Phase 4 GCC sweep: {fault_label} ({total} total runs)..."
    )
    print(
        f"GCC constraints: {gcc_cfg.latency_ms}ms latency, "
        f"{gcc_cfg.tpm_limit} TPM cap"
    )

    results = asyncio.run(run(run_cfg, fault_types, gcc_cfg))
    passed = sum(r.success for r in results)

    print(f"\nComplete: {passed}/{len(results)} runs succeeded.")
    print(f"Artifacts: {run_cfg.artifacts_dir}/")
    print("\nRecovery summary (GCC-constrained):")
    print(
        f"  {'Fault Type':<25} {'Runs':>5} {'Auto':>6} "
        f"{'Manual':>7} {'Unrecov':>8} {'Avg latency_s':>15}"
    )
    print(f"  {'-'*25} {'-'*5} {'-'*6} {'-'*7} {'-'*8} {'-'*15}")
    by_fault: dict[str, list] = {}
    for r in results:
        ft = r.metrics.get("failure_type", "unknown")
        by_fault.setdefault(ft, []).append(r)
    for ft, rs in by_fault.items():
        auto = sum(1 for r in rs if r.metrics.get("recovery_mode") == "automatic")
        manual = sum(1 for r in rs if r.metrics.get("recovery_mode") == "manual")
        unrecov = sum(
            1 for r in rs if r.metrics.get("recovery_mode") == "unrecoverable"
        )
        latencies = [r.metrics.get("total_latency_s", 0) for r in rs]
        avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
        print(
            f"  {ft:<25} {len(rs):>5} {auto:>6} "
            f"{manual:>7} {unrecov:>8} {avg_lat:>15.2f}s"
        )

    # Phase 3 vs Phase 4 comparison hint
    print("\n--- Phase 3 vs Phase 4 comparison ---")
    print("Run the analysis stage to compare recovery modes with/without GCC constraints.")


if __name__ == "__main__":
    main()
