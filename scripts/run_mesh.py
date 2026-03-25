"""CLI entry point for the mesh sweep.

Usage:
    python scripts/run_mesh.py --config config/mesh.yaml
    python scripts/run_mesh.py --config config/mesh.yaml --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()  # Load .env before config resolves ${ENV_VAR} placeholders

from aip_intern.baseline.runner import RunConfig
from aip_intern.core.config import load_config
from aip_intern.mesh.runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 mesh sweep")
    parser.add_argument("--config", default="config/mesh.yaml")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run once for smoke test"
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

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

    n_label = "dry-run" if args.dry_run else f"{run_cfg.n_runs}-run"
    print(f"Starting {n_label} mesh sweep...")
    results = asyncio.run(run(run_cfg))
    passed = sum(r.success for r in results)
    print(f"\nComplete: {passed}/{len(results)} runs succeeded.")
    print(f"Artifacts: {run_cfg.artifacts_dir}/")


if __name__ == "__main__":
    main()
