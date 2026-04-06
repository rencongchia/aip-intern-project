"""Load and merge artifact run directories into a pandas DataFrame.

This is the primary data layer for notebooks. Run 01_run.ipynb to generate
artifacts/, then import this module in 02_analysis.ipynb.

Usage:
    import sys; sys.path.insert(0, "../../")  # repo root
    from analysis.aggregate import load_runs
    df = load_runs("artifacts/", prefix="baseline")
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_runs(artifacts_dir: str | Path, prefix: str = "") -> pd.DataFrame:
    """Load all metrics.json files from artifacts_dir matching the given prefix.

    Args:
        artifacts_dir: Path to the artifacts/ directory.
        prefix: Filter to run_ids starting with this prefix (e.g. "baseline", "mesh").
                Empty string loads all runs.

    Returns:
        DataFrame with one row per run. Columns: run_id, total_latency_s,
        total_prompt_tokens, total_completion_tokens, error, step_trace_len,
        message_count, state_size_bytes.
    """
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        return pd.DataFrame()
    rows = []
    for run_dir in sorted(artifacts_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if prefix and not run_dir.name.startswith(prefix):
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        m = json.loads(metrics_path.read_text())
        rows.append({
            "run_id": m["run_id"],
            "total_latency_s": m.get("total_latency_s", 0),
            "total_prompt_tokens": m.get("total_prompt_tokens", 0),
            "total_completion_tokens": m.get("total_completion_tokens", 0),
            "error": m.get("error"),
            "step_trace_len": len(m.get("step_trace", [])),
            "message_count": m.get("message_count"),
            "state_size_bytes": m.get("state_size_bytes"),
            # Phase 3 fault injection fields (None for Phase 1-2 runs)
            "failure_type": m.get("failure_type"),
            "recovery_time_s": m.get("recovery_time_s"),
            "output_quality": m.get("output_quality"),
            "recovery_mode": m.get("recovery_mode"),
            "notes": m.get("notes", ""),
            # Phase 4 GCC simulation flag
            "gcc_sim": m.get("gcc_sim", False),
        })
    return pd.DataFrame(rows)
