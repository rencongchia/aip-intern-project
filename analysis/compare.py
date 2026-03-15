"""Baseline vs mesh metric comparison.

Used in Phase 2's 02_analysis.ipynb and Phase 3's failure analysis.

Usage:
    from analysis.aggregate import load_runs
    from analysis.compare import compare_phases

    baseline_df = load_runs("artifacts/", prefix="baseline")
    mesh_df = load_runs("artifacts/", prefix="mesh")
    summary = compare_phases(baseline_df, mesh_df)
    print(summary)
"""

from __future__ import annotations

import pandas as pd


def compare_phases(baseline: pd.DataFrame, mesh: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics comparing baseline vs mesh across key metrics.

    Returns a DataFrame with metrics as rows and (baseline, mesh, delta) as columns.
    Delta = mesh - baseline (positive = mesh is higher/slower/more expensive).
    """
    metrics = ["total_latency_s", "total_prompt_tokens", "total_completion_tokens"]
    rows = []
    for m in metrics:
        b_mean = baseline[m].mean() if len(baseline) else float("nan")
        m_mean = mesh[m].mean() if len(mesh) else float("nan")
        rows.append({
            "metric": m,
            "baseline_mean": round(b_mean, 3),
            "mesh_mean": round(m_mean, 3),
            "delta": round(m_mean - b_mean, 3),
        })
    # Error rate
    b_err = baseline["error"].notna().mean() if len(baseline) else float("nan")
    m_err = mesh["error"].notna().mean() if len(mesh) else float("nan")
    rows.append({
        "metric": "error_rate",
        "baseline_mean": round(b_err, 3),
        "mesh_mean": round(m_err, 3),
        "delta": round(m_err - b_err, 3),
    })
    return pd.DataFrame(rows).set_index("metric")
