"""Reusable plotting functions for all phases.

Notebooks import from here rather than re-implementing plot logic.

Usage:
    from analysis.plots import plot_latency_distribution, plot_phase_comparison
    fig = plot_latency_distribution(baseline_df, label="Baseline Phase 1")
    fig.savefig("latency.png")
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_latency_distribution(df: pd.DataFrame, label: str = "") -> plt.Figure:
    """Histogram of end-to-end latency across runs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["total_latency_s"].dropna(), bins=10, edgecolor="black")
    ax.set_xlabel("Latency (s)")
    ax.set_ylabel("Runs")
    ax.set_title(f"End-to-end latency distribution — {label}")
    fig.tight_layout()
    return fig


def plot_phase_comparison(summary: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing baseline vs mesh on each metric."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(summary))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x], summary["baseline_mean"], width, label="Baseline"
    )
    ax.bar(
        [i + width / 2 for i in x], summary["mesh_mean"], width, label="Mesh"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary.index, rotation=20, ha="right")
    ax.legend()
    ax.set_title("Phase 1 vs Phase 2 metric comparison")
    fig.tight_layout()
    return fig


def plot_token_cost(df: pd.DataFrame, label: str = "") -> plt.Figure:
    """Stacked bar of prompt vs completion tokens per run."""
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = range(len(df))
    ax.bar(idx, df["total_prompt_tokens"], label="Prompt tokens")
    ax.bar(
        idx,
        df["total_completion_tokens"],
        bottom=df["total_prompt_tokens"],
        label="Completion tokens",
    )
    ax.set_xlabel("Run")
    ax.set_ylabel("Tokens")
    ax.set_title(f"Token cost per run — {label}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_recovery_modes(df: pd.DataFrame, label: str = "") -> plt.Figure:
    """Stacked bar of recovery modes per fault type."""
    fault_types = df["failure_type"].unique()
    modes = ["automatic", "manual", "unrecoverable"]
    colors = {"automatic": "#4CAF50", "manual": "#FF9800", "unrecoverable": "#F44336"}

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = [0] * len(fault_types)
    for mode in modes:
        counts = [
            (df[df["failure_type"] == ft]["recovery_mode"] == mode).sum()
            for ft in fault_types
        ]
        ax.bar(fault_types, counts, bottom=bottom, label=mode, color=colors[mode])
        bottom = [b + c for b, c in zip(bottom, counts)]

    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Runs")
    ax.set_title(f"Recovery modes by fault type — {label}")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_fault_latency(df: pd.DataFrame, label: str = "") -> plt.Figure:
    """Box plot of latency per fault type."""
    fault_types = sorted(df["failure_type"].dropna().unique())
    data = [df[df["failure_type"] == ft]["total_latency_s"].dropna() for ft in fault_types]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data, labels=fault_types)
    ax.set_xlabel("Fault Type")
    ax.set_ylabel("Latency (s)")
    ax.set_title(f"Latency distribution by fault type — {label}")
    fig.tight_layout()
    return fig


def plot_phase3_vs_phase4(p3: pd.DataFrame, p4: pd.DataFrame) -> plt.Figure:
    """Grouped bar comparing Phase 3 vs Phase 4 avg latency per fault type."""
    fault_types = sorted(p3["failure_type"].dropna().unique())
    p3_lat = [p3[p3["failure_type"] == ft]["total_latency_s"].mean() for ft in fault_types]
    p4_lat = [p4[p4["failure_type"] == ft]["total_latency_s"].mean() for ft in fault_types]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(fault_types))
    width = 0.35
    ax.bar([i - width / 2 for i in x], p3_lat, width, label="Phase 3 (no GCC)")
    ax.bar([i + width / 2 for i in x], p4_lat, width, label="Phase 4 (GCC)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(fault_types, rotation=20, ha="right")
    ax.set_ylabel("Avg Latency (s)")
    ax.set_title("Phase 3 vs Phase 4 — Latency by fault type")
    ax.legend()
    fig.tight_layout()
    return fig
