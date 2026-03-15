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
