"""Recovery scoring — Phase 3 implementation.

Scoring rubric (from spec):
- recovery_time_s: time from exception raise to graph reaching END (or giving up)
- output_quality: 0.0–1.0, based on whether expected output files were produced
- recovery_mode: "automatic" (graph handled it) | "manual" (human intervention needed)

recovery_mode taxonomy:
- "automatic": AIPInternError was caught by the node itself (crew_node's except Exception
  block) and stored in state['error']; the graph reached END normally.
- "manual": AIPInternError propagated to the runner (runner's except AIPInternError block);
  graph did NOT reach END. Human must restart the run.
- "unrecoverable": reserved for Phase 4 GCC runs where even retry won't help (e.g.,
  failure cascades corrupt state permanently). Assigned post-hoc in Phase 4 comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_EXPECTED_FILES = ["triage.csv", "brief.md", "response_templates.md"]


@dataclass
class RecoveryScore:
    failure_type: str                        # e.g. "AgentTimeoutError"
    recovery_time_s: float                   # seconds from fault injection to END
    output_quality: float                    # 0.0 = no outputs, 1.0 = all 3 files present + non-empty
    recovery_mode: Literal["automatic", "manual", "unrecoverable"]
    notes: str = ""


def score_output_quality(outputs_path: Path) -> float:
    """Score output completeness: fraction of expected files produced and non-empty.

    Expected files: triage.csv, brief.md, response_templates.md
    Returns: 0.0, 0.33, 0.67, or 1.0
    """
    if not outputs_path.exists():
        return 0.0
    produced = sum(
        1 for fname in _EXPECTED_FILES
        if (outputs_path / fname).exists() and (outputs_path / fname).stat().st_size > 0
    )
    return round(produced / len(_EXPECTED_FILES), 2)


def score_recovery(
    failure_type: str,
    t_fault: float,
    t_end: float,
    outputs_path: Path,
    graph_reached_end: bool = False,
) -> RecoveryScore:
    """Score a single failure injection run.

    Args:
        failure_type: Name of the injected fault (e.g. "AgentTimeoutError").
        t_fault: Unix timestamp when the fault was injected (recorded by injector).
        t_end: Unix timestamp when the graph reached END (or gave up).
        outputs_path: Path to artifacts/{run_id}/outputs/.
        graph_reached_end: True if the graph completed normally (error stored in state);
            False if an AIPInternError propagated to the runner.

    Returns:
        RecoveryScore with recovery_time_s, output_quality, recovery_mode.
    """
    recovery_time_s = max(0.0, t_end - t_fault)
    output_quality = score_output_quality(outputs_path)

    if graph_reached_end:
        # Node caught the exception internally; graph reached END with degraded output
        recovery_mode: Literal["automatic", "manual", "unrecoverable"] = "automatic"
    else:
        # AIPInternError propagated to runner; human restart required
        recovery_mode = "manual"

    notes = ""
    if failure_type == "checkpoint_loss" and output_quality > 0:
        notes = (
            "Output files exist on disk but LangGraph state has no record — "
            "demonstrates checkpoint asymmetry: files written before state update lost."
        )

    return RecoveryScore(
        failure_type=failure_type,
        recovery_time_s=recovery_time_s,
        output_quality=output_quality,
        recovery_mode=recovery_mode,
        notes=notes,
    )
