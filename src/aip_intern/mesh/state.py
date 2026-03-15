"""MeshState — LangGraph state schema for Phase 2.

Extends BaselineState with mesh-specific observability fields.
These two extra fields (message_count, state_size_bytes) are the
primary measurements comparing Phase 1 vs Phase 2.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class MeshState(TypedDict):
    """State for the LangGraph + CrewAI mesh graph."""

    run_id: str
    task_description: str

    # Same outputs as baseline
    triage_result: Optional[str]
    brief_result: Optional[str]
    response_result: Optional[str]
    error: Optional[str]
    step_trace: list[str]

    # Mesh-specific observability (Phase 2 measurements)
    message_count: int      # inter-agent messages within CrewAI crew
    state_size_bytes: int   # serialised LangGraph state size at crew_node return
