"""BaselineState — the LangGraph state schema for Phase 1.

This TypedDict is the interface contract between nodes.
Import state from here; never import the full graph just to get state.

Phase 3 interns: add failure tracking fields here if you need them in state.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class BaselineState(TypedDict):
    """Shared state passed between triage_node, brief_node, and response_node."""

    run_id: str
    task_description: str

    # Inputs (populated by runner before graph.ainvoke)
    feedback_files: list[str]     # list of relative paths, e.g. ["data/feedback/msg_001.txt"]
    policy_content: str           # raw contents of policy_snippets.md

    # Outputs (populated by nodes)
    triage_result: Optional[str]   # path to outputs/triage.csv (relative)
    brief_result: Optional[str]    # path to outputs/brief.md (relative)
    response_result: Optional[str] # path to outputs/response_templates.md (relative)

    # Error tracking
    error: Optional[str]           # non-None if any node raised AIPInternError

    # Observability
    step_trace: list[str]          # ordered list of node names visited, e.g. ["triage_node", ...]

    # Token estimation (tiktoken cl100k_base, accumulated across nodes)
    prompt_tokens: int
    completion_tokens: int
