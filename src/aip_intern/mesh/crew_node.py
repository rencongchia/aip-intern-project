"""crew_node — wraps a CrewAI Crew as a single async LangGraph node.

The supervisor_node routes unconditionally to this node (Phase 2).
crew_node calls crew.kickoff_async() and returns state updates.

CrewAI's kickoff_async() is used (not kickoff()) to avoid blocking
the async LangGraph execution loop.

After the crew finishes, output files are copied from workspace/outputs/
(where the tools write them) into the per-run artifacts directory so each
run's outputs are preserved independently.

Inter-agent message count is read from the CrewAI crew result.
State transfer size is measured by serialising the state dict before returning.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from aip_intern.core.exceptions import AIPInternError
from aip_intern.mesh.state import MeshState

if TYPE_CHECKING:
    from crewai import Crew


_OUTPUT_FILES = ["triage.csv", "brief.md", "response_templates.md"]


def _collect_outputs(workspace_root: Path, artifacts_outputs: Path) -> None:
    """Copy crew output files from workspace/outputs/ to artifacts/<run_id>/outputs/.

    CrewAI agents write to workspace/outputs/ via the write_file tool.
    The runner needs copies in artifacts/<run_id>/outputs/ so each run's
    outputs are preserved independently (rather than overwriting each other).
    """
    ws_outputs = workspace_root / "outputs"
    if not ws_outputs.exists():
        return
    artifacts_outputs.mkdir(parents=True, exist_ok=True)
    for fname in _OUTPUT_FILES:
        src = ws_outputs / fname
        if src.exists():
            shutil.copy2(src, artifacts_outputs / fname)


async def crew_node(
    state: MeshState,
    crew: "Crew",
    workspace_root: Path | None = None,
    artifacts_outputs: Path | None = None,
) -> dict:
    """Invoke a CrewAI Crew and write results back to LangGraph state.

    Args:
        state: Current MeshState.
        crew: Compiled CrewAI Crew (Triage Specialist + Brief+Response Specialist).
        workspace_root: Path to workspace directory (for collecting outputs).
        artifacts_outputs: Path to artifacts/<run_id>/outputs/ for this run.

    Returns:
        Dict of state updates including outputs, message_count, state_size_bytes.
    """
    try:
        result = await crew.kickoff_async(
            inputs={
                "run_id": state["run_id"],
                "task_description": state["task_description"],
            }
        )

        # Copy output files from workspace/outputs/ → artifacts/<run_id>/outputs/
        ws_root = workspace_root or Path("workspace/")
        if artifacts_outputs is not None:
            _collect_outputs(ws_root, artifacts_outputs)

        # Measure state transfer size (serialised state at this checkpoint)
        state_snapshot = {**state, "step_trace": state["step_trace"] + ["crew_node"]}
        state_size = len(json.dumps(state_snapshot, default=str).encode())

        # Inter-agent handoff count = number of completed crew tasks
        # (one per specialist hand-off: Triage → Brief+Response = 2 tasks)
        msg_count = (
            len(result.tasks_output)
            if hasattr(result, "tasks_output") and result.tasks_output
            else len(crew.tasks)
        )

        # Token usage — reported by the LLM provider (e.g. OpenRouter),
        # aggregated across all agent turns by CrewAI's UsageMetrics.
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(result, "token_usage") and result.token_usage:
            usage = result.token_usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        return {
            "triage_result": "outputs/triage.csv",
            "brief_result": "outputs/brief.md",
            "response_result": "outputs/response_templates.md",
            "message_count": msg_count,
            "state_size_bytes": state_size,
            "step_trace": state["step_trace"] + ["crew_node"],
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
    except AIPInternError:
        raise
    except Exception as e:
        return {
            "error": f"crew_node: {e}",
            "step_trace": state["step_trace"] + ["crew_node"],
            "state_size_bytes": 0,
        }
