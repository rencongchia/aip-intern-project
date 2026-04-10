"""Baseline LangGraph node functions.

Each node is a pure async function: (state, llm, tools, outputs_dir, workspace_root)
→ dict of state updates. Nodes can be tested independently using the mock_llm fixture.

Workspace files are read in Python and passed as context to the LLM. The LLM
generates text output which is written to disk by the node. No tool-calling
required — works reliably with any model including small local models.

Error handling: unexpected exceptions are caught and re-raised as AIPInternError
subclasses so Phase 3 can intercept and score them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from aip_intern.baseline.state import BaselineState
from aip_intern.core.exceptions import AIPInternError

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


def _estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken cl100k_base (approximate)."""
    if _ENC is None or not text:
        return 0
    return len(_ENC.encode(text))

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text().strip()


def _read_workspace_files(workspace_root: Path) -> str:
    """Read all feedback files and policy snippets from workspace."""
    lines = []

    feedback_dir = workspace_root / "data" / "feedback"
    if feedback_dir.exists():
        for f in sorted(feedback_dir.iterdir()):
            if f.is_file() and f.suffix == ".txt":
                lines.append(f"--- {f.name} ---")
                lines.append(f.read_text(errors="replace").strip())

    policy_file = workspace_root / "data" / "policy_snippets.md"
    if policy_file.exists():
        lines.append("--- policy_snippets.md ---")
        lines.append(policy_file.read_text(errors="replace").strip())

    return "\n\n".join(lines)


def _build_context(state: BaselineState, workspace_content: str = "") -> str:
    """Build a context string summarising available state for downstream nodes."""
    lines = [f"Task: {state['task_description']}"]
    if workspace_content:
        lines.append(f"\n## Workspace Files\n\n{workspace_content}")
    if state.get("triage_result"):
        lines.append(f"\n## Triage Output\n\n{state['triage_result']}")
    if state.get("brief_result"):
        lines.append(f"\n## Brief Output\n\n{state['brief_result']}")
    return "\n".join(lines)


async def triage_node(
    state: BaselineState,
    llm: "BaseChatModel",
    outputs_dir: Path | None = None,
    workspace_root: Path | None = None,
    **kwargs,
) -> dict:
    """Read feedback files and produce triage CSV.

    Returns a state update dict with triage_result (the CSV content) and step_trace.
    """
    try:
        ws_root = workspace_root or Path("workspace/")
        workspace_content = _read_workspace_files(ws_root)
        messages = [
            SystemMessage(content=_load_prompt("triage.txt")),
            HumanMessage(content=_build_context(state, workspace_content)),
        ]
        prompt_text = "\n".join(m.content for m in messages)
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        # Write output to disk if outputs_dir provided
        if outputs_dir and outputs_dir.exists():
            (outputs_dir / "triage.csv").write_text(content)

        return {
            "triage_result": content,
            "step_trace": state["step_trace"] + ["triage_node"],
            "prompt_tokens": _estimate_tokens(prompt_text),
            "completion_tokens": _estimate_tokens(content),
        }
    except AIPInternError:
        raise
    except Exception as e:
        return {
            "error": f"triage_node: {e}",
            "step_trace": state["step_trace"] + ["triage_node"],
        }


async def brief_node(
    state: BaselineState,
    llm: "BaseChatModel",
    outputs_dir: Path | None = None,
    workspace_root: Path | None = None,
    **kwargs,
) -> dict:
    """Generate action brief from triage data."""
    if not state.get("triage_result"):
        return {"error": "brief_node: triage_result not set — run triage_node first"}
    try:
        messages = [
            SystemMessage(content=_load_prompt("brief.txt")),
            HumanMessage(content=_build_context(state)),
        ]
        prompt_text = "\n".join(m.content for m in messages)
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        if outputs_dir and outputs_dir.exists():
            (outputs_dir / "brief.md").write_text(content)

        return {
            "brief_result": content,
            "step_trace": state["step_trace"] + ["brief_node"],
            "prompt_tokens": state.get("prompt_tokens", 0) + _estimate_tokens(prompt_text),
            "completion_tokens": state.get("completion_tokens", 0) + _estimate_tokens(content),
        }
    except AIPInternError:
        raise
    except Exception as e:
        return {
            "error": f"brief_node: {e}",
            "step_trace": state["step_trace"] + ["brief_node"],
        }


async def response_node(
    state: BaselineState,
    llm: "BaseChatModel",
    outputs_dir: Path | None = None,
    workspace_root: Path | None = None,
    **kwargs,
) -> dict:
    """Generate response templates from triage + brief data."""
    if not state.get("brief_result"):
        return {"error": "response_node: brief_result not set — run brief_node first"}
    try:
        messages = [
            SystemMessage(content=_load_prompt("response.txt")),
            HumanMessage(content=_build_context(state)),
        ]
        prompt_text = "\n".join(m.content for m in messages)
        response = await llm.ainvoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        if outputs_dir and outputs_dir.exists():
            (outputs_dir / "response_templates.md").write_text(content)

        return {
            "response_result": content,
            "step_trace": state["step_trace"] + ["response_node"],
            "prompt_tokens": state.get("prompt_tokens", 0) + _estimate_tokens(prompt_text),
            "completion_tokens": state.get("completion_tokens", 0) + _estimate_tokens(content),
        }
    except AIPInternError:
        raise
    except Exception as e:
        return {
            "error": f"response_node: {e}",
            "step_trace": state["step_trace"] + ["response_node"],
        }
