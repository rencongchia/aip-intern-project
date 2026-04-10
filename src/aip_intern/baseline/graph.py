"""LangGraph StateGraph for the Phase 1 baseline.

Graph topology:
    START → triage_node → brief_node → response_node → END

build_graph() returns a compiled graph ready for ainvoke().
The LLM, outputs_dir, and workspace_root are injected at construction time
so they can be swapped for mocks in tests.

Usage:
    graph = build_graph(llm, outputs_dir=Path("artifacts/run_1/outputs"),
                        workspace_root=Path("workspace/"))
    result = await graph.ainvoke(initial_state, config=invoke_config)
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from aip_intern.baseline.nodes import brief_node, response_node, triage_node
from aip_intern.baseline.state import BaselineState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def build_graph(
    llm: "BaseChatModel",
    tools: list | None = None,
    outputs_dir: Path | None = None,
    workspace_root: Path | None = None,
):
    """Build and compile the baseline StateGraph.

    Args:
        llm: Chat model instance (real or mock).
        tools: Ignored (kept for backward compatibility). Pass None.
        outputs_dir: Directory to write output files (triage.csv, brief.md, etc.).
        workspace_root: Path to workspace directory with input data.

    Returns:
        Compiled LangGraph ready for ainvoke().
    """
    builder = StateGraph(BaselineState)

    builder.add_node("triage_node", partial(
        triage_node, llm=llm, outputs_dir=outputs_dir, workspace_root=workspace_root,
    ))
    builder.add_node("brief_node", partial(
        brief_node, llm=llm, outputs_dir=outputs_dir, workspace_root=workspace_root,
    ))
    builder.add_node("response_node", partial(
        response_node, llm=llm, outputs_dir=outputs_dir, workspace_root=workspace_root,
    ))

    builder.add_edge(START, "triage_node")
    builder.add_edge("triage_node", "brief_node")
    builder.add_edge("brief_node", "response_node")
    builder.add_edge("response_node", END)

    return builder.compile()
