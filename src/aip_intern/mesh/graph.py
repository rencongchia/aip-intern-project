"""LangGraph StateGraph for the Phase 2 mesh.

Graph topology:
    START → supervisor_node → crew_node → END

build_graph() returns a compiled graph ready for ainvoke().

Usage:
    graph = build_graph(llm_cfg, workspace_root=Path("workspace/"))
    result = await graph.ainvoke(initial_state, config=invoke_config)
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from crewai import LLM, Crew, Process
from langgraph.graph import END, START, StateGraph

from aip_intern.mesh.crew.agents import (
    make_brief_response_specialist,
    make_triage_specialist,
)
from aip_intern.mesh.crew.tasks import make_brief_task, make_triage_task
from aip_intern.mesh.crew_node import crew_node
from aip_intern.mesh.nodes import supervisor_node
from aip_intern.mesh.state import MeshState
from aip_intern.mesh.tools import get_tools, set_workspace_root


def _build_crew(llm_cfg, workspace_root: Path) -> Crew:
    """Assemble the CrewAI Crew from agents and tasks.

    Agents are given workspace tools (read_file, write_file, list_directory)
    and must call them to access feedback files and write outputs. Workspace
    content is intentionally NOT pre-injected into task descriptions —
    measuring real tool-call traffic is the point of the mesh experiment.
    """
    set_workspace_root(workspace_root)
    tools = get_tools()

    crewai_llm = LLM(
        model=llm_cfg.model,
        base_url=llm_cfg.base_url,
        api_key=llm_cfg.api_key,
        temperature=llm_cfg.temperature,
    )

    triage_agent = make_triage_specialist(crewai_llm, tools)
    brief_agent = make_brief_response_specialist(crewai_llm, tools)
    triage_task = make_triage_task(triage_agent)
    brief_task = make_brief_task(brief_agent, triage_task)

    return Crew(
        agents=[triage_agent, brief_agent],
        tasks=[triage_task, brief_task],
        process=Process.sequential,
        verbose=False,
    )


def build_graph(
    llm_cfg,
    workspace_root: Path = Path("workspace/"),
    artifacts_outputs: Path | None = None,
):
    """Build and compile the mesh StateGraph.

    Args:
        llm_cfg: LLMCfg dataclass with model, base_url, api_key etc.
        workspace_root: Path to workspace directory for CrewAI tools.
        artifacts_outputs: Per-run artifacts/outputs/ dir; if provided,
            crew_node will copy output files there after the crew finishes.

    Returns:
        Compiled LangGraph ready for ainvoke().
    """
    crew = _build_crew(llm_cfg, workspace_root)

    builder = StateGraph(MeshState)
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node(
        "crew_node",
        partial(
            crew_node,
            crew=crew,
            workspace_root=workspace_root,
            artifacts_outputs=artifacts_outputs,
        ),
    )

    builder.add_edge(START, "supervisor_node")
    builder.add_edge("supervisor_node", "crew_node")
    builder.add_edge("crew_node", END)

    return builder.compile()
