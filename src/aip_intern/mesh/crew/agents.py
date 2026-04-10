"""CrewAI Agent definitions for the Phase 2 mesh.

Two specialists replace the 6-agent phase3 supervisor loop.
agents.py is correct here (not nodes.py) — CrewAI uses Agent objects.

These agents are assembled into a Crew in crew_node.py.
"""

from __future__ import annotations

from pathlib import Path

from crewai import LLM, Agent

_PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text().strip()


def make_triage_specialist(llm: LLM, tools: list) -> Agent:
    """Agent responsible for reading feedback files and writing triage.csv."""
    return Agent(
        role="Triage Specialist",
        goal="Classify all citizen feedback by category, urgency, and department owner",
        backstory=_load_prompt("triage_specialist.txt"),
        llm=llm,
        tools=tools,
        verbose=False,
        max_iter=10,
    )


def make_brief_response_specialist(llm: LLM, tools: list) -> Agent:
    """Agent responsible for writing brief.md and response_templates.md."""
    return Agent(
        role="Brief and Response Specialist",
        goal="Produce an action brief and category-based response templates",
        backstory=_load_prompt("brief_response_specialist.txt"),
        llm=llm,
        tools=tools,
        verbose=False,
        max_iter=10,
    )
