"""CrewAI Task definitions for the Phase 2 mesh.

Tasks describe the work and the workspace layout, but DO NOT pre-inject
file contents. Agents must use their tools (read_file, list_directory,
write_file) to discover and read inputs and to write outputs — that's the
inter-agent / tool-call traffic the mesh experiment is designed to measure.
"""

from __future__ import annotations

from crewai import Agent, Task

_WORKSPACE_LAYOUT = (
    "## Workspace layout (relative to workspace root)\n"
    "- Feedback messages: `data/feedback/msg_001.txt`, `data/feedback/msg_002.txt`, "
    "`data/feedback/msg_003.txt`\n"
    "- Policy guidelines: `data/policy_snippets.md`\n"
    "- Output directory: `outputs/`\n\n"
    "Use the file tools (`read_file`, `list_directory`, `write_file`) to read "
    "inputs and write outputs. Paths must be relative to workspace root.\n"
)


def make_triage_task(agent: Agent) -> Task:
    desc = (
        "Classify each citizen feedback item and produce a triage CSV.\n\n"
        "Output format (CSV with header):\n"
        "id,category,urgency,owner,summary,pii_flagged\n\n"
        "Categories: Infrastructure / Noise & Nuisance / Parks & Recreation / "
        "Public Safety\n"
        "Urgency: HIGH / MEDIUM / LOW\n\n"
        + _WORKSPACE_LAYOUT
        + "\nRead each feedback file and the policy with `read_file`, then write "
        "the final CSV to `outputs/triage.csv` using `write_file`.\n"
    )
    return Task(
        description=desc,
        expected_output="A CSV with one row per feedback item",
        agent=agent,
    )


def make_brief_task(
    agent: Agent,
    triage_task: Task | None = None,
) -> Task:
    desc = (
        "Using the triage results from the previous task, write:\n"
        "1. An action brief with: executive summary, urgent items, "
        "theme analysis, recommended actions, statistics\n"
        "2. Response templates: one per category with subject, opening, body, closing\n"
        "Use [PLACEHOLDER] for variable content.\n\n"
        + _WORKSPACE_LAYOUT
        + "\nWrite the brief to `outputs/brief.md` and the response templates to "
        "`outputs/response_templates.md` using `write_file`. If you need to "
        "re-read the policy or feedback for context, use `read_file`.\n"
    )
    return Task(
        description=desc,
        expected_output="Action brief and response templates as markdown text",
        agent=agent,
        context=[triage_task] if triage_task else [],
    )
